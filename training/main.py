"""
Побатчевое обучение бустингов из объединённого train_dataset.parquet.
Чтение идёт батчами через pyarrow + tqdm, без загрузки всего датасета в RAM.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_LGB_PATH, MODEL_XGB_PATH, TRAIN_DATASET_PATH, resolve_model_input_columns
from training.config import CATBOOST_PARAMS, LGBM_PARAMS, VAL_RATIO, XGB_PARAMS

logger = logging.getLogger(__name__)


def _prepare_batch(dfb: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dfb[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=False)
    y = pd.to_numeric(dfb["target"], errors="coerce").fillna(0).astype(np.int32).to_numpy(copy=False)
    w = pd.to_numeric(dfb["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32).to_numpy(copy=False)
    return x, y, w


def _detect_columns(path: Path) -> tuple[list[str], str]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    feature_cols = resolve_model_input_columns(names)
    if "event_dttm" not in names:
        raise ValueError("В датасете нет колонки event_dttm")
    if "target" not in names or "sample_weight" not in names:
        raise ValueError("В датасете нет target/sample_weight")
    return feature_cols, "event_dttm"


def _find_time_cutoff(path: Path, val_ratio: float, batch_size: int = 2_500_000) -> pd.Timestamp:
    """Определяем time-cutoff по дням без полной загрузки датасета."""
    pf = pq.ParquetFile(path)
    by_day: Counter[pd.Timestamp] = Counter()
    total = 0
    for rb in tqdm(pf.iter_batches(columns=["event_dttm"], batch_size=batch_size), desc="Скан дат", unit="batch"):
        s = pd.to_datetime(rb.column(0).to_pandas(), errors="coerce").dt.floor("D")
        vc = s.value_counts(dropna=True)
        for k, v in vc.items():
            by_day[k] += int(v)
            total += int(v)
    if total == 0:
        raise ValueError("Не удалось прочитать event_dttm для split по времени.")
    val_target = max(1, int(total * val_ratio))
    acc = 0
    cutoff = None
    for day, cnt in sorted(by_day.items(), reverse=True):
        acc += cnt
        cutoff = day
        if acc >= val_target:
            break
    assert cutoff is not None
    logger.info("Time split cutoff day: %s (val target rows ~= %d)", cutoff.date(), val_target)
    return cutoff


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. "
            "Сначала соберите датасет: ./dataset_cpp/build/build_dataset ."
        )
    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)
    pf = pq.ParquetFile(TRAIN_DATASET_PATH)

    results: list[tuple[str, float]] = []

    logger.warning("CatBoost отключен в streaming-режиме: нет стабильного побатчевого обучения без загрузки всего train в память.")

    # --- XGBoost: побатчевое дообучение ---
    try:
        import xgboost as xgb

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "eta": XGB_PARAMS.get("learning_rate", 0.05),
            "max_depth": XGB_PARAMS.get("max_depth", 8),
            "subsample": XGB_PARAMS.get("subsample", 0.8),
            "colsample_bytree": XGB_PARAMS.get("colsample_bytree", 0.8),
            "tree_method": XGB_PARAMS.get("tree_method", "hist"),
            "seed": XGB_PARAMS.get("random_state", 42),
        }
        booster = None
        rounds_per_batch = 1
        y_val_all: list[np.ndarray] = []
        p_val_all: list[np.ndarray] = []
        for rb in tqdm(
            pf.iter_batches(columns=feature_cols + ["target", "sample_weight", dttm_col], batch_size=2_000_000),
            desc="XGBoost stream fit",
            unit="batch",
        ):
            dfb = rb.to_pandas()
            dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
            train_mask = dttm < cutoff_day
            val_mask = ~train_mask

            if train_mask.any():
                xtr, ytr, wtr = _prepare_batch(dfb.loc[train_mask], feature_cols)
                dtrain = xgb.DMatrix(xtr, label=ytr, weight=wtr, feature_names=feature_cols)
                booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=rounds_per_batch, xgb_model=booster)

            if val_mask.any() and booster is not None:
                xva, yva, _ = _prepare_batch(dfb.loc[val_mask], feature_cols)
                dval = xgb.DMatrix(xva, label=yva, feature_names=feature_cols)
                p = booster.predict(dval)
                y_val_all.append(yva)
                p_val_all.append(p.astype(np.float32))

        if booster is None:
            raise RuntimeError("XGBoost не обучился: нет train-батчей после split.")
        pr = average_precision_score(np.concatenate(y_val_all), np.concatenate(p_val_all))
        results.append(("XGBoost", float(pr)))
        booster.save_model(str(MODEL_XGB_PATH))
        logger.info("XGBoost PR-AUC (val): %.6f → %s", pr, MODEL_XGB_PATH)
    except Exception as e:
        logger.warning("XGBoost пропущен: %s", e)

    # --- LightGBM: побатчевое дообучение ---
    try:
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "metric": "average_precision",
            "learning_rate": LGBM_PARAMS.get("learning_rate", 0.05),
            "num_leaves": LGBM_PARAMS.get("num_leaves", 64),
            "max_depth": LGBM_PARAMS.get("max_depth", -1),
            "feature_fraction": LGBM_PARAMS.get("colsample_bytree", 0.8),
            "bagging_fraction": LGBM_PARAMS.get("subsample", 0.8),
            "verbose": -1,
            "seed": LGBM_PARAMS.get("random_state", 42),
        }
        booster = None
        rounds_per_batch = 1
        y_val_all = []
        p_val_all = []
        for rb in tqdm(
            pf.iter_batches(columns=feature_cols + ["target", "sample_weight", dttm_col], batch_size=2_000_000),
            desc="LightGBM stream fit",
            unit="batch",
        ):
            dfb = rb.to_pandas()
            dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
            train_mask = dttm < cutoff_day
            val_mask = ~train_mask

            if train_mask.any():
                xtr, ytr, wtr = _prepare_batch(dfb.loc[train_mask], feature_cols)
                Xtr = pd.DataFrame(xtr, columns=feature_cols)
                dtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=True)
                booster = lgb.train(params, dtrain, num_boost_round=rounds_per_batch, init_model=booster, keep_training_booster=True)

            if val_mask.any() and booster is not None:
                xva, yva, _ = _prepare_batch(dfb.loc[val_mask], feature_cols)
                Xva = pd.DataFrame(xva, columns=feature_cols)
                p = booster.predict(Xva)
                y_val_all.append(yva)
                p_val_all.append(np.asarray(p, dtype=np.float32))

        if booster is None:
            raise RuntimeError("LightGBM не обучился: нет train-батчей после split.")
        pr = average_precision_score(np.concatenate(y_val_all), np.concatenate(p_val_all))
        results.append(("LightGBM", float(pr)))
        booster.save_model(str(MODEL_LGB_PATH))
        logger.info("LightGBM PR-AUC (val): %.6f → %s", pr, MODEL_LGB_PATH)
    except Exception as e:
        logger.warning("LightGBM пропущен: %s", e)

    logger.info("=== Сводка PR-AUC (валидация по времени) ===")
    for name, pr in sorted(results, key=lambda x: -x[1]):
        logger.info("  %s: %.6f", name, pr)
    if results:
        best = max(results, key=lambda x: x[1])
        logger.info("Лучшая по PR-AUC: %s (%.6f)", best[0], best[1])


if __name__ == "__main__":
    main()
