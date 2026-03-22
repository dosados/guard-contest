"""
Стриминговое обучение Random Forest на train_dataset.parquet.

Подход: проход по parquet батчами, на каждом train-батче добавляем несколько деревьев
в RandomForestClassifier(warm_start=True). После обучения отдельным проходом считаем
PR-AUC на временной валидации.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_RF_PATH, TRAIN_DATASET_PATH, resolve_model_input_columns
from training.config import RF_PARAMS, VAL_RATIO

logger = logging.getLogger(__name__)


def _prepare_batch(dfb: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dfb[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=False)
    y = pd.to_numeric(dfb["target"], errors="coerce").fillna(0).astype(np.int32).to_numpy(copy=False)
    if "sample_weight" in dfb.columns:
        w = pd.to_numeric(dfb["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32).to_numpy(copy=False)
    else:
        w = np.ones(shape=(len(dfb),), dtype=np.float32)
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

    trees_total = int(RF_PARAMS.get("n_estimators_total", 400))
    trees_per_batch = int(RF_PARAMS.get("trees_per_batch", 8))
    if trees_total <= 0 or trees_per_batch <= 0:
        raise ValueError("RF_PARAMS: n_estimators_total и trees_per_batch должны быть > 0")

    model = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        max_depth=RF_PARAMS.get("max_depth", None),
        max_features=RF_PARAMS.get("max_features", "sqrt"),
        min_samples_leaf=RF_PARAMS.get("min_samples_leaf", 1),
        n_jobs=RF_PARAMS.get("n_jobs", -1),
        random_state=RF_PARAMS.get("random_state", 42),
    )

    trees_added = 0
    train_rows = 0
    for rb in tqdm(
        pf.iter_batches(columns=feature_cols + ["target", "sample_weight", dttm_col], batch_size=2_000_000),
        desc="RandomForest stream fit",
        unit="batch",
    ):
        if trees_added >= trees_total:
            break
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        train_mask = dttm < cutoff_day
        if not train_mask.any():
            continue

        xtr, ytr, wtr = _prepare_batch(dfb.loc[train_mask], feature_cols)
        if xtr.shape[0] == 0:
            continue

        add_now = min(trees_per_batch, trees_total - trees_added)
        model.set_params(n_estimators=trees_added + add_now)
        Xtr = pd.DataFrame(xtr, columns=feature_cols)
        model.fit(Xtr, ytr, sample_weight=wtr)
        trees_added += add_now
        train_rows += xtr.shape[0]

    if trees_added == 0:
        raise RuntimeError("RandomForest не обучился: нет train-батчей после split.")

    logger.info("Обучено деревьев: %d (train rows processed: %d)", trees_added, train_rows)

    y_val_all: list[np.ndarray] = []
    p_val_all: list[np.ndarray] = []
    for rb in tqdm(
        pf.iter_batches(columns=feature_cols + ["target", dttm_col], batch_size=2_000_000),
        desc="RandomForest val predict",
        unit="batch",
    ):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        val_mask = dttm >= cutoff_day
        if not val_mask.any():
            continue
        xva, yva, _ = _prepare_batch(dfb.loc[val_mask], feature_cols)
        if xva.shape[0] == 0:
            continue
        Xva = pd.DataFrame(xva, columns=feature_cols)
        pr = model.predict_proba(Xva)[:, 1].astype(np.float32)
        y_val_all.append(yva)
        p_val_all.append(pr)

    if not y_val_all:
        raise RuntimeError("Валидация пуста: не удалось собрать val-батчи.")
    pr_auc = average_precision_score(np.concatenate(y_val_all), np.concatenate(p_val_all))
    logger.info("RandomForest PR-AUC (val): %.6f", pr_auc)

    MODEL_RF_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_RF_PATH)
    logger.info("Модель сохранена: %s", MODEL_RF_PATH)


if __name__ == "__main__":
    main()

