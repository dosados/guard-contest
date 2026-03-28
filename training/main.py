"""
Обучение XGBoost на объединённом full_dataset.parquet.
Сплит train/val по времени; батчи parquet через DataIter + ExtMemQuantileDMatrix (XGBoost 3.x):
квантили и промежуточные данные выносятся на диск (cache_prefix), без загрузки всего parquet в RAM.
При отсутствии ExtMemQuantileDMatrix — откат на QuantileDMatrix (может потребовать много RAM).
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from xgboost.core import DataIter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import (
    MODEL_XGB_PATH,
    MODEL_XGB_HIGH_TR_AMOUNT_PATH,
    MODEL_XGB_LOW_TR_AMOUNT_PATH,
    OUTPUT_DIR,
    TRAIN_DATASET_PATH,
    remap_sample_weight_from_dataset,
    resolve_model_input_columns,
)
from training.config import (
    VAL_RATIO,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_EVAL_VERBOSE_EVERY,
    XGB_EXTERNAL_PARQUET_BATCH_ROWS,
    XGB_MODEL_HYPERPARAMS,
    XGB_PARAMS,
)

logger = logging.getLogger(__name__)
TR_AMOUNT_SPLIT_THRESHOLD = 30.0


def _prepare_batch(
    dfb: pd.DataFrame,
    feature_cols: list[str],
    *,
    remap_weight2_positive_label_to_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dfb[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=False)
    y = pd.to_numeric(dfb["target"], errors="coerce").fillna(0).astype(np.int32).to_numpy(copy=False)
    w_raw = pd.to_numeric(dfb["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32).to_numpy(copy=False)
    if remap_weight2_positive_label_to_zero:
        y = y.copy()
        soft = (y == 1) & np.isclose(w_raw, np.float32(2.0), rtol=0.0, atol=1e-5)
        y[soft] = 0
    w = remap_sample_weight_from_dataset(w_raw)
    return x, y, w


def _detect_columns(path: Path) -> tuple[list[str], str]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    feature_cols = resolve_model_input_columns(names)
    if "event_dttm" not in names:
        raise ValueError("В датасете нет колонки event_dttm")
    if "target" not in names or "sample_weight" not in names:
        raise ValueError("В датасете нет target/sample_weight")
    if "tr_amount" not in feature_cols:
        raise ValueError("В датасете/фичах нет tr_amount, невозможно обучить двухсегментные модели.")
    return feature_cols, "event_dttm"


def _find_time_cutoff(path: Path, val_ratio: float, batch_size: int = 2_500_000) -> pd.Timestamp:
    """Определяем time-cutoff по дням без полной загрузки датасета (для grid search и др.)."""
    pf = pq.ParquetFile(path)
    by_day: Counter[pd.Timestamp] = Counter()
    total = 0
    for rb in pf.iter_batches(columns=["event_dttm"], batch_size=batch_size):
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


def _count_val_rows(
    path: Path,
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    tr_amount_col: str,
    tr_amount_lower_exclusive: float | None = None,
    tr_amount_upper_inclusive: float | None = None,
    batch_size: int = 2_500_000,
) -> int:
    """Число val-строк (дttm >= cutoff по маске ~train), один столбец — лёгкий скан."""
    pf = pq.ParquetFile(path)
    n_val = 0
    for rb in pf.iter_batches(columns=[dttm_col, tr_amount_col], batch_size=batch_size):
        dfb = rb.to_pandas()
        s = pd.to_datetime(dfb[dttm_col], errors="coerce")
        mask = ~(s < cutoff_day)
        tr_values = pd.to_numeric(dfb[tr_amount_col], errors="coerce")
        if tr_amount_lower_exclusive is not None:
            mask &= tr_values > float(tr_amount_lower_exclusive)
        if tr_amount_upper_inclusive is not None:
            mask &= tr_values <= float(tr_amount_upper_inclusive)
        n_val += int(mask.sum())
    return n_val


def _build_xgb_train_params(config_mode: str) -> dict[str, float | int | str]:
    defaults = {
        "learning_rate": float(XGB_PARAMS.get("learning_rate", 0.05)),
        "max_depth": int(XGB_PARAMS.get("max_depth", 8)),
        "subsample": float(XGB_PARAMS.get("subsample", 0.8)),
        "colsample_bytree": float(XGB_PARAMS.get("colsample_bytree", 0.8)),
        "min_child_weight": float(XGB_PARAMS.get("min_child_weight", 1.0)),
        "gamma": float(XGB_PARAMS.get("gamma", 0.0)),
        "reg_alpha": float(XGB_PARAMS.get("reg_alpha", 0.0)),
        "reg_lambda": float(XGB_PARAMS.get("reg_lambda", 1.0)),
    }
    selected = XGB_MODEL_HYPERPARAMS if config_mode == "best" else defaults

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": float(selected["learning_rate"]),
        "max_depth": int(selected["max_depth"]),
        "subsample": float(selected["subsample"]),
        "colsample_bytree": float(selected["colsample_bytree"]),
        "min_child_weight": float(selected["min_child_weight"]),
        "gamma": float(selected["gamma"]),
        "alpha": float(selected["reg_alpha"]),
        "lambda": float(selected["reg_lambda"]),
        "tree_method": str(XGB_PARAMS.get("tree_method", "hist")),
        "seed": int(XGB_PARAMS.get("random_state", 42)),
    }
    return params


def _xgb_predict_with_best_iteration(booster, dmatrix) -> np.ndarray:
    """
    После early stopping предсказание по лучшей итерации (иначе по умолчанию могут браться все деревья).
    Совместимость XGBoost 1.x (best_ntree_limit / ntree_limit) и 2.x (iteration_range).
    """
    bi = getattr(booster, "best_iteration", None)
    if bi is not None and bi >= 0:
        try:
            return np.asarray(
                booster.predict(dmatrix, iteration_range=(0, int(bi) + 1)),
                dtype=np.float32,
            )
        except TypeError:
            pass
    bnl = getattr(booster, "best_ntree_limit", None)
    if bnl is not None and bnl > 0:
        try:
            return np.asarray(
                booster.predict(dmatrix, iteration_range=(0, int(bnl))),
                dtype=np.float32,
            )
        except TypeError:
            return np.asarray(booster.predict(dmatrix, ntree_limit=int(bnl)), dtype=np.float32)
    return np.asarray(booster.predict(dmatrix), dtype=np.float32)


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class ParquetTrainValDataIter(DataIter):
    """Батчи parquet → только train или только val; reset перевычитывает файл с начала."""

    def __init__(
        self,
        parquet_path: Path,
        feature_cols: list[str],
        dttm_col: str,
        tr_amount_col: str,
        cutoff_day: pd.Timestamp,
        batch_rows: int,
        mode: Literal["train", "val"],
        *,
        cache_prefix: str | None,
        tr_amount_lower_exclusive: float | None = None,
        tr_amount_upper_inclusive: float | None = None,
        remap_weight2_positive_label_to_zero: bool = False,
    ) -> None:
        self._parquet_path = parquet_path
        self._feature_cols = feature_cols
        self._dttm_col = dttm_col
        self._tr_amount_col = tr_amount_col
        self._cutoff_day = cutoff_day
        self._batch_rows = batch_rows
        self._mode = mode
        self._tr_amount_lower_exclusive = tr_amount_lower_exclusive
        self._tr_amount_upper_inclusive = tr_amount_upper_inclusive
        self._remap_weight2_positive_label_to_zero = remap_weight2_positive_label_to_zero
        self._cols = feature_cols + ["target", "sample_weight", dttm_col]
        self._batch_iter = None
        if cache_prefix is not None:
            super().__init__(cache_prefix=cache_prefix, release_data=True)
        else:
            super().__init__()

    def reset(self) -> None:
        pf = pq.ParquetFile(self._parquet_path)
        self._batch_iter = iter(pf.iter_batches(columns=self._cols, batch_size=self._batch_rows))

    def next(self, input_data: Callable[..., None]) -> bool:
        train_mode = self._mode == "train"
        while True:
            try:
                rb = next(self._batch_iter)
            except StopIteration:
                return False
            dfb = rb.to_pandas()
            dttm = pd.to_datetime(dfb[self._dttm_col], errors="coerce")
            mask = dttm < self._cutoff_day if train_mode else ~(dttm < self._cutoff_day)
            tr_values = pd.to_numeric(dfb[self._tr_amount_col], errors="coerce")
            if self._tr_amount_lower_exclusive is not None:
                mask &= tr_values > float(self._tr_amount_lower_exclusive)
            if self._tr_amount_upper_inclusive is not None:
                mask &= tr_values <= float(self._tr_amount_upper_inclusive)
            if not mask.any():
                continue
            x, y, w = _prepare_batch(
                dfb.loc[mask],
                self._feature_cols,
                remap_weight2_positive_label_to_zero=self._remap_weight2_positive_label_to_zero,
            )
            input_data(
                data=x,
                label=y.astype(np.float32),
                weight=w,
                feature_names=self._feature_cols,
            )
            return True


def train_xgb_streaming_prauc(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    tr_amount_col: str,
    cutoff_day: pd.Timestamp,
    params: dict[str, float | int | str],
    *,
    ext_train_cache: Path,
    ext_val_cache: Path,
    tr_amount_lower_exclusive: float | None = None,
    tr_amount_upper_inclusive: float | None = None,
    segment_name: str = "all",
    num_boost_round: int | None = None,
    batch_rows: int | None = None,
    early_stopping_rounds: int | None = None,
    verbose_eval: int | None = None,
    remap_weight2_positive_label_to_zero: bool = False,
) -> tuple[float, Any, Callable[[], None]]:
    """
    Один прогон как в main: QuantileDMatrix / ExtMemQuantileDMatrix, eval + early stopping, PR-AUC (val) с sample_weight.
    При remap_weight2_positive_label_to_zero=True метки на train/val совпадают с флагом --xgb-remap-weight2-positives-as-zero.
    Возвращает (pr_auc, booster, cleanup). После save_model booster вызовите cleanup() (освобождает DMatrix и дисковый кэш).
    """
    import xgboost as xgb

    rounds = max(1, int(num_boost_round if num_boost_round is not None else XGB_PARAMS.get("n_estimators", 600)))
    br = int(batch_rows if batch_rows is not None else XGB_EXTERNAL_PARQUET_BATCH_ROWS)
    es = int(XGB_EARLY_STOPPING_ROUNDS if early_stopping_rounds is None else early_stopping_rounds)
    ve = int(XGB_EVAL_VERBOSE_EVERY if verbose_eval is None else verbose_eval)

    use_extmem = hasattr(xgb, "ExtMemQuantileDMatrix")
    if not use_extmem:
        logger.warning(
            "В этой сборке XGBoost нет ExtMemQuantileDMatrix — используется QuantileDMatrix "
            "(без дискового кэша итератора, возможен большой расход RAM)."
        )

    train_cp = str((ext_train_cache / "qdm").resolve()) if use_extmem else None
    val_cp = str((ext_val_cache / "qdm").resolve()) if use_extmem else None

    if use_extmem:
        _rm_tree(ext_train_cache)
        _rm_tree(ext_val_cache)
        ext_train_cache.mkdir(parents=True, exist_ok=True)
        ext_val_cache.mkdir(parents=True, exist_ok=True)
        logger.info(
            "XGBoost: ExtMemQuantileDMatrix train (batch_rows=%d, disk cache=%s) …",
            br,
            ext_train_cache,
        )
    else:
        logger.info("XGBoost: QuantileDMatrix train (batch_rows=%d) …", br)

    train_it = ParquetTrainValDataIter(
        parquet_path,
        feature_cols,
        dttm_col,
        tr_amount_col,
        cutoff_day,
        br,
        "train",
        cache_prefix=train_cp,
        tr_amount_lower_exclusive=tr_amount_lower_exclusive,
        tr_amount_upper_inclusive=tr_amount_upper_inclusive,
        remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero,
    )
    if use_extmem:
        dtrain = xgb.ExtMemQuantileDMatrix(train_it, missing=np.nan, enable_categorical=False)
    else:
        dtrain = xgb.QuantileDMatrix(train_it)

    train_kw: dict = {"params": params, "dtrain": dtrain, "num_boost_round": rounds}
    dval = None
    yva = None
    val_it = None

    logger.info("Подсчёт val-строк по event_dttm + tr_amount (сегмент: %s) …", segment_name)
    n_val = _count_val_rows(
        parquet_path,
        dttm_col,
        cutoff_day,
        tr_amount_col=tr_amount_col,
        tr_amount_lower_exclusive=tr_amount_lower_exclusive,
        tr_amount_upper_inclusive=tr_amount_upper_inclusive,
        batch_size=br,
    )
    logger.info("Val rows (оценка по time split, сегмент %s): %d", segment_name, n_val)

    if n_val > 0:
        if use_extmem:
            logger.info("XGBoost: ExtMemQuantileDMatrix eval (disk cache=%s, ref=train) …", ext_val_cache)
        else:
            logger.info("XGBoost: QuantileDMatrix eval …")
        val_it = ParquetTrainValDataIter(
            parquet_path,
            feature_cols,
            dttm_col,
            tr_amount_col,
            cutoff_day,
            br,
            "val",
            cache_prefix=val_cp,
            tr_amount_lower_exclusive=tr_amount_lower_exclusive,
            tr_amount_upper_inclusive=tr_amount_upper_inclusive,
            remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero,
        )
        if use_extmem:
            dval = xgb.ExtMemQuantileDMatrix(
                val_it,
                missing=np.nan,
                enable_categorical=False,
                ref=dtrain,
            )
        else:
            dval = xgb.QuantileDMatrix(val_it, ref=dtrain)
        train_kw["evals"] = [(dval, "eval")]
        train_kw["verbose_eval"] = ve if ve > 0 else False
        if es > 0:
            train_kw["early_stopping_rounds"] = es
        yva = np.asarray(dval.get_label(), dtype=np.float32)

    booster = xgb.train(**train_kw)

    pr = 0.0
    if dval is not None and yva is not None:
        p = _xgb_predict_with_best_iteration(booster, dval)
        y_int = yva.astype(np.int32)
        try:
            w_eval = np.asarray(dval.get_weight(), dtype=np.float64)
            if w_eval.size == y_int.size:
                pr = float(average_precision_score(y_int, p, sample_weight=w_eval))
                logger.info("PR-AUC (val): sklearn с sample_weight как в DMatrix eval")
            else:
                pr = float(average_precision_score(y_int, p))
                logger.warning("PR-AUC (val): sklearn без весов (размер weight не совпал с label)")
        except Exception:
            pr = float(average_precision_score(y_int, p))
            logger.warning("PR-AUC (val): sklearn без весов (get_weight недоступен)")
        bi = getattr(booster, "best_iteration", None)
        if bi is not None and bi >= 0:
            logger.info("XGBoost best_iteration (лучший aucpr на eval по логу train): %s", bi)
    else:
        logger.warning("XGBoost: нет val-строк — PR-AUC не считается.")

    def cleanup() -> None:
        if not use_extmem:
            return
        nonlocal booster, dtrain, train_it, dval, val_it
        booster = None
        dtrain = None
        train_it = None
        if n_val > 0:
            dval = None
            val_it = None
        gc.collect()
        _rm_tree(ext_train_cache)
        _rm_tree(ext_val_cache)
        logger.info("Дисковый кэш ExtMemQuantileDMatrix удалён (%s, %s).", ext_train_cache.name, ext_val_cache.name)

    return pr, booster, cleanup


def evaluate_segmented_val_prauc(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    low_booster: Any,
    high_booster: Any,
    *,
    tr_amount_col: str = "tr_amount",
    threshold: float = TR_AMOUNT_SPLIT_THRESHOLD,
    batch_size: int = 1_000_000,
    remap_weight2_positive_label_to_zero: bool = False,
) -> float:
    """Единый PR-AUC на всей val: запись роутится в low/high модель по tr_amount, как в submission."""
    import xgboost as xgb

    pf = pq.ParquetFile(parquet_path)
    columns = feature_cols + ["target", "sample_weight", dttm_col, tr_amount_col]

    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    total_rows = 0

    for rb in pf.iter_batches(columns=columns, batch_size=batch_size):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        val_mask = ~(dttm < cutoff_day)
        if not bool(val_mask.any()):
            continue
        val_df = dfb.loc[val_mask].copy()
        if val_df.empty:
            continue

        tr_values = pd.to_numeric(val_df[tr_amount_col], errors="coerce")
        low_mask = tr_values <= float(threshold)
        high_mask = ~low_mask

        y = pd.to_numeric(val_df["target"], errors="coerce").fillna(0).astype(np.int32).to_numpy(copy=False)
        w_raw = pd.to_numeric(val_df["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32).to_numpy(
            copy=False
        )
        if remap_weight2_positive_label_to_zero:
            y = y.copy()
            soft = (y == 1) & np.isclose(w_raw, np.float32(2.0), rtol=0.0, atol=1e-5)
            y[soft] = 0
        w = remap_sample_weight_from_dataset(w_raw).astype(np.float64, copy=False)
        p = np.zeros(len(val_df), dtype=np.float32)

        if bool(low_mask.any()):
            x_low = (
                val_df.loc[low_mask, feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=False)
            )
            d_low = xgb.DMatrix(x_low, feature_names=feature_cols)
            p[low_mask.to_numpy()] = _xgb_predict_with_best_iteration(low_booster, d_low)
        if bool(high_mask.any()):
            x_high = (
                val_df.loc[high_mask, feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=False)
            )
            d_high = xgb.DMatrix(x_high, feature_names=feature_cols)
            p[high_mask.to_numpy()] = _xgb_predict_with_best_iteration(high_booster, d_high)

        y_all.append(y)
        p_all.append(p.astype(np.float64, copy=False))
        w_all.append(w)
        total_rows += len(val_df)

    if total_rows == 0:
        logger.warning("Segmented PR-AUC: валидационные строки не найдены.")
        return 0.0

    y_concat = np.concatenate(y_all, axis=0)
    p_concat = np.concatenate(p_all, axis=0)
    w_concat = np.concatenate(w_all, axis=0)
    pr = float(average_precision_score(y_concat, p_concat, sample_weight=w_concat))
    logger.info("PR-AUC (val, routed как в submission): %.6f на %d строках", pr, total_rows)
    return pr


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xgb-config",
        choices=("best", "default"),
        default="best",
        help="best (по умолчанию): гиперпараметры из training/grid_search/xgb_best_params.json; "
        "default: базовые XGB_PARAMS из training/config.py",
    )
    parser.add_argument(
        "--xgb-segmented",
        action="store_true",
        help=(
            "Включить двухмодельный режим XGBoost по tr_amount "
            f"(<= {int(TR_AMOUNT_SPLIT_THRESHOLD)} и > {int(TR_AMOUNT_SPLIT_THRESHOLD)}). "
            "По умолчанию обучается одна модель в legacy-путь."
        ),
    )
    parser.add_argument(
        "--xgb-remap-weight2-positives-as-zero",
        action="store_true",
        help=(
            "При обучении XGBoost: target=0 без изменений; target=1 с sample_weight=2 (как в parquet, до remap) "
            "подавать в модель как класс 0; target=1 с sample_weight=5 остаётся классом 1. "
            "Веса обучения по-прежнему через remap_sample_weight_from_dataset."
        ),
    )
    args = parser.parse_args()

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. "
            "Сначала соберите датасет: ./dataset_cpp/build/build_dataset ."
        )
    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)

    results: list[tuple[str, float]] = []

    logger.warning("CatBoost в этом скрипте не обучается — отдельный пайплайн.")

    try:
        params = _build_xgb_train_params(args.xgb_config)
        rounds = max(1, int(XGB_PARAMS.get("n_estimators", 600)))
        logger.info("XGBoost config mode: %s", args.xgb_config)
        if args.xgb_remap_weight2_positives_as_zero:
            logger.info(
                "XGBoost: включено переназначение меток — target=1 и sample_weight=2 → label 0 при обучении/val-метриках."
            )
        logger.info("XGBoost train params: %s, num_boost_round=%d", params, rounds)

        if args.xgb_segmented:
            segments = [
                {
                    "name": f"tr_amount<={int(TR_AMOUNT_SPLIT_THRESHOLD)}",
                    "tr_amount_lower_exclusive": None,
                    "tr_amount_upper_inclusive": TR_AMOUNT_SPLIT_THRESHOLD,
                    "model_path": MODEL_XGB_LOW_TR_AMOUNT_PATH,
                    "ext_train_cache": OUTPUT_DIR / "xgb_extmem_train_tr_amount_le_30",
                    "ext_val_cache": OUTPUT_DIR / "xgb_extmem_val_tr_amount_le_30",
                },
                {
                    "name": f"tr_amount>{int(TR_AMOUNT_SPLIT_THRESHOLD)}",
                    "tr_amount_lower_exclusive": TR_AMOUNT_SPLIT_THRESHOLD,
                    "tr_amount_upper_inclusive": None,
                    "model_path": MODEL_XGB_HIGH_TR_AMOUNT_PATH,
                    "ext_train_cache": OUTPUT_DIR / "xgb_extmem_train_tr_amount_gt_30",
                    "ext_val_cache": OUTPUT_DIR / "xgb_extmem_val_tr_amount_gt_30",
                },
            ]

            trained: dict[str, Any] = {}
            cleanups: list[Callable[[], None]] = []
            for seg in segments:
                logger.info("=== Обучение XGBoost сегмента: %s ===", seg["name"])
                pr, booster, cleanup = train_xgb_streaming_prauc(
                    TRAIN_DATASET_PATH,
                    feature_cols,
                    dttm_col,
                    "tr_amount",
                    cutoff_day,
                    params,
                    ext_train_cache=seg["ext_train_cache"],
                    ext_val_cache=seg["ext_val_cache"],
                    tr_amount_lower_exclusive=seg["tr_amount_lower_exclusive"],
                    tr_amount_upper_inclusive=seg["tr_amount_upper_inclusive"],
                    segment_name=seg["name"],
                    num_boost_round=rounds,
                    remap_weight2_positive_label_to_zero=args.xgb_remap_weight2_positives_as_zero,
                )
                results.append((f"XGBoost ({seg['name']})", pr))
                booster.save_model(str(seg["model_path"]))
                logger.info("XGBoost PR-AUC (val, %s): %.6f → %s", seg["name"], pr, seg["model_path"])
                trained[seg["name"]] = booster
                cleanups.append(cleanup)

            low_name = f"tr_amount<={int(TR_AMOUNT_SPLIT_THRESHOLD)}"
            high_name = f"tr_amount>{int(TR_AMOUNT_SPLIT_THRESHOLD)}"
            if low_name in trained and high_name in trained:
                pr_routed = evaluate_segmented_val_prauc(
                    TRAIN_DATASET_PATH,
                    feature_cols,
                    dttm_col,
                    cutoff_day,
                    trained[low_name],
                    trained[high_name],
                    tr_amount_col="tr_amount",
                    threshold=TR_AMOUNT_SPLIT_THRESHOLD,
                    batch_size=XGB_EXTERNAL_PARQUET_BATCH_ROWS,
                    remap_weight2_positive_label_to_zero=args.xgb_remap_weight2_positives_as_zero,
                )
                results.append(("XGBoost (val routed all rows)", pr_routed))

            for cleanup in cleanups:
                cleanup()

            logger.info(
                "Одиночный путь модели не перезаписывается: %s (двухсегментные веса сохранены отдельно).",
                MODEL_XGB_PATH,
            )
        else:
            logger.info("=== Обучение XGBoost (одна модель, legacy default) ===")
            pr, booster, cleanup = train_xgb_streaming_prauc(
                TRAIN_DATASET_PATH,
                feature_cols,
                dttm_col,
                "tr_amount",
                cutoff_day,
                params,
                ext_train_cache=OUTPUT_DIR / "xgb_extmem_train_single",
                ext_val_cache=OUTPUT_DIR / "xgb_extmem_val_single",
                segment_name="all",
                num_boost_round=rounds,
                remap_weight2_positive_label_to_zero=args.xgb_remap_weight2_positives_as_zero,
            )
            results.append(("XGBoost (single model)", pr))
            booster.save_model(str(MODEL_XGB_PATH))
            logger.info("XGBoost PR-AUC (val, single): %.6f → %s", pr, MODEL_XGB_PATH)
            cleanup()

    except Exception as e:
        logger.warning("XGBoost пропущен: %s", e)
        _rm_tree(OUTPUT_DIR / "xgb_extmem_train_single")
        _rm_tree(OUTPUT_DIR / "xgb_extmem_val_single")
        _rm_tree(OUTPUT_DIR / "xgb_extmem_train_tr_amount_le_30")
        _rm_tree(OUTPUT_DIR / "xgb_extmem_val_tr_amount_le_30")
        _rm_tree(OUTPUT_DIR / "xgb_extmem_train_tr_amount_gt_30")
        _rm_tree(OUTPUT_DIR / "xgb_extmem_val_tr_amount_gt_30")

    logger.info("=== Сводка PR-AUC (валидация по времени) ===")
    for name, pr in sorted(results, key=lambda x: -x[1]):
        logger.info("  %s: %.6f", name, pr)
    if results:
        best = max(results, key=lambda x: x[1])
        logger.info("Лучшая по PR-AUC: %s (%.6f)", best[0], best[1])


if __name__ == "__main__":
    main()
