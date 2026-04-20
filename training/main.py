from __future__ import annotations

import os
import gc
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from xgboost.core import DataIter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import (
    MODEL_XGB_PATH,
    TRAIN_DATASET_PATH,
    XGB_EXTMEM_TRAIN_SINGLE_DIR,
    XGB_EXTMEM_VAL_SINGLE_DIR,
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

# full_dataset.parquet (C++ build_dataset): event_dttm as utf8, e.g. "2024-10-01 05:29:14"
_EVENT_DTTM_STRING_FORMAT = "%Y-%m-%d %H:%M:%S"
_EVENT_DTTM_TS_TYPE = pa.timestamp("s")


def _pd_timestamp_cutoff_to_arrow_scalar_ts_s(cutoff_day: pd.Timestamp) -> pa.Scalar:
    ts = cutoff_day
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return pa.scalar(ts.to_pydatetime(), type=_EVENT_DTTM_TS_TYPE)


def _event_dttm_array_to_timestamp_seconds(arr: pa.Array) -> pa.Array:
    # event_dttm → timestamp[s] for pyarrow.compute; invalid → null
    if len(arr) == 0:
        return pa.array([], type=_EVENT_DTTM_TS_TYPE)
    if pa.types.is_dictionary(arr.type):
        arr = pc.dictionary_decode(arr)
    t = arr.type
    if pa.types.is_timestamp(t):
        return pc.cast(arr, _EVENT_DTTM_TS_TYPE)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pc.strptime(arr, format=_EVENT_DTTM_STRING_FORMAT, unit="s")
    logger.warning("event_dttm: unexpected type %s, parsing via pandas", t)
    p = pd.to_datetime(arr.to_pandas(), errors="coerce")
    return pc.cast(pa.array(p, from_pandas=True), _EVENT_DTTM_TS_TYPE)


def _train_val_mask_from_event_dttm(
    dttm_arr: pa.Array,
    cutoff_ts_scalar: pa.Scalar,
    *,
    train_mode: bool,
) -> np.ndarray:
    # train: dttm < cutoff; val: else
    ts = _event_dttm_array_to_timestamp_seconds(dttm_arr)
    lt = pc.less(ts, cutoff_ts_scalar)
    lt_pd = pc.fill_null(lt, False)
    before = np.asarray(lt_pd, dtype=np.bool_)
    return before if train_mode else ~before


def _arrow_array_to_float64(arr: pa.Array) -> np.ndarray:
    # Arrow column → float64
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    t = arr.type
    if pa.types.is_dictionary(t):
        arr = pc.dictionary_decode(arr)
        t = arr.type
    if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
        v = pc.cast(arr, pa.float64())
        return np.asarray(v.to_numpy(zero_copy_only=False), dtype=np.float64)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        try:
            v = pc.cast(arr, pa.float64())
            return np.asarray(v.to_numpy(zero_copy_only=False), dtype=np.float64)
        except pa.ArrowInvalid:
            pass
    s = pd.to_numeric(pd.Series(arr.to_pandas(), dtype=object), errors="coerce")
    return s.to_numpy(dtype=np.float64, na_value=np.nan)


def _finalize_yw(
    y_f: np.ndarray,
    w_raw: np.ndarray,
    *,
    remap_weight2_positive_label_to_zero: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.nan_to_num(y_f, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.rint(y).astype(np.int32, copy=False)
    w_raw = np.nan_to_num(w_raw, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)
    if remap_weight2_positive_label_to_zero:
        y = y.copy()
        soft = (y == 1) & np.isclose(w_raw, np.float32(2.0), rtol=0.0, atol=1e-5)
        y[soft] = 0
    w = remap_sample_weight_from_dataset(w_raw)
    return y, w


def _prepare_batch_from_recordbatch(
    rb: pa.RecordBatch,
    mask: np.ndarray,
    nfeat: int,
    j_target: int,
    j_sw: int,
    *,
    remap_weight2_positive_label_to_zero: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask)
    rb_sub = rb.take(pa.array(idx, type=pa.int32()))
    cols = [_arrow_array_to_float64(rb_sub.column(j)) for j in range(nfeat)]
    x = np.column_stack(cols).astype(np.float32, copy=False)
    y_f = _arrow_array_to_float64(rb_sub.column(j_target))
    w_f = _arrow_array_to_float64(rb_sub.column(j_sw))
    y, w = _finalize_yw(y_f, w_f, remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero)
    return x, y, w


def _detect_columns(path: Path) -> tuple[list[str], str]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    feature_cols = resolve_model_input_columns(names)
    if "event_dttm" not in names:
        raise ValueError("Dataset has no event_dttm column")
    if "target" not in names or "sample_weight" not in names:
        raise ValueError("Dataset has no target/sample_weight columns")
    return feature_cols, "event_dttm"


def _find_time_cutoff(path: Path, val_ratio: float, batch_size: int = 2_500_000) -> pd.Timestamp:
    # time cutoff by day without loading full dataset
    pf = pq.ParquetFile(path)
    by_day: Counter[pd.Timestamp] = Counter()
    total = 0
    for rb in pf.iter_batches(columns=["event_dttm"], batch_size=batch_size):
        ts = _event_dttm_array_to_timestamp_seconds(rb.column(0))
        floored = pc.floor_temporal(ts, unit="day")
        vc_struct = pc.value_counts(floored)
        vals = vc_struct.field(0)
        cnts = vc_struct.field(1)
        for v, c in zip(vals.to_pylist(), cnts.to_pylist()):
            if v is None:
                continue
            day = pd.Timestamp(v).normalize()
            by_day[day] += int(c)
            total += int(c)
    if total == 0:
        raise ValueError("Could not read event_dttm for time-based split.")
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
    batch_size: int = 2_500_000,
) -> int:
    # count val rows (dttm >= cutoff)
    pf = pq.ParquetFile(path)
    n_val = 0
    cutoff_scalar = _pd_timestamp_cutoff_to_arrow_scalar_ts_s(cutoff_day)
    for rb in pf.iter_batches(columns=[dttm_col], batch_size=batch_size):
        m = _train_val_mask_from_event_dttm(rb.column(0), cutoff_scalar, train_mode=False)
        n_val += int(m.sum())
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
    def __init__(
        self,
        parquet_path: Path,
        feature_cols: list[str],
        dttm_col: str,
        cutoff_day: pd.Timestamp,
        batch_rows: int,
        mode: Literal["train", "val"],
        *,
        cache_prefix: str | None,
        remap_weight2_positive_label_to_zero: bool = False,
    ) -> None:
        self._parquet_path = parquet_path
        self._feature_cols = feature_cols
        self._dttm_col = dttm_col
        self._batch_rows = batch_rows
        self._mode = mode
        self._remap_weight2_positive_label_to_zero = remap_weight2_positive_label_to_zero
        self._cols = feature_cols + ["target", "sample_weight", dttm_col]
        self._nfeat = len(feature_cols)
        self._j_target = self._nfeat
        self._j_sw = self._nfeat + 1
        self._j_dttm = self._nfeat + 2
        self._batch_iter = None
        self._cutoff_ts_scalar = _pd_timestamp_cutoff_to_arrow_scalar_ts_s(cutoff_day)
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
            if rb.num_rows == 0:
                continue
            mask = _train_val_mask_from_event_dttm(
                rb.column(self._j_dttm),
                self._cutoff_ts_scalar,
                train_mode=train_mode,
            )
            if not mask.any():
                continue
            x, y, w = _prepare_batch_from_recordbatch(
                rb,
                mask,
                self._nfeat,
                self._j_target,
                self._j_sw,
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
    cutoff_day: pd.Timestamp,
    params: dict[str, float | int | str],
    *,
    ext_train_cache: Path,
    ext_val_cache: Path,
    num_boost_round: int | None = None,
    batch_rows: int | None = None,
    early_stopping_rounds: int | None = None,
    verbose_eval: int | None = None,
    remap_weight2_positive_label_to_zero: bool = False,
) -> tuple[float, Any, Callable[[], None]]:
    import xgboost as xgb

    rounds = max(1, int(num_boost_round if num_boost_round is not None else XGB_PARAMS.get("n_estimators", 600)))
    br = int(batch_rows if batch_rows is not None else XGB_EXTERNAL_PARQUET_BATCH_ROWS)
    if batch_rows is None and os.environ.get("GUARD_XGB_PARQUET_BATCH_ROWS"):
        br = max(50_000, int(os.environ["GUARD_XGB_PARQUET_BATCH_ROWS"]))
    es = int(XGB_EARLY_STOPPING_ROUNDS if early_stopping_rounds is None else early_stopping_rounds)
    ve = int(XGB_EVAL_VERBOSE_EVERY if verbose_eval is None else verbose_eval)

    use_extmem = hasattr(xgb, "ExtMemQuantileDMatrix")
    if not use_extmem:
        logger.warning(
            "This XGBoost build has no ExtMemQuantileDMatrix - using QuantileDMatrix "
            "(no on-disk iterator cache; high RAM use possible)."
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
        cutoff_day,
        br,
        "train",
        cache_prefix=train_cp,
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

    logger.info("Counting val rows by event_dttm …")
    n_val = _count_val_rows(parquet_path, dttm_col, cutoff_day, batch_size=br)
    logger.info("Val rows (time-split estimate): %d", n_val)

    if n_val > 0:
        if use_extmem:
            logger.info("XGBoost: ExtMemQuantileDMatrix eval (disk cache=%s, ref=train) …", ext_val_cache)
        else:
            logger.info("XGBoost: QuantileDMatrix eval …")
        val_it = ParquetTrainValDataIter(
            parquet_path,
            feature_cols,
            dttm_col,
            cutoff_day,
            br,
            "val",
            cache_prefix=val_cp,
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
                logger.info("PR-AUC (val): sklearn with sample_weight like DMatrix eval")
            else:
                pr = float(average_precision_score(y_int, p))
                logger.warning("PR-AUC (val): sklearn without weights (weight size mismatch vs label)")
        except Exception:
            pr = float(average_precision_score(y_int, p))
            logger.warning("PR-AUC (val): sklearn without weights (get_weight unavailable)")
        bi = getattr(booster, "best_iteration", None)
        if bi is not None and bi >= 0:
            logger.info("XGBoost best_iteration (best aucpr on eval from train log): %s", bi)
    else:
        logger.warning("XGBoost: no val rows - PR-AUC not computed.")

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
        logger.info("ExtMemQuantileDMatrix disk cache removed (%s, %s).", ext_train_cache.name, ext_val_cache.name)

    return pr, booster, cleanup


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    xgb_config = "best"
    xgb_remap_weight2_positives_as_zero = False

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Not found: {TRAIN_DATASET_PATH}. "
            "Build the dataset first (conda env guard-cpp): build_dataset -> output/datasets/train/full_dataset.parquet"
        )
    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    MODEL_XGB_PATH.parent.mkdir(parents=True, exist_ok=True)
    XGB_EXTMEM_TRAIN_SINGLE_DIR.parent.mkdir(parents=True, exist_ok=True)
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)

    results: list[tuple[str, float]] = []

    try:
        params = _build_xgb_train_params(xgb_config)
        rounds = max(1, int(XGB_PARAMS.get("n_estimators", 600)))
        if os.environ.get("GUARD_XGB_NUM_BOOST_ROUND"):
            rounds = max(1, int(os.environ["GUARD_XGB_NUM_BOOST_ROUND"]))
        logger.info("XGBoost config mode: %s", xgb_config)
        if xgb_remap_weight2_positives_as_zero:
            logger.info(
                "XGBoost: label remap enabled - target=1 and sample_weight=2 -> label 0 for train/val metrics."
            )
        logger.info("XGBoost train params: %s, num_boost_round=%d", params, rounds)

        logger.info("=== Training XGBoost ===")
        pr, booster, cleanup = train_xgb_streaming_prauc(
            TRAIN_DATASET_PATH,
            feature_cols,
            dttm_col,
            cutoff_day,
            params,
            ext_train_cache=XGB_EXTMEM_TRAIN_SINGLE_DIR,
            ext_val_cache=XGB_EXTMEM_VAL_SINGLE_DIR,
            num_boost_round=rounds,
            remap_weight2_positive_label_to_zero=xgb_remap_weight2_positives_as_zero,
        )
        results.append(("XGBoost", pr))
        booster.save_model(str(MODEL_XGB_PATH))
        logger.info("XGBoost PR-AUC (val): %.6f -> %s", pr, MODEL_XGB_PATH)
        cleanup()

    except Exception as e:
        logger.warning("XGBoost skipped: %s", e)
        _rm_tree(XGB_EXTMEM_TRAIN_SINGLE_DIR)
        _rm_tree(XGB_EXTMEM_VAL_SINGLE_DIR)

    logger.info("=== PR-AUC summary (time-based validation) ===")
    for name, pr in sorted(results, key=lambda x: -x[1]):
        logger.info("  %s: %.6f", name, pr)
    if results:
        best = max(results, key=lambda x: x[1])
        logger.info("Best by PR-AUC: %s (%.6f)", best[0], best[1])


if __name__ == "__main__":
    main()
