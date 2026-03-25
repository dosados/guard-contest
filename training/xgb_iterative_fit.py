"""
Один цикл обучения XGBoost: ExtMemQuantileDMatrix / QuantileDMatrix + DataIter по parquet
(как training/main.py). Train с весами, val без весов — невзвешенный PR-AUC (pos_label=1).
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from training.parquet_io import count_val_rows, prepare_batch

logger = logging.getLogger(__name__)


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def xgb_predict_with_best_iteration(booster: Any, dmatrix: Any) -> np.ndarray:
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


def fit_xgb_parquet_iterative(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    xgb_params: dict[str, Any],
    *,
    num_boost_round: int,
    batch_rows: int,
    train_cache_dir: Path,
    val_cache_dir: Path,
    early_stopping_rounds: int = 0,
    verbose_eval: int | bool = False,
) -> tuple[Any, float]:
    """
    Обучает XGBoost, возвращает (booster, невзвешенный PR-AUC на val, pos_label=1).
    Каталоги train_cache_dir / val_cache_dir очищаются перед стартом и после (внутри функции).
    """
    import xgboost as xgb
    from xgboost.core import DataIter

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
            emit_sample_weight: bool = True,
        ) -> None:
            self._parquet_path = parquet_path
            self._feature_cols = feature_cols
            self._dttm_col = dttm_col
            self._cutoff_day = cutoff_day
            self._batch_rows = batch_rows
            self._mode = mode
            self._emit_weight = emit_sample_weight
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
                if not mask.any():
                    continue
                x, y, w = prepare_batch(dfb.loc[mask], self._feature_cols)
                if self._emit_weight:
                    input_data(
                        data=x,
                        label=y.astype(np.float32),
                        weight=w,
                        feature_names=self._feature_cols,
                    )
                else:
                    input_data(
                        data=x,
                        label=y.astype(np.float32),
                        feature_names=self._feature_cols,
                    )
                return True

    use_extmem = hasattr(xgb, "ExtMemQuantileDMatrix")
    train_cp = str((train_cache_dir / "qdm").resolve()) if use_extmem else None
    val_cp = str((val_cache_dir / "qdm").resolve()) if use_extmem else None

    if use_extmem:
        _rm_tree(train_cache_dir)
        _rm_tree(val_cache_dir)
        train_cache_dir.mkdir(parents=True, exist_ok=True)
        val_cache_dir.mkdir(parents=True, exist_ok=True)

    train_it = ParquetTrainValDataIter(
        parquet_path,
        feature_cols,
        dttm_col,
        cutoff_day,
        batch_rows,
        "train",
        cache_prefix=train_cp,
        emit_sample_weight=True,
    )
    if use_extmem:
        dtrain = xgb.ExtMemQuantileDMatrix(train_it, missing=np.nan, enable_categorical=False)
    else:
        dtrain = xgb.QuantileDMatrix(train_it)

    train_kw: dict[str, Any] = {
        "params": xgb_params,
        "dtrain": dtrain,
        "num_boost_round": num_boost_round,
    }
    dval = None
    yva = None
    n_val = count_val_rows(parquet_path, dttm_col, cutoff_day, batch_size=batch_rows)

    if n_val > 0:
        val_it = ParquetTrainValDataIter(
            parquet_path,
            feature_cols,
            dttm_col,
            cutoff_day,
            batch_rows,
            "val",
            cache_prefix=val_cp,
            emit_sample_weight=False,
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
        train_kw["verbose_eval"] = verbose_eval if verbose_eval else False
        if early_stopping_rounds > 0:
            train_kw["early_stopping_rounds"] = int(early_stopping_rounds)
        yva = np.asarray(dval.get_label(), dtype=np.float32)

    booster = xgb.train(**train_kw)

    pr = 0.0
    if dval is not None and yva is not None:
        p = xgb_predict_with_best_iteration(booster, dval)
        y_int = yva.astype(np.int32)
        pr = float(average_precision_score(y_int, p, pos_label=1))

    if use_extmem:
        del dtrain
        del train_it
        if n_val > 0:
            del dval
            del val_it
        gc.collect()
        _rm_tree(train_cache_dir)
        _rm_tree(val_cache_dir)

    return booster, pr
