from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from boosting_lr_stack.model import build_meta_features, load_base_boosters, save_stack_model
from shared.config import MODEL_LR_BOOST_STACK_PATH, TRAIN_DATASET_PATH
from training.config import LR_PARAMS, VAL_RATIO
from training.main import _detect_columns, _find_time_cutoff, _prepare_batch

logger = logging.getLogger(__name__)


def _make_classifier() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty=str(LR_PARAMS.get("penalty", "l2")),
        alpha=float(LR_PARAMS.get("alpha", 1e-4)),
        fit_intercept=bool(LR_PARAMS.get("fit_intercept", True)),
        max_iter=int(LR_PARAMS.get("max_iter_per_batch", 1)),
        tol=None,
        random_state=int(LR_PARAMS.get("random_state", 42)),
    )


def _find_meta_cutoff_in_boosting_val(
    dataset_path: Path,
    dttm_col: str,
    boosting_val_cutoff_day: pd.Timestamp,
    *,
    meta_val_ratio_inside_boosting_val: float,
    batch_rows: int,
) -> tuple[pd.Timestamp, int]:
    """
    Внутренний cutoff по времени только на boosting-val части:
    - dttm < meta_cutoff: train метамодели;
    - dttm >= meta_cutoff: собственная валидация метамодели.
    """
    if not (0.0 < meta_val_ratio_inside_boosting_val < 1.0):
        raise ValueError("meta_val_ratio_inside_boosting_val должен быть в диапазоне (0, 1).")

    pf = pq.ParquetFile(dataset_path)
    by_day: Counter[pd.Timestamp] = Counter()
    total_val_rows = 0
    for rb in pf.iter_batches(columns=[dttm_col], batch_size=batch_rows):
        s = pd.to_datetime(rb.column(0).to_pandas(), errors="coerce").dt.floor("D")
        val_s = s[~(s < boosting_val_cutoff_day)]
        if val_s.empty:
            continue
        vc = val_s.value_counts(dropna=True)
        for k, v in vc.items():
            by_day[k] += int(v)
            total_val_rows += int(v)

    if total_val_rows == 0:
        raise RuntimeError("Не удалось найти строки boosting-валидации для обучения stack LR.")

    meta_val_target = max(1, int(total_val_rows * meta_val_ratio_inside_boosting_val))
    acc = 0
    meta_cutoff = None
    for day, cnt in sorted(by_day.items(), reverse=True):
        acc += cnt
        meta_cutoff = day
        if acc >= meta_val_target:
            break
    assert meta_cutoff is not None
    return meta_cutoff, total_val_rows


def _fit_stack_lr(
    dataset_path: Path,
    *,
    use_segmented_xgb: bool,
    batch_rows: int,
    meta_val_ratio_inside_boosting_val: float,
) -> tuple[SGDClassifier, float]:
    feature_cols, dttm_col = _detect_columns(dataset_path)
    boosting_val_cutoff_day = _find_time_cutoff(dataset_path, VAL_RATIO, batch_size=batch_rows)
    meta_cutoff_day, boosting_val_rows = _find_meta_cutoff_in_boosting_val(
        dataset_path,
        dttm_col,
        boosting_val_cutoff_day,
        meta_val_ratio_inside_boosting_val=meta_val_ratio_inside_boosting_val,
        batch_rows=batch_rows,
    )
    logger.info(
        "Stack LR split: boosting-val rows=%d, meta-train day < %s, meta-val day >= %s",
        boosting_val_rows,
        meta_cutoff_day.date(),
        meta_cutoff_day.date(),
    )

    pf = pq.ParquetFile(dataset_path)
    cols = feature_cols + ["target", "sample_weight", dttm_col]

    boosters = load_base_boosters(use_segmented_xgb=use_segmented_xgb)
    lr_model = _make_classifier()
    initialized = False
    meta_train_rows = 0

    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        # Учим метамодель только на части boosting-валидации.
        meta_train_mask = (~(dttm < boosting_val_cutoff_day)) & (dttm < meta_cutoff_day)
        if not bool(meta_train_mask.any()):
            continue
        train_df = dfb.loc[meta_train_mask].copy()
        X_train_df = train_df[feature_cols]
        meta_x = build_meta_features(X_train_df, boosters)
        _, y, w = _prepare_batch(train_df, feature_cols)
        if not initialized:
            lr_model.partial_fit(meta_x, y, classes=np.asarray([0, 1], dtype=np.int32), sample_weight=w)
            initialized = True
        else:
            lr_model.partial_fit(meta_x, y, sample_weight=w)
        meta_train_rows += len(train_df)

    if not initialized:
        raise RuntimeError("Не удалось обучить stack LR: в выбранной части boosting-валидации нет train-строк.")

    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []

    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        # Собственная валидация метамодели: более свежий хвост boosting-валидации.
        meta_val_mask = ~(dttm < meta_cutoff_day)
        if not bool(meta_val_mask.any()):
            continue
        val_df = dfb.loc[meta_val_mask].copy()
        X_val_df = val_df[feature_cols]
        meta_x = build_meta_features(X_val_df, boosters)
        _, y, w = _prepare_batch(val_df, feature_cols)
        p = np.asarray(lr_model.predict_proba(meta_x)[:, 1], dtype=np.float64)
        y_all.append(y.astype(np.int32, copy=False))
        p_all.append(p)
        w_all.append(w.astype(np.float64, copy=False))

    if not y_all:
        logger.warning("Meta-val строки не найдены, PR-AUC = 0.0")
        return lr_model, 0.0

    pr_auc = float(average_precision_score(np.concatenate(y_all), np.concatenate(p_all), sample_weight=np.concatenate(w_all)))
    logger.info("Stack LR trained rows (on boosting-val subset): %d", meta_train_rows)
    logger.info("Stack LR PR-AUC (meta-val): %.6f", pr_auc)
    return lr_model, pr_auc


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Logistic Regression поверх CatBoost + XGBoost выходов")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=TRAIN_DATASET_PATH,
        help="Путь к train parquet (по умолчанию shared.config.TRAIN_DATASET_PATH).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_LR_BOOST_STACK_PATH,
        help="Куда сохранить stack LR модель.",
    )
    parser.add_argument(
        "--xgb-segmented",
        action="store_true",
        help="Использовать 2 XGBoost-модели по tr_amount (<=30 и >30).",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=1_000_000,
        help="Размер parquet батча.",
    )
    parser.add_argument(
        "--meta-val-ratio-inside-boosting-val",
        type=float,
        default=0.2,
        help="Доля самых свежих строк boosting-валидации под собственную валидацию stack LR (0..1).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Не найден train parquet: {args.dataset}")

    model, pr_auc = _fit_stack_lr(
        args.dataset,
        use_segmented_xgb=args.xgb_segmented,
        batch_rows=int(args.batch_rows),
        meta_val_ratio_inside_boosting_val=float(args.meta_val_ratio_inside_boosting_val),
    )
    save_stack_model(args.output, model, use_segmented_xgb=args.xgb_segmented)
    logger.info("Stack LR сохранена: %s (PR-AUC %.6f)", args.output, pr_auc)


if __name__ == "__main__":
    main()

