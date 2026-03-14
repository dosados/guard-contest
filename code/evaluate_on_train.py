"""
Проверка модели на train в условиях, идентичных реальному использованию:
pretrain → агрегация фичей, затем проход по train с обновлением агрегатов и предсказанием
только для размеченных событий. Метрики считаются по этим предсказаниям.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    MODEL_PATH,
    PRETRAIN_PATHS,
    TRAIN_LABELS_PATH,
    TRAIN_PATHS,
)
from features import FEATURE_NAMES, compute_features
from parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    build_user_aggregates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"


def load_train_labels(path: Path) -> dict[int, int]:
    """Загружает разметку train из parquet: event_id -> target (0 — жёлтая, 1 — целевой класс)."""
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    df = pd.read_parquet(path)
    if "event_id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Labels parquet must have event_id and target, got {list(df.columns)}")
    return df.set_index("event_id")["target"].astype(int).to_dict()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

    logger.info("Loading model from %s", MODEL_PATH)
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    labels = load_train_labels(TRAIN_LABELS_PATH)
    logger.info("Loaded %s labeled events", len(labels))

    pretrain_paths = [p for p in PRETRAIN_PATHS if p.exists()]
    if not pretrain_paths:
        logger.warning("No pretrain files at %s, starting with empty aggregates", PRETRAIN_PATHS)
    else:
        logger.info("Building aggregates from pretrain")
    aggregates = build_user_aggregates(
        pretrain_paths, batch_size=BATCH_SIZE, show_progress=True
    )
    logger.info("Pretrain done, %s users", len(aggregates))

    train_paths = [p for p in TRAIN_PATHS if p.exists()]
    if not train_paths:
        raise FileNotFoundError(f"No train files at {TRAIN_PATHS}")

    columns_to_read = list(FEATURE_COLUMNS) + [EVENT_ID_COLUMN]
    y_true: list[int] = []
    pred_scores: list[float] = []

    for path in tqdm(train_paths, desc="Train files", unit="file"):
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        columns = [c for c in columns_to_read if c in schema.names]
        for batch in pf.iter_batches(columns=columns, batch_size=BATCH_SIZE):
            col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = batch.num_rows
            for i in range(n):
                row = {name: col_lists[name][i] for name in batch.schema.names}
                cid = row.get(CUSTOMER_ID_COLUMN)
                if cid is None:
                    continue
                event_id = row.get(EVENT_ID_COLUMN)
                if event_id not in labels:
                    aggregates[cid].update(row)
                    continue
                feats = compute_features(aggregates[cid], row)
                aggregates[cid].update(row)
                x = [feats[name] for name in FEATURE_NAMES]
                pred = model.predict([x], prediction_type="RawFormulaVal")
                pred_val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
                y_true.append(int(labels[event_id]))
                pred_scores.append(pred_val)

    if not y_true:
        raise ValueError("No labeled rows found. Check TRAIN_PATHS and TRAIN_LABELS_PATH.")

    pr_auc = average_precision_score(y_true, pred_scores)
    roc_auc = roc_auc_score(y_true, pred_scores)
    pred_class = [1 if s > 0.5 else 0 for s in pred_scores]
    acc = accuracy_score(y_true, pred_class)

    logger.info("--- Metrics on train (same pipeline as inference) ---")
    logger.info("PR-AUC:   %.4f", pr_auc)
    logger.info("ROC-AUC:  %.4f", roc_auc)
    logger.info("Accuracy: %.4f", acc)
    logger.info("Samples:  %s (0=%s, 1=%s)", len(y_true), y_true.count(0), y_true.count(1))


if __name__ == "__main__":
    main()
