"""
Шаг 1 пайплайна: агрегация на pretrain, проход по train с подсчётом фичей, сохранение на диск.
Запуск: из папки code/ или из корня с PYTHONPATH=code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    PRETRAIN_PATHS,
    TRAIN_FEATURES_PATH,
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
    pretrain_paths = [p for p in PRETRAIN_PATHS if p.exists()]
    if not pretrain_paths:
        logger.warning("No pretrain files found at %s, starting with empty aggregates", PRETRAIN_PATHS)
    else:
        logger.info("Building aggregates from pretrain: %s", pretrain_paths)
    aggregates = build_user_aggregates(
        pretrain_paths, batch_size=BATCH_SIZE, show_progress=True
    )
    logger.info("Pretrain done, %s users", len(aggregates))

    labels = load_train_labels(TRAIN_LABELS_PATH)
    logger.info("Loaded %s labeled events", len(labels))

    train_paths = [p for p in TRAIN_PATHS if p.exists()]
    if not train_paths:
        raise FileNotFoundError(f"No train files at {TRAIN_PATHS}")

    rows: list[dict] = []
    columns_to_read = list(FEATURE_COLUMNS) + [EVENT_ID_COLUMN]

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
                out = {**feats, EVENT_ID_COLUMN: event_id, TARGET_COLUMN: int(labels[event_id])}
                rows.append(out)

    if not rows:
        raise ValueError("No labeled rows collected. Check TRAIN_PATHS and TRAIN_LABELS_PATH.")

    df = pd.DataFrame(rows)
    TRAIN_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRAIN_FEATURES_PATH, index=False)
    logger.info("Saved %s rows to %s", len(df), TRAIN_FEATURES_PATH)


if __name__ == "__main__":
    main()
