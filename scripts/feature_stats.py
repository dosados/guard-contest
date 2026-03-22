"""
Скрипт для сравнения средних значений фичей на train и test.

Считает средние по колонкам MODEL_INPUT_FEATURES (shared/config.py) на train и test:
  - train: из output/train_dataset_part_*.parquet или output/train_dataset.parquet
  - test: фичи считаются по pretest + test (как в submission), затем mean() по строкам.

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/feature_stats.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

from shared.config import BATCH_SIZE, MODEL_INPUT_FEATURES, PRETEST_PATH, TEST_PATH
from shared.train_dataset import load_train_dataframe, train_dataset_is_available
from shared.features import FEATURE_NAMES, compute_features
from shared.parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    UserAggregates,
    build_user_aggregates,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # --- Train: из датасета ---
    if not train_dataset_is_available():
        logger.error("Train dataset not found. Соберите датасет (dataset_cpp/build_dataset).")
        sys.exit(1)
    df_train = load_train_dataframe()
    for c in FEATURE_NAMES:
        if c not in df_train.columns:
            logger.error("Missing column in train: %s", c)
            sys.exit(1)
    mean_train = df_train[MODEL_INPUT_FEATURES].mean()
    logger.info("Train: %d строк, средние по фичам посчитаны", len(df_train))

    # --- Test: агрегаты по pretest, затем фичи по каждой строке test ---
    pretest_paths = [PRETEST_PATH] if isinstance(PRETEST_PATH, Path) else list(PRETEST_PATH)
    pretest_paths = [p for p in pretest_paths if p.exists()]
    if not pretest_paths:
        logger.warning("Pretest not found, test features will use empty aggregates.")
    aggregates = defaultdict(
        UserAggregates,
        build_user_aggregates(pretest_paths, batch_size=BATCH_SIZE, show_progress=True)
        if pretest_paths
        else {},
    )
    if not TEST_PATH.exists():
        logger.error("Test file not found: %s", TEST_PATH)
        sys.exit(1)
    pf = pq.ParquetFile(TEST_PATH)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS + [EVENT_ID_COLUMN] if c in schema.names]
    if EVENT_ID_COLUMN not in columns:
        columns.append(EVENT_ID_COLUMN)
    rows: list[dict[str, float]] = []
    total_rows = int(pf.metadata.num_rows)
    with tqdm(total=total_rows, desc="test rows (features)", unit="row") as pbar:
        for batch in pf.iter_batches(columns=columns, batch_size=BATCH_SIZE):
            col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = batch.num_rows
            for i in range(n):
                row = {name: col_lists[name][i] for name in batch.schema.names}
                cid = row.get(CUSTOMER_ID_COLUMN)
                if cid is None:
                    pbar.update(1)
                    continue
                feats = compute_features(aggregates[cid], row)
                aggregates[cid].update(row)
                rows.append(feats)
                pbar.update(1)
    df_test = pd.DataFrame(rows)[MODEL_INPUT_FEATURES]
    mean_test = df_test.mean()

    # --- Вывод ---
    logger.info("=" * 70)
    logger.info("Feature means: train vs test")
    logger.info("=" * 70)
    out = pd.DataFrame({"train": mean_train, "test": mean_test})
    out["diff"] = out["test"] - out["train"]
    print(out.to_string())
    logger.info("=" * 70)
    logger.info("Train rows: %d, Test rows: %d", len(df_train), len(df_test))


if __name__ == "__main__":
    main()
