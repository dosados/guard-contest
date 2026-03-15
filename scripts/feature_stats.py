"""
Скрипт для сравнения средних значений фичей на train и test.

Считает df_train[FEATURE_NAMES].mean() и df_test[FEATURE_NAMES].mean():
  - train: из output/train_dataset.parquet
  - test: фичи считаются по pretest + test (как в submission), затем mean() по строкам.

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/feature_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

from shared.config import (
    BATCH_SIZE,
    PRETEST_PATH,
    TEST_PATH,
    TRAIN_DATASET_PATH,
)
from shared.features import FEATURE_NAMES, compute_features
from shared.parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    UserAggregates,
    build_user_aggregates,
)


def main() -> None:
    # --- Train: из датасета ---
    if not TRAIN_DATASET_PATH.exists():
        print(f"Train dataset not found: {TRAIN_DATASET_PATH}. Run dataset/main.py first.")
        sys.exit(1)
    df_train = pd.read_parquet(TRAIN_DATASET_PATH)
    for c in FEATURE_NAMES:
        if c not in df_train.columns:
            print(f"Missing column in train: {c}")
            sys.exit(1)
    mean_train = df_train[FEATURE_NAMES].mean()

    # --- Test: агрегаты по pretest, затем фичи по каждой строке test ---
    pretest_paths = [PRETEST_PATH] if isinstance(PRETEST_PATH, Path) else list(PRETEST_PATH)
    pretest_paths = [p for p in pretest_paths if p.exists()]
    if not pretest_paths:
        print("Pretest not found, test features will use empty aggregates.")
    aggregates = defaultdict(
        UserAggregates,
        build_user_aggregates(pretest_paths, batch_size=BATCH_SIZE, show_progress=True)
        if pretest_paths
        else {},
    )
    if not TEST_PATH.exists():
        print(f"Test file not found: {TEST_PATH}")
        sys.exit(1)
    pf = pq.ParquetFile(TEST_PATH)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS + [EVENT_ID_COLUMN] if c in schema.names]
    if EVENT_ID_COLUMN not in columns:
        columns.append(EVENT_ID_COLUMN)
    rows: list[dict[str, float]] = []
    for batch in pf.iter_batches(columns=columns, batch_size=BATCH_SIZE):
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            feats = compute_features(aggregates[cid], row)
            aggregates[cid].update(row)
            rows.append(feats)
    df_test = pd.DataFrame(rows)[FEATURE_NAMES]
    mean_test = df_test.mean()

    # --- Вывод ---
    print("=" * 70)
    print("Feature means: train vs test")
    print("=" * 70)
    out = pd.DataFrame({"train": mean_train, "test": mean_test})
    out["diff"] = out["test"] - out["train"]
    print(out.to_string())
    print("=" * 70)
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")


if __name__ == "__main__":
    main()
