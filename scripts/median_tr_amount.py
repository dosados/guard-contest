#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import TRAIN_DATASET_PATH


def main() -> None:
    dataset_path = TRAIN_DATASET_PATH
    batch_size = 100_000

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)

    col = "log_1_plus_transactions_seen"
    if col not in parquet_file.schema.names:
        raise KeyError(f"Column '{col}' was not found in dataset.")

    chunks: list[np.ndarray] = []
    for batch in parquet_file.iter_batches(
        columns=[col],
        batch_size=batch_size,
    ):
        values = batch.column(0).drop_null().to_numpy(zero_copy_only=False)
        if values.size > 0:
            chunks.append(values)

    if not chunks:
        raise ValueError(f"Column '{col}' contains no numeric values.")

    all_values = np.concatenate(chunks)
    percentile_levels = np.arange(0, 101, 10)
    percentile_values = np.percentile(all_values, percentile_levels)

    print(f"{col} percentiles:")
    for level, value in zip(percentile_levels, percentile_values):
        print(f"P{int(level):>3}: {float(value)}")


if __name__ == "__main__":
    main()
