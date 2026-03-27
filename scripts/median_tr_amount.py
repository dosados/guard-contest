#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print median value of 'tr_amount' from a parquet dataset."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="output/full_dataset.parquet",
        help="Path to parquet file (default: output/full_dataset.parquet).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for parquet reading (default: 100000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)

    if "tr_amount" not in parquet_file.schema.names:
        raise KeyError("Column 'tr_amount' was not found in dataset.")

    tr_amount_chunks: list[np.ndarray] = []
    for batch in parquet_file.iter_batches(
        columns=["tr_amount"],
        batch_size=args.batch_size,
    ):
        values = batch.column(0).drop_null().to_numpy(zero_copy_only=False)
        if values.size > 0:
            tr_amount_chunks.append(values)

    if not tr_amount_chunks:
        raise ValueError("Column 'tr_amount' contains no numeric values.")

    median_value = float(np.median(np.concatenate(tr_amount_chunks)))
    print(f"Median tr_amount: {median_value}")


if __name__ == "__main__":
    main()
