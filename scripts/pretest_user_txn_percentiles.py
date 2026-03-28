#!/usr/bin/env python3
"""
Считает перцентили (0..100 с шагом 10) по числу транзакций на пользователя в pretest.parquet.

Запуск из корня репозитория:
  PYTHONPATH=. python3 scripts/pretest_user_txn_percentiles.py
  PYTHONPATH=. python3 scripts/pretest_user_txn_percentiles.py --path data/test/pretest.parquet
  PYTHONPATH=. python3 scripts/pretest_user_txn_percentiles.py --batch-size 200000
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import PRETEST_PATH
from shared.dataset_settings import WINDOW_TRANSACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print percentiles (0..100 step 10) of transactions-per-user for pretest parquet."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=PRETEST_PATH,
        help=f"Path to pretest parquet (default: {PRETEST_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for parquet reading (default: 100000).",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=WINDOW_TRANSACTIONS,
        help=(
            "Optional cap for transactions-per-user before percentile calculation "
            f"(default: {WINDOW_TRANSACTIONS}). Use <=0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path: Path = args.path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    pf = pq.ParquetFile(path)
    if "customer_id" not in pf.schema.names:
        raise KeyError("Column 'customer_id' was not found in dataset.")

    tx_per_user: defaultdict[object, int] = defaultdict(int)
    for batch in pf.iter_batches(columns=["customer_id"], batch_size=args.batch_size):
        users = batch.column(0).to_numpy(zero_copy_only=False)
        valid_users = users[pd.notna(users)]
        for user_id in valid_users:
            tx_per_user[user_id] += 1

    if not tx_per_user:
        raise ValueError("No valid customer_id values found.")

    counts = np.fromiter(tx_per_user.values(), dtype=np.int64, count=len(tx_per_user))
    if args.cap > 0:
        counts = np.minimum(counts, args.cap)
    percentile_levels = np.arange(0, 101, 10)
    percentile_values = np.percentile(counts, percentile_levels)

    print(f"File: {path}")
    print(f"Users with at least 1 transaction: {len(tx_per_user)}")
    if args.cap > 0:
        print(f"Applied cap: {args.cap}")
    print("Transactions-per-user percentiles:")
    for level, value in zip(percentile_levels, percentile_values):
        print(f"P{int(level):>3}: {float(value)}")


if __name__ == "__main__":
    main()
