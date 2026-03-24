"""
Считает по parquet-датасету:
  - число строк, где is_amount_high == 1;
  - число строк, где operation_amt (или legacy operaton_amt) задан.

По умолчанию: output/full_dataset.parquet (shared.config.TRAIN_DATASET_PATH).
Читает только эти две колонки, побатчево по row groups.

Запуск из корня репозитория:
  PYTHONPATH=. python3 scripts/count_is_amount_high.py
  PYTHONPATH=. python3 scripts/count_is_amount_high.py --path /path/to/file.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import TRAIN_DATASET_PATH


def count_valid_numeric(arr: np.ndarray) -> int:
    """Не null / не NaN; для вещественных — только конечные значения (не inf)."""
    kind = arr.dtype.kind
    if kind == "f":
        return int(np.isfinite(arr).sum())
    if kind in "iu":
        return int(arr.size)
    return int(pd.notna(arr).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подсчёт is_amount_high == 1 и валидных operation_amt"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=TRAIN_DATASET_PATH,
        help=f"Parquet-файл (по умолчанию {TRAIN_DATASET_PATH})",
    )
    args = parser.parse_args()
    path: Path = args.path

    if not path.is_file():
        print(f"Файл не найден: {path}", file=sys.stderr)
        sys.exit(1)

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if "is_amount_high" not in names:
        print(f"В {path} нет колонки is_amount_high", file=sys.stderr)
        sys.exit(1)
    amt_col = "operation_amt" if "operation_amt" in names else "operaton_amt"
    if amt_col not in names:
        print(f"В {path} нет колонки operation_amt / operaton_amt", file=sys.stderr)
        sys.exit(1)

    count_one = 0
    count_op_amt_valid = 0
    total = 0
    cols = ["is_amount_high", amt_col]
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i, columns=cols)
        hi = table["is_amount_high"].combine_chunks().to_numpy(zero_copy_only=False)
        amt = table[amt_col].combine_chunks().to_numpy(zero_copy_only=False)
        total += hi.size
        count_one += int(np.sum(hi == 1))
        count_op_amt_valid += count_valid_numeric(amt)

    print(f"Файл: {path}")
    print(f"Всего строк: {total}")
    print(f"is_amount_high == 1: {count_one}")
    if total:
        print(f"Доля is_amount_high == 1: {count_one / total:.6f}")
    print(f"{amt_col} — заданных числовых значений (не null/NaN, конечных): {count_op_amt_valid}")
    if total:
        missing_amt = total - count_op_amt_valid
        print(f"{amt_col} — без значения (null/NaN/inf): {missing_amt}")
        print(f"Доля валидных {amt_col}: {count_op_amt_valid / total:.6f}")


if __name__ == "__main__":
    main()
