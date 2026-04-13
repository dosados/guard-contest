#!/usr/bin/env python3
"""Печать схемы output/full_dataset.parquet и примеров типов (event_dttm и др.)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pyarrow.parquet as pq  # noqa: E402

from shared.config import TRAIN_DATASET_PATH  # noqa: E402


def main() -> int:
    path = TRAIN_DATASET_PATH
    if not path.is_file():
        print(f"Нет файла: {path}", file=sys.stderr)
        return 1
    pf = pq.ParquetFile(path)
    print("Schema:")
    print(pf.schema_arrow)
    names = pf.schema_arrow.names
    sample_cols = [c for c in ("event_dttm", "target", "sample_weight", "event_descr") if c in names]
    t = pf.read(columns=sample_cols, use_threads=True)
    print("\nSample columns:", sample_cols)
    for name in sample_cols:
        col = t.column(name)
        print(f"  {name}: type={col.type!s} examples={col.slice(0, 3).to_pylist()!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
