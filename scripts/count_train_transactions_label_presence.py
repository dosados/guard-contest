#!/usr/bin/env python3
"""
Count train transactions whose event_id appears in train_labels vs not.

Reads Parquet with PyArrow in batches. Intended to run with conda env guard-contest:

  conda run -n guard-contest python scripts/count_train_transactions_label_presence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def contiguous(col: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks()
    return col


def load_label_event_ids(labels_path: Path, batch_size: int) -> pa.Array:
    """Sorted int64 array (unique) for use with pc.is_in."""
    ids: list[int] = []
    pf = pq.ParquetFile(labels_path)
    for batch in pf.iter_batches(columns=["event_id"], batch_size=batch_size):
        col = batch.column(0)
        if col.null_count > 0:
            col = pc.drop_null(col)
        if len(col) == 0:
            continue
        col = contiguous(col)
        if pa.types.is_dictionary(col.type):
            col = pc.dictionary_decode(col)
        col = pc.cast(col, pa.int64())
        ids.extend(col.to_pylist())
    if not ids:
        return pa.array([], type=pa.int64())
    arr = pa.array(sorted(set(ids)), type=pa.int64())
    return arr


def normalize_event_id_column(col: pa.Array | pa.ChunkedArray) -> pa.Array:
    col = contiguous(col)
    if col.null_count == len(col):
        return col
    if pa.types.is_dictionary(col.type):
        col = pc.dictionary_decode(col)
    # int / float string from parquet — cast to int64 where possible
    try:
        col = pc.cast(col, pa.int64())
    except pa.ArrowInvalid:
        col = pc.cast(pc.cast(col, pa.float64()), pa.int64())
    return col


def count_train_file(
    path: Path,
    label_values: pa.Array,
    batch_size: int,
) -> tuple[int, int, int]:
    """
    Returns (in_labels, not_in_labels, null_event_id_rows).
    Rows with null event_id are counted only in null_event_id_rows (not in not_in_labels).
    """
    in_labels = 0
    not_in_labels = 0
    null_rows = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["event_id"], batch_size=batch_size):
        col = batch.column(0)
        null_rows += int(col.null_count)
        if col.null_count == len(col):
            continue
        non_null = pc.drop_null(col)
        if len(non_null) == 0:
            continue
        nn = normalize_event_id_column(non_null)
        if len(label_values) == 0:
            not_in_labels += len(nn)
            continue
        mask = pc.is_in(nn, value_set=label_values)
        # is_in: null input -> null; we dropped nulls
        in_labels += int(pc.sum(mask).as_py())
        not_in_labels += int(len(nn) - pc.sum(mask).as_py())
    return in_labels, not_in_labels, null_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root() / "data" / "train",
        help="Directory with train_part_*.parquet",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_root() / "data" / "train_labels.parquet",
        help="train_labels.parquet path",
    )
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()

    if not args.labels.is_file():
        print(f"Labels file not found: {args.labels}", file=sys.stderr)
        return 1

    paths = sorted(args.data_dir.glob("train_part_*.parquet"))
    if not paths:
        print(f"No train_part_*.parquet under {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Labels: {args.labels}")
    label_values = load_label_event_ids(args.labels, args.batch_size)
    print(f"Unique labeled event_id count: {len(label_values):,}")

    total_in = 0
    total_out = 0
    total_null = 0

    for p in paths:
        a, b, c = count_train_file(p, label_values, args.batch_size)
        total_in += a
        total_out += b
        total_null += c
        print(f"{p.name}: in_labels={a:,} not_in_labels={b:,} null_event_id={c:,}")

    total_tx = total_in + total_out + total_null
    print("---")
    print(f"TOTAL rows: {total_tx:,}")
    print(f"  event_id in train_labels:     {total_in:,} ({100.0 * total_in / total_tx:.4f} %)" if total_tx else "  (no rows)")
    print(
        f"  event_id not in train_labels: {total_out:,} ({100.0 * total_out / total_tx:.4f} %)" if total_tx else ""
    )
    print(f"  null event_id:                {total_null:,} ({100.0 * total_null / total_tx:.4f} %)" if total_tx else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
