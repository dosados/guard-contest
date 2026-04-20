#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import OUTPUT_DIR, TRAIN_DATASET_PATH  # noqa: E402


def main() -> int:
    inp = TRAIN_DATASET_PATH
    out = OUTPUT_DIR / "slice_head_sample.parquet"
    rows = 5000
    if rows < 1:
        raise SystemExit("rows must be >= 1")
    pf = pq.ParquetFile(inp)
    chunks: list[pa.RecordBatch] = []
    n = 0
    for batch in pf.iter_batches(batch_size=min(65_536, rows)):
        need = rows - n
        if batch.num_rows <= need:
            chunks.append(batch)
            n += batch.num_rows
        else:
            chunks.append(batch.slice(0, need))
            n = rows
            break
        if n >= rows:
            break
    if not chunks:
        raise SystemExit("empty input")
    table = pa.Table.from_batches(chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")
    print(f"Wrote {table.num_rows} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
