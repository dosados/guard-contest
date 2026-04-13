#!/usr/bin/env python3
"""
Сканирует event_dttm в full_dataset.parquet и считает успешный парсинг тем же правилом,
что training/main._event_dttm_array_to_timestamp_seconds (strptime с фиксированным форматом).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import TRAIN_DATASET_PATH  # noqa: E402

# Синхронно с training/main.py
_EVENT_DTTM_STRING_FORMAT = "%Y-%m-%d %H:%M:%S"
_EVENT_DTTM_TS_TYPE = pa.timestamp("s")


def _parse_like_training(arr: pa.Array) -> pa.Array:
    if len(arr) == 0:
        return pa.array([], type=_EVENT_DTTM_TS_TYPE)
    if pa.types.is_dictionary(arr.type):
        arr = pc.dictionary_decode(arr)
    t = arr.type
    if pa.types.is_timestamp(t):
        return pc.cast(arr, _EVENT_DTTM_TS_TYPE)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pc.strptime(arr, format=_EVENT_DTTM_STRING_FORMAT, unit="s")
    raise TypeError(f"Неожиданный тип event_dttm: {t}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=TRAIN_DATASET_PATH,
        help="Parquet с колонкой event_dttm",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=15_000_000,
        help="Максимум строк для проверки (0 = весь файл)",
    )
    p.add_argument("--batch-rows", type=int, default=2_000_000, help="Размер батча чтения")
    args = p.parse_args()

    path = args.dataset
    if not path.is_file():
        print(f"Нет файла: {path}", file=sys.stderr)
        return 1

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if "event_dttm" not in names:
        print("В схеме нет event_dttm", file=sys.stderr)
        return 1

    total = 0
    ok = 0
    remaining = None if args.max_rows == 0 else args.max_rows

    for rb in pf.iter_batches(columns=["event_dttm"], batch_size=args.batch_rows):
        col = rb.column(0)
        n = len(col)
        if remaining is not None:
            if remaining <= 0:
                break
            if n > remaining:
                col = col.slice(0, remaining)
                n = len(col)
            remaining -= n

        parsed = _parse_like_training(col)
        total += n
        ok += n - parsed.null_count

    fail = total - ok
    pct = (100.0 * ok / total) if total else 0.0
    print(f"Строк проверено: {total}")
    print(f"Успешный парсинг (как в training): {ok} ({pct:.6f} %)")
    print(f"Null после strptime (не распарсилось): {fail}")
    if total:
        print(f"Доля ошибок: {100.0 * fail / total:.6f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
