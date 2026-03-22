"""
Скрипт объединяет output/train_dataset_part_*.parquet в один output/full_dataset.

Исходные part-файлы НЕ удаляются.

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/merge_train_dataset_parts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import OUTPUT_DIR, TRAIN_DATASET_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=TRAIN_DATASET_PATH,
        help="Куда писать объединённый parquet (по умолчанию output/full_dataset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200_000,
        help="Размер batch при перекладывании row groups.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="Сжатие для выходного parquet (zstd/snappy/gzip/none).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(OUTPUT_DIR.glob("train_dataset_part_*.parquet"))
    if not parts:
        print("Не найдены output/train_dataset_part_*.parquet")
        sys.exit(1)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression.lower() == "none" else args.compression
    writer: pq.ParquetWriter | None = None
    rows_total = 0

    for part in tqdm(parts, desc="merge parquet parts", unit="file"):
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
            writer.write_table(table)
            rows_total += table.num_rows

    if writer is None:
        print("Part-файлы найдены, но данных для записи нет.")
        sys.exit(1)

    writer.close()
    print(f"Merged {len(parts)} files -> {out_path} (rows: {rows_total})")
    print("Исходные part-файлы сохранены без изменений.")


if __name__ == "__main__":
    main()
