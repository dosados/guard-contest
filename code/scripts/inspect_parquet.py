#!/usr/bin/env python3
"""Читает несколько строк из parquet-файла и выводит названия колонок и их типы."""

import sys
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    if len(sys.argv) < 2:
        # путь по умолчанию относительно скрипта
        default = Path(__file__).resolve().parent.parent / "data" / "train" / "train_part_1.parquet"
        path = default
        if not path.exists():
            print(f"Использование: {sys.argv[0]} <путь_к_parquet>")
            print(f"Пример: {sys.argv[0]} data/train/train_part_1.parquet")
            sys.exit(1)
    else:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"Файл не найден: {path}")
            sys.exit(1)

    # Схема читается без загрузки всех данных
    schema = pq.read_schema(path)
    n_cols = len(schema)

    # Для превью читаем только первый row group (мало данных для больших файлов)
    pf = pq.ParquetFile(path)
    table = pf.read_row_group(0) if pf.metadata.num_row_groups > 0 else pf.read()
    n_rows = table.num_rows

    print(f"Файл: {path}")
    print(f"Строк в первом row group: {n_rows}, колонок: {n_cols}")
    if pf.metadata.num_row_groups > 1:
        print(f"Всего row groups: {pf.metadata.num_row_groups}")
    print()
    print("Колонки и типы:")
    for field in schema:
        print(f"  {field.name}: {field.type}")

    # первые несколько строк (превью)
    max_preview = 5
    if n_rows > 0:
        print()
        print(f"Первые {min(max_preview, n_rows)} строк (превью):")
        for r in range(min(max_preview, n_rows)):
            row_str = []
            for c in range(n_cols):
                val = table.column(c)[r]
                row_str.append(f"{schema.field(c).name}={val}")
            print("  ", " | ".join(row_str))


if __name__ == "__main__":
    main()
