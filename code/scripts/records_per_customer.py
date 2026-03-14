#!/usr/bin/env python3
"""
Читает parquet-файл батчами и считает среднее число записей на один customer_id.
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

CUSTOMER_ID_COLUMN = "customer_id"


def avg_records_per_customer(
    path: str | Path,
    batch_size: int = 100_000,
    customer_column: str = CUSTOMER_ID_COLUMN,
) -> tuple[int, int, float]:
    """
    Читает parquet по батчам, считает записи по customer_id.
    Возвращает (всего_записей, уникальных_клиентов, среднее_записей_на_клиента).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    pf = pq.ParquetFile(path)
    # Счётчик записей по каждому customer_id
    counts: dict[int | str, int] = {}

    for batch in pf.iter_batches(columns=[customer_column], batch_size=batch_size):
        col = batch.column(customer_column)
        for i in range(batch.num_rows):
            cid = col[i].as_py()
            counts[cid] = counts.get(cid, 0) + 1

    total_records = sum(counts.values())
    num_customers = len(counts)
    if num_customers == 0:
        return 0, 0, 0.0
    avg = total_records / num_customers
    return total_records, num_customers, avg


def avg_records_per_customer_multi(
    paths: list[Path],
    batch_size: int = 100_000,
    customer_column: str = CUSTOMER_ID_COLUMN,
) -> tuple[int, int, float]:
    """
    То же, что avg_records_per_customer, но для нескольких parquet-файлов (общий подсчёт).
    """
    counts: dict[int | str, int] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=[customer_column], batch_size=batch_size):
            col = batch.column(customer_column)
            for i in range(batch.num_rows):
                cid = col[i].as_py()
                counts[cid] = counts.get(cid, 0) + 1
    total_records = sum(counts.values())
    num_customers = len(counts)
    if num_customers == 0:
        return 0, 0, 0.0
    return total_records, num_customers, total_records / num_customers


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

    if len(sys.argv) >= 2:
        # Один файл из аргумента
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"Файл не найден: {path}")
            sys.exit(1)
        print(f"Читаю {path} батчами по {batch_size}...")
        total, customers, avg = avg_records_per_customer(path, batch_size=batch_size)
        print(f"Всего записей:     {total}")
        print(f"Уникальных клиентов (customer_id): {customers}")
        print(f"В среднем записей на одного customer_id: {avg:.2f}")
        return

    # По одному файлу из train, pretrain, pretest, test
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    datasets: list[tuple[str, Path]] = [
        ("train", train_dir / "train_part_1.parquet"),
        ("pretrain", train_dir / "pretrain_part_1.parquet"),
        ("pretest", test_dir / "pretest.parquet"),
        ("test", test_dir / "test.parquet"),
    ]

    for name, path in datasets:
        if not path.exists():
            print(f"\n--- {name} --- файл не найден: {path}, пропуск")
            continue
        print(f"\n--- {name} --- {path.name}")
        total, customers, avg = avg_records_per_customer(path, batch_size=batch_size)
        print(f"Всего записей:     {total}")
        print(f"Уникальных клиентов (customer_id): {customers}")
        print(f"В среднем записей на одного customer_id: {avg:.2f}")

    print()


if __name__ == "__main__":
    main()
