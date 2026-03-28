#!/usr/bin/env python3
"""
Печатает имена колонок и Arrow-типы для parquet-файлов (схема на уровне файла).

Примеры из корня репозитория:
  PYTHONPATH=. python3 scripts/inspect_parquet_schema.py data/train/train_part_1.parquet
  PYTHONPATH=. python3 scripts/inspect_parquet_schema.py data/train --recursive
  PYTHONPATH=. python3 scripts/inspect_parquet_schema.py output/full_dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _collect_parquet_paths(root: Path, *, recursive: bool) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() != ".parquet":
            raise ValueError(f"Не parquet: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Нет такого пути: {root}")
    if recursive:
        return sorted(root.rglob("*.parquet"))
    return sorted(root.glob("*.parquet"))


def _print_schema(path: Path) -> None:
    schema = pq.read_schema(path)
    print(f"\n{'=' * 72}")
    print(f"{path}")
    print(f"{'=' * 72}")
    print(f"Колонок: {len(schema.names)}")
    for i, name in enumerate(schema.names):
        field = schema.field(i)
        null = "nullable" if field.nullable else "required"
        print(f"  {i:3d}  {name!s}")
        print(f"       type: {field.type}  ({null})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Вывести схему parquet: колонка → Arrow-тип (и nullable)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Файл .parquet или каталог с .parquet",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Если path — каталог: искать *.parquet рекурсивно",
    )
    args = parser.parse_args()

    paths = _collect_parquet_paths(args.path.resolve(), recursive=args.recursive)
    if not paths:
        print(f"Не найдено *.parquet в {args.path}", file=sys.stderr)
        sys.exit(1)
    for p in paths:
        _print_schema(p)


if __name__ == "__main__":
    main()
