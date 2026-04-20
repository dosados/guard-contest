#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import TRAIN_DATA_ROOT  # noqa: E402


def _collect_parquet_paths(root: Path, *, recursive: bool) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() != ".parquet":
            raise ValueError(f"Not a parquet file: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"No such path: {root}")
    if recursive:
        return sorted(root.rglob("*.parquet"))
    return sorted(root.glob("*.parquet"))


def _print_schema(path: Path) -> None:
    schema = pq.read_schema(path)
    print(f"\n{'=' * 72}")
    print(f"{path}")
    print(f"{'=' * 72}")
    print(f"Columns: {len(schema.names)}")
    for i, name in enumerate(schema.names):
        field = schema.field(i)
        null = "nullable" if field.nullable else "required"
        print(f"  {i:3d}  {name!s}")
        print(f"       type: {field.type}  ({null})")


def main() -> None:
    path = TRAIN_DATA_ROOT
    recursive = False

    paths = _collect_parquet_paths(path.resolve(), recursive=recursive)
    if not paths:
        print(f"No *.parquet found in {path}", file=sys.stderr)
        sys.exit(1)
    for p in paths:
        _print_schema(p)


if __name__ == "__main__":
    main()
