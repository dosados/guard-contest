#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _cell_to_str(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def unique_customer_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    if "customer_id" not in schema_names:
        raise ValueError(f"{path} has no customer_id column; columns: {sorted(schema_names)}")
    out: set[str] = set()
    for batch in pf.iter_batches(columns=["customer_id"], batch_size=65536):
        col = batch.column(0)
        # to_pylist() decodes dictionary arrays and yields None for nulls
        for s in col.to_pylist():
            if s is None:
                continue
            t = _cell_to_str(s).strip()
            if t:
                out.add(t)
    return out


def _part_index_from_tag(tag: str) -> int:
    # pretrain_part_k / train_part_k → k
    prefix = "pretrain_part_" if tag.startswith("pretrain") else "train_part_"
    return int(tag.removeprefix(prefix))


def _pair_report(
    sa: set[str],
    sb: set[str],
    *,
    sample_overlap: int,
) -> tuple[int, list[str]]:
    inter = sa & sb
    n = len(inter)
    sample: list[str] = []
    if n and sample_overlap > 0:
        sample = sorted(inter)[:sample_overlap]
    return n, sample


def main() -> int:
    repo_root = _PROJECT_ROOT
    sample_overlap = 15
    root: Path = repo_root.resolve()
    train_dir = root / "data" / "train"

    pre_paths = [train_dir / f"pretrain_part_{i}.parquet" for i in (1, 2, 3)]
    tr_paths = [train_dir / f"train_part_{i}.parquet" for i in (1, 2, 3)]
    all_paths = [(f"pretrain_part_{i}", pre_paths[i - 1]) for i in (1, 2, 3)] + [
        (f"train_part_{i}", tr_paths[i - 1]) for i in (1, 2, 3)
    ]

    print(f"repo_root={root}")
    sets: dict[str, set[str]] = {}
    for tag, path in all_paths:
        print(f"Reading unique customer_id: {tag} … ({path})")
        sets[tag] = unique_customer_ids(path)
        print(f"  unique: {len(sets[tag]):,}")

    tags_and_paths: list[tuple[str, Path]] = [
        (f"pretrain_part_{i}", pre_paths[i - 1]) for i in (1, 2, 3)
    ] + [(f"train_part_{i}", tr_paths[i - 1]) for i in (1, 2, 3)]

    bad_overlap = False
    print("\n--- Pairs with different part index (expected intersection = 0) ---")
    for i, (ta, _pa) in enumerate(tags_and_paths):
        for tb, _pb in tags_and_paths[i + 1 :]:
            ia, ib = _part_index_from_tag(ta), _part_index_from_tag(tb)
            if ia == ib:
                continue
            n, sample = _pair_report(sets[ta], sets[tb], sample_overlap=sample_overlap)
            status = "OK" if n == 0 else "ERROR"
            if n:
                bad_overlap = True
            print(f"{ta}  ∩  {tb}:  {n:,}  [{status}]")
            if sample:
                print(f"  sample ids: {sample}")

    print("\n--- Pairs with same part index (pretrain_k ∩ train_k, informational) ---")
    for k in (1, 2, 3):
        ta = f"pretrain_part_{k}"
        tb = f"train_part_{k}"
        n, sample = _pair_report(sets[ta], sets[tb], sample_overlap=sample_overlap)
        print(f"{ta}  ∩  {tb}:  {n:,}  [expected shared users]")
        if sample and n:
            print(f"  sample ids: {sample}")

    if bad_overlap:
        print(
            "\nResult: customer_id overlap between files with **different** part indices - "
            "hypothesis fails."
        )
        return 1
    print(
        "\nResult: no overlaps for all pairs with **different** part indices - hypothesis holds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
