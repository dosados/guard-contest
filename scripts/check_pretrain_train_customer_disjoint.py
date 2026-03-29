#!/usr/bin/env python3
"""
Проверка гипотезы: при **разных номерах части** множества customer_id не пересекаются
(между pretrain_part_i / train_part_j для i ≠ j, а также pretrain_i ∩ pretrain_j, train_i ∩ train_j).

Пары с **одинаковым** номером (pretrain_part_1 и train_part_1) обычно — одни и те же
пользователи; для них печатается размер пересечения, но это **не** считается ошибкой.

Ожидаемые пути относительно корня репозитория (как в dataset_cpp/build_dataset.cpp):
  data/train/pretrain_part_{1,2,3}.parquet
  data/train/train_part_{1,2,3}.parquet

Читает только колонку customer_id батчами. Код выхода 1, если есть пересечение
хотя бы у одной пары файлов с **разными** номерами части.

Примеры:
  python3 scripts/check_pretrain_train_customer_disjoint.py
  python3 scripts/check_pretrain_train_customer_disjoint.py --repo-root /path/to/guard-contest
  python3 scripts/check_pretrain_train_customer_disjoint.py --sample-overlap 20
"""

from __future__ import annotations

import argparse
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
        raise FileNotFoundError(f"Нет файла: {path}")
    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    if "customer_id" not in schema_names:
        raise ValueError(f"В {path} нет колонки customer_id, есть: {sorted(schema_names)}")
    out: set[str] = set()
    for batch in pf.iter_batches(columns=["customer_id"], batch_size=65536):
        col = batch.column(0)
        # to_pylist() корректно раскодирует dictionary и даёт None для null
        for s in col.to_pylist():
            if s is None:
                continue
            t = _cell_to_str(s).strip()
            if t:
                out.add(t)
    return out


def _part_index_from_tag(tag: str) -> int:
    """pretrain_part_2 -> 2, train_part_3 -> 3."""
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
    p = argparse.ArgumentParser(description="Пересечения customer_id между pretrain/train частями.")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Корень репозитория (по умолчанию родитель scripts/).",
    )
    p.add_argument(
        "--sample-overlap",
        type=int,
        default=15,
        help="Сколько примеров пересечения показать на пару (0 — не показывать).",
    )
    args = p.parse_args()
    root: Path = args.repo_root.resolve()
    train_dir = root / "data" / "train"

    pre_paths = [train_dir / f"pretrain_part_{i}.parquet" for i in (1, 2, 3)]
    tr_paths = [train_dir / f"train_part_{i}.parquet" for i in (1, 2, 3)]
    all_paths = [(f"pretrain_part_{i}", pre_paths[i - 1]) for i in (1, 2, 3)] + [
        (f"train_part_{i}", tr_paths[i - 1]) for i in (1, 2, 3)
    ]

    print(f"repo_root={root}")
    sets: dict[str, set[str]] = {}
    for tag, path in all_paths:
        print(f"Читаю уникальные customer_id: {tag} … ({path})")
        sets[tag] = unique_customer_ids(path)
        print(f"  уникальных: {len(sets[tag]):,}")

    tags_and_paths: list[tuple[str, Path]] = [
        (f"pretrain_part_{i}", pre_paths[i - 1]) for i in (1, 2, 3)
    ] + [(f"train_part_{i}", tr_paths[i - 1]) for i in (1, 2, 3)]

    bad_overlap = False
    print("\n--- Пары с разным номером части (ожидается пересечение = 0) ---")
    for i, (ta, _pa) in enumerate(tags_and_paths):
        for tb, _pb in tags_and_paths[i + 1 :]:
            ia, ib = _part_index_from_tag(ta), _part_index_from_tag(tb)
            if ia == ib:
                continue
            n, sample = _pair_report(sets[ta], sets[tb], sample_overlap=args.sample_overlap)
            status = "OK" if n == 0 else "ОШИБКА"
            if n:
                bad_overlap = True
            print(f"{ta}  ∩  {tb}:  {n:,}  [{status}]")
            if sample:
                print(f"  примеры id: {sample}")

    print("\n--- Пары с одним номером части (pretrain_k ∩ train_k, справочно) ---")
    for k in (1, 2, 3):
        ta = f"pretrain_part_{k}"
        tb = f"train_part_{k}"
        n, sample = _pair_report(sets[ta], sets[tb], sample_overlap=args.sample_overlap)
        print(f"{ta}  ∩  {tb}:  {n:,}  [ожидаемо общие пользователи]")
        if sample and n:
            print(f"  примеры id: {sample}")

    if bad_overlap:
        print(
            "\nИтог: есть пересечения customer_id между файлами с **разными** номерами части — "
            "гипотеза не выполняется."
        )
        return 1
    print(
        "\nИтог: для всех пар с **разными** номерами части пересечений нет — гипотеза выполняется."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
