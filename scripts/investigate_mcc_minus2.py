#!/usr/bin/env python3
"""
Диагностика mcc_code -> -2 (MCC_MISSING_KEY): сырые train parquet и output/global_aggregates.

Запуск из корня репозитория:
  conda activate guard-contest
  PYTHONPATH=. python3 scripts/investigate_mcc_minus2.py

Опции:
  --sample-rows 8   сколько примеров «проблемных» строк показать на файл
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCC_MISSING = -2
MCC_GLOBAL = -(2**63)


def _parse_like_cpp_mcc(v) -> int:
    """Упрощённо как parse_mcc_from_column + parse_mcc_key_string (C++)."""
    if v is None:
        return MCC_MISSING
    if isinstance(v, bool):
        return MCC_MISSING
    if isinstance(v, (int, np.integer)):
        x = int(v)
        if x == MCC_GLOBAL or x == MCC_MISSING:
            return MCC_MISSING
        return x
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if not math.isfinite(x):
            return MCC_MISSING
        r = round(x)
        if abs(x - r) > 1e-6 * (1.0 + abs(x)):
            return MCC_MISSING
        vi = int(r)
        if vi == MCC_GLOBAL or vi == MCC_MISSING:
            return MCC_MISSING
        return vi
    t = str(v).strip()
    if not t:
        return MCC_MISSING
    try:
        v = int(t, 10)
        if v == MCC_GLOBAL or v == MCC_MISSING:
            return MCC_MISSING
        return v
    except ValueError:
        pass
    try:
        x = float(t)
        if not math.isfinite(x):
            return MCC_MISSING
        r = round(x)
        if abs(x - r) > 1e-6 * (1.0 + abs(x)):
            return MCC_MISSING
        vi = int(r)
        if vi == MCC_GLOBAL or vi == MCC_MISSING:
            return MCC_MISSING
        return vi
    except ValueError:
        return MCC_MISSING


def _rel(p: Path) -> Path:
    try:
        return p.relative_to(_PROJECT_ROOT)
    except ValueError:
        return p


def _cell_py(arr: pa.Array, i: int):
    s = arr[i]
    if not s.is_valid:
        return None
    return s.as_py()


def _analyze_train_parquet(path: Path, sample_rows: int) -> None:
    print("\n" + "=" * 88)
    print(f"TRAIN/PRETRAIN: {_rel(path)}")
    print("=" * 88)
    pf = pq.ParquetFile(path)
    nrows = pf.metadata.num_rows
    print(f"rows={nrows}  row_groups={pf.metadata.num_row_groups}")
    sch = pf.schema_arrow
    if "mcc_code" not in sch.names:
        print("Нет колонки mcc_code")
        return
    idx = sch.get_field_index("mcc_code")
    print(f"mcc_code schema type: {sch.field(idx).type}")

    cols = ["mcc_code"]
    has_cid = "customer_id" in sch.names
    if has_cid:
        cols.append("customer_id")

    total = 0
    nulls = 0
    n_minus2 = 0
    sub_n = 0
    sub_m2 = 0
    bad_raw_counter: Counter[str] = Counter()
    samples: list[tuple[object, object]] = []

    for batch in pf.iter_batches(columns=cols, batch_size=262144):
        df = batch.to_pandas()
        bn = len(df)
        total += bn
        mcc_s = df["mcc_code"]
        nulls += int(mcc_s.isna().sum())
        parsed = mcc_s.map(_parse_like_cpp_mcc)
        n_minus2 += int((parsed == MCC_MISSING).sum())
        bad = mcc_s[parsed == MCC_MISSING]
        for val, cnt in bad.value_counts(dropna=False).head(200).items():
            bad_raw_counter[repr(val)] += int(cnt)
        if has_cid and len(samples) < sample_rows:
            cid_s = df["customer_id"]
            has_c = cid_s.notna() & (cid_s.astype(str).str.strip() != "")
            sub_n += int(has_c.sum())
            sub_m2 += int((parsed == MCC_MISSING)[has_c].sum())
            mask = has_c & (parsed == MCC_MISSING)
            if mask.any():
                take = df.loc[mask, ["customer_id", "mcc_code"]].head(sample_rows - len(samples))
                for _, row in take.iterrows():
                    if len(samples) >= sample_rows:
                        break
                    samples.append((row["customer_id"], row["mcc_code"]))

    print(f"mcc_code nulls (Arrow): {nulls}")
    print(f"после парсера как в C++: MCC_MISSING (-2) = {n_minus2} ({100.0 * n_minus2 / max(total, 1):.4f}%)")
    if bad_raw_counter:
        top = bad_raw_counter.most_common(25)
        print("топ repr сырых значений среди распарсенных -2 (число строк, до 25):")
        for k, v in top:
            print(f"  {k}  count={v}")

    if has_cid:
        print(
            f"только строки с непустым customer_id: rows={sub_n}  "
            f"MCC_MISSING={sub_m2} ({100.0 * sub_m2 / max(sub_n, 1):.4f}%)"
        )
        if samples:
            print(f"примеры (customer_id, raw mcc_code) среди -2 с непустым customer_id:")
            for a, b in samples:
                print(f"  customer_id={a!r}  mcc_code={b!r}")


def _analyze_aggregate_file(path: Path) -> None:
    print("\n" + "=" * 88)
    print(f"AGGREGATE: {path.relative_to(_PROJECT_ROOT) if path.is_relative_to(_PROJECT_ROOT) else path}")
    print("=" * 88)
    t = pq.read_table(path, memory_map=True)
    print(f"rows={t.num_rows}  columns={t.column_names}")
    if "mcc_code" not in t.column_names:
        return
    col = t.column("mcc_code")
    if pa.types.is_integer(col.type):
        arr = col.combine_chunks().to_pylist()
        c = Counter(arr)
        print(f"mcc_code value counts (top 15): {c.most_common(15)}")
        n2 = c.get(MCC_MISSING, 0)
        if n2:
            print(f"  MCC_MISSING (-2): {n2} ({100.0 * n2 / max(len(arr), 1):.4f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=8)
    ap.add_argument("--train-dir", type=Path, default=_PROJECT_ROOT / "data" / "train")
    ap.add_argument("--agg-dir", type=Path, default=_PROJECT_ROOT / "output" / "global_aggregates")
    args = ap.parse_args()

    train_dir = args.train_dir.resolve()
    agg_dir = args.agg_dir.resolve()

    print("Почему остаётся -2:")
    print("  • null / пустая строка / нечисловой mcc_code")
    print("  • значения, равные зарезервированным MCC_GLOBAL_KEY / MCC_MISSING_KEY")
    print("  • строка не целиком целое и не парсится как «целое float» (например '12a')")
    print()

    if train_dir.is_dir():
        for p in sorted(train_dir.glob("*.parquet")):
            _analyze_train_parquet(p, args.sample_rows)
    else:
        print(f"Нет {train_dir}", file=sys.stderr)

    if agg_dir.is_dir():
        for name in (
            "mcc.parquet",
            "mcc_totals.parquet",
            "mcc_tz_joint.parquet",
            "mcc_channel_joint.parquet",
            "mcc_currency_joint.parquet",
            "channel_mcc_pair.parquet",
            "channel_mcc_top3.parquet",
        ):
            p = agg_dir / name
            if p.is_file():
                _analyze_aggregate_file(p)
    else:
        print(f"Нет {agg_dir}", file=sys.stderr)

    print("\nГотово.")


if __name__ == "__main__":
    main()
