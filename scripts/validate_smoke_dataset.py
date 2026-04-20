#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.types as pat

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_INPUT_FEATURES, TRAIN_DATASET_META_COLUMNS, TRAIN_DATASET_PATH

# Often missing in raw data; high NaN rate does not imply a broken join.
_HIGH_NAN_OK_FEATURES = frozenset({"battery_level"})


def main() -> int:
    path = TRAIN_DATASET_PATH
    nan_frac_warn = 0.99
    if not path.is_file():
        print(f"SKIP: missing file {path}", file=sys.stderr)
        return 0

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    schema = pf.schema_arrow
    missing_meta = [c for c in TRAIN_DATASET_META_COLUMNS if c not in names]
    if missing_meta:
        print(f"ERROR: missing meta columns: {missing_meta}")
        return 1
    missing_feat = [c for c in MODEL_INPUT_FEATURES if c not in names]
    if missing_feat:
        print(f"ERROR: missing features: {missing_feat[:20]}{'…' if len(missing_feat) > 20 else ''}")
        return 1

    ed = pf.read(columns=["event_dttm"]).column(0)
    null_frac = float(pc.sum(pc.cast(pc.is_null(ed), "int64")).as_py() or 0) / max(ed.length(), 1)
    if null_frac > 0.01:
        print(f"WARN: event_dttm null fraction {null_frac:.4f}")

    bad_inf: list[str] = []
    bad_nan: list[str] = []
    for col in MODEL_INPUT_FEATURES:
        if col not in names:
            continue
        typ = schema.field(col).type
        if not (pat.is_floating(typ) or pat.is_integer(typ)):
            continue
        c = pf.read(columns=[col]).column(0)
        arr = pc.cast(c, "float64", safe=False)
        n = arr.length()
        if n == 0:
            continue
        inf_m = pc.or_(pc.equal(arr, math.inf), pc.equal(arr, -math.inf))
        n_inf = int(pc.sum(pc.cast(inf_m, "int64")).as_py() or 0)
        if n_inf:
            bad_inf.append(f"{col}:{n_inf}")
        n_nan = int(pc.sum(pc.cast(pc.is_nan(arr), "int64")).as_py() or 0)
        frac = n_nan / n
        if frac >= nan_frac_warn and col not in _HIGH_NAN_OK_FEATURES:
            bad_nan.append(f"{col}:{frac:.3f}")

    if bad_inf:
        print(f"WARN: inf values in columns: {bad_inf[:15]}")
    if bad_nan:
        print(f"WARN: very high NaN fraction: {bad_nan[:15]}")

    eid = pf.read(columns=["event_id"]).column(0)
    n_rows = eid.length()
    n_unique = int(pc.count_distinct(eid).as_py() or 0)
    if n_unique != n_rows:
        print(f"WARN: event_id duplicates? rows={n_rows} distinct={n_unique}")

    print(f"OK: {path} rows={n_rows} features={len(MODEL_INPUT_FEATURES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
