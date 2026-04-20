#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import GLOBAL_CATEGORY_AGGREGATES_DIR, TRAIN_DATASET_PATH
from shared.global_category_aggregates import GLOBAL_CATEGORY_FEATURE_NAMES, GlobalCategoryLookups


def main() -> int:
    dataset = TRAIN_DATASET_PATH
    aggregates_dir = GLOBAL_CATEGORY_AGGREGATES_DIR
    n_rows = 200
    rtol = 1e-5
    atol = 1e-6
    if not dataset.is_file():
        print(f"SKIP: missing {dataset}", file=sys.stderr)
        return 0
    if not aggregates_dir.is_dir():
        print(f"SKIP: aggregates directory missing {aggregates_dir}", file=sys.stderr)
        return 0

    lookups = GlobalCategoryLookups(aggregates_dir)
    cols = list(GLOBAL_CATEGORY_FEATURE_NAMES) + [
        "mcc_code",
        "channel_indicator_type",
        "channel_indicator_subtype",
        "currency_iso_cd",
        "timezone",
        "event_type_nm",
        "event_descr",
        "event_desc",
        "pos_cd",
        "operation_amt",
    ]
    pf = pq.ParquetFile(dataset)
    available = [c for c in cols if c in pf.schema_arrow.names]
    t = pf.read(columns=available).slice(0, n_rows).to_pandas()

    mismatches = 0
    for _, row in t.iterrows():
        rd = row.to_dict()
        op = float(rd.get("operation_amt", float("nan")))
        py = lookups.features_for_row(rd, op)
        for name in GLOBAL_CATEGORY_FEATURE_NAMES:
            a = float(rd.get(name, float("nan")))
            b = float(py[name])
            if not (math.isfinite(a) and math.isfinite(b)):
                if math.isnan(a) and math.isnan(b):
                    continue
                mismatches += 1
                break
            if not np.isclose(a, b, rtol=rtol, atol=atol):
                mismatches += 1
                print(f"MISMATCH {name}: parquet={a} python={b}")
                break

    if mismatches:
        print(f"FAIL: {mismatches} rows differ (checked first mismatch per row)")
        return 1
    print(f"OK: {len(t)} rows, {len(GLOBAL_CATEGORY_FEATURE_NAMES)} global features each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
