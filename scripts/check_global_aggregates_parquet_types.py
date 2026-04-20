#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import GLOBAL_CATEGORY_AGGREGATES_DIR

MCC_GLOBAL_KEY = -(2**63)
MCC_MISSING_KEY = -2


def _arrow_type_str(t: pa.DataType) -> str:
    s = str(t)
    if pa.types.is_dictionary(t):
        return f"dictionary(index={t.index_type}, value={t.value_type})"
    return s


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(_PROJECT_ROOT)
    except ValueError:
        return path


def _print_schema_compact(path: Path) -> None:
    schema = pq.read_schema(path)
    pf = pq.ParquetFile(path)
    nrows = pf.metadata.num_rows
    print(f"\n{'=' * 80}\n{_rel(path)}")
    print(f"num_rows (metadata): {nrows}  columns: {len(schema.names)}")
    for i, name in enumerate(schema.names):
        f = schema.field(i)
        print(f"  {i:3d}  {name:44s}  {_arrow_type_str(f.type):32s}  nullable={f.nullable}")


def _summarize_one_column(path: Path, colname: str) -> None:
    schema = pq.read_schema(path)
    if colname not in schema.names:
        print(f"  (no column {colname!r})")
        return
    print(f"  --- column {colname!r} ---")
    try:
        table = pq.read_table(path, columns=[colname], memory_map=True)
    except Exception as e:
        print(f"  ERROR read: {e}")
        return
    arr = table.column(0)
    t = arr.type
    print(f"  physical type: {_arrow_type_str(t)}")
    if pa.types.is_dictionary(t):
        arr = arr.combine_chunks()
        arr = arr.dictionary_decode()
        t = arr.type
        print(f"  after dictionary_decode: {_arrow_type_str(t)}")
    n = len(arr)
    nulls = pc.sum(pc.cast(pc.is_null(arr), pa.int64())).as_py()
    print(f"  rows: {n}  nulls: {nulls}")
    mask = pc.is_valid(arr)
    if not pc.any(mask).as_py():
        print("  (all null)")
        return
    v = pc.filter(arr, mask)
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        print(f"  min: {pc.min(v).as_py()}  max: {pc.max(v).as_py()}")
        if pa.types.is_integer(t):
            for sentinel, label in [(MCC_MISSING_KEY, "MCC_MISSING_KEY"), (MCC_GLOBAL_KEY, "MCC_GLOBAL_KEY")]:
                c = pc.sum(pc.cast(pc.equal(arr, sentinel), pa.int64())).as_py()
                if c:
                    print(f"  count == {sentinel} ({label}): {c}")
    elif pa.types.is_string(t) or pa.types.is_large_string(t):
        lens = pc.utf8_length(v)
        print(
            f"  utf8_length: min={pc.min(lens).as_py()} max={pc.max(lens).as_py()} "
            f"mean={pc.mean(pc.cast(lens, pa.float64())).as_py():.2f}"
        )
        show = min(5, len(v))
        print(f"  first values: {[v[i].as_py() for i in range(show)]}")


def _summarize_column(path: Path, name: str, arr: pa.ChunkedArray) -> None:
    t = arr.type
    n = arr.length()
    nulls = pc.sum(pc.cast(pc.is_null(arr), pa.int64())).as_py()
    print(f"    dtype: {_arrow_type_str(t)}")
    print(f"    rows: {n}  nulls: {nulls}")

    if pa.types.is_integer(t) or pa.types.is_floating(t):
        mask = pc.is_valid(arr)
        if not pc.any(mask).as_py():
            print("    (all null)")
            return
        v = pc.filter(arr, mask)
        mn = pc.min(v)
        mx = pc.max(v)
        print(f"    min: {mn.as_py()}  max: {mx.as_py()}")
        if pa.types.is_integer(t):
            for sentinel, label in [
                (MCC_MISSING_KEY, "MCC_MISSING_KEY (-2)"),
                (MCC_GLOBAL_KEY, "MCC_GLOBAL_KEY (-2**63)"),
            ]:
                cnt = pc.sum(pc.cast(pc.equal(arr, sentinel), pa.int64())).as_py()
                if cnt:
                    print(f"    count == {sentinel} ({label}): {cnt}")
        mn_v, mx_v = mn.as_py(), mx.as_py()
        if mn_v is not None and mx_v is not None and pa.types.is_integer(t):
            if mn_v < -10_000_000 or mx_v > 10_000_000:
                print("    *** suspicious numeric range ***")
        return

    if pa.types.is_string(t) or pa.types.is_large_string(t):
        lens = pc.utf8_length(pc.fill_null(arr, ""))
        mask = pc.is_valid(arr)
        if pc.any(mask).as_py():
            lv = pc.filter(lens, mask)
            print(
                f"    utf8_length: min={pc.min(lv).as_py()} max={pc.max(lv).as_py()} "
                f"mean={pc.mean(pc.cast(lv, pa.float64())).as_py():.2f}"
            )
        return

    print(f"    (no numeric summary for type {t})")


def _inspect_file_all_columns(path: Path, *, max_string_uniques: int = 12) -> None:
    print(f"\n{'=' * 80}\n{_rel(path)}\n{'=' * 80}")
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print(f"num_row_groups: {meta.num_row_groups}  num_rows: {meta.num_rows}")
    schema = pf.schema_arrow
    for i, name in enumerate(schema.names):
        field = schema.field(i)
        print(f"\n  [{i}] {name}")
        print(f"      schema type: {_arrow_type_str(field.type)}  nullable={field.nullable}")
        st_parts: list[str] = []
        for rg in range(meta.num_row_groups):
            col = meta.row_group(rg).column(i)
            st = col.statistics
            if st is None or not st.has_min_max:
                continue
            try:
                st_parts.append(f"rg{rg}[{st.min},{st.max}]")
            except Exception:
                st_parts.append(f"rg{rg}[stats?]")
        if st_parts:
            print(f"      parquet stats min/max: {'; '.join(st_parts[:4])}{'…' if len(st_parts) > 4 else ''}")
        try:
            table = pq.read_table(path, columns=[name], memory_map=True)
        except Exception as e:
            print(f"      ERROR read column: {e}")
            continue
        col = table.column(0)
        _summarize_column(path, name, col)
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            uniq = pc.unique(col)
            nu = len(uniq)
            print(f"    approx distinct strings: {nu}")
            if nu <= max_string_uniques and nu > 0:
                vals = [uniq[j].as_py() for j in range(nu)]
                print(f"    values: {vals}")


def _mcc_type_from_schema(path: Path) -> None:
    # mcc_code type from parquet schema only
    sch = pq.read_schema(path)
    if "mcc_code" not in sch.names:
        print("  mcc_code: (column missing from schema)")
        return
    f = sch.field("mcc_code")
    print(f"  mcc_code (parquet schema type): {_arrow_type_str(f.type)}  nullable={f.nullable}")
    # Parquet stats from first row groups when present
    pf = pq.ParquetFile(path)
    idx = sch.get_field_index("mcc_code")
    st_txt: list[str] = []
    for rg in range(min(pf.metadata.num_row_groups, 3)):
        col = pf.metadata.row_group(rg).column(idx)
        st = col.statistics
        if st is not None and getattr(st, "has_min_max", None) and st.has_min_max:
            try:
                st_txt.append(f"rg{rg}[{st.min},{st.max}]")
            except Exception:
                st_txt.append(f"rg{rg}[?]")
    if st_txt:
        print(f"  parquet min/max (up to 3 RG): {'; '.join(st_txt)}")


def _scan_training_sources(train_dir: Path, labels_path: Path, *, train_mcc_full_stats: bool) -> None:
    print("\n" + "#" * 80)
    print("# SOURCES: data/train/*.parquet")
    print("#" * 80)
    if not train_dir.is_dir():
        print(f"Directory missing: {train_dir}", file=sys.stderr)
    else:
        for p in sorted(train_dir.glob("*.parquet")):
            _print_schema_compact(p)
            print("  mcc_code:")
            _mcc_type_from_schema(p)
            if train_mcc_full_stats:
                print("  (full read of mcc_code column - may be slow)")
                _summarize_one_column(p, "mcc_code")
            sch = pq.read_schema(p)
            if "event_type_nm" in sch.names:
                print("  event_type_nm (schema type):", _arrow_type_str(sch.field("event_type_nm").type))
                if train_mcc_full_stats:
                    _summarize_one_column(p, "event_type_nm")

    print("\n" + "#" * 80)
    print("# SOURCE: train_labels.parquet")
    print("#" * 80)
    if not labels_path.is_file():
        print(f"File missing: {labels_path}", file=sys.stderr)
    else:
        _print_schema_compact(labels_path)
        for c in ("event_id", "target"):
            sch = pq.read_schema(labels_path)
            if c in sch.names:
                print(f"  {c} (schema type): {_arrow_type_str(sch.field(c).type)}")
                if train_mcc_full_stats or labels_path.stat().st_size < 50_000_000:
                    _summarize_one_column(labels_path, c)


def _scan_aggregates(agg_dir: Path, *, mode: str) -> None:
    print("\n" + "#" * 80)
    print("# OUTPUT: output/datasets/global_aggregates/*.parquet")
    print("#" * 80)
    if not agg_dir.is_dir():
        print(f"Directory missing: {agg_dir}", file=sys.stderr)
        return
    paths = sorted(agg_dir.glob("*.parquet"))
    if not paths:
        print(f"No *.parquet in {agg_dir}", file=sys.stderr)
        return
    for p in paths:
        _print_schema_compact(p)
        if mode == "schema":
            continue
        if mode == "light":
            sch = pq.read_schema(p)
            for hint in ("mcc_code", "cnt", "n_rows", "ch_row_total", "top1_mcc", "top2_mcc", "top3_mcc"):
                if hint in sch.names:
                    _summarize_one_column(p, hint)
            continue
        _inspect_file_all_columns(p)


def main() -> None:
    train_dir = _PROJECT_ROOT / "data" / "train"
    labels = _PROJECT_ROOT / "data" / "train_labels.parquet"
    aggregates_dir = GLOBAL_CATEGORY_AGGREGATES_DIR
    aggregates_schema_only = False
    aggregates_light = False
    no_train = False
    no_aggregates = False
    train_mcc_full_stats = False

    if aggregates_schema_only:
        agg_mode = "schema"
    elif aggregates_light:
        agg_mode = "light"
    else:
        agg_mode = "full"

    if not no_train:
        _scan_training_sources(
            train_dir.resolve(),
            labels.resolve(),
            train_mcc_full_stats=train_mcc_full_stats,
        )
    if not no_aggregates:
        _scan_aggregates(aggregates_dir.resolve(), mode=agg_mode)

    print("\nHint: if train mcc_code is INT64/FLOAT but build_global_aggregates reads it via "
          "col_get_str(), MCC parsing may yield empty strings -> joins use -2 (MCC_MISSING_KEY) everywhere.\n")
    print("Done.")


if __name__ == "__main__":
    main()
