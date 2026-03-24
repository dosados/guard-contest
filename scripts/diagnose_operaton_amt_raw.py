"""
Диагностика сырых train/pretrain parquet для поля суммы (operaton_amt / operation_amt).

Читает только нужные колонки, побатчево по row groups — укладывается в память на больших файлах.

Запуск из корня репозитория:
  PYTHONPATH=. python3 scripts/diagnose_operaton_amt_raw.py
  PYTHONPATH=. python3 scripts/diagnose_operaton_amt_raw.py --train /path/train_part_1.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import PRETRAIN_PATHS, TRAIN_PATHS

CANDIDATE_NAMES = ("operaton_amt", "operation_amt")


def _print_schema(names: list[str]) -> None:
    print(f"  Все колонки ({len(names)}):")
    for n in names:
        print(f"    - {n}")


def _columns_like_amt(names: list[str]) -> list[str]:
    out = []
    for n in names:
        ln = n.lower()
        if "amt" in ln or "operat" in ln:
            out.append(n)
    return sorted(set(out))


def _chunk_path_key(path_in_schema: object) -> str:
    if path_in_schema is None:
        return ""
    if isinstance(path_in_schema, str):
        return path_in_schema
    if isinstance(path_in_schema, (list, tuple)):
        return ".".join(str(x) for x in path_in_schema)
    return str(path_in_schema)


def _parquet_chunk_stats(path: Path, col_name: str) -> None:
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print(f"  Row groups: {meta.num_row_groups}, rows (metadata): {meta.num_rows}")
    for rg in range(meta.num_row_groups):
        rg_meta = meta.row_group(rg)
        for col_idx in range(rg_meta.num_columns):
            col_meta = rg_meta.column(col_idx)
            if _chunk_path_key(col_meta.path_in_schema) != col_name:
                continue
            st = col_meta.statistics
            print(f"  Row group {rg}: statistics for {col_name!r}:")
            if st is None or not st.has_min_max:
                print("    (нет min/max в метаданных)")
            else:
                print(f"    min={st.min} max={st.max}")
            if hasattr(st, "null_count") and st.null_count is not None:
                print(f"    null_count (chunk stat)={st.null_count}")
            return
    print(f"  (в метаданных row groups нет колонки {col_name!r} — редкий случай)")


def _sum_true_mask(mask: pa.Array | pa.ChunkedArray) -> int:
    """Сумма True в boolean-маске (null → False)."""
    filled = pc.fill_null(mask, False)
    s = pc.sum(pc.cast(filled, pa.int64()))
    v = s.as_py()
    return int(v) if v is not None else 0


def _scan_column(path: Path, col_name: str) -> None:
    pf = pq.ParquetFile(path)
    if col_name not in pf.schema_arrow.names:
        return

    field = pf.schema_arrow.field(col_name)
    print(f"  Тип Arrow: {field.type} (nullable={field.nullable})")

    total = 0
    arrow_null = 0
    # для float: не-NaN и конечные
    finite_ct = 0
    nan_ct = 0
    inf_ct = 0
    nonempty_str = 0
    first_example: str | None = None
    warned_unsupported_type = False

    n_rg = pf.num_row_groups
    for rg in range(n_rg):
        try:
            col = pf.read_row_group(rg, columns=[col_name]).column(0)
        except Exception as e:
            print(f"  [!] Row group {rg}/{n_rg}: ошибка чтения: {e}")
            raise
        total += col.length()
        arrow_null += col.null_count

        if pa.types.is_floating(field.type):
            # Без combine_chunks / to_numpy — меньше пиков памяти на огромных row groups
            valid = pc.is_valid(col)
            finite_ct += _sum_true_mask(pc.and_(valid, pc.is_finite(col)))
            nan_ct += _sum_true_mask(pc.and_(valid, pc.is_nan(col)))
            inf_ct += _sum_true_mask(pc.and_(valid, pc.is_inf(col)))
        elif pa.types.is_integer(field.type):
            finite_ct += col.length() - col.null_count
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            for chunk in col.chunks:
                for v in chunk.to_pylist():
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s:
                        nonempty_str += 1
                        if first_example is None:
                            first_example = s[:80] + ("…" if len(s) > 80 else "")
        else:
            warned_unsupported_type = True

    if warned_unsupported_type:
        print(
            f"  [!] Тип {field.type} не обрабатывается col_get_str в build_dataset.cpp "
            "(string/int32/int64) — сумма для C++ будет как пустая."
        )

    print(f"  Строк всего: {total}")
    print(f"  Arrow null: {arrow_null}")
    if pa.types.is_floating(field.type):
        print(f"  Значения float: finite={finite_ct}, NaN={nan_ct}, Inf={inf_ct}")
        print(
            "  (build_dataset читает сумму через col_get_str → float/double в C++ сейчас не поддерживаются)"
        )
    elif pa.types.is_integer(field.type):
        print(f"  Целые не-null (как увидит C++ после to_string): {finite_ct}")
    elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        print(f"  Непустых строк после trim: {nonempty_str}")
        if first_example is not None:
            print(f"  Пример значения: {first_example!r}")


def analyze_file(path: Path, label: str) -> None:
    print()
    print("=" * 72)
    print(f"{label}: {path}")
    print("=" * 72)
    if not path.is_file():
        print("  Файл не найден.")
        return

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    _print_schema(names)

    amt_like = _columns_like_amt(names)
    if amt_like:
        print("  Колонки с 'amt' или 'operat' в имени:")
        for n in amt_like:
            t = pf.schema_arrow.field(n).type
            print(f"    - {n}: {t}")

    for cand in CANDIDATE_NAMES:
        if cand in names:
            print()
            print(f"--- Колонка {cand!r} ---")
            _parquet_chunk_stats(path, cand)
            _scan_column(path, cand)
        else:
            print()
            print(f"--- Колонка {cand!r}: отсутствует в схеме ---")


def main() -> None:
    p = argparse.ArgumentParser(description="Диагностика operaton_amt в сырых parquet")
    p.add_argument("--train", type=Path, default=TRAIN_PATHS[0], help="train_part_1.parquet")
    p.add_argument("--pretrain", type=Path, default=PRETRAIN_PATHS[0], help="pretrain_part_1.parquet")
    args = p.parse_args()

    print("Сырые данные: проверка колонок суммы для build_dataset (ожидается имя operaton_amt).")
    analyze_file(args.train, "TRAIN")
    analyze_file(args.pretrain, "PRETRAIN")

    print()
    print("=" * 72)
    print("Итог для отладки пустого operaton_amt в full_dataset.parquet:")
    print("  1) В схеме должна быть колонка operaton_amt (не operation_amt), если так задумано в C++.")
    print("  2) Тип string или int32/int64 — C++ col_get_str умеет; float/double — нет (всегда пусто → NaN).")
    print("  3) Смотрите Arrow null и NaN для float — оба дают 'пустую' сумму на выходе.")
    print("=" * 72)


if __name__ == "__main__":
    main()
