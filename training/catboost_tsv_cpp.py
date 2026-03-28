"""
Вызов C++ parquet_to_catboost_tsv: быстрый экспорт full_dataset.parquet → train.tsv / val.tsv.
Python только пишет список фич и запускает бинарник (cutoff считает C++ как _find_time_cutoff + VAL_RATIO).
При отсутствии бинарника — откат на _stream_parquet_to_tsv_splits из catboost_train.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from shared.config import OUTPUT_DIR, PROJECT_ROOT, TRAIN_DATASET_PATH

logger = logging.getLogger(__name__)

_TRAINING_DIR = Path(__file__).resolve().parent
_DEFAULT_CPP_EXE = _TRAINING_DIR / "cpp" / "build" / "parquet_to_catboost_tsv"

# Совпадает с константами в training/cpp/parquet_to_catboost_tsv.cpp (cwd = PROJECT_ROOT).
CPP_TSV_CACHE_DIR = OUTPUT_DIR / "cat_train_tsv_cache"
CPP_FEATURES_FILE = CPP_TSV_CACHE_DIR / "catboost_export_features.txt"
CPP_TRAIN_TSV = CPP_TSV_CACHE_DIR / "train.tsv"
CPP_VAL_TSV = CPP_TSV_CACHE_DIR / "val.tsv"


def resolve_cpp_exe() -> Path | None:
    env = os.environ.get("GUARD_CONTEST_PARQUET_TO_CATBOOST_TSV")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    return _DEFAULT_CPP_EXE if _DEFAULT_CPP_EXE.is_file() else None


def write_feature_order(path: Path, feature_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")


def export_via_cpp(
    feature_cols: list[str],
    *,
    threads: int = 0,
    exe: Path | None = None,
) -> None:
    """Бинарник читает parquet, считает cutoff (VAL_RATIO), пишет TSV в output/cat_train_tsv_cache/."""
    bin_path = exe or resolve_cpp_exe()
    if bin_path is None or not bin_path.is_file():
        raise FileNotFoundError(
            "Нет бинарника parquet_to_catboost_tsv. Соберите:\n"
            "  cmake -S training/cpp -B training/cpp/build && cmake --build training/cpp/build -j\n"
            f"или задайте GUARD_CONTEST_PARQUET_TO_CATBOOST_TSV (ожидали {_DEFAULT_CPP_EXE})."
        )
    CPP_TSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_feature_order(CPP_FEATURES_FILE, feature_cols)
    cmd = [str(bin_path)]
    if threads and threads > 0:
        cmd.extend(["--threads", str(int(threads))])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def export_train_val_tsv(
    parquet_path: Path,
    train_tsv: Path,
    val_tsv: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    batch_rows: int,
    *,
    prefer_cpp: bool = True,
    cpp_threads: int = 0,
    remap_weight2_positive_label_to_zero: bool = False,
) -> None:
    """
    prefer_cpp: при наличии собранного parquet_to_catboost_tsv — только он; иначе Python-стриминг.
    C++ всегда режет по колонке event_dttm (как в датасете).
    remap_weight2_positive_label_to_zero: только Python-стриминг (C++ не меняет метки по весу).
    """
    use_cpp = prefer_cpp and not remap_weight2_positive_label_to_zero
    if remap_weight2_positive_label_to_zero and prefer_cpp:
        logger.info(
            "Переназначение target (sample_weight=2 в parquet → label 0): экспорт через Python, не C++."
        )
    if use_cpp:
        exe = resolve_cpp_exe()
        if exe is not None:
            if train_tsv.resolve() != CPP_TRAIN_TSV.resolve() or val_tsv.resolve() != CPP_VAL_TSV.resolve():
                raise ValueError(
                    "C++ экспортёр пишет только в output/cat_train_tsv_cache/train.tsv и val.tsv "
                    f"(ожидалось train={CPP_TRAIN_TSV}, val={CPP_VAL_TSV})."
                )
            if parquet_path.resolve() != TRAIN_DATASET_PATH.resolve():
                raise ValueError(
                    f"C++ экспортёр читает только {TRAIN_DATASET_PATH} (получено {parquet_path})."
                )
            logger.info("Экспорт TSV через C++ (%s) …", exe)
            export_via_cpp(feature_cols, threads=cpp_threads, exe=exe)
            return
        logger.warning(
            "C++ parquet_to_catboost_tsv не найден (%s), экспорт через Python (медленнее).",
            _DEFAULT_CPP_EXE,
        )
    from training.catboost_train import _stream_parquet_to_tsv_splits

    _stream_parquet_to_tsv_splits(
        parquet_path,
        feature_cols,
        dttm_col,
        cutoff_day,
        batch_rows,
        train_tsv,
        val_tsv,
        remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero,
    )
