"""
CatBoost: один проход parquet → train.tsv и val.tsv на диске, затем один вызов
Pool(data=path, delimiter='\\t', has_header=True, column_description=...) и model.fit(train_pool).
Данные для обучения CatBoost читает с файла (не через огромный DataFrame в Python).

Отличие от training/catboost_train.py: train Pool квантуется (quantize + границы на val при --use-eval-pool),
CatBoost с thread_count=10 и used_ram_limit='30gb'.

При --use-eval-pool val тоже через Pool; иначе PR-AUC на val — потоково по parquet (без полного val Pool в RAM).
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_PATH, OUTPUT_DIR, TRAIN_DATASET_PATH
from training.catboost_tsv_cpp import export_train_val_tsv
from training.catboost_train import _streaming_val_prauc_from_parquet, _write_column_description
from training.config import (
    CATBOOST_PARAMS,
    RANDOM_SEED,
    VAL_RATIO,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_EXTERNAL_PARQUET_BATCH_ROWS,
)
from training.main import _count_val_rows, _detect_columns, _find_time_cutoff, _rm_tree

logger = logging.getLogger(__name__)

# Жёсткие настройки только для этого entrypoint (не трогаем training/catboost_train.py).
CATBOOST_THREAD_COUNT = 10
CATBOOST_USED_RAM_LIMIT = "30gb"


def _train_catboost_from_tsv_quantized(
    train_tsv: Path,
    val_tsv: Path,
    column_description: Path,
    quantization_borders_path: Path,
    *,
    catboost_params: dict,
    early_stopping_rounds: int,
    use_eval_pool: bool,
) -> tuple[float, CatBoostClassifier]:
    """Pool из TSV → quantize(train) → save_quantization_borders → quantize(val) с теми же границами → fit."""
    train_pool = Pool(
        data=str(train_tsv),
        column_description=str(column_description),
        delimiter="\t",
        has_header=True,
        thread_count=CATBOOST_THREAD_COUNT,
    )
    train_pool.quantize(
        used_ram_limit=CATBOOST_USED_RAM_LIMIT,
        random_seed=RANDOM_SEED,
    )
    quantization_borders_path.parent.mkdir(parents=True, exist_ok=True)
    train_pool.save_quantization_borders(str(quantization_borders_path))
    logger.info(
        "Квантизация train Pool, границы: %s (used_ram_limit=%s, threads=%d)",
        quantization_borders_path.name,
        CATBOOST_USED_RAM_LIMIT,
        CATBOOST_THREAD_COUNT,
    )

    val_pool = None
    if use_eval_pool and val_tsv.exists() and val_tsv.stat().st_size > 0:
        val_pool = Pool(
            data=str(val_tsv),
            column_description=str(column_description),
            delimiter="\t",
            has_header=True,
            thread_count=CATBOOST_THREAD_COUNT,
        )
        val_pool.quantize(input_borders=str(quantization_borders_path))
        logger.info("Квантизация val Pool по границам train")

    params = {k: v for k, v in catboost_params.items() if k != "verbose"}
    verbose = int(catboost_params.get("verbose", 100))

    model = CatBoostClassifier(**params)

    fit_kw: dict = {"verbose": verbose}
    if val_pool is not None:
        fit_kw["eval_set"] = val_pool
        if early_stopping_rounds > 0:
            fit_kw["early_stopping_rounds"] = early_stopping_rounds
            fit_kw["use_best_model"] = True

    model.fit(train_pool, **fit_kw)

    pr = 0.0
    if val_pool is not None:
        p = np.asarray(model.predict_proba(val_pool)[:, 1], dtype=np.float64)
        yva = np.asarray(val_pool.get_label(), dtype=np.float64).astype(np.int32)
        try:
            w_eval = np.asarray(val_pool.get_weight(), dtype=np.float64)
            if w_eval.size == yva.size:
                pr = float(average_precision_score(yva, p, sample_weight=w_eval))
                logger.info("PR-AUC (val): sklearn с sample_weight из Pool")
            else:
                pr = float(average_precision_score(yva, p))
                logger.warning("PR-AUC (val): sklearn без весов (размер weight не совпал)")
        except Exception:
            pr = float(average_precision_score(yva, p))
            logger.warning("PR-AUC (val): sklearn без весов (get_weight недоступен)")
        try:
            logger.info("CatBoost best_iteration: %s", model.get_best_iteration())
        except Exception:
            pass
    else:
        logger.info("Val Pool отключен: PR-AUC посчитаем потоково после fit.")

    return pr, model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="CatBoost: parquet → TSV на диске → Pool(path) → fit (один train Pool)"
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=None,
        help=f"батч pyarrow при чтении parquet (по умолчанию {XGB_EXTERNAL_PARQUET_BATCH_ROWS})",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help=f"0 — без early stopping (по умолчанию как XGB: {XGB_EARLY_STOPPING_ROUNDS})",
    )
    parser.add_argument(
        "--use-eval-pool",
        action="store_true",
        help="eval_set из val.tsv через Pool (больше RAM; иначе PR-AUC потоком по parquet после fit)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="каталог для TSV и column_description.cd (по умолчанию output/cat_train_tsv_cache; "
        "с C++-экспортом другой путь недопустим)",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="не удалять TSV после обучения",
    )
    parser.add_argument(
        "--no-cpp",
        action="store_true",
        help="не вызывать C++ parquet_to_catboost_tsv, только Python (pyarrow/pandas)",
    )
    parser.add_argument(
        "--cpp-threads",
        type=int,
        default=CATBOOST_THREAD_COUNT,
        help=f"число потоков C++-экспортёра (по умолчанию {CATBOOST_THREAD_COUNT}; 0 = авто по ядрам/row groups)",
    )
    parser.add_argument(
        "--remap-weight2-positives-as-zero",
        action="store_true",
        help=(
            "Как --xgb-remap-weight2-positives-as-zero в training/main: target=1 и sample_weight=2 (в parquet, "
            "до remap) в TSV как класс 0; веса через remap_sample_weight_from_dataset. При включении экспорт только "
            "через Python (C++ не поддерживает)."
        ),
    )
    args = parser.parse_args()

    batch_rows = int(args.batch_rows if args.batch_rows is not None else XGB_EXTERNAL_PARQUET_BATCH_ROWS)
    es = int(XGB_EARLY_STOPPING_ROUNDS if args.early_stopping_rounds is None else args.early_stopping_rounds)

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. Сначала соберите датасет: ./dataset_cpp/build/build_dataset ."
        )

    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)

    cache_dir = args.cache_dir if args.cache_dir is not None else OUTPUT_DIR / "cat_train_tsv_cache"
    _rm_tree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = cache_dir / "train.tsv"
    val_tsv = cache_dir / "val.tsv"
    cd_path = cache_dir / "column_description.cd"

    logger.info("Parquet → train.tsv / val.tsv, cutoff=%s …", cutoff_day.date())
    if args.remap_weight2_positives_as_zero:
        logger.info(
            "CatBoost: переназначение меток — target=1 и sample_weight=2 → label 0 в train/val TSV (как в XGB main)."
        )
    _write_column_description(cd_path, feature_cols)
    export_train_val_tsv(
        TRAIN_DATASET_PATH,
        train_tsv,
        val_tsv,
        feature_cols,
        dttm_col,
        cutoff_day,
        batch_rows,
        prefer_cpp=not args.no_cpp,
        cpp_threads=args.cpp_threads,
        remap_weight2_positive_label_to_zero=args.remap_weight2_positives_as_zero,
    )

    logger.info("Подсчёт val-строк по time split …")
    n_val = _count_val_rows(
        TRAIN_DATASET_PATH,
        dttm_col,
        cutoff_day,
        "tr_amount",
        batch_size=batch_rows,
    )
    logger.info("Val rows (оценка): %d", n_val)

    cat_params = {
        **CATBOOST_PARAMS,
        "thread_count": CATBOOST_THREAD_COUNT,
        "used_ram_limit": CATBOOST_USED_RAM_LIMIT,
    }

    try:
        pr, model = _train_catboost_from_tsv_quantized(
            train_tsv,
            val_tsv,
            cd_path,
            cache_dir / "quantization_borders.dat",
            catboost_params=cat_params,
            early_stopping_rounds=es,
            use_eval_pool=args.use_eval_pool,
        )
        if not args.use_eval_pool and n_val > 0:
            logger.info("Потоковый PR-AUC на val (без val Pool) …")
            pr = _streaming_val_prauc_from_parquet(
                model,
                TRAIN_DATASET_PATH,
                feature_cols,
                dttm_col,
                cutoff_day,
                batch_rows,
                remap_weight2_positive_label_to_zero=args.remap_weight2_positives_as_zero,
            )
            logger.info("PR-AUC (val): sklearn с sample_weight, потоковый расчёт")
        model.save_model(str(MODEL_PATH))
        logger.info("CatBoost PR-AUC (val): %.6f → %s", pr, MODEL_PATH)
    finally:
        gc.collect()
        if not args.keep_cache:
            _rm_tree(cache_dir)
            logger.info("Временный каталог TSV удалён: %s", cache_dir.name)
        else:
            logger.info("TSV оставлены в %s", cache_dir)

    logger.info("Готово.")


if __name__ == "__main__":
    main()
