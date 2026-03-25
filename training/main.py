"""
Обучение XGBoost на объединённом full_dataset.parquet.
Сплит train/val по времени; батчи parquet через DataIter + ExtMemQuantileDMatrix (XGBoost 3.x):
квантили и промежуточные данные выносятся на диск (cache_prefix), без загрузки всего parquet в RAM.
При отсутствии ExtMemQuantileDMatrix — откат на QuantileDMatrix (может потребовать много RAM).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_XGB_PATH, OUTPUT_DIR, TRAIN_DATASET_PATH
from training.config import (
    VAL_RATIO,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_EVAL_VERBOSE_EVERY,
    XGB_EXTERNAL_PARQUET_BATCH_ROWS,
    XGB_MODEL_HYPERPARAMS,
    XGB_PARAMS,
)
from training.parquet_io import detect_columns, find_time_cutoff
from training.xgb_iterative_fit import fit_xgb_parquet_iterative

logger = logging.getLogger(__name__)


def _build_xgb_train_params(config_mode: str) -> dict[str, float | int | str]:
    defaults = {
        "learning_rate": float(XGB_PARAMS.get("learning_rate", 0.05)),
        "max_depth": int(XGB_PARAMS.get("max_depth", 8)),
        "subsample": float(XGB_PARAMS.get("subsample", 0.8)),
        "colsample_bytree": float(XGB_PARAMS.get("colsample_bytree", 0.8)),
    }
    selected = XGB_MODEL_HYPERPARAMS if config_mode == "best" else defaults

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": float(selected["learning_rate"]),
        "max_depth": int(selected["max_depth"]),
        "subsample": float(selected["subsample"]),
        "colsample_bytree": float(selected["colsample_bytree"]),
        "min_child_weight": float(XGB_PARAMS.get("min_child_weight", 1.0)),
        "gamma": float(XGB_PARAMS.get("gamma", 0.0)),
        "alpha": float(XGB_PARAMS.get("reg_alpha", 0.0)),
        "lambda": float(XGB_PARAMS.get("reg_lambda", 1.0)),
        "tree_method": str(XGB_PARAMS.get("tree_method", "hist")),
        "seed": int(XGB_PARAMS.get("random_state", 42)),
    }
    return params


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xgb-config",
        choices=("best", "default"),
        default="default",
        help="default (по умолчанию): базовые XGB_PARAMS из training/config.py; "
        "best: гиперпараметры из training/grid_search/xgb_best_params.json",
    )
    args = parser.parse_args()

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. "
            "Сначала соберите датасет: ./dataset_cpp/build/build_dataset ."
        )
    feature_cols, dttm_col = detect_columns(TRAIN_DATASET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_day = find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)

    ext_train_cache = OUTPUT_DIR / "xgb_extmem_train"
    ext_val_cache = OUTPUT_DIR / "xgb_extmem_val"

    results: list[tuple[str, float]] = []

    logger.warning("CatBoost в этом скрипте не обучается — отдельный пайплайн.")

    try:
        import xgboost as xgb

        use_extmem = hasattr(xgb, "ExtMemQuantileDMatrix")
        if not use_extmem:
            logger.warning(
                "В этой сборке XGBoost нет ExtMemQuantileDMatrix — используется QuantileDMatrix "
                "(без дискового кэша итератора, возможен большой расход RAM)."
            )

        params = _build_xgb_train_params(args.xgb_config)
        rounds = max(1, int(XGB_PARAMS.get("n_estimators", 600)))
        logger.info("XGBoost config mode: %s", args.xgb_config)
        logger.info("XGBoost train params: %s, num_boost_round=%d", params, rounds)

        br = int(XGB_EXTERNAL_PARQUET_BATCH_ROWS)
        if use_extmem:
            logger.info(
                "XGBoost: ExtMemQuantileDMatrix train (batch_rows=%d, disk cache=%s) …",
                br,
                ext_train_cache,
            )
        else:
            logger.info("XGBoost: QuantileDMatrix train (batch_rows=%d) …", br)

        booster, pr = fit_xgb_parquet_iterative(
            TRAIN_DATASET_PATH,
            feature_cols,
            dttm_col,
            cutoff_day,
            params,
            num_boost_round=rounds,
            batch_rows=br,
            train_cache_dir=ext_train_cache,
            val_cache_dir=ext_val_cache,
            early_stopping_rounds=int(XGB_EARLY_STOPPING_ROUNDS),
            verbose_eval=int(XGB_EVAL_VERBOSE_EVERY),
        )

        bi = getattr(booster, "best_iteration", None)
        if bi is not None and bi >= 0:
            logger.info("XGBoost best_iteration (лучший невзвешенный aucpr на eval): %s", bi)

        results.append(("XGBoost", pr))
        booster.save_model(str(MODEL_XGB_PATH))
        logger.info("XGBoost PR-AUC (val, невзвешенный, pos_label=1): %.6f → %s", pr, MODEL_XGB_PATH)

        if use_extmem:
            logger.info(
                "Дисковый кэш ExtMemQuantileDMatrix удалён (%s, %s).",
                ext_train_cache.name,
                ext_val_cache.name,
            )

    except Exception as e:
        logger.warning("XGBoost пропущен: %s", e)
        for p in (ext_train_cache, ext_val_cache):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    logger.info("=== Сводка PR-AUC (валидация по времени) ===")
    for name, pr in sorted(results, key=lambda x: -x[1]):
        logger.info("  %s: %.6f", name, pr)
    if results:
        best = max(results, key=lambda x: x[1])
        logger.info("Лучшая по PR-AUC: %s (%.6f)", best[0], best[1])


if __name__ == "__main__":
    main()
