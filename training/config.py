"""Параметры обучения и валидации."""

from __future__ import annotations

import json
import logging
from pathlib import Path

# Список признаков на вход модели: shared.config.MODEL_INPUT_FEATURES (все FEATURE_NAMES).

logger = logging.getLogger(__name__)

# Доля валидации по времени (последние VAL_RATIO строк после сортировки по event_dttm)
VAL_RATIO = 0.15
RANDOM_SEED = 42

GRID_SEARCH_DIR = Path(__file__).resolve().parent / "grid_search"
XGB_BEST_PARAMS_PATH = GRID_SEARCH_DIR / "xgb_best_params.json"

CATBOOST_PARAMS = {
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "random_seed": RANDOM_SEED,
    "verbose": 100,
}

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
    "eval_metric": "aucpr",
}

# Мониторинг eval_metric (aucpr) на val при обучении в training/main.py
XGB_EARLY_STOPPING_ROUNDS = 50  # 0 — только evals, без остановки
XGB_EVAL_VERBOSE_EVERY = 50  # печать каждые N раундов; 0 — тихо

# Размер батча pyarrow при чтении parquet для QuantileDMatrix + DataIter (training/main.py)
XGB_EXTERNAL_PARQUET_BATCH_ROWS = 1_000_000


def _load_xgb_hyperparams_from_grid_search() -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        "learning_rate": float(XGB_PARAMS["learning_rate"]),
        "max_depth": int(XGB_PARAMS["max_depth"]),
        "subsample": float(XGB_PARAMS["subsample"]),
        "colsample_bytree": float(XGB_PARAMS["colsample_bytree"]),
    }
    if not XGB_BEST_PARAMS_PATH.exists():
        return defaults
    try:
        payload = json.loads(XGB_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        hp = payload.get("hyperparameters", {})
        return {
            "learning_rate": float(hp.get("learning_rate", defaults["learning_rate"])),
            "max_depth": int(hp.get("max_depth", defaults["max_depth"])),
            "subsample": float(hp.get("subsample", defaults["subsample"])),
            "colsample_bytree": float(hp.get("colsample_bytree", defaults["colsample_bytree"])),
        }
    except Exception as exc:
        logger.warning("Не удалось прочитать %s: %s", XGB_BEST_PARAMS_PATH, exc)
        return defaults


XGB_MODEL_HYPERPARAMS = _load_xgb_hyperparams_from_grid_search()

LGBM_PARAMS = {
    "n_estimators": 800,
    "max_depth": -1,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators_total": 400,
    "trees_per_batch": 8,
    "max_depth": 18,
    "max_features": "sqrt",
    "min_samples_leaf": 8,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

LR_PARAMS = {
    "alpha": 1e-4,
    "penalty": "l2",
    "fit_intercept": True,
    "max_iter_per_batch": 1,
    "random_state": RANDOM_SEED,
}
