from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Model input feature list: shared.config.MODEL_INPUT_FEATURES 

logger = logging.getLogger(__name__)

VAL_RATIO = 0.15
RANDOM_SEED = 42

GRID_SEARCH_DIR = Path(__file__).resolve().parent / "grid_search"
XGB_BEST_PARAMS_PATH = GRID_SEARCH_DIR / "xgb_best_params.json"
CAT_BEST_PARAMS_PATH = GRID_SEARCH_DIR / "cat_best_params.json"

XGB_PARAM_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.05],
    "max_depth": [6,5,4],
    "subsample": [0.9],
    "colsample_bytree": [0.8],
    "min_child_weight": [7, 9],
    "gamma": [5.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [5.0],
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

XGB_EARLY_STOPPING_ROUNDS = 50  # 0 = eval only, no early stopping
XGB_EVAL_VERBOSE_EVERY = 50  # log every N rounds; 0 = silent

# Parquet batch rows for QuantileDMatrix DataIter; override via GUARD_XGB_PARQUET_BATCH_ROWS.
XGB_EXTERNAL_PARQUET_BATCH_ROWS = 1_000_000

CAT_MAX_TRAIN_ROWS = 2_000_000
CAT_MAX_VAL_ROWS = 600_000
CAT_PARQUET_BATCH_SIZE = 250_000
CATBOOST_ITERATIONS = 600
CATBOOST_GRID_ITERATIONS = 200


def _load_xgb_hyperparams_from_grid_search() -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        "learning_rate": float(XGB_PARAMS["learning_rate"]),
        "max_depth": int(XGB_PARAMS["max_depth"]),
        "subsample": float(XGB_PARAMS["subsample"]),
        "colsample_bytree": float(XGB_PARAMS["colsample_bytree"]),
        "min_child_weight": float(XGB_PARAMS.get("min_child_weight", 1.0)),
        "gamma": float(XGB_PARAMS.get("gamma", 0.0)),
        "reg_alpha": float(XGB_PARAMS.get("reg_alpha", 0.0)),
        "reg_lambda": float(XGB_PARAMS.get("reg_lambda", 1.0)),
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
            "min_child_weight": float(hp.get("min_child_weight", defaults["min_child_weight"])),
            "gamma": float(hp.get("gamma", defaults["gamma"])),
            "reg_alpha": float(hp.get("reg_alpha", defaults["reg_alpha"])),
            "reg_lambda": float(hp.get("reg_lambda", defaults["reg_lambda"])),
        }
    except Exception as exc:
        logger.warning("Could not read %s: %s", XGB_BEST_PARAMS_PATH, exc)
        return defaults


XGB_MODEL_HYPERPARAMS = _load_xgb_hyperparams_from_grid_search()

CAT_PARAM_GRID: dict[str, list[Any]] = {
    "depth": [6, 8],
    "learning_rate": [0.05, 0.03],
    "l2_leaf_reg": [3.0, 5.0],
}

_CAT_DEFAULTS: dict[str, float | int] = {
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_SEED,
}


def _load_cat_hyperparams_from_grid_search() -> dict[str, float | int]:
    defaults = dict(_CAT_DEFAULTS)
    if not CAT_BEST_PARAMS_PATH.exists():
        return defaults
    try:
        payload = json.loads(CAT_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        hp = payload.get("hyperparameters", {})
        return {
            "depth": int(hp.get("depth", defaults["depth"])),
            "learning_rate": float(hp.get("learning_rate", defaults["learning_rate"])),
            "l2_leaf_reg": float(hp.get("l2_leaf_reg", defaults["l2_leaf_reg"])),
            "random_seed": int(hp.get("random_seed", defaults["random_seed"])),
        }
    except Exception as exc:
        logger.warning("Could not read %s: %s", CAT_BEST_PARAMS_PATH, exc)
        return defaults


CAT_MODEL_HYPERPARAMS = _load_cat_hyperparams_from_grid_search()
