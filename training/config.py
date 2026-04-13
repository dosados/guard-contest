"""Параметры обучения и валидации."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Список признаков на вход модели: shared.config.MODEL_INPUT_FEATURES (все FEATURE_NAMES).

logger = logging.getLogger(__name__)

# Доля валидации по времени (последние VAL_RATIO строк после сортировки по event_dttm)
VAL_RATIO = 0.15
RANDOM_SEED = 42

GRID_SEARCH_DIR = Path(__file__).resolve().parent / "grid_search"
XGB_BEST_PARAMS_PATH = GRID_SEARCH_DIR / "xgb_best_params.json"

# Единственное место, где задаётся сетка гиперпараметров для полного перебора
# (training/xgb_grid_search.py). Ключи — те же, что использует training/main._build_xgb_train_params
# (через XGB_MODEL_HYPERPARAMS после сохранения лучшего JSON).
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

# Мониторинг eval_metric (aucpr) на val при обучении в training/main.py
XGB_EARLY_STOPPING_ROUNDS = 50  # 0 — только evals, без остановки
XGB_EVAL_VERBOSE_EVERY = 50  # печать каждые N раундов; 0 — тихо

# Размер батча pyarrow при чтении parquet для QuantileDMatrix + DataIter (training/main.py).
# Больше — меньше вызовов итератора и накладных расходов, выше пик RAM на батч (~n_rows * n_features * 4 байт
# для float32 X плюс служебное). Типичный диапазон 500k–2M; при OOM уменьшить.
XGB_EXTERNAL_PARQUET_BATCH_ROWS = 5_000_000


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
        logger.warning("Не удалось прочитать %s: %s", XGB_BEST_PARAMS_PATH, exc)
        return defaults


XGB_MODEL_HYPERPARAMS = _load_xgb_hyperparams_from_grid_search()

RF_PARAMS = {
    "n_estimators_total": 400,
    "trees_per_batch": 8,
    "max_depth": 18,
    "max_features": "sqrt",
    "min_samples_leaf": 8,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
