"""Параметры обучения и валидации."""

from __future__ import annotations

import logging

# Список признаков на вход модели и исключения: shared.config.MODEL_INPUT_FEATURES, MODEL_FEATURES_EXCLUDED.

logger = logging.getLogger(__name__)

# Доля валидации по времени (последние VAL_RATIO строк после сортировки по event_dttm)
VAL_RATIO = 0.15
RANDOM_SEED = 42

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
