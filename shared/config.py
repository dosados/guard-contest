from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from shared.dataset_settings import DATASET_MODE, WINDOW_TRANSACTIONS, WINDOW_TRANSACTIONS_MODE
from shared.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DATA_ROOT = DATA_ROOT / "train"
TEST_DATA_ROOT = DATA_ROOT / "test"

PRETRAIN_PATHS = [
    TRAIN_DATA_ROOT / "pretrain_part_1.parquet",
    TRAIN_DATA_ROOT / "pretrain_part_2.parquet",
    TRAIN_DATA_ROOT / "pretrain_part_3.parquet",
]
TRAIN_PATHS = [
    TRAIN_DATA_ROOT / "train_part_1.parquet",
    TRAIN_DATA_ROOT / "train_part_2.parquet",
    TRAIN_DATA_ROOT / "train_part_3.parquet",
]

TRAIN_LABELS_PATH = DATA_ROOT / "train_labels.parquet"
PRETEST_PATH = TEST_DATA_ROOT / "pretest.parquet"
TEST_PATH = TEST_DATA_ROOT / "test.parquet"

# Artifact root; datasets and models live under subdirs (see dataset_cpp build_*).
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASETS_DIR = OUTPUT_DIR / "datasets"
TRAIN_DATASET_DIR = DATASETS_DIR / "train"
GLOBAL_CATEGORY_AGGREGATES_DIR = DATASETS_DIR / "global_aggregates"
TRAIN_DATASET_PATH = TRAIN_DATASET_DIR / "full_dataset.parquet"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


# Optional env overrides: GUARD_TRAIN_DATASET, GUARD_GLOBAL_AGGREGATES_DIR, GUARD_MODEL_*.
if (_p := _env_path("GUARD_TRAIN_DATASET")) is not None:
    TRAIN_DATASET_PATH = _p
if (_p := _env_path("GUARD_GLOBAL_AGGREGATES_DIR")) is not None:
    GLOBAL_CATEGORY_AGGREGATES_DIR = _p

MODELS_DIR = OUTPUT_DIR / "models"
MODEL_XGB_DIR = MODELS_DIR / "xgb"
MODEL_CAT_DIR = MODELS_DIR / "cat"
MODEL_XGB_PATH = MODEL_XGB_DIR / "model_xgb.json"
MODEL_CAT_PATH = MODEL_CAT_DIR / "model_cat.cbm"

if (_p := _env_path("GUARD_MODEL_XGB_PATH")) is not None:
    MODEL_XGB_PATH = _p
    MODEL_XGB_DIR = _p.parent
if (_p := _env_path("GUARD_MODEL_CAT_PATH")) is not None:
    MODEL_CAT_PATH = _p
    MODEL_CAT_DIR = _p.parent

RESEARCH_DIR = OUTPUT_DIR / "research"
RESEARCH_XGB_DIR = RESEARCH_DIR / "xgb"
RESEARCH_CAT_DIR = RESEARCH_DIR / "cat"

SUBMISSION_DIR = OUTPUT_DIR / "submission"
SUBMISSION_XGB_PATH = SUBMISSION_DIR / "xgb_submission.csv"
SUBMISSION_CAT_PATH = SUBMISSION_DIR / "cat_submission.csv"

CACHE_DIR = OUTPUT_DIR / "cache"
XGB_EXTMEM_TRAIN_SINGLE_DIR = CACHE_DIR / "xgb_extmem_train_single"
XGB_EXTMEM_VAL_SINGLE_DIR = CACHE_DIR / "xgb_extmem_val_single"
XGB_EXTMEM_GRID_TRAIN_DIR = CACHE_DIR / "xgb_grid_extmem_train"
XGB_EXTMEM_GRID_VAL_DIR = CACHE_DIR / "xgb_grid_extmem_val"
CATBOOST_TRAIN_DIR = CACHE_DIR / "cat_train_logs"

# Meta columns in C++ full_dataset (not MODEL_INPUT_FEATURES).
TRAIN_DATASET_META_COLUMNS: tuple[str, ...] = (
    "mcc_code",
    "event_type_nm",
    "event_descr",
    "currency_iso_cd",
    "pos_cd",
    "accept_language",
    "screen_size",
    "timezone",
    "channel_indicator_type",
    "channel_indicator_subtype",
    "customer_id",
    "event_id",
    "target",
    "sample_weight",
    "event_dttm",
)

# Parquet read batch size
BATCH_SIZE = 65_536
WINDOWED_BATCH_SIZE = BATCH_SIZE

# WINDOW_TRANSACTIONS, DATASET_MODE, WINDOW_TRANSACTIONS_MODE live in shared.dataset_settings

# Label join weights (see dataset build).
WEIGHT_UNLABELED = 1.0
WEIGHT_LABELED_0 = 2.0
WEIGHT_LABELED_1 = 5.0


def remap_sample_weight_from_dataset(weights: np.ndarray) -> np.ndarray:
    # parquet weights 1/2/5 → training 1/5/10 (order-sensitive)
    orig = np.asarray(weights, dtype=np.float32)
    out = orig.copy()
    close = np.isclose(orig, np.float32(5.0), rtol=0.0, atol=1e-5)
    out[close] = np.float32(10.0)
    close2 = np.isclose(orig, np.float32(2.0), rtol=0.0, atol=1e-5)
    out[close2] = np.float32(5.0)
    return out


# All dataset features in FEATURE_NAMES order (training and submission).
MODEL_INPUT_FEATURES: list[str] = list(FEATURE_NAMES)


def validate_model_input_dataframe(X: pd.DataFrame) -> None:
    # Inference: columns must match MODEL_INPUT_FEATURES order
    expected = MODEL_INPUT_FEATURES
    n_exp = len(expected)
    if X.shape[1] != n_exp:
        raise ValueError(
            f"Feature matrix: got {X.shape[1]} columns, expected {n_exp} (MODEL_INPUT_FEATURES). "
            "Rebuild dataset and models if FEATURE_NAMES changed."
        )
    actual = list(X.columns)
    if actual != expected:
        for i, pair in enumerate(zip(actual, expected)):
            a, e = pair
            if a != e:
                raise ValueError(
                    f"Column #{i}: got '{a}', expected '{e}'. "
                    "Use X = pd.DataFrame(rows)[MODEL_INPUT_FEATURES] after compute_features."
                )
        raise ValueError("Column set of X does not match MODEL_INPUT_FEATURES.")


def validate_xgboost_booster_feature_count(booster: Any) -> None:
    # booster.num_features vs len(MODEL_INPUT_FEATURES)
    expected_n = len(MODEL_INPUT_FEATURES)
    try:
        nf = int(booster.num_features())
    except (AttributeError, TypeError, ValueError):
        return
    if nf != expected_n:
        raise ValueError(
            f"XGBoost: booster.num_features()={nf}, code has len(MODEL_INPUT_FEATURES)={expected_n}. "
            "Retrain models on the current full_dataset.parquet."
        )


def resolve_model_input_columns(parquet_schema_names: Sequence[str]) -> list[str]:
    # Require FEATURE_NAMES in schema; return MODEL_INPUT_FEATURES order
    available = frozenset(parquet_schema_names)
    missing = [c for c in FEATURE_NAMES if c not in available]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing}")
    return list(MODEL_INPUT_FEATURES)
