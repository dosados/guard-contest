"""Пути к данным и параметры пайплайна."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

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

OUTPUT_DIR = PROJECT_ROOT / "output"
TRAIN_DATASET_PATH = OUTPUT_DIR / "full_dataset.parquet"
# Колонки в full_dataset.parquet от C++ build_dataset, не признаки модели (см. MODEL_INPUT_FEATURES).
TRAIN_DATASET_META_COLUMNS: tuple[str, ...] = ("customer_id",)

# Побатчевое чтение parquet
BATCH_SIZE = 65_536
WINDOWED_BATCH_SIZE = BATCH_SIZE

# WINDOW_TRANSACTIONS, DATASET_MODE, WINDOW_TRANSACTIONS_MODE — в shared.dataset_settings

# Веса обучения: при event_id в train_labels — target в датасете всегда 1, вес по исходному target в parquet (0/1); без записи в labels — target 0, WEIGHT_UNLABELED
WEIGHT_UNLABELED = 1.0
WEIGHT_LABELED_0 = 2.0  # «жёлтый свет» в исходной разметке
WEIGHT_LABELED_1 = 5.0  # «красный», целевой класс в исходной разметке


def remap_sample_weight_from_dataset(weights: np.ndarray) -> np.ndarray:
    """
    Дискретные sample_weight из parquet (1 / 2 / 5) → веса для обучения (1 / 5 / 10).
    Сопоставление по исходному массиву, чтобы 2→5 не попало под правило 5→10.
    """
    orig = np.asarray(weights, dtype=np.float32)
    out = orig.copy()
    close = np.isclose(orig, np.float32(5.0), rtol=0.0, atol=1e-5)
    out[close] = np.float32(10.0)
    close2 = np.isclose(orig, np.float32(2.0), rtol=0.0, atol=1e-5)
    out[close2] = np.float32(5.0)
    return out


# Обучение
MODEL_PATH = OUTPUT_DIR / "model.cbm"
MODEL_XGB_PATH = OUTPUT_DIR / "model_xgb.json"
MODEL_LGB_PATH = OUTPUT_DIR / "model_lgb.txt"
MODEL_RF_PATH = OUTPUT_DIR / "model_rf.joblib"
MODEL_LR_PATH = OUTPUT_DIR / "model_lr.joblib"
MODEL_TORCH_PATH = OUTPUT_DIR / "weights" / "model_torch.pt"

# Все фичи датасета в порядке FEATURE_NAMES (обучение и submission).
MODEL_INPUT_FEATURES: list[str] = list(FEATURE_NAMES)


def resolve_model_input_columns(parquet_schema_names: Sequence[str]) -> list[str]:
    """
    Проверяет, что в parquet есть все колонки из FEATURE_NAMES,
    и возвращает их в порядке MODEL_INPUT_FEATURES.
    """
    available = frozenset(parquet_schema_names)
    missing = [c for c in FEATURE_NAMES if c not in available]
    if missing:
        raise ValueError(f"В датасете нет колонок фичей: {missing}")
    return list(MODEL_INPUT_FEATURES)
