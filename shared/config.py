"""Пути к данным и параметры пайплайна."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

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
TRAIN_DATASET_PATH = OUTPUT_DIR / "full_dataset"

# Побатчевое чтение parquet
BATCH_SIZE = 65_536
WINDOWED_BATCH_SIZE = BATCH_SIZE

# WINDOW_TRANSACTIONS, DATASET_MODE, WINDOW_TRANSACTIONS_MODE — в shared.dataset_settings

# Веса обучения: при event_id в train_labels — target в датасете всегда 1, вес по исходному target в parquet (0/1); без записи в labels — target 0, WEIGHT_UNLABELED
WEIGHT_UNLABELED = 1.0
WEIGHT_LABELED_0 = 2.0  # «жёлтый свет» в исходной разметке
WEIGHT_LABELED_1 = 5.0  # «красный», целевой класс в исходной разметке

# Обучение
MODEL_PATH = OUTPUT_DIR / "model.cbm"
MODEL_XGB_PATH = OUTPUT_DIR / "model_xgb.json"
MODEL_LGB_PATH = OUTPUT_DIR / "model_lgb.txt"
MODEL_RF_PATH = OUTPUT_DIR / "model_rf.joblib"
MODEL_LR_PATH = OUTPUT_DIR / "model_lr.joblib"

# Не подаём на вход модели (наименее полезные по permutation importance, см. output/research/feature_importance_report.txt).
MODEL_FEATURES_EXCLUDED: tuple[str, ...] = (
    "day_of_week",
    "is_night_transaction",
    "is_weekend",
    "transactions_last_24h_norm",
    "transactions_last_10m_to_1h",
    "transactions_last_24h",
    "transactions_last_1h_to_24h",
    "transactions_last_1h_norm",
    "sum_amount_last_24h",
    "sum_amount_last_1h_norm",
    "operation_amt",
    "sum_1h_to_24h",
    "sum_amount_last_1h",
    "transactions_last_1h",
    "transactions_last_10m_norm",
)

_MODEL_FEATURES_EXCLUDED_SET = frozenset(MODEL_FEATURES_EXCLUDED)
_unknown_excluded = _MODEL_FEATURES_EXCLUDED_SET - frozenset(FEATURE_NAMES)
if _unknown_excluded:
    raise RuntimeError(
        "MODEL_FEATURES_EXCLUDED содержит имена вне FEATURE_NAMES: "
        f"{sorted(_unknown_excluded)}"
    )

# Порядок колонок на вход модели = порядок в FEATURE_NAMES, минус исключённые.
MODEL_INPUT_FEATURES: list[str] = [f for f in FEATURE_NAMES if f not in _MODEL_FEATURES_EXCLUDED_SET]


def resolve_model_input_columns(parquet_schema_names: Sequence[str]) -> list[str]:
    """
    Проверяет, что в parquet есть все колонки из FEATURE_NAMES (полный датасет),
    и возвращает список колонок для обучения/инференса в порядке MODEL_INPUT_FEATURES.
    """
    available = frozenset(parquet_schema_names)
    missing = [c for c in FEATURE_NAMES if c not in available]
    if missing:
        raise ValueError(f"В датасете нет колонок фичей: {missing}")
    return list(MODEL_INPUT_FEATURES)
