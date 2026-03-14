"""
Шаг 2 пайплайна: загрузка сохранённых фичей и обучение CatBoost.
Модель сохраняется на диск; её можно заменить другой (например, sklearn), поменяв только этот модуль.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from catboost import CatBoostClassifier

from config import CATBOOST_PARAMS, MODEL_PATH, TRAIN_FEATURES_PATH
from features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"


def main() -> None:
    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Run prepare_training_data.py first to create {TRAIN_FEATURES_PATH}"
        )
    logger.info("Loading training data from %s", TRAIN_FEATURES_PATH)
    df = pd.read_parquet(TRAIN_FEATURES_PATH)
    for name in FEATURE_NAMES:
        if name not in df.columns:
            raise ValueError(f"Missing feature column: {name}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {TARGET_COLUMN}")

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]
    logger.info("Training CatBoost: %s samples, %s features", len(X), len(FEATURE_NAMES))

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X, y, verbose=CATBOOST_PARAMS.get("verbose", 100))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
