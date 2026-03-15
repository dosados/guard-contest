"""
Точка входа 2: разбиение данных, обучение моделей (CatBoost, XGBoost, LightGBM),
валидация, сохранение весов и вывод метрик по всем моделям.

Загружает датасет из output/train_dataset.parquet (созданный dataset/main.py),
делит на train/val, обучает четыре модели (CatBoost, XGBoost, LightGBM, PyTorch MLP),
считает PR-AUC на валидации для каждой, сохраняет веса в output/ и выводит сводную таблицу метрик.

Конфиг моделей и VAL_RATIO — в training/config.py.

Запуск из корня репозитория:
  PYTHONPATH=. python training/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Корень проекта в sys.path до импортов shared/training
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
import xgboost as xgb
import lightgbm as lgb

from shared.config import (
    MODEL_PATH,
    MODEL_LGB_PATH,
    MODEL_PYTORCH_PATH,
    MODEL_XGB_PATH,
    TRAIN_DATASET_PATH,
)
from training.config import (
    CATBOOST_PARAMS,
    LGBM_PARAMS,
    PYTORCH_PARAMS,
    VAL_RATIO,
    XGB_PARAMS,
)
from training.pytorch_model import train_and_validate, save_model as save_pytorch_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"


def main() -> None:
    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Run dataset/main.py first to create {TRAIN_DATASET_PATH}"
        )
    logger.info("Loading dataset from %s", TRAIN_DATASET_PATH)
    df = pd.read_parquet(TRAIN_DATASET_PATH)
    non_feature_columns = {"event_id", TARGET_COLUMN, "event_dttm"}
    feature_columns = [c for c in df.columns if c not in non_feature_columns]
    if not feature_columns:
        raise ValueError("No feature columns in dataset (expected columns other than event_id, target, event_dttm)")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {TARGET_COLUMN}")
    if "event_dttm" not in df.columns:
        raise ValueError(
            "Dataset must contain event_dttm for time-based split. Re-run dataset/main.py."
        )

    # Разбиение по времени: сортируем по event_dttm, val = последние VAL_RATIO
    df = df.copy()
    df["event_dttm"] = pd.to_datetime(df["event_dttm"], errors="coerce")
    df = df.sort_values("event_dttm", na_position="first").reset_index(drop=True)
    n = len(df)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val
    train_idx = df.index[:n_train].tolist()
    val_idx = df.index[n_train:].tolist()
    X_train = df.loc[train_idx, feature_columns]
    y_train = df.loc[train_idx, TARGET_COLUMN]
    X_val = df.loc[val_idx, feature_columns]
    y_val = df.loc[val_idx, TARGET_COLUMN]
    logger.info(
        "Time-based split: train=%s (first by time), val=%s (last by time)",
        len(X_train),
        len(X_val),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics: list[tuple[str, float]] = []

    # --- CatBoost ---
    logger.info("Training CatBoost...")
    model_cb = CatBoostClassifier(**CATBOOST_PARAMS)
    model_cb.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=CATBOOST_PARAMS.get("verbose", 100),
    )
    logger.info("CatBoost fit done, computing validation PR-AUC...")
    X_val_cb = np.ascontiguousarray(X_val.fillna(0).values)
    pr_auc_cb = average_precision_score(y_val, model_cb.predict_proba(X_val_cb)[:, 1])
    metrics.append(("CatBoost", pr_auc_cb))
    logger.info("Saving CatBoost model...")
    model_cb.save_model(str(MODEL_PATH))
    logger.info("CatBoost PR-AUC: %.4f, saved to %s", pr_auc_cb, MODEL_PATH)

    # --- XGBoost ---
    logger.info("Training XGBoost...")
    model_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    pr_auc_xgb = average_precision_score(y_val, model_xgb.predict_proba(X_val)[:, 1])
    metrics.append(("XGBoost", pr_auc_xgb))
    model_xgb.save_model(str(MODEL_XGB_PATH))
    logger.info("XGBoost PR-AUC: %.4f, saved to %s", pr_auc_xgb, MODEL_XGB_PATH)

    # --- LightGBM ---
    logger.info("Training LightGBM...")
    model_lgb = lgb.LGBMClassifier(**LGBM_PARAMS)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(100)],
    )
    pr_auc_lgb = average_precision_score(y_val, model_lgb.predict_proba(X_val)[:, 1])
    metrics.append(("LightGBM", pr_auc_lgb))
    model_lgb.booster_.save_model(str(MODEL_LGB_PATH))
    logger.info("LightGBM PR-AUC: %.4f, saved to %s", pr_auc_lgb, MODEL_LGB_PATH)

    # --- PyTorch MLP ---
    logger.info("Training PyTorch MLP...")
    X_train_np = X_train.fillna(0).values
    X_val_np = X_val.fillna(0).values
    y_train_np = y_train.values
    y_val_np = y_val.values
    model_pt, pr_auc_pt = train_and_validate(
        X_train_np,
        y_train_np,
        X_val_np,
        y_val_np,
        **PYTORCH_PARAMS,
    )
    metrics.append(("PyTorch MLP", pr_auc_pt))
    save_pytorch_model(model_pt, MODEL_PYTORCH_PATH)
    logger.info("PyTorch MLP PR-AUC: %.4f, saved to %s", pr_auc_pt, MODEL_PYTORCH_PATH)

    # Сводка метрик
    logger.info("=" * 50)
    logger.info("Validation PR-AUC (all models):")
    for name, pr_auc in metrics:
        logger.info("  %s: %.4f", name, pr_auc)
    best_name, best_auc = max(metrics, key=lambda x: x[1])
    logger.info("Best: %s (PR-AUC %.4f)", best_name, best_auc)
    logger.info("=" * 50)
    print("\n--- Validation PR-AUC ---")
    print(f"{'Model':<12}  PR-AUC")
    print("-" * 24)
    for name, pr_auc in metrics:
        print(f"{name:<12}  {pr_auc:.4f}")
    print("-" * 24)
    print(f"Best: {best_name} ({best_auc:.4f})")
    print()


if __name__ == "__main__":
    main()
