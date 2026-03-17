"""
Точка входа 3: построение файла для сдачи в систему.

Загружает выбранную обученную модель, агрегирует по pretest, проходит по test,
считает фичи и предсказания, сохраняет submission.csv в формате: event_id, predict.

Выбор модели — аргумент командной строки (--model / -m):
  catboost, xgboost, lightgbm, pytorch

Запуск из корня репозитория:
  PYTHONPATH=. python submission/main.py --model catboost
  PYTHONPATH=. python submission/main.py -m lightgbm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

# Добавляем корень проекта в путь
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import (
    MODEL_LGB_PATH,
    MODEL_PATH,
    MODEL_PYTORCH_PATH,
    MODEL_XGB_PATH,
    PREDICTIONS_PATH,
    PRETEST_PATH,
    TEST_LABELS_PATH,
    TEST_PATH,
    WINDOW_TRANSACTIONS,
    WINDOWED_BATCH_SIZE,
)
from shared.features import FEATURE_NAMES, compute_features
from shared.parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    WindowedAggregates,
    build_windowed_aggregates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Требование платформы: ровно столько операций в тесте
EXPECTED_TEST_ROWS = 633_683

MODEL_CHOICES = ("catboost", "xgboost", "lightgbm", "pytorch")


def _get_feature_names() -> list[str]:
    """Список фичей в том же порядке, что и в датасете."""
    return FEATURE_NAMES


def _load_model_and_predictor(
    model_name: str,
    feature_names: list[str],
) -> tuple[object, Callable[[list[float]], float]]:
    """Загружает модель и возвращает (model, predict_fn). predict_fn(x: list[float]) -> float."""
    name = model_name.lower().strip()
    if name == "catboost":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"CatBoost model not found: {MODEL_PATH}. Run training first.")
        m = CatBoostClassifier()
        m.load_model(str(MODEL_PATH))
        logger.info("Loaded CatBoost from %s", MODEL_PATH)

        def predict_fn(x: list[float]) -> float:
            pred = m.predict([x], prediction_type="RawFormulaVal")
            return float(pred[0]) if hasattr(pred, "__len__") else float(pred)

        return m, predict_fn

    if name == "xgboost":
        if not MODEL_XGB_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found: {MODEL_XGB_PATH}. Run training first.")
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_XGB_PATH))
        logger.info("Loaded XGBoost from %s", MODEL_XGB_PATH)

        def predict_fn(x: list[float]) -> float:
            a = np.array([x], dtype=np.float32)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # output_margin=True даёт "сырые" логиты, которые могут быть >1 или <0
            margin = m.predict(a, output_margin=True)
            return float(margin[0])

        return m, predict_fn

    if name == "lightgbm":
        if not MODEL_LGB_PATH.exists():
            raise FileNotFoundError(f"LightGBM model not found: {MODEL_LGB_PATH}. Run training first.")
        m = lgb.Booster(model_file=str(MODEL_LGB_PATH))
        logger.info("Loaded LightGBM from %s", MODEL_LGB_PATH)

        def predict_fn(x: list[float]) -> float:
            # Booster.predict ожидает 2D array
            a = np.array([x], dtype=np.float32)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # raw_score=True — логиты (до сигмоиды), могут быть >1 или <0
            return float(m.predict(a, raw_score=True)[0])

        return m, predict_fn

    if name == "pytorch":
        if not MODEL_PYTORCH_PATH.exists():
            raise FileNotFoundError(f"PyTorch model not found: {MODEL_PYTORCH_PATH}. Run training first.")
        from training.config import PYTORCH_PARAMS
        from training.pytorch_model import load_model as load_pytorch_model
        input_size = len(feature_names)
        hidden = tuple(PYTORCH_PARAMS["hidden_sizes"])
        m = load_pytorch_model(MODEL_PYTORCH_PATH, input_size=input_size, hidden_sizes=hidden)
        m.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = m.to(dev)
        logger.info("Loaded PyTorch MLP from %s", MODEL_PYTORCH_PATH)

        def predict_fn(x: list[float]) -> float:
            arr = np.nan_to_num(np.array([x], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            t = torch.from_numpy(arr).to(dev)
            with torch.no_grad():
                logit = m(t).squeeze()
            # Возвращаем логит: >1 — высокая уверенность в целевом классе, <0 — в нецелевом
            return float(logit.item())

        return m, predict_fn

    raise ValueError(
        f"Unknown model: {model_name!r}. Choose one of: {', '.join(MODEL_CHOICES)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build submission CSV using a trained model (event_id, predict)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="catboost",
        choices=MODEL_CHOICES,
        help="Model to use for predictions: catboost, xgboost, lightgbm, pytorch (default: catboost)",
    )
    args = parser.parse_args()

    feature_names = _get_feature_names()
    logger.info("Submission: window=%s, feature count: %s", WINDOW_TRANSACTIONS, len(feature_names))

    _, predict_fn = _load_model_and_predictor(args.model, feature_names)

    pretest_paths = [PRETEST_PATH] if isinstance(PRETEST_PATH, Path) else list(PRETEST_PATH)
    pretest_paths = [p for p in pretest_paths if p.exists()]

    if not pretest_paths:
        logger.warning("Pretest not found at %s, starting with empty windowed aggregates", PRETEST_PATH)
    windowed_aggregates = build_windowed_aggregates(
        pretest_paths,
        window_size=WINDOW_TRANSACTIONS,
        batch_size=WINDOWED_BATCH_SIZE,
        show_progress=True,
    )
    logger.info("Pretest done (window=%s), %s users", WINDOW_TRANSACTIONS, len(windowed_aggregates))

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
    pf = pq.ParquetFile(TEST_PATH)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS + [EVENT_ID_COLUMN] if c in schema.names]
    if EVENT_ID_COLUMN not in columns:
        columns.append(EVENT_ID_COLUMN)

    predictions: list[tuple[int, float]] = []

    for batch in pf.iter_batches(columns=columns, batch_size=WINDOWED_BATCH_SIZE):
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            event_id = row.get(EVENT_ID_COLUMN)
            if cid not in windowed_aggregates:
                windowed_aggregates[cid] = WindowedAggregates(WINDOW_TRANSACTIONS)
            agg = windowed_aggregates[cid].get_aggregates()
            tr_amount = min(len(windowed_aggregates[cid]), WINDOW_TRANSACTIONS)
            feats = compute_features(agg, row, tr_amount=tr_amount)
            windowed_aggregates[cid].add(row)
            x = [feats[name] for name in feature_names]
            pred_val = predict_fn(x)
            predictions.append((event_id, pred_val))

    out_df = pd.DataFrame(predictions, columns=["event_id", "predict"])
    if len(out_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Submission must contain exactly {EXPECTED_TEST_ROWS} rows (platform requirement), "
            f"got {len(out_df)}. Check that no test rows are skipped (e.g. due to missing customer_id)."
        )
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Saved %s predictions to %s using model '%s' (OK)", len(out_df), PREDICTIONS_PATH, args.model)

    if TEST_LABELS_PATH and Path(TEST_LABELS_PATH).exists():
        labels_df = pd.read_csv(TEST_LABELS_PATH)
        if "event_id" in labels_df.columns and "target" in labels_df.columns:
            merged = out_df.merge(
                labels_df[["event_id", "target"]], on="event_id", how="inner"
            )
            from sklearn.metrics import average_precision_score
            pr_auc = average_precision_score(merged["target"], merged["predict"])
            logger.info("PR-AUC (on labeled test): %.4f", pr_auc)
        else:
            logger.warning("Test labels file missing event_id or target columns")
    else:
        logger.info("No test labels path set, skipping PR-AUC")


if __name__ == "__main__":
    main()
