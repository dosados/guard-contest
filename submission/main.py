"""
Построение output/submission.csv: pretest → агрегаты, test → те же фичи, что при обучении, предсказание логита.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import (
    BATCH_SIZE,
    MODEL_INPUT_FEATURES,
    MODEL_LGB_PATH,
    MODEL_LR_PATH,
    MODEL_PATH,
    MODEL_RF_PATH,
    MODEL_TORCH_PATH,
    MODEL_XGB_PATH,
    OUTPUT_DIR,
    PRETEST_PATH,
    TEST_PATH,
)
from shared.features import compute_features
from shared.parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    UserAggregates,
    _existing_columns,
    build_windowed_aggregates,
    iter_parquet_rows,
)

EXPECTED_ROWS = 633_683

logger = logging.getLogger(__name__)


def proba_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


def load_predictor(model_name: str):
    model_name = model_name.lower().strip()
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier()
        m.load_model(str(MODEL_PATH))

        def pred(X: pd.DataFrame) -> np.ndarray:
            pr = m.predict_proba(X)[:, 1]
            return proba_to_logit(pr)

        return pred

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        m = XGBClassifier()
        m.load_model(str(MODEL_XGB_PATH))

        def pred(X: pd.DataFrame) -> np.ndarray:
            pr = m.predict_proba(X)[:, 1]
            return proba_to_logit(pr)

        return pred

    if model_name in ("lightgbm", "lgbm"):
        import lightgbm as lgb

        m = lgb.Booster(model_file=str(MODEL_LGB_PATH))

        def pred(X: pd.DataFrame) -> np.ndarray:
            pr = m.predict(X)
            return proba_to_logit(np.asarray(pr, dtype=np.float64))

        return pred

    if model_name in ("random_forest", "rf", "randomforest"):
        import joblib

        saved = joblib.load(MODEL_RF_PATH)
        m = saved["model"]
        rf_features = saved.get("features", MODEL_INPUT_FEATURES)

        def pred(X: pd.DataFrame) -> np.ndarray:
            pr = m.predict_proba(X[rf_features])[:, 1]
            return proba_to_logit(np.asarray(pr, dtype=np.float64))

        return pred

    if model_name in ("logistic_regression", "logreg", "lr"):
        import joblib

        saved = joblib.load(MODEL_LR_PATH)
        m = saved["model"]
        lr_features = saved.get("features", MODEL_INPUT_FEATURES)

        def pred(X: pd.DataFrame) -> np.ndarray:
            pr = m.predict_proba(X[lr_features])[:, 1]
            return proba_to_logit(np.asarray(pr, dtype=np.float64))

        return pred

    raise ValueError(
        f"Неизвестная модель: {model_name}. Используйте catboost, xgboost, lightgbm, random_forest, "
        "logistic_regression или torch (см. --model torch)."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        default="catboost",
        help="catboost | xgboost | lightgbm | random_forest | logistic_regression | torch",
    )
    args = parser.parse_args()

    model_key = args.model.lower().strip()
    use_torch = model_key in ("torch", "lstm", "tabular_lstm")
    logger.info("Модель: %s", args.model)
    predict_fn = None if use_torch else load_predictor(args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pretest_paths = [PRETEST_PATH] if PRETEST_PATH.exists() else []
    aggregates: defaultdict[object, UserAggregates] = defaultdict(lambda: UserAggregates(unlimited=True))
    if pretest_paths:
        logger.info("Построение агрегатов по pretest (окно без лимита по числу транзакций): %s", pretest_paths)
        built = build_windowed_aggregates(
            pretest_paths,
            batch_size=BATCH_SIZE,
            show_progress=True,
            unlimited_window=True,
        )
        for k, v in built.items():
            aggregates[k] = v
    else:
        logger.warning("Pretest не найден (%s), агрегаты пустые", PRETEST_PATH)

    cols = _existing_columns(TEST_PATH, list(FEATURE_COLUMNS))
    if CUSTOMER_ID_COLUMN not in cols:
        cols = [CUSTOMER_ID_COLUMN] + cols
    if EVENT_ID_COLUMN not in cols:
        cols.append(EVENT_ID_COLUMN)

    event_ids: list[int] = []
    feature_matrix: list[dict[str, float]] = []
    customer_row_ids: list[object] = []

    logger.info("Разбор test и расчёт фич: %s", TEST_PATH)
    for row in iter_parquet_rows(
        [TEST_PATH],
        columns=cols,
        batch_size=BATCH_SIZE,
        show_progress=True,
        progress_desc="test rows",
    ):
        cid = row.get(CUSTOMER_ID_COLUMN)
        if cid is None:
            continue
        if isinstance(cid, str) and not cid.strip():
            continue
        agg = aggregates[cid]
        tr_amt = float(agg.transactions_before_current_count())
        feats = compute_features(agg, row, tr_amount=tr_amt)
        eid_raw = row.get(EVENT_ID_COLUMN)
        if eid_raw is None:
            continue
        event_ids.append(int(eid_raw))
        feature_matrix.append(feats)
        customer_row_ids.append(cid)
        agg.update(row)

    logger.info("Предсказание: матрица %d × %d", len(feature_matrix), len(MODEL_INPUT_FEATURES))
    if use_torch:
        import torch

        from torch_model.predict import load_tabular_lstm, logits_stateful_sequence

        if not MODEL_TORCH_PATH.exists():
            raise FileNotFoundError(f"Нет чекпоинта {MODEL_TORCH_PATH}. Обучите: PYTHONPATH=. python torch_model/train.py")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm = load_tabular_lstm(MODEL_TORCH_PATH, device)
        logits_arr = logits_stateful_sequence(
            lstm, device, customer_row_ids, feature_matrix, MODEL_INPUT_FEATURES
        )
    else:
        assert predict_fn is not None
        X = pd.DataFrame(feature_matrix)[MODEL_INPUT_FEATURES]
        logits_arr = predict_fn(X)

    out = pd.DataFrame({"event_id": event_ids, "predict": logits_arr.astype(np.float64)})
    out_path = OUTPUT_DIR / "submission.csv"
    out.to_csv(out_path, index=False)

    if len(out) != EXPECTED_ROWS:
        logger.warning("Строк %d, ожидалось %d.", len(out), EXPECTED_ROWS)
    logger.info("Сохранено %d строк в %s", len(out), out_path)


if __name__ == "__main__":
    main()
