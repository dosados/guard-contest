"""
Построение output/submission.csv: pretest → агрегаты, test → те же фичи, что при обучении, предсказание логита (XGBoost).
"""

from __future__ import annotations

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
    GLOBAL_CATEGORY_AGGREGATES_DIR,
    MODEL_INPUT_FEATURES,
    MODEL_XGB_PATH,
    OUTPUT_DIR,
    PRETEST_PATH,
    TEST_PATH,
    validate_model_input_dataframe,
    validate_xgboost_booster_feature_count,
)
from shared.features import compute_features
from shared.global_category_aggregates import GlobalCategoryLookups
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


def _xgboost_predict_probs(booster, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Согласовано с training/main.py: при early stopping — только деревья до best_iteration."""
    import xgboost as xgb

    d = xgb.DMatrix(X[feature_cols], feature_names=feature_cols)
    bi = getattr(booster, "best_iteration", None)
    if bi is not None and bi >= 0:
        try:
            return np.asarray(booster.predict(d, iteration_range=(0, int(bi) + 1)), dtype=np.float64)
        except TypeError:
            pass
    bnl = getattr(booster, "best_ntree_limit", None)
    if bnl is not None and bnl > 0:
        try:
            return np.asarray(booster.predict(d, iteration_range=(0, int(bnl))), dtype=np.float64)
        except TypeError:
            return np.asarray(booster.predict(d, ntree_limit=int(bnl)), dtype=np.float64)
    return np.asarray(booster.predict(d), dtype=np.float64)


def proba_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


def load_xgboost_predictor():
    from xgboost import XGBClassifier

    if not MODEL_XGB_PATH.exists():
        raise FileNotFoundError(f"Не найдена XGBoost модель: {MODEL_XGB_PATH}")
    m = XGBClassifier()
    m.load_model(str(MODEL_XGB_PATH))
    booster = m.get_booster()
    validate_xgboost_booster_feature_count(booster)
    logger.info("XGBoost: %s", MODEL_XGB_PATH.name)

    def pred(X: pd.DataFrame) -> np.ndarray:
        validate_model_input_dataframe(X)
        pr = _xgboost_predict_probs(booster, X, MODEL_INPUT_FEATURES)
        return proba_to_logit(pr)

    return pred


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    predict_fn = load_xgboost_predictor()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_cat = GlobalCategoryLookups(GLOBAL_CATEGORY_AGGREGATES_DIR)
    logger.info("Глобальные агрегаты: %s", GLOBAL_CATEGORY_AGGREGATES_DIR)

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
        op_amt = float(feats.get("operation_amt", float("nan")))
        feats.update(global_cat.features_for_row(row, op_amt))
        eid_raw = row.get(EVENT_ID_COLUMN)
        if eid_raw is None:
            continue
        event_ids.append(int(eid_raw))
        feature_matrix.append(feats)
        customer_row_ids.append(cid)
        agg.update(row)

    logger.info("Предсказание: матрица %d × %d", len(feature_matrix), len(MODEL_INPUT_FEATURES))
    X = pd.DataFrame(feature_matrix)[MODEL_INPUT_FEATURES]
    validate_model_input_dataframe(X)
    logits_arr = predict_fn(X)

    out = pd.DataFrame({"event_id": event_ids, "predict": logits_arr.astype(np.float64)})
    out_path = OUTPUT_DIR / "submission.csv"
    out.to_csv(out_path, index=False)

    if len(out) != EXPECTED_ROWS:
        logger.warning("Строк %d, ожидалось %d.", len(out), EXPECTED_ROWS)
    logger.info("Сохранено %d строк в %s", len(out), out_path)


if __name__ == "__main__":
    main()
