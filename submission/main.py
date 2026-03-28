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
    MODEL_LR_BOOST_STACK_PATH,
    MODEL_PATH,
    MODEL_RF_PATH,
    MODEL_TORCH_MLP_PATH,
    MODEL_TORCH_PATH,
    MODEL_XGB_PATH,
    MODEL_XGB_HIGH_TR_AMOUNT_PATH,
    MODEL_XGB_LOW_TR_AMOUNT_PATH,
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
TR_AMOUNT_SPLIT_THRESHOLD = 30.0

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


def load_predictor(model_name: str, *, use_segmented_xgb: bool = False):
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

        use_segmented_models = use_segmented_xgb
        if use_segmented_models:
            if not (MODEL_XGB_LOW_TR_AMOUNT_PATH.exists() and MODEL_XGB_HIGH_TR_AMOUNT_PATH.exists()):
                raise FileNotFoundError(
                    "Для --xgb-segmented-inference нужны обе модели: "
                    f"{MODEL_XGB_LOW_TR_AMOUNT_PATH} и {MODEL_XGB_HIGH_TR_AMOUNT_PATH}"
                )
            m_low = XGBClassifier()
            m_low.load_model(str(MODEL_XGB_LOW_TR_AMOUNT_PATH))
            booster_low = m_low.get_booster()

            m_high = XGBClassifier()
            m_high.load_model(str(MODEL_XGB_HIGH_TR_AMOUNT_PATH))
            booster_high = m_high.get_booster()
            logger.info(
                "XGBoost: используем 2 модели по tr_amount (<=%.0f: %s, >%.0f: %s)",
                TR_AMOUNT_SPLIT_THRESHOLD,
                MODEL_XGB_LOW_TR_AMOUNT_PATH.name,
                TR_AMOUNT_SPLIT_THRESHOLD,
                MODEL_XGB_HIGH_TR_AMOUNT_PATH.name,
            )
        else:
            if not MODEL_XGB_PATH.exists():
                raise FileNotFoundError(f"Не найдена XGBoost модель: {MODEL_XGB_PATH}")
            m = XGBClassifier()
            m.load_model(str(MODEL_XGB_PATH))
            booster = m.get_booster()
            logger.info("XGBoost: используем legacy одиночную модель %s", MODEL_XGB_PATH.name)

        def pred(X: pd.DataFrame) -> np.ndarray:
            if use_segmented_models:
                if "tr_amount" not in X.columns:
                    raise ValueError("Для двухмодельного XGBoost в инференсе требуется колонка tr_amount.")
                tr_values = pd.to_numeric(X["tr_amount"], errors="coerce")
                low_mask = tr_values <= TR_AMOUNT_SPLIT_THRESHOLD
                high_mask = ~low_mask
                pr = np.zeros(len(X), dtype=np.float64)
                if bool(low_mask.any()):
                    pr[low_mask.to_numpy()] = _xgboost_predict_probs(
                        booster_low,
                        X.loc[low_mask],
                        MODEL_INPUT_FEATURES,
                    )
                if bool(high_mask.any()):
                    pr[high_mask.to_numpy()] = _xgboost_predict_probs(
                        booster_high,
                        X.loc[high_mask],
                        MODEL_INPUT_FEATURES,
                    )
            else:
                pr = _xgboost_predict_probs(booster, X, MODEL_INPUT_FEATURES)
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

    if model_name in ("boosting_stack_lr", "stack_lr", "boosting_logreg"):
        from boosting_lr_stack.model import build_meta_features, load_base_boosters
        import joblib

        saved = joblib.load(MODEL_LR_BOOST_STACK_PATH)
        m = saved["model"]
        use_segmented = bool(saved.get("use_segmented_xgb", use_segmented_xgb))
        boosters = load_base_boosters(use_segmented_xgb=use_segmented)

        def pred(X: pd.DataFrame) -> np.ndarray:
            meta_x = build_meta_features(X, boosters)
            pr = m.predict_proba(meta_x)[:, 1]
            return proba_to_logit(np.asarray(pr, dtype=np.float64))

        return pred

    raise ValueError(
        f"Неизвестная модель: {model_name}. Используйте catboost, xgboost, lightgbm, random_forest, "
        "logistic_regression, boosting_stack_lr, torch (LSTM по сырому test) или torch_mlp (MLP по фичам как у бустингов)."
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
        help="catboost | catboost_disk | xgboost | lightgbm | random_forest | logistic_regression | boosting_stack_lr | torch | torch_mlp",
    )
    parser.add_argument(
        "--xgb-segmented-inference",
        action="store_true",
        help=(
            "Для --model xgboost: использовать роутинг на 2 модели по tr_amount "
            f"(<= {int(TR_AMOUNT_SPLIT_THRESHOLD)} и > {int(TR_AMOUNT_SPLIT_THRESHOLD)}). "
            "По умолчанию используется одна legacy-модель."
        ),
    )
    args = parser.parse_args()

    model_key = args.model.lower().strip()
    use_torch_lstm = model_key in ("torch", "lstm", "tabular_lstm")
    use_torch_mlp = model_key in ("torch_mlp", "mlp", "tabular_mlp")
    use_catboost_disk = model_key in ("catboost_disk", "catboost-tsv")
    logger.info("Модель: %s", args.model)
    predict_fn = None
    if not use_torch_lstm and not use_torch_mlp and not use_catboost_disk:
        predict_fn = load_predictor(
            args.model,
            use_segmented_xgb=args.xgb_segmented_inference,
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if use_catboost_disk:
        from submission.catboost_disk_infer import run_catboost_disk_submission

        run_catboost_disk_submission(OUTPUT_DIR / "submission.csv")
        return

    if use_torch_lstm:
        import torch
        from torch_logic.predict import predict_logits

        if not MODEL_TORCH_PATH.exists():
            raise FileNotFoundError(
                f"Нет чекпоинта {MODEL_TORCH_PATH}. "
                "Обучите: PYTHONPATH=. python torch_logic/train.py"
            )
        if not TEST_PATH.exists():
            raise FileNotFoundError(f"Не найден test parquet: {TEST_PATH}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = predict_logits(MODEL_TORCH_PATH, TEST_PATH, device)
        out_path = OUTPUT_DIR / "submission.csv"
        out.to_csv(out_path, index=False)
        if len(out) != EXPECTED_ROWS:
            logger.warning("Строк %d, ожидалось %d.", len(out), EXPECTED_ROWS)
        logger.info("Сохранено %d строк в %s", len(out), out_path)
        return

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
    X = pd.DataFrame(feature_matrix)[MODEL_INPUT_FEATURES]
    if use_torch_mlp:
        import torch
        from torch_logic.predict_mlp import predict_mlp_dataframe

        if not MODEL_TORCH_MLP_PATH.exists():
            raise FileNotFoundError(
                f"Нет чекпоинта MLP: {MODEL_TORCH_MLP_PATH}. "
                "Обучите: PYTHONPATH=. python torch_logic/train_mlp.py"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            logger.warning("torch_mlp: CUDA недоступна, инференс на CPU.")
        logits_arr = predict_mlp_dataframe(X, MODEL_TORCH_MLP_PATH, device)
    else:
        assert predict_fn is not None
        logits_arr = predict_fn(X)

    out = pd.DataFrame({"event_id": event_ids, "predict": logits_arr.astype(np.float64)})
    out_path = OUTPUT_DIR / "submission.csv"
    out.to_csv(out_path, index=False)

    if len(out) != EXPECTED_ROWS:
        logger.warning("Строк %d, ожидалось %d.", len(out), EXPECTED_ROWS)
    logger.info("Сохранено %d строк в %s", len(out), out_path)


if __name__ == "__main__":
    main()
