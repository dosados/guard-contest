from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from shared.config import (
    MODEL_INPUT_FEATURES,
    MODEL_PATH,
    MODEL_XGB_HIGH_TR_AMOUNT_PATH,
    MODEL_XGB_LOW_TR_AMOUNT_PATH,
    MODEL_XGB_PATH,
)

TR_AMOUNT_SPLIT_THRESHOLD = 30.0
META_FEATURES = ["catboost_proba", "xgboost_proba"]


def proba_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


def _xgboost_predict_probs(booster: Any, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
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


def load_base_boosters(*, use_segmented_xgb: bool) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена CatBoost модель: {MODEL_PATH}")
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODEL_PATH))

    if use_segmented_xgb:
        if not (MODEL_XGB_LOW_TR_AMOUNT_PATH.exists() and MODEL_XGB_HIGH_TR_AMOUNT_PATH.exists()):
            raise FileNotFoundError(
                "Для segmented XGBoost нужны обе модели: "
                f"{MODEL_XGB_LOW_TR_AMOUNT_PATH} и {MODEL_XGB_HIGH_TR_AMOUNT_PATH}"
            )
        m_low = XGBClassifier()
        m_low.load_model(str(MODEL_XGB_LOW_TR_AMOUNT_PATH))
        m_high = XGBClassifier()
        m_high.load_model(str(MODEL_XGB_HIGH_TR_AMOUNT_PATH))
        return {
            "catboost": cat_model,
            "use_segmented_xgb": True,
            "xgb_low": m_low.get_booster(),
            "xgb_high": m_high.get_booster(),
        }

    if not MODEL_XGB_PATH.exists():
        raise FileNotFoundError(f"Не найдена XGBoost модель: {MODEL_XGB_PATH}")
    m = XGBClassifier()
    m.load_model(str(MODEL_XGB_PATH))
    return {
        "catboost": cat_model,
        "use_segmented_xgb": False,
        "xgb_single": m.get_booster(),
    }


def build_meta_features(X: pd.DataFrame, boosters: dict[str, Any]) -> np.ndarray:
    cat_model: CatBoostClassifier = boosters["catboost"]
    cat_pr = np.asarray(cat_model.predict_proba(X[MODEL_INPUT_FEATURES])[:, 1], dtype=np.float64)

    if bool(boosters.get("use_segmented_xgb", False)):
        if "tr_amount" not in X.columns:
            raise ValueError("Для segmented XGBoost требуется колонка tr_amount.")
        low_booster = boosters["xgb_low"]
        high_booster = boosters["xgb_high"]
        tr_values = pd.to_numeric(X["tr_amount"], errors="coerce")
        low_mask = tr_values <= TR_AMOUNT_SPLIT_THRESHOLD
        high_mask = ~low_mask
        xgb_pr = np.zeros(shape=(len(X),), dtype=np.float64)
        if bool(low_mask.any()):
            xgb_pr[low_mask.to_numpy()] = _xgboost_predict_probs(low_booster, X.loc[low_mask], MODEL_INPUT_FEATURES)
        if bool(high_mask.any()):
            xgb_pr[high_mask.to_numpy()] = _xgboost_predict_probs(high_booster, X.loc[high_mask], MODEL_INPUT_FEATURES)
    else:
        xgb_pr = _xgboost_predict_probs(boosters["xgb_single"], X, MODEL_INPUT_FEATURES)

    return np.column_stack([cat_pr, xgb_pr]).astype(np.float64, copy=False)


def save_stack_model(path: Path, model: Any, *, use_segmented_xgb: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "meta_features": list(META_FEATURES),
        "use_segmented_xgb": bool(use_segmented_xgb),
    }
    joblib.dump(payload, path)


def load_stack_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Не найдена stack LR модель: {path}")
    payload = joblib.load(path)
    if "model" not in payload:
        raise ValueError(f"Некорректный формат stack LR модели: {path}")
    return payload

