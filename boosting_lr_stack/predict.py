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

from boosting_lr_stack.model import build_meta_features, load_base_boosters, load_stack_model, proba_to_logit
from shared.config import BATCH_SIZE, MODEL_INPUT_FEATURES, MODEL_LR_BOOST_STACK_PATH, OUTPUT_DIR, PRETEST_PATH, TEST_PATH
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


def build_submission(model_path: Path, output_path: Path) -> pd.DataFrame:
    saved = load_stack_model(model_path)
    model = saved["model"]
    use_segmented_xgb = bool(saved.get("use_segmented_xgb", False))
    boosters = load_base_boosters(use_segmented_xgb=use_segmented_xgb)

    pretest_paths = [PRETEST_PATH] if PRETEST_PATH.exists() else []
    aggregates: defaultdict[object, UserAggregates] = defaultdict(lambda: UserAggregates(unlimited=True))
    if pretest_paths:
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
        agg.update(row)

    X = pd.DataFrame(feature_matrix)[MODEL_INPUT_FEATURES]
    meta_x = build_meta_features(X, boosters)
    proba = np.asarray(model.predict_proba(meta_x)[:, 1], dtype=np.float64)
    logits = proba_to_logit(proba)
    out = pd.DataFrame({"event_id": event_ids, "predict": logits})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Инференс stack LR (CatBoost+XGBoost -> LogisticRegression)")
    parser.add_argument("--model-path", type=Path, default=MODEL_LR_BOOST_STACK_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "submission.csv")
    args = parser.parse_args()

    out = build_submission(args.model_path, args.output)
    if len(out) != EXPECTED_ROWS:
        logger.warning("Строк %d, ожидалось %d.", len(out), EXPECTED_ROWS)
    logger.info("Сохранено %d строк в %s", len(out), args.output)


if __name__ == "__main__":
    main()

