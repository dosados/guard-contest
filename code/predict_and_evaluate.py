"""
Шаг 3 пайплайна: загрузка модели, агрегация на pretest, проход по test с предсказаниями.
Сохраняет submission.csv и при наличии разметки считает PR-AUC.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from catboost import CatBoostClassifier

from config import (
    BATCH_SIZE,
    MODEL_PATH,
    PREDICTIONS_PATH,
    PRETEST_PATH,
    TEST_LABELS_PATH,
    TEST_PATH,
)
from features import FEATURE_NAMES, compute_features
from parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    build_user_aggregates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train model first: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    logger.info("Loaded model from %s", MODEL_PATH)

    pretest_paths = [PRETEST_PATH] if isinstance(PRETEST_PATH, Path) else list(PRETEST_PATH)
    pretest_paths = [p for p in pretest_paths if p.exists()]
    if not pretest_paths:
        logger.warning("Pretest not found at %s, starting with empty aggregates", PRETEST_PATH)
    aggregates = build_user_aggregates(
        pretest_paths, batch_size=BATCH_SIZE, show_progress=True
    )
    logger.info("Pretest done, %s users", len(aggregates))

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
    pf = pq.ParquetFile(TEST_PATH)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS + [EVENT_ID_COLUMN] if c in schema.names]
    if EVENT_ID_COLUMN not in columns:
        columns.append(EVENT_ID_COLUMN)

    predictions: list[tuple[int, float]] = []
    for batch in pf.iter_batches(columns=columns, batch_size=BATCH_SIZE):
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            event_id = row.get(EVENT_ID_COLUMN)
            feats = compute_features(aggregates[cid], row)
            aggregates[cid].update(row)
            x = [feats[name] for name in FEATURE_NAMES]
            pred = model.predict([x], prediction_type="RawFormulaVal")
            pred_val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            predictions.append((event_id, pred_val))

    out_df = pd.DataFrame(predictions, columns=["event_id", "predict"])
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Saved %s predictions to %s", len(out_df), PREDICTIONS_PATH)

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
