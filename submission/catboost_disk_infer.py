"""
CatBoost: тот же пайплайн, что submission/main.py (pretest → агрегаты, test → фичи),
но предсказание батчами: фичи пишутся во временный TSV → Pool(data=path) → predict_proba,
чтобы не держать всю матрицу признаков в RAM.

Запуск: PYTHONPATH=. python submission/catboost_disk_infer.py
или --model catboost_disk в submission/main.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from catboost import CatBoostClassifier, Pool

from shared.config import (
    BATCH_SIZE,
    MODEL_INPUT_FEATURES,
    MODEL_PATH,
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

logger = logging.getLogger(__name__)
EXPECTED_ROWS = 633_683


def _proba_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


def _write_feature_cd(path: Path, feature_cols: list[str]) -> None:
    lines = [f"{i}\tNum\t{name}" for i, name in enumerate(feature_cols)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_chunk_tsv(path: Path, rows: list[dict[str, float]], feature_cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(feature_cols)
        for feats in rows:
            w.writerow([feats.get(c, float("nan")) for c in feature_cols])


def run_catboost_disk_submission(out_path: Path | None = None, *, chunk_rows: int = 32768) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Нет модели CatBoost: {MODEL_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Не найден test: {TEST_PATH}")

    out_path = out_path or OUTPUT_DIR / "submission.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pretest_paths = [PRETEST_PATH] if PRETEST_PATH.exists() else []
    aggregates: defaultdict[object, UserAggregates] = defaultdict(lambda: UserAggregates(unlimited=True))
    if pretest_paths:
        logger.info("Агрегаты по pretest: %s", pretest_paths)
        built = build_windowed_aggregates(
            pretest_paths,
            batch_size=BATCH_SIZE,
            show_progress=True,
            unlimited_window=True,
        )
        for k, v in built.items():
            aggregates[k] = v
    else:
        logger.warning("Pretest не найден (%s)", PRETEST_PATH)

    cols = _existing_columns(TEST_PATH, list(FEATURE_COLUMNS))
    if CUSTOMER_ID_COLUMN not in cols:
        cols = [CUSTOMER_ID_COLUMN] + cols
    if EVENT_ID_COLUMN not in cols:
        cols.append(EVENT_ID_COLUMN)

    cd_path = OUTPUT_DIR / "catboost_infer_features.cd"
    chunk_tsv = OUTPUT_DIR / "catboost_infer_chunk.tsv"
    _write_feature_cd(cd_path, MODEL_INPUT_FEATURES)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    event_ids: list[int] = []
    logits_parts: list[np.ndarray] = []

    chunk: list[dict[str, float]] = []

    def predict_chunk() -> None:
        if not chunk:
            return
        _write_chunk_tsv(chunk_tsv, chunk, MODEL_INPUT_FEATURES)
        pool = Pool(
            data=str(chunk_tsv),
            column_description=str(cd_path),
            delimiter="\t",
            has_header=True,
        )
        try:
            pr = np.asarray(model.predict_proba(pool)[:, 1], dtype=np.float64)
        finally:
            del pool
            gc.collect()
        logits_parts.append(_proba_to_logit(pr))
        chunk.clear()

    logger.info("Test → TSV-чанки → Pool, chunk_rows=%d: %s", chunk_rows, TEST_PATH)
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
        chunk.append(feats)
        agg.update(row)
        if len(chunk) >= chunk_rows:
            predict_chunk()

    predict_chunk()

    if chunk_tsv.exists():
        chunk_tsv.unlink(missing_ok=True)

    logits_arr = np.concatenate(logits_parts) if logits_parts else np.array([], dtype=np.float64)
    if len(event_ids) != len(logits_arr):
        raise RuntimeError(f"event_ids={len(event_ids)} vs logits={len(logits_arr)}")

    out = pd.DataFrame({"event_id": event_ids, "predict": logits_arr.astype(np.float64)})
    out.to_csv(out_path, index=False)

    if len(out) != EXPECTED_ROWS:
        logger.warning("Строк %d, ожидалось %d.", len(out), EXPECTED_ROWS)
    logger.info("Сохранено %d строк в %s", len(out), out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="CatBoost: pretest → агрегаты, test → временный TSV + Pool → submission.csv"
    )
    parser.add_argument("--output", type=Path, default=None, help="путь submission.csv")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=32768,
        help="строк на один TSV перед Pool.predict (меньше — ниже пик RAM)",
    )
    args = parser.parse_args()
    run_catboost_disk_submission(args.output, chunk_rows=args.chunk_rows)


if __name__ == "__main__":
    main()
