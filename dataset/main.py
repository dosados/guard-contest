"""
Точка входа 1: создание датасета и сохранение на диск.

Файлы, из которых собирается датасет:
  - Pretrain (агрегаты по пользователям): data/train/pretrain_part_0.parquet,
    pretrain_part_1.parquet, pretrain_part_2.parquet, pretrain_part_3.parquet
  - Train (строки датасета): data/train/train_part_0.parquet, train_part_1.parquet,
    train_part_2.parquet, train_part_3.parquet
  - Разметка: data/train_labels.parquet

Алгоритм:
  1) Проход по pretrain → построение агрегированных фичей по пользователям.
  2) Для каждой записи в train: считаем фичи по текущим агрегатам, добавляем строку
     в датасет (все фичи + event_id + target), затем обновляем агрегаты этой записью.
  3) Метки target берутся из data/train_labels.parquet.
  4) Результат сохраняется в output/train_dataset.parquet.

Запуск из корня репозитория:
  PYTHONPATH=. python dataset/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Добавляем корень проекта в путь для импорта shared
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import (
    BATCH_SIZE,
    DATASET_MODE,
    PRETRAIN_N_WORKERS,
    PRETRAIN_PATHS,
    TRAIN_DATASET_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATHS,
    WINDOW_TRANSACTIONS,
    WINDOWED_BATCH_SIZE,
)
from shared.features import FEATURE_NAMES, FEATURE_NAMES_FULL, compute_features
from shared.parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_DTTM_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    build_user_aggregates,
    build_windowed_aggregates,
    WindowedAggregates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"


def load_train_labels(path: Path) -> dict[int, int]:
    """Загружает разметку train из parquet: event_id -> target (0 — жёлтая, 1 — целевой класс)."""
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    df = pd.read_parquet(path)
    if "event_id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Labels parquet must have event_id and target, got {list(df.columns)}")
    return df.set_index("event_id")["target"].astype(int).to_dict()


def main() -> None:
    use_full = DATASET_MODE == "full"
    use_window = DATASET_MODE == "window_50"
    if not use_full and not use_window:
        raise ValueError(
            f"DATASET_MODE must be 'full' or 'window_50', got {DATASET_MODE!r}. "
            "Set shared.config.DATASET_MODE."
        )
    logger.info("Режим датасета: %s", DATASET_MODE)

    pretrain_paths = [p for p in PRETRAIN_PATHS if p.exists()]
    if not pretrain_paths:
        logger.warning("No pretrain files found at %s, starting with empty aggregates", PRETRAIN_PATHS)

    if use_full:
        if pretrain_paths:
            logger.info("Агрегаты (full) строятся из pretrain: %s", [str(p.name) for p in pretrain_paths])
        aggregates = build_user_aggregates(
            pretrain_paths,
            batch_size=BATCH_SIZE,
            show_progress=(PRETRAIN_N_WORKERS <= 1),
            n_workers=PRETRAIN_N_WORKERS,
        )
        windowed_aggregates = None
        logger.info("Pretrain done, %s users", len(aggregates))
    else:
        if pretrain_paths:
            logger.info(
                "Оконные агрегаты (window=%s) строятся из pretrain: %s",
                WINDOW_TRANSACTIONS,
                [str(p.name) for p in pretrain_paths],
            )
        windowed_aggregates = build_windowed_aggregates(
            pretrain_paths,
            window_size=WINDOW_TRANSACTIONS,
            batch_size=WINDOWED_BATCH_SIZE,
            show_progress=True,
        )
        aggregates = None
        logger.info("Pretrain done, %s users", len(windowed_aggregates))

    labels = load_train_labels(TRAIN_LABELS_PATH)
    logger.info("Loaded %s labeled events", len(labels))

    train_paths = [p for p in TRAIN_PATHS if p.exists()]
    if not train_paths:
        raise FileNotFoundError(f"No train files at {TRAIN_PATHS}")

    logger.info("Датасет собирается из train-файлов: %s", [str(p.name) for p in train_paths])

    rows: list[dict] = []
    columns_to_read = list(FEATURE_COLUMNS) + [EVENT_ID_COLUMN]
    # В режиме full: сколько размеченных строк по клиенту уже добавлено в датасет
    per_customer_labeled_count: dict[int | str, int] = {}

    batch_size_train = WINDOWED_BATCH_SIZE if use_window else BATCH_SIZE
    for path in train_paths:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        columns = [c for c in columns_to_read if c in schema.names]
        num_rows = pf.metadata.num_rows if pf.metadata else None
        pbar = tqdm(
            pf.iter_batches(columns=columns, batch_size=batch_size_train),
            desc=path.name,
            total=(num_rows + batch_size_train - 1) // batch_size_train if num_rows is not None else None,
            unit="batch",
        )
        for batch in pbar:
            col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = batch.num_rows
            for i in range(n):
                row = {name: col_lists[name][i] for name in batch.schema.names}
                cid = row.get(CUSTOMER_ID_COLUMN)
                if cid is None:
                    continue
                event_id = row.get(EVENT_ID_COLUMN)
                if event_id not in labels:
                    if use_full:
                        aggregates[cid].update(row)
                    else:
                        if cid not in windowed_aggregates:
                            windowed_aggregates[cid] = WindowedAggregates(WINDOW_TRANSACTIONS)
                        windowed_aggregates[cid].add(row)
                    continue
                if use_full:
                    transactions_seen = per_customer_labeled_count.get(cid, 0)
                    feats = compute_features(aggregates[cid], row, transactions_seen=transactions_seen)
                    aggregates[cid].update(row)
                    per_customer_labeled_count[cid] = transactions_seen + 1
                else:
                    if cid not in windowed_aggregates:
                        windowed_aggregates[cid] = WindowedAggregates(WINDOW_TRANSACTIONS)
                    agg = windowed_aggregates[cid].get_aggregates()
                    feats = compute_features(agg, row)
                    windowed_aggregates[cid].add(row)
                out = {
                    **feats,
                    EVENT_ID_COLUMN: event_id,
                    TARGET_COLUMN: int(labels[event_id]),
                    "event_dttm": row.get(EVENT_DTTM_COLUMN),
                }
                rows.append(out)

    if not rows:
        raise ValueError("No labeled rows collected. Check TRAIN_PATHS and TRAIN_LABELS_PATH.")

    df = pd.DataFrame(rows)
    TRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRAIN_DATASET_PATH, index=False)
    logger.info("Saved %s rows to %s", len(df), TRAIN_DATASET_PATH)


if __name__ == "__main__":
    main()
