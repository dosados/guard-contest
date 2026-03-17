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
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Добавляем корень проекта в путь для импорта shared
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import (
    DATASET_N_WORKERS,
    PRETRAIN_PATHS,
    TRAIN_DATASET_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATHS,
    WINDOW_TRANSACTIONS,
    WINDOWED_BATCH_SIZE,
)
from shared.features import FEATURE_NAMES, compute_features
from shared.parquet_batch_aggregates import (
    CUSTOMER_ID_COLUMN,
    EVENT_DTTM_COLUMN,
    EVENT_ID_COLUMN,
    FEATURE_COLUMNS,
    build_windowed_aggregates,
    WindowedAggregates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"
WEIGHT_COLUMN = "sample_weight"

# Порядок колонок в выходном датасете (для колоночного накопления и слияния частей)
OUTPUT_COLUMNS = list(FEATURE_NAMES) + [EVENT_ID_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN, "event_dttm"]

# Веса примеров для обучения:
# - события без обратной связи (нет в train_labels.parquet)
# - события с обратной связью target=0 (подтверждённые клиентом)
# - события с обратной связью target=1 (неподтверждённые / целевые)
WEIGHT_UNLABELED = 1.0
WEIGHT_LABELED_0 = 2.0
WEIGHT_LABELED_1 = 3.0


def load_train_labels(path: Path) -> dict[int, int]:
    """
    Загружает разметку train из parquet: event_id -> метка обратной связи (0/1).

    В новом формате:
      - сам факт наличия event_id в этом файле означает, что операция имеет обратную связь
        и должна считаться целевым классом (target=1) в обучающем датасете;
      - значение target (0 или 1) используется только для задания веса примера:
        примеры с target=1 получают больший вес, чем с target=0.
    """
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    df = pd.read_parquet(path)
    if "event_id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Labels parquet must have event_id and target, got {list(df.columns)}")
    return df.set_index("event_id")["target"].astype(int).to_dict()


def _pool_worker_init() -> None:
    """В дочернем процессе: отключаем буфер stdout/stderr, чтобы прогресс был виден."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def _process_one_part(
    pretrain_path: Path | str,
    train_path: Path | str,
    labels: dict[int, int],
    window_size: int,
    batch_size: int,
) -> dict[str, list]:
    """
    Обрабатывает одну пару (pretrain_part_i, train_part_i): строит оконные агрегаты
    из pretrain, проходится по train, считает фичи и накапливает строки в колоночном виде.
    Возвращает dict[имя_колонки] -> list значений (без изменения логики агрегатов).
    """
    pretrain_path = Path(pretrain_path)
    train_path = Path(train_path)
    logger.info("Часть %s: pretrain -> агрегаты", train_path.name)
    windowed_aggregates = build_windowed_aggregates(
        [pretrain_path] if pretrain_path.exists() else [],
        window_size=window_size,
        batch_size=batch_size,
        show_progress=False,
    )
    columnar: dict[str, list] = {c: [] for c in OUTPUT_COLUMNS}
    columns_to_read = list(FEATURE_COLUMNS) + [EVENT_ID_COLUMN]
    pf = pq.ParquetFile(train_path)
    schema = pf.schema_arrow
    columns = [c for c in columns_to_read if c in schema.names]
    num_rows = pf.metadata.num_rows if pf.metadata else None
    total_batches = (num_rows + batch_size - 1) // batch_size if num_rows is not None else None
    batch_iter = tqdm(
        pf.iter_batches(columns=columns, batch_size=batch_size),
        desc=train_path.name,
        total=total_batches,
        unit="batch",
    )
    for batch in batch_iter:
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            if cid not in windowed_aggregates:
                windowed_aggregates[cid] = WindowedAggregates(window_size)
            agg = windowed_aggregates[cid].get_aggregates()
            tr_amount = min(len(windowed_aggregates[cid]), window_size)
            feats = compute_features(agg, row, tr_amount=tr_amount)
            windowed_aggregates[cid].add(row)
            event_id = row.get(EVENT_ID_COLUMN)
            is_labeled = event_id in labels
            target = 1 if is_labeled else 0
            if not is_labeled:
                weight = WEIGHT_UNLABELED
            else:
                label_val = int(labels[event_id])
                weight = WEIGHT_LABELED_0 if label_val == 0 else WEIGHT_LABELED_1
            for name in FEATURE_NAMES:
                columnar[name].append(feats[name])
            columnar[EVENT_ID_COLUMN].append(event_id)
            columnar[TARGET_COLUMN].append(target)
            columnar[WEIGHT_COLUMN].append(float(weight))
            columnar["event_dttm"].append(row.get(EVENT_DTTM_COLUMN))
    return columnar


def _run_sequential(
    pretrain_paths: list[Path],
    train_paths: list[Path],
    labels: dict[int, int],
) -> dict[str, list]:
    """Последовательный проход: один общий агрегат по всем pretrain, затем все train. Колоночное накопление."""
    windowed_aggregates = build_windowed_aggregates(
        pretrain_paths,
        window_size=WINDOW_TRANSACTIONS,
        batch_size=WINDOWED_BATCH_SIZE,
        show_progress=True,
    )
    columnar: dict[str, list] = {c: [] for c in OUTPUT_COLUMNS}
    columns_to_read = list(FEATURE_COLUMNS) + [EVENT_ID_COLUMN]
    for path in train_paths:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        columns = [c for c in columns_to_read if c in schema.names]
        num_rows = pf.metadata.num_rows if pf.metadata else None
        pbar = tqdm(
            pf.iter_batches(columns=columns, batch_size=WINDOWED_BATCH_SIZE),
            desc=path.name,
            total=(num_rows + WINDOWED_BATCH_SIZE - 1) // WINDOWED_BATCH_SIZE if num_rows is not None else None,
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
                if cid not in windowed_aggregates:
                    windowed_aggregates[cid] = WindowedAggregates(WINDOW_TRANSACTIONS)
                agg = windowed_aggregates[cid].get_aggregates()
                tr_amount = min(len(windowed_aggregates[cid]), WINDOW_TRANSACTIONS)
                feats = compute_features(agg, row, tr_amount=tr_amount)
                windowed_aggregates[cid].add(row)
                event_id = row.get(EVENT_ID_COLUMN)
                is_labeled = event_id in labels
                target = 1 if is_labeled else 0
                if not is_labeled:
                    weight = WEIGHT_UNLABELED
                else:
                    label_val = int(labels[event_id])
                    weight = WEIGHT_LABELED_0 if label_val == 0 else WEIGHT_LABELED_1
                for name in FEATURE_NAMES:
                    columnar[name].append(feats[name])
                columnar[EVENT_ID_COLUMN].append(event_id)
                columnar[TARGET_COLUMN].append(target)
                columnar[WEIGHT_COLUMN].append(float(weight))
                columnar["event_dttm"].append(row.get(EVENT_DTTM_COLUMN))
    return columnar


def main() -> None:
    logger.info("Оконный режим: window=%s", WINDOW_TRANSACTIONS)

    pretrain_paths = [p for p in PRETRAIN_PATHS if p.exists()]
    if not pretrain_paths:
        logger.warning("No pretrain files found at %s, starting with empty aggregates", PRETRAIN_PATHS)

    train_paths = [p for p in TRAIN_PATHS if p.exists()]
    if not train_paths:
        raise FileNotFoundError(f"No train files at {TRAIN_PATHS}")

    labels = load_train_labels(TRAIN_LABELS_PATH)
    logger.info("Loaded %s labeled events", len(labels))

    use_parallel = (
        DATASET_N_WORKERS > 0
        and len(pretrain_paths) == len(train_paths)
        and len(pretrain_paths) >= 1
    )
    if use_parallel:
        n_workers = min(DATASET_N_WORKERS, len(pretrain_paths))
        logger.info(
            "Параллельное создание датасета: %s процессов, пары (pretrain_part_i, train_part_i)",
            n_workers,
        )
        args_list = [
            (pretrain_paths[i], train_paths[i], labels, WINDOW_TRANSACTIONS, WINDOWED_BATCH_SIZE)
            for i in range(len(pretrain_paths))
        ]
        with Pool(processes=n_workers, initializer=_pool_worker_init) as pool:
            logger.info("Запуск частей, ожидание завершения...")
            parts = pool.starmap(_process_one_part, args_list)
        columnar = {c: [] for c in OUTPUT_COLUMNS}
        for part in parts:
            for c in OUTPUT_COLUMNS:
                columnar[c].extend(part[c])
    else:
        if pretrain_paths:
            logger.info(
                "Оконные агрегаты (window=%s) строятся из pretrain: %s",
                WINDOW_TRANSACTIONS,
                [str(p.name) for p in pretrain_paths],
            )
        logger.info("Датасет собирается из train-файлов: %s", [str(p.name) for p in train_paths])
        columnar = _run_sequential(pretrain_paths, train_paths, labels)

    n_rows = len(columnar[OUTPUT_COLUMNS[0]])
    if n_rows == 0:
        raise ValueError("No rows collected. Check TRAIN_PATHS and TRAIN_LABELS_PATH.")

    df = pd.DataFrame(columnar, columns=OUTPUT_COLUMNS)
    TRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRAIN_DATASET_PATH, index=False)
    logger.info("Saved %s rows to %s", len(df), TRAIN_DATASET_PATH)


if __name__ == "__main__":
    main()
