#!/usr/bin/env python3
"""
Для каждого parquet-файла в data/ с колонкой customer_id:
  - выводится короткая статистика по распределению «записей на пользователя»
    (среднее, медиана, мин, макс, Q1, Q3);
  - сохраняется одна гистограмма распределения (бины могут объединять несколько значений).
Чтение parquet — побатчево.
Запуск: PYTHONPATH=. python scripts/data_transactions_distribution.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
DATA_ROOT = PROJECT_ROOT / "data"
CUSTOMER_ID_COLUMN = "customer_id"
BATCH_SIZE = 500_000
PLOTS_DIR = PROJECT_ROOT / "output" / "transaction_distribution_plots"


def collect_parquet_paths(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*.parquet"))
    return [p for p in paths if p.is_file()]


def schema_has_column(path: Path, column: str) -> bool:
    schema = pq.read_schema(path)
    return column in [f.name for f in schema]


def count_transactions_per_user(path: Path) -> list[int]:
    """Читает parquet побатчево по customer_id, возвращает список «транзакций на пользователя»."""
    file_ = pq.ParquetFile(path)
    user_counts: dict[int, int] = {}
    nrows = int(file_.metadata.num_rows)
    with tqdm(total=nrows, desc=f"read {path.name}", unit="row", leave=False) as pbar:
        for batch in file_.iter_batches(columns=[CUSTOMER_ID_COLUMN], batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            vc = df[CUSTOMER_ID_COLUMN].value_counts()
            for uid, cnt in vc.items():
                if pd.isna(uid):
                    continue
                user_counts[int(uid)] = user_counts.get(int(uid), 0) + cnt
            pbar.update(batch.num_rows)
    return list(user_counts.values())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not DATA_ROOT.is_dir():
        logger.error("Каталог данных не найден: %s", DATA_ROOT)
        sys.exit(1)

    paths = collect_parquet_paths(DATA_ROOT)
    if not paths:
        logger.warning("Parquet-файлы в data/ не найдены.")
        sys.exit(0)

    logger.info("Найдено parquet-файлов: %d", len(paths))
    for path in tqdm(paths, desc="parquet files", unit="file"):
        if not schema_has_column(path, CUSTOMER_ID_COLUMN):
            continue

        per_user_counts = count_transactions_per_user(path)
        if not per_user_counts:
            continue

        arr = np.array(per_user_counts)
        rel = path.relative_to(PROJECT_ROOT)

        # Короткая статистика
        logger.info("%s", rel)
        logger.info(
            "  среднее: %.2f  медиана: %.2f",
            float(np.mean(arr)),
            float(np.median(arr)),
        )
        logger.info("  мин: %d  макс: %d", int(np.min(arr)), int(np.max(arr)))
        logger.info(
            "  Q1: %.2f  Q3: %.2f",
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
        )

        # Одна гистограмма: все данные, бины объединяют диапазон значений
        for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
            try:
                plt.style.use(style)
                break
            except OSError:
                continue
        plt.rcParams["figure.facecolor"] = "#fafafa"
        plt.rcParams["axes.facecolor"] = "#ffffff"

        fig, ax = plt.subplots(figsize=(10, 5))
        lo, hi = int(arr.min()), int(arr.max())
        n_bins = min(80, max(15, hi - lo + 1))  # бины охватывают весь диапазон
        ax.hist(arr, bins=n_bins, range=(lo, hi), color="#2ecc71", edgecolor="white", alpha=0.85, linewidth=0.5)
        ax.set_xlabel("Количество транзакций на пользователя", fontsize=11)
        ax.set_ylabel("Число пользователей", fontsize=11)
        ax.set_title(str(rel), fontsize=12)
        ax.tick_params(axis="both", labelsize=9)
        fig.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / f"{path.stem}_hist.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
