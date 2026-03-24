#!/usr/bin/env python3
"""
Подробный анализ распределения "числа записей на пользователя" по наборам:
  - pretrain_part_*.parquet (вместе и по каждому part)
  - train_part_*.parquet (вместе и по каждому part)
  - pretest.parquet
  - test.parquet

Скрипт сохраняет все результаты в директорию, где расположен сам:
  - user_record_distribution_report.txt
  - *_distribution.csv
  - *_top_users.csv
  - *.png (гистограммы)

Запуск из корня проекта:
  PYTHONPATH=. python scripts/user_record_distribution/analyze_user_record_distribution.py
"""

from __future__ import annotations

import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import PRETEST_PATH, PRETRAIN_PATHS, TEST_PATH, TRAIN_PATHS
from shared.parquet_batch_aggregates import CUSTOMER_ID_COLUMN

logger = logging.getLogger(__name__)
BATCH_SIZE = 500_000
OUTPUT_DIR = Path(__file__).resolve().parent
REPORT_PATH = OUTPUT_DIR / "user_record_distribution_report.txt"


@dataclass
class DatasetResult:
    name: str
    counts_by_user: dict[int, int]
    source_paths: list[Path]

    @property
    def counts_array(self) -> np.ndarray:
        return np.array(list(self.counts_by_user.values()), dtype=np.int64)


def _read_user_counts_from_parquet(path: Path) -> dict[int, int]:
    file_ = pq.ParquetFile(path)
    total_rows = int(file_.metadata.num_rows)
    user_counts: dict[int, int] = defaultdict(int)

    with tqdm(
        total=total_rows,
        desc=f"read {path.name}",
        unit="row",
        leave=False,
    ) as pbar:
        for batch in file_.iter_batches(columns=[CUSTOMER_ID_COLUMN], batch_size=BATCH_SIZE):
            ids = batch.column(CUSTOMER_ID_COLUMN).to_pylist()
            for uid in ids:
                if uid is None or (isinstance(uid, float) and math.isnan(uid)):
                    continue
                user_counts[int(uid)] += 1
            pbar.update(batch.num_rows)

    return dict(user_counts)


def collect_counts(paths: Iterable[Path], dataset_name: str) -> DatasetResult:
    merged_counts: dict[int, int] = defaultdict(int)
    used_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            logger.warning("Файл не найден, пропускаю: %s", path)
            continue
        used_paths.append(path)
        current = _read_user_counts_from_parquet(path)
        for uid, cnt in current.items():
            merged_counts[uid] += cnt
    return DatasetResult(
        name=dataset_name,
        counts_by_user=dict(merged_counts),
        source_paths=used_paths,
    )


def _style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def save_histogram(result: DatasetResult) -> None:
    arr = result.counts_array
    if arr.size == 0:
        return

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    lo, hi = int(arr.min()), int(arr.max())
    n_bins_full = min(140, max(25, hi - lo + 1))

    axes[0].hist(
        arr,
        bins=n_bins_full,
        range=(lo, hi),
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].set_title(f"{result.name}: полный диапазон", fontsize=12)
    axes[0].set_xlabel("Записей на пользователя")
    axes[0].set_ylabel("Число пользователей")
    axes[0].grid(alpha=0.3)

    # Увеличенный фокус на "теле" распределения: до p99.
    upper_focus = int(np.quantile(arr, 0.99))
    upper_focus = max(upper_focus, lo + 1)
    n_bins_focus = min(120, max(20, upper_focus - lo + 1))
    axes[1].hist(
        arr,
        bins=n_bins_focus,
        range=(lo, upper_focus),
        color="#F58518",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_title(f"{result.name}: увеличенный масштаб (до p99={upper_focus})", fontsize=12)
    axes[1].set_xlabel("Записей на пользователя")
    axes[1].set_ylabel("Число пользователей")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{result.name}_hist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_distribution_csv(result: DatasetResult) -> None:
    arr = result.counts_array
    if arr.size == 0:
        return
    values, freqs = np.unique(arr, return_counts=True)
    df = pd.DataFrame(
        {
            "records_per_user": values,
            "users_count": freqs,
            "users_share": freqs / freqs.sum(),
        }
    )
    df.to_csv(OUTPUT_DIR / f"{result.name}_distribution.csv", index=False)


def save_top_users_csv(result: DatasetResult, top_n: int = 200) -> None:
    if not result.counts_by_user:
        return
    rows = sorted(result.counts_by_user.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    df = pd.DataFrame(rows, columns=["customer_id", "records_count"])
    df.to_csv(OUTPUT_DIR / f"{result.name}_top_users.csv", index=False)


def build_report(results: list[DatasetResult]) -> str:
    lines: list[str] = []
    lines.append("Подробный отчёт по распределению количества записей на пользователя")
    lines.append("=" * 78)
    lines.append("")

    for res in results:
        arr = res.counts_array
        lines.append(f"[{res.name}]")
        lines.append("-" * 78)
        lines.append("Источники:")
        for p in res.source_paths:
            lines.append(f"  - {p.relative_to(PROJECT_ROOT)}")

        if arr.size == 0:
            lines.append("Нет данных (пустой набор или файлы не найдены).")
            lines.append("")
            continue

        q = np.quantile(arr, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        lines.append(f"Всего записей (суммарно): {int(arr.sum())}")
        lines.append(f"Уникальных пользователей: {int(arr.size)}")
        lines.append(f"Среднее: {arr.mean():.4f}")
        lines.append(f"Std: {arr.std():.4f}")
        lines.append(f"Медиана: {np.median(arr):.4f}")
        lines.append(
            "Квантили: "
            f"min={q[0]:.0f}, p1={q[1]:.0f}, p5={q[2]:.0f}, p10={q[3]:.0f}, "
            f"p25={q[4]:.0f}, p50={q[5]:.0f}, p75={q[6]:.0f}, p90={q[7]:.0f}, "
            f"p95={q[8]:.0f}, p99={q[9]:.0f}, max={q[10]:.0f}"
        )

        values, freqs = np.unique(arr, return_counts=True)
        lines.append("")
        lines.append("Топ-100 значений 'записей на пользователя' по числу пользователей:")
        order = np.argsort(-freqs)
        for idx in order[:100]:
            v = int(values[idx])
            f = int(freqs[idx])
            share = 100.0 * f / arr.size
            lines.append(f"  records={v:>6d} | users={f:>8d} | share={share:>7.3f}%")

        top_users = sorted(res.counts_by_user.items(), key=lambda x: (-x[1], x[0]))[:20]
        lines.append("")
        lines.append("Топ-20 пользователей по числу записей:")
        for uid, cnt in top_users:
            lines.append(f"  customer_id={uid:>10d} | records={cnt:>8d}")

        lines.append("")
        lines.append("Сгенерированные файлы:")
        lines.append(f"  - {res.name}_hist.png")
        lines.append(f"  - {res.name}_distribution.csv")
        lines.append(f"  - {res.name}_top_users.csv")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    groups: list[tuple[str, list[Path]]] = [
        ("pretrain_parts", list(PRETRAIN_PATHS)),
        ("train_parts", list(TRAIN_PATHS)),
        ("pretest", [PRETEST_PATH]),
        ("test", [TEST_PATH]),
    ]

    part_level: list[tuple[str, list[Path]]] = []
    for p in PRETRAIN_PATHS:
        part_level.append((p.stem, [p]))
    for p in TRAIN_PATHS:
        part_level.append((p.stem, [p]))

    all_defs = groups + part_level
    results: list[DatasetResult] = []

    for name, paths in tqdm(all_defs, desc="datasets", unit="set"):
        logger.info("Обработка %s", name)
        result = collect_counts(paths, name)
        results.append(result)
        save_histogram(result)
        save_distribution_csv(result)
        save_top_users_csv(result)

    report_text = build_report(results)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Отчёт сохранён: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
