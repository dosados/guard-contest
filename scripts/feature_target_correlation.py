"""
Скрипт для анализа фичей, коррелирующих с таргетом.

После создания датасета (dataset_cpp/build_dataset):
  - загружает train_dataset (части C++ или один parquet);
  - считает корреляцию каждой фичи с target (point-biserial для бинарного таргета);
  - строит графики и сохраняет таблицу в output/.

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/feature_target_correlation.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from shared.config import OUTPUT_DIR
from shared.train_dataset import load_train_dataframe, train_dataset_is_available
from tqdm import tqdm

logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"
NON_FEATURE_COLUMNS = {"event_id", TARGET_COLUMN, "event_dttm", "sample_weight"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def correlation_with_target(df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Корреляция каждой фичи с бинарным target (Pearson = point-biserial)."""
    y = df[TARGET_COLUMN].astype(float)
    corrs = {}
    for col in tqdm(feature_columns, desc="corr(feature, target)", unit="feat"):
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 10:
            corrs[col] = np.nan
            continue
        corrs[col] = x.loc[valid].corr(y.loc[valid])
    return pd.Series(corrs)


def plot_correlation_bars(
    corrs: pd.Series,
    out_path: Path,
    title: str = "Корреляция фичей с target",
    top_n: int = 40,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    sorted_idx = corrs.abs().sort_values(ascending=True).tail(top_n).index
    plot_corrs = corrs.loc[sorted_idx]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in plot_corrs]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_corrs) * 0.22)))
    ax.barh(range(len(plot_corrs)), plot_corrs.values, color=colors, height=0.75)
    ax.set_yticks(range(len(plot_corrs)))
    ax.set_yticklabels(plot_corrs.index, fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Корреляция с target")
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Feature-target correlation (Pearson vs binary target)")
    parser.add_argument("--no-plot", action="store_true", help="Only save CSV, do not build plots")
    parser.add_argument("--top", type=int, default=40, help="Number of features to show on bar charts (default: 40)")
    args = parser.parse_args()

    if not train_dataset_is_available():
        logger.error("Dataset not found. Соберите датасет (dataset_cpp/build_dataset).")
        sys.exit(1)

    logger.info("Загрузка датасета …")
    df = load_train_dataframe()
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        logger.error("No feature columns in dataset.")
        sys.exit(1)
    if TARGET_COLUMN not in df.columns:
        logger.error("Missing column: %s", TARGET_COLUMN)
        sys.exit(1)

    logger.info("Корреляции с target (%d фичей) …", len(feature_columns))
    corrs = correlation_with_target(df, feature_columns)

    out_table = pd.DataFrame({"feature": corrs.index, "correlation": corrs.values})
    out_table = out_table.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "feature_target_correlation.csv"
    out_table.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    if not args.no_plot:
        try:
            plot_correlation_bars(
                corrs,
                OUTPUT_DIR / "feature_target_correlation.png",
                title="Корреляция фичей с target (топ по |r|)",
                top_n=args.top,
            )
            logger.info("Saved: %s", OUTPUT_DIR / "feature_target_correlation.png")
        except ImportError:
            logger.warning("matplotlib not available, skipping correlation plot.")

    logger.info("Top 15 features by |correlation| with target:")
    print(out_table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
