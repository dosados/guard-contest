"""
Скрипт для анализа фичей, коррелирующих с таргетом.

После создания датасета (dataset/main.py) и при необходимости обучения (training/main.py):
  - загружает output/train_dataset.parquet;
  - считает корреляцию каждой фичи с target (point-biserial для бинарного таргета);
  - если есть обученная модель CatBoost (output/model.cbm), добавляет важность фичей;
  - строит графики и сохраняет таблицу в output/.

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/feature_target_correlation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from shared.config import OUTPUT_DIR, TRAIN_DATASET_PATH, MODEL_PATH

TARGET_COLUMN = "target"
NON_FEATURE_COLUMNS = {"event_id", TARGET_COLUMN, "event_dttm"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def correlation_with_target(df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Корреляция каждой фичи с бинарным target (Pearson = point-biserial)."""
    y = df[TARGET_COLUMN].astype(float)
    corrs = {}
    for col in feature_columns:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 10:
            corrs[col] = np.nan
            continue
        corrs[col] = x.loc[valid].corr(y.loc[valid])
    return pd.Series(corrs)


def load_catboost_importance(model_path: Path, feature_columns: list[str]) -> pd.Series | None:
    """Загружает важность фичей из CatBoost, если путь существует."""
    if not model_path.exists():
        return None
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        imp = model.get_feature_importance()
        # CatBoost возвращает в порядке фичей при обучении
        if len(imp) != len(feature_columns):
            return None
        return pd.Series(dict(zip(feature_columns, imp)))
    except Exception:
        return None


def plot_correlation_bars(
    corrs: pd.Series,
    out_path: Path,
    title: str = "Корреляция фичей с target",
    top_n: int = 40,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    # Сортируем по абсолютной корреляции, берём top_n
    sorted_idx = corrs.abs().sort_values(ascending=True).tail(top_n).index
    plot_corrs = corrs.loc[sorted_idx]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in plot_corrs]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_corrs) * 0.22)))
    bars = ax.barh(range(len(plot_corrs)), plot_corrs.values, color=colors, height=0.75)
    ax.set_yticks(range(len(plot_corrs)))
    ax.set_yticklabels(plot_corrs.index, fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Корреляция с target")
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_importance_bars(importance: pd.Series, out_path: Path, title: str = "Важность фичей (CatBoost)", top_n: int = 40) -> None:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    sorted_idx = importance.sort_values(ascending=True).tail(top_n).index
    plot_imp = importance.loc[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_imp) * 0.22)))
    ax.barh(range(len(plot_imp)), plot_imp.values, color="#27ae60", height=0.75)
    ax.set_yticks(range(len(plot_imp)))
    ax.set_yticklabels(plot_imp.index, fontsize=9)
    ax.set_xlabel("Важность (CatBoost)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-target correlation and optional CatBoost importance")
    parser.add_argument("--no-plot", action="store_true", help="Only save CSV, do not build plots")
    parser.add_argument("--top", type=int, default=40, help="Number of features to show on bar charts (default: 40)")
    args = parser.parse_args()

    if not TRAIN_DATASET_PATH.exists():
        print(f"Dataset not found: {TRAIN_DATASET_PATH}. Run dataset/main.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset...")
    df = pd.read_parquet(TRAIN_DATASET_PATH)
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        print("No feature columns in dataset.", file=sys.stderr)
        sys.exit(1)
    if TARGET_COLUMN not in df.columns:
        print(f"Missing column: {TARGET_COLUMN}", file=sys.stderr)
        sys.exit(1)

    print("Computing correlations with target...")
    corrs = correlation_with_target(df, feature_columns)

    out_table = pd.DataFrame({"feature": corrs.index, "correlation": corrs.values})
    out_table = out_table.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "feature_target_correlation.csv"
    out_table.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if not args.no_plot:
        try:
            plot_correlation_bars(
                corrs,
                OUTPUT_DIR / "feature_target_correlation.png",
                title="Корреляция фичей с target (топ по |r|)",
                top_n=args.top,
            )
            print(f"Saved: {OUTPUT_DIR / 'feature_target_correlation.png'}")
        except ImportError:
            print("matplotlib not available, skipping correlation plot.", file=sys.stderr)

    importance = load_catboost_importance(MODEL_PATH, feature_columns)
    if importance is not None:
        imp_table = pd.DataFrame({"feature": importance.index, "importance": importance.values})
        imp_table = imp_table.sort_values("importance", ascending=False).reset_index(drop=True)
        imp_csv = OUTPUT_DIR / "feature_importance_catboost.csv"
        imp_table.to_csv(imp_csv, index=False)
        print(f"CatBoost importance saved: {imp_csv}")
        if not args.no_plot:
            try:
                plot_importance_bars(
                    importance,
                    OUTPUT_DIR / "feature_importance_catboost.png",
                    title="Важность фичей (CatBoost)",
                    top_n=args.top,
                )
                print(f"Saved: {OUTPUT_DIR / 'feature_importance_catboost.png'}")
            except ImportError:
                print("matplotlib not available, skipping importance plot.", file=sys.stderr)
    else:
        print("CatBoost model not found, skipping feature importance. Run training/main.py to train.")

    print("\nTop 15 features by |correlation| with target:")
    print(out_table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
