"""
Permutation importance по train-датасету для XGBoost.

Датасет: используется только `output/full_dataset.parquet`.

Признаки определяются динамически из parquet: берутся все колонки кроме служебных
(`target`, `event_dttm`, `sample_weight`, `event_id`, `customer_id`).
Это позволяет автоматически работать с актуальным состоянием датасета.

Важность: permutation importance на временной валидации:
importance = baseline_pr_auc - pr_auc_after_shuffle(feature)

Запуск из корня: PYTHONPATH=. python research/main.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import OUTPUT_DIR, TRAIN_DATASET_META_COLUMNS, TRAIN_DATASET_PATH, remap_sample_weight_from_dataset
from training.config import VAL_RATIO, XGB_PARAMS

logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _format_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def _estimate_sample_memory_bytes(n_rows: int, n_features: int, include_weight: bool) -> int:
    x_bytes = n_rows * n_features * np.dtype(np.float32).itemsize
    y_bytes = n_rows * np.dtype(np.int32).itemsize
    w_bytes = n_rows * np.dtype(np.float32).itemsize if include_weight else 0
    return x_bytes + y_bytes + w_bytes


def _apply_memory_budget(
    requested_train_rows: int,
    requested_val_rows: int,
    n_features: int,
    max_memory_gb: float,
    safety_factor: float,
) -> tuple[int, int]:
    """
    Подбираем безопасные размеры train/val выборок под лимит памяти.
    """
    budget_bytes = int(max_memory_gb * (1024 ** 3) * safety_factor)
    req_bytes = (
        _estimate_sample_memory_bytes(requested_train_rows, n_features, include_weight=True)
        + _estimate_sample_memory_bytes(requested_val_rows, n_features, include_weight=False)
    )
    if req_bytes <= budget_bytes:
        logger.info(
            "Оценка памяти сэмплов: %s (бюджет %s) — оставляем запрошенные размеры.",
            _format_gb(req_bytes),
            _format_gb(budget_bytes),
        )
        return requested_train_rows, requested_val_rows

    scale = budget_bytes / max(req_bytes, 1)
    safe_train = max(50_000, int(requested_train_rows * scale))
    safe_val = max(25_000, int(requested_val_rows * scale))
    safe_bytes = (
        _estimate_sample_memory_bytes(safe_train, n_features, include_weight=True)
        + _estimate_sample_memory_bytes(safe_val, n_features, include_weight=False)
    )
    logger.warning(
        "Запрошенные train/val сэмплы требуют %s > бюджета %s. "
        "Авто-уменьшение до train=%d, val=%d (оценка %s).",
        _format_gb(req_bytes),
        _format_gb(budget_bytes),
        safe_train,
        safe_val,
        _format_gb(safe_bytes),
    )
    return safe_train, safe_val


def _prepare_batch(dfb: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dfb[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=False)
    y = pd.to_numeric(dfb["target"], errors="coerce").fillna(0).astype(np.int32).to_numpy(copy=False)
    if "sample_weight" in dfb.columns:
        w = pd.to_numeric(dfb["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32).to_numpy(copy=False)
        w = remap_sample_weight_from_dataset(w)
    else:
        w = np.ones(shape=(len(dfb),), dtype=np.float32)
    return x, y, w


def _detect_columns(path: Path) -> tuple[list[str], str]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    ignored_cols = {"target", "event_dttm", "sample_weight", "event_id", *TRAIN_DATASET_META_COLUMNS}
    feature_cols = [name for name in names if name not in ignored_cols]
    if "event_dttm" not in names:
        raise ValueError("В датасете нет колонки event_dttm")
    if "target" not in names:
        raise ValueError("В датасете нет target")
    if not feature_cols:
        raise ValueError("Не найдены колонки признаков: после исключения служебных колонок список пуст.")
    return feature_cols, "event_dttm"


def _batch_read_columns(path: Path, feature_cols: list[str], dttm_col: str) -> list[str]:
    """Колонки для iter_batches: фичи + target + время + sample_weight при наличии."""
    available = frozenset(pq.ParquetFile(path).schema_arrow.names)
    cols: list[str] = list(feature_cols) + ["target", dttm_col]
    if "sample_weight" in available:
        cols.append("sample_weight")
    missing = [c for c in cols if c not in available]
    if missing:
        raise ValueError(f"В {path} нет колонок: {missing}")
    return cols


def _find_time_cutoff_paths(paths: list[Path], val_ratio: float, batch_size: int = 2_500_000) -> pd.Timestamp:
    by_day: Counter[pd.Timestamp] = Counter()
    total = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        tag = path.name
        for rb in tqdm(
            pf.iter_batches(columns=["event_dttm"], batch_size=batch_size),
            desc=f"Скан дат ({tag})",
            unit="batch",
        ):
            s = pd.to_datetime(rb.column(0).to_pandas(), errors="coerce").dt.floor("D")
            vc = s.value_counts(dropna=True)
            for k, v in vc.items():
                by_day[k] += int(v)
                total += int(v)
    if total == 0:
        raise ValueError("Не удалось прочитать event_dttm для split по времени.")
    val_target = max(1, int(total * val_ratio))
    acc = 0
    cutoff = None
    for day, cnt in sorted(by_day.items(), reverse=True):
        acc += cnt
        cutoff = day
        if acc >= val_target:
            break
    assert cutoff is not None
    logger.info("Time split cutoff day: %s (val target rows ~= %d, total rows %d)", cutoff.date(), val_target, total)
    return cutoff


def _find_time_cutoff(path: Path, val_ratio: float, batch_size: int = 2_500_000) -> pd.Timestamp:
    return _find_time_cutoff_paths([path], val_ratio=val_ratio, batch_size=batch_size)


def _fit_xgb(x_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, feature_cols: list[str]):
    import xgboost as xgb

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": XGB_PARAMS.get("learning_rate", 0.05),
        "max_depth": XGB_PARAMS.get("max_depth", 8),
        "subsample": XGB_PARAMS.get("subsample", 0.8),
        "colsample_bytree": XGB_PARAMS.get("colsample_bytree", 0.8),
        "tree_method": XGB_PARAMS.get("tree_method", "hist"),
        "seed": XGB_PARAMS.get("random_state", 42),
    }
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train, feature_names=feature_cols)
    rounds = int(XGB_PARAMS.get("n_estimators", 600))
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=max(50, rounds))


def _predict_proba(model, x: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    import xgboost as xgb

    d = xgb.DMatrix(x, feature_names=feature_cols)
    return np.asarray(model.predict(d), dtype=np.float32)


def _compute_permutation_importance(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    repeats: int,
    random_seed: int,
) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(random_seed)
    baseline_pred = _predict_proba(model, x_val, feature_cols)
    baseline = float(average_precision_score(y_val, baseline_pred))
    logger.info("xgboost baseline PR-AUC: %.6f", baseline)

    importances: dict[str, float] = {}
    for j, feat in enumerate(tqdm(feature_cols, desc="xgboost: permutation", unit="feature")):
        drops: list[float] = []
        original = x_val[:, j].copy()
        for _ in range(repeats):
            perm_idx = rng.permutation(x_val.shape[0])
            x_val[:, j] = original[perm_idx]
            pred = _predict_proba(model, x_val, feature_cols)
            score = float(average_precision_score(y_val, pred))
            drops.append(baseline - score)
        x_val[:, j] = original
        importances[feat] = float(np.mean(drops))
    return baseline, importances


def _sample_from_stream(
    path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    max_train_rows: int,
    max_val_rows: int,
    batch_size: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    train_seen = 0
    val_seen = 0
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    w_train: np.ndarray | None = None
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None

    pf = pq.ParquetFile(path)
    read_cols = _batch_read_columns(path, feature_cols, dttm_col)
    for rb in tqdm(
        pf.iter_batches(columns=read_cols, batch_size=batch_size),
        desc=f"Сэмплирование ({path.name})",
        unit="batch",
    ):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        train_mask = dttm < cutoff_day
        val_mask = ~train_mask

        if train_mask.any():
            xtr, ytr, wtr = _prepare_batch(dfb.loc[train_mask], feature_cols)
            for i in range(xtr.shape[0]):
                train_seen += 1
                if x_train is None:
                    x_train = np.empty((max_train_rows, len(feature_cols)), dtype=np.float32)
                    y_train = np.empty((max_train_rows,), dtype=np.int32)
                    w_train = np.empty((max_train_rows,), dtype=np.float32)
                if train_seen <= max_train_rows:
                    idx = train_seen - 1
                    x_train[idx] = xtr[i]
                    y_train[idx] = ytr[i]
                    w_train[idx] = wtr[i]
                else:
                    j = int(rng.integers(0, train_seen))
                    if j < max_train_rows:
                        x_train[j] = xtr[i]
                        y_train[j] = ytr[i]
                        w_train[j] = wtr[i]

        if val_mask.any():
            xva, yva, _ = _prepare_batch(dfb.loc[val_mask], feature_cols)
            for i in range(xva.shape[0]):
                val_seen += 1
                if x_val is None:
                    x_val = np.empty((max_val_rows, len(feature_cols)), dtype=np.float32)
                    y_val = np.empty((max_val_rows,), dtype=np.int32)
                if val_seen <= max_val_rows:
                    idx = val_seen - 1
                    x_val[idx] = xva[i]
                    y_val[idx] = yva[i]
                else:
                    j = int(rng.integers(0, val_seen))
                    if j < max_val_rows:
                        x_val[j] = xva[i]
                        y_val[j] = yva[i]

    if x_train is None or y_train is None or w_train is None or x_val is None or y_val is None:
        raise RuntimeError("Не удалось собрать train/val сэмплы из датасета.")

    n_train = min(train_seen, max_train_rows)
    n_val = min(val_seen, max_val_rows)
    logger.info("Собран train sample: %d (из %d), val sample: %d (из %d)", n_train, train_seen, n_val, val_seen)
    return x_train[:n_train], y_train[:n_train], w_train[:n_train], x_val[:n_val], y_val[:n_val]


def _importances_to_df(model_name: str, baseline: float, importances: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "feature": list(importances.keys()),
            "importance_drop_pr_auc": list(importances.values()),
        }
    )
    df["model"] = model_name
    df["baseline_pr_auc"] = baseline
    return df.sort_values("importance_drop_pr_auc", ascending=False, ignore_index=True)


def _plot_summary(summary: pd.DataFrame, out_dir: Path, top_k: int) -> tuple[Path, Path]:
    top_df = summary.head(top_k).copy()
    low_df = summary.tail(top_k).copy().sort_values("mean_importance_drop_pr_auc", ascending=True)

    top_path = out_dir / "top_features.png"
    low_path = out_dir / "least_features.png"

    plt.figure(figsize=(12, 8))
    plt.barh(top_df["feature"][::-1], top_df["mean_importance_drop_pr_auc"][::-1], color="#1f77b4")
    plt.xlabel("Mean drop in PR-AUC after permutation")
    plt.ylabel("Feature")
    plt.title("Most important features")
    plt.tight_layout()
    plt.savefig(top_path, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.barh(low_df["feature"], low_df["mean_importance_drop_pr_auc"], color="#ff7f0e")
    plt.xlabel("Mean drop in PR-AUC after permutation")
    plt.ylabel("Feature")
    plt.title("Least important features")
    plt.tight_layout()
    plt.savefig(low_path, dpi=160)
    plt.close()
    return top_path, low_path


def _write_markdown_report(
    out_path: Path,
    train_rows: int,
    val_rows: int,
    repeats: int,
    max_memory_gb: float,
    summary: pd.DataFrame,
    top_png: Path,
    low_png: Path,
) -> None:
    top_k = min(15, len(summary))
    top_df = summary.head(top_k)
    low_df = summary.tail(top_k).iloc[::-1].reset_index(drop=True)

    lines: list[str] = []
    lines.append("# Feature Importance Research")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Model: `xgboost`")
    lines.append(f"- Train rows used: `{train_rows}`")
    lines.append(f"- Validation rows used: `{val_rows}`")
    lines.append(f"- Permutation repeats: `{repeats}`")
    lines.append(f"- Memory budget target: `{max_memory_gb:.1f} GB`")
    lines.append("")
    lines.append("## How to read importance")
    lines.append("- Value = `baseline_pr_auc - pr_auc_after_shuffle(feature)`.")
    lines.append("- Bigger positive value => feature is more useful for quality.")
    lines.append("- Near zero => feature contributes little in this setup.")
    lines.append("- Negative value => possible noise / interaction artifact; candidate for review.")
    lines.append("")
    lines.append("## Charts")
    lines.append(f"- Most important: `{top_png.name}`")
    lines.append(f"- Least important: `{low_png.name}`")
    lines.append("")
    lines.append("## Top most important features")
    for i, row in top_df.iterrows():
        lines.append(f"{i + 1}. `{row['feature']}`: `{row['mean_importance_drop_pr_auc']:.8f}`")
    lines.append("")
    lines.append("## Top least important features")
    for i, row in low_df.iterrows():
        lines.append(f"{i + 1}. `{row['feature']}`: `{row['mean_importance_drop_pr_auc']:.8f}`")
    lines.append("")
    lines.append("## Practical interpretation")
    lines.append("- Keep top features as priority signals in future iterations.")
    lines.append("- For least-important features: test ablation/removal and compare PR-AUC + stability.")
    lines.append("- Re-run this analysis after major feature-engineering or dataset rebuild.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=2_000_000, help="Макс. число строк train для сэмпла")
    parser.add_argument("--max-val-rows", type=int, default=600_000, help="Макс. число строк val для сэмпла")
    parser.add_argument("--batch-size", type=int, default=250_000, help="Размер parquet batch (меньше => безопаснее по RAM)")
    parser.add_argument("--repeats", type=int, default=5, help="Число повторов permutation для каждой фичи")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-memory-gb", type=float, default=20.0, help="Верхний предел RAM для процесса")
    parser.add_argument(
        "--memory-safety-factor",
        type=float,
        default=0.55,
        help="Доля RAM-бюджета под train/val массивы (остальное под pandas/модель)",
    )
    args = parser.parse_args()

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. "
            "Сначала соберите датасет: ./dataset_cpp/build/build_dataset ."
        )
    logger.info("Файл датасета: %s", TRAIN_DATASET_PATH)

    if args.max_memory_gb <= 1.0:
        raise ValueError("--max-memory-gb должен быть > 1")
    if not (0.2 <= args.memory_safety_factor <= 0.9):
        raise ValueError("--memory-safety-factor должен быть в диапазоне [0.2, 0.9]")

    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    logger.info("Признаков для анализа: %d (%s …)", len(feature_cols), feature_cols[0])
    safe_train_rows, safe_val_rows = _apply_memory_budget(
        requested_train_rows=args.max_train_rows,
        requested_val_rows=args.max_val_rows,
        n_features=len(feature_cols),
        max_memory_gb=args.max_memory_gb,
        safety_factor=args.memory_safety_factor,
    )
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)
    x_train, y_train, w_train, x_val, y_val = _sample_from_stream(
        TRAIN_DATASET_PATH,
        feature_cols=feature_cols,
        dttm_col=dttm_col,
        cutoff_day=cutoff_day,
        max_train_rows=safe_train_rows,
        max_val_rows=safe_val_rows,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    logger.info("=== Обучение и importance: xgboost ===")
    model = _fit_xgb(x_train, y_train, w_train, feature_cols)
    baseline, importances = _compute_permutation_importance(
        model=model,
        x_val=x_val.copy(),
        y_val=y_val,
        feature_cols=feature_cols,
        repeats=args.repeats,
        random_seed=args.seed,
    )
    result_frames = [_importances_to_df("xgboost", baseline, importances)]

    all_results = pd.concat(result_frames, ignore_index=True)
    summary = (
        all_results.groupby("feature", as_index=False)["importance_drop_pr_auc"]
        .mean()
        .rename(columns={"importance_drop_pr_auc": "mean_importance_drop_pr_auc"})
        .sort_values("mean_importance_drop_pr_auc", ascending=False, ignore_index=True)
    )

    out_dir = OUTPUT_DIR / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_path = out_dir / "feature_importance_per_model.csv"
    summary_path = out_dir / "feature_importance_summary.csv"
    txt_path = out_dir / "feature_importance_report.txt"
    md_path = out_dir / "feature_importance_report.md"

    all_results.to_csv(per_model_path, index=False)
    summary.to_csv(summary_path, index=False)

    top_k = min(15, len(summary))
    top_df = summary.head(top_k)
    low_df = summary.tail(top_k).iloc[::-1].reset_index(drop=True)

    lines = []
    lines.append("Feature importance report (permutation importance on time-based validation)")
    lines.append("Model: xgboost")
    lines.append(f"Rows used: train={len(x_train)}, val={len(x_val)}")
    lines.append(f"Permutation repeats: {args.repeats}")
    lines.append("")
    lines.append("TOP most important features:")
    for i, row in top_df.iterrows():
        lines.append(f"{i + 1:2d}. {row['feature']}: {row['mean_importance_drop_pr_auc']:.8f}")
    lines.append("")
    lines.append("TOP least important features:")
    for i, row in low_df.iterrows():
        lines.append(f"{i + 1:2d}. {row['feature']}: {row['mean_importance_drop_pr_auc']:.8f}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    top_png, low_png = _plot_summary(summary, out_dir=out_dir, top_k=top_k)
    _write_markdown_report(
        out_path=md_path,
        train_rows=len(x_train),
        val_rows=len(x_val),
        repeats=args.repeats,
        max_memory_gb=args.max_memory_gb,
        summary=summary,
        top_png=top_png,
        low_png=low_png,
    )

    logger.info("Сохранено: %s", per_model_path)
    logger.info("Сохранено: %s", summary_path)
    logger.info("Сохранено: %s", txt_path)
    logger.info("Сохранено: %s", md_path)
    logger.info("Сохранено: %s", top_png)
    logger.info("Сохранено: %s", low_png)
    logger.info("Самая важная фича: %s", top_df.iloc[0]["feature"])
    logger.info("Самая неважная фича: %s", low_df.iloc[0]["feature"])


if __name__ == "__main__":
    main()
