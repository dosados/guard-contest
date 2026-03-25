"""
Анализ важности признаков для XGBoost, согласованный с training/main.py.

- Признаки: `training.parquet_io.detect_columns` (порядок как при обучении).
- Time split: `training.parquet_io.find_time_cutoff` (как training.main), батчи признаков —
  `prepare_batch` из того же модуля (веса через remap_sample_weight_from_dataset).
- Предсказания: `xgb_predict_with_best_iteration` (как метрика после обучения в training).
- По умолчанию загружается `MODEL_XGB_PATH` после training.main. Permutation на val-сэмпле:
  baseline − PR-AUC после shuffle, `pos_label=1`.
- Нативные важности бустера (gain, weight, cover, total_gain, total_cover) — в отдельном CSV.
- Артефакты: CSV, PNG, MD, TXT; полное ранжирование всех признаков — в `feature_ranking_all.txt`.

`--refit-on-sample`: те же params + `n_estimators`, eval на val-сэмпле **без** весов,
`early_stopping_rounds` из training.config (как в fit_xgb_parquet_iterative).

Запуск из корня: PYTHONPATH=. python research/main.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

from shared.config import MODEL_XGB_PATH, OUTPUT_DIR, TRAIN_DATASET_PATH
from training.config import VAL_RATIO, XGB_EARLY_STOPPING_ROUNDS, XGB_PARAMS
from training.main import _build_xgb_train_params
from training.parquet_io import detect_columns, find_time_cutoff, prepare_batch
from training.xgb_iterative_fit import xgb_predict_with_best_iteration

logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_NATIVE_IMPORTANCE_TYPES = ("gain", "weight", "cover", "total_gain", "total_cover")


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


def _batch_read_columns(path: Path, feature_cols: list[str], dttm_col: str) -> list[str]:
    available = frozenset(pq.ParquetFile(path).schema_arrow.names)
    cols: list[str] = list(feature_cols) + ["target", dttm_col]
    if "sample_weight" in available:
        cols.append("sample_weight")
    missing = [c for c in cols if c not in available]
    if missing:
        raise ValueError(f"В {path} нет колонок: {missing}")
    return cols


def _find_time_cutoff(path: Path, val_ratio: float, batch_size: int = 2_500_000) -> pd.Timestamp:
    """Делегирует в training.parquet_io.find_time_cutoff (как training.main); имя сохранено для xgb_forward_selection."""
    return find_time_cutoff(path, val_ratio, batch_size=batch_size)


def _load_xgb_booster(path: Path):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def _verify_booster_features(booster, feature_cols: list[str]) -> None:
    names = getattr(booster, "feature_names", None)
    if not names:
        return
    got = list(names)
    if got != feature_cols:
        raise ValueError(
            "Имена признаков в модели не совпадают с датасетом (порядок/состав). "
            f"model n={len(got)}, dataset n={len(feature_cols)}. "
            "Переобучите модель или используйте тот же full_dataset."
        )


def _train_xgb_on_sample(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    xgb_config: str,
) -> object:
    import xgboost as xgb

    params = _build_xgb_train_params(xgb_config)
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train, feature_names=feature_cols)
    rounds = max(1, int(XGB_PARAMS.get("n_estimators", 600)))
    train_kw: dict = {"params": params, "dtrain": dtrain, "num_boost_round": rounds, "verbose_eval": False}
    n_val = int(x_val.shape[0])
    if n_val > 0:
        dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_cols)
        train_kw["evals"] = [(dval, "eval")]
        es = int(XGB_EARLY_STOPPING_ROUNDS)
        if es > 0:
            train_kw["early_stopping_rounds"] = es
    logger.info(
        "Refit на сэмпле: num_boost_round=%d, xgb-config=%s, val_rows=%d, early_stopping=%s",
        rounds,
        xgb_config,
        n_val,
        int(XGB_EARLY_STOPPING_ROUNDS) if n_val > 0 else 0,
    )
    return xgb.train(**train_kw)


def _predict_proba(booster, x: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    import xgboost as xgb

    d = xgb.DMatrix(x, feature_names=feature_cols)
    return xgb_predict_with_best_iteration(booster, d)


def _compute_permutation_importance(
    booster,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    repeats: int,
    random_seed: int,
) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(random_seed)
    baseline_pred = _predict_proba(booster, x_val, feature_cols)
    baseline = float(average_precision_score(y_val, baseline_pred, pos_label=1))
    logger.info("XGBoost baseline PR-AUC (val sample, pos_label=1): %.6f", baseline)

    importances: dict[str, float] = {}
    for j, feat in enumerate(tqdm(feature_cols, desc="XGB permutation", unit="feature")):
        drops: list[float] = []
        original = x_val[:, j].copy()
        for _ in range(repeats):
            perm_idx = rng.permutation(x_val.shape[0])
            x_val[:, j] = original[perm_idx]
            pred = _predict_proba(booster, x_val, feature_cols)
            score = float(average_precision_score(y_val, pred, pos_label=1))
            drops.append(baseline - score)
        x_val[:, j] = original
        importances[feat] = float(np.mean(drops))
    return baseline, importances


def _compute_full_val_baseline(
    booster,
    path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    batch_size: int,
) -> tuple[float, int]:
    """PR-AUC на всем val (dttm >= cutoff) без permutation."""
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    pf = pq.ParquetFile(path)
    read_cols = _batch_read_columns(path, feature_cols, dttm_col)
    for rb in tqdm(
        pf.iter_batches(columns=read_cols, batch_size=batch_size),
        desc="Full-val baseline",
        unit="batch",
    ):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        val_mask = ~(dttm < cutoff_day)
        if not val_mask.any():
            continue
        xva, yva, _ = prepare_batch(dfb.loc[val_mask], feature_cols)
        pred = _predict_proba(booster, xva, feature_cols)
        y_parts.append(yva.astype(np.int32, copy=False))
        p_parts.append(pred.astype(np.float32, copy=False))
    if not y_parts:
        return 0.0, 0
    y_all = np.concatenate(y_parts)
    p_all = np.concatenate(p_parts)
    score = float(average_precision_score(y_all, p_all, pos_label=1))
    return score, int(y_all.shape[0])


def _native_importance_table(booster, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for feat in feature_cols:
        row: dict[str, str | float] = {"feature": feat}
        for itype in _NATIVE_IMPORTANCE_TYPES:
            try:
                sc = booster.get_score(importance_type=itype)
            except Exception:
                sc = {}
            if isinstance(sc, dict):
                row[itype] = float(sc.get(feat, 0.0))
            else:
                row[itype] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


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
            xtr, ytr, wtr = prepare_batch(dfb.loc[train_mask], feature_cols)
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
            xva, yva, _ = prepare_batch(dfb.loc[val_mask], feature_cols)
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


def _permutation_to_df(baseline: float, importances: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "feature": list(importances.keys()),
            "importance_drop_pr_auc": list(importances.values()),
        }
    )
    df["model"] = "xgboost"
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
    plt.title("Most important features (XGB)")
    plt.tight_layout()
    plt.savefig(top_path, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.barh(low_df["feature"], low_df["mean_importance_drop_pr_auc"], color="#ff7f0e")
    plt.xlabel("Mean drop in PR-AUC after permutation")
    plt.ylabel("Feature")
    plt.title("Least important features (XGB)")
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
    model_source: str,
    model_path: str | None,
    xgb_config: str,
    summary: pd.DataFrame,
    top_png: Path,
    low_png: Path,
    native_csv: Path,
    ranking_txt: Path,
) -> None:
    top_k = min(15, len(summary))
    top_df = summary.head(top_k)
    low_df = summary.tail(top_k).iloc[::-1].reset_index(drop=True)

    lines: list[str] = []
    lines.append("# Feature importance (XGBoost)")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Model: **XGBoost**")
    lines.append(f"- Model source: `{model_source}`")
    if model_path:
        lines.append(f"- Saved model path: `{model_path}`")
    if model_source == "refit_on_sample":
        lines.append(f"- `--xgb-config` for refit: `{xgb_config}`")
    lines.append(f"- Train rows used: `{train_rows}`")
    lines.append(f"- Validation rows used: `{val_rows}`")
    lines.append(f"- Permutation repeats: `{repeats}`")
    lines.append(f"- Memory budget target: `{max_memory_gb:.1f} GB`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- Permutation (CSV): `feature_importance_per_model.csv`, `feature_importance_summary.csv`")
    lines.append(f"- Native booster scores (CSV): `{native_csv.name}`")
    lines.append(f"- **Full ranking (all features, TXT): `{ranking_txt.name}`**")
    lines.append(f"- Charts: `{top_png.name}`, `{low_png.name}`")
    lines.append("")
    lines.append("## How to read permutation importance")
    lines.append("- Value = `baseline_pr_auc - pr_auc_after_shuffle(feature)` (PR-AUC, `pos_label=1`).")
    lines.append("- Larger positive ⇒ more useful on this val sample.")
    lines.append("- Near zero ⇒ little contribution.")
    lines.append("- Negative ⇒ noise / interactions; review.")
    lines.append("")
    lines.append("## Charts")
    lines.append(f"- Most important: `{top_png.name}`")
    lines.append(f"- Least important: `{low_png.name}`")
    lines.append("")
    lines.append("## Top features (permutation, preview)")
    for i, row in top_df.iterrows():
        lines.append(f"{i + 1}. `{row['feature']}`: `{row['mean_importance_drop_pr_auc']:.8f}`")
    lines.append("")
    lines.append("## Bottom features (permutation, preview)")
    for i, row in low_df.iterrows():
        lines.append(f"{i + 1}. `{row['feature']}`: `{row['mean_importance_drop_pr_auc']:.8f}`")
    lines.append("")
    lines.append("Полный список признаков — в текстовом файле ранжирования.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_full_ranking_txt(
    out_path: Path,
    perm_df: pd.DataFrame,
    native_df: pd.DataFrame,
    baseline: float,
    train_rows: int,
    val_rows: int,
    repeats: int,
    model_source: str,
) -> None:
    """Полное ранжирование: сначала permutation (все фичи), затем по native gain."""
    lines: list[str] = []
    lines.append("XGBoost — полное ранжирование признаков")
    lines.append(f"Источник модели: {model_source}")
    lines.append(f"Baseline PR-AUC (permutation, val sample, pos_label=1): {baseline:.8f}")
    lines.append(f"Строки: train={train_rows}, val={val_rows}; повторы permutation={repeats}")
    lines.append("")
    lines.append("=== Ранжирование по permutation importance (убывание) ===")
    lines.append("rank\tfeature\timportance_drop_pr_auc")
    for rank, row in enumerate(perm_df.itertuples(index=False), start=1):
        lines.append(f"{rank}\t{row.feature}\t{row.importance_drop_pr_auc:.8f}")
    lines.append("")
    nat_sorted = native_df.sort_values("gain", ascending=False, ignore_index=True)
    lines.append("=== Ранжирование по native gain (убывание) ===")
    lines.append("rank\tfeature\tgain\tweight\tcover\ttotal_gain\ttotal_cover")
    for rank, row in enumerate(nat_sorted.itertuples(index=False), start=1):
        lines.append(
            f"{rank}\t{row.feature}\t{row.gain:.6f}\t{row.weight:.6f}\t{row.cover:.6f}\t"
            f"{row.total_gain:.6f}\t{row.total_cover:.6f}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Важность признаков XGBoost (permutation + native).")
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
    parser.add_argument(
        "--refit-on-sample",
        action="store_true",
        help="Не загружать MODEL_XGB_PATH, а обучить XGB на RAM-сэмпле (params как в training.main).",
    )
    parser.add_argument(
        "--xgb-config",
        choices=("best", "default"),
        default="default",
        help="При --refit-on-sample: default или best из grid_search (как training.main).",
    )
    parser.add_argument(
        "--full-val-baseline",
        action="store_true",
        help="Дополнительно посчитать baseline PR-AUC на всем val (без permutation).",
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

    feature_cols, dttm_col = detect_columns(TRAIN_DATASET_PATH)
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

    if args.refit_on_sample:
        booster = _train_xgb_on_sample(
            x_train, y_train, w_train, x_val, y_val, feature_cols, args.xgb_config
        )
        model_source = "refit_on_sample"
        model_path_str = None
        logger.info("Модель: обучена на сэмпле (--refit-on-sample), xgb-config=%s", args.xgb_config)
    else:
        if not MODEL_XGB_PATH.exists():
            raise FileNotFoundError(
                f"Нет файла модели {MODEL_XGB_PATH}. Сначала обучите: PYTHONPATH=. python -m training.main "
                "или запустите research с --refit-on-sample."
            )
        booster = _load_xgb_booster(MODEL_XGB_PATH)
        _verify_booster_features(booster, feature_cols)
        model_source = "saved_model"
        model_path_str = str(MODEL_XGB_PATH.resolve())
        logger.info("Модель: загружена из %s", model_path_str)

    native_df = _native_importance_table(booster, feature_cols)

    baseline, importances = _compute_permutation_importance(
        booster=booster,
        x_val=x_val.copy(),
        y_val=y_val,
        feature_cols=feature_cols,
        repeats=args.repeats,
        random_seed=args.seed,
    )
    full_val_baseline = None
    full_val_rows = 0
    if args.full_val_baseline:
        full_val_baseline, full_val_rows = _compute_full_val_baseline(
            booster=booster,
            path=TRAIN_DATASET_PATH,
            feature_cols=feature_cols,
            dttm_col=dttm_col,
            cutoff_day=cutoff_day,
            batch_size=args.batch_size,
        )
        logger.info(
            "XGBoost baseline PR-AUC (full val, pos_label=1): %.6f (rows=%d)",
            full_val_baseline,
            full_val_rows,
        )
    perm_full = _permutation_to_df(baseline, importances)

    summary = (
        perm_full.groupby("feature", as_index=False)["importance_drop_pr_auc"]
        .mean()
        .rename(columns={"importance_drop_pr_auc": "mean_importance_drop_pr_auc"})
        .sort_values("mean_importance_drop_pr_auc", ascending=False, ignore_index=True)
    )

    out_dir = OUTPUT_DIR / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_path = out_dir / "feature_importance_per_model.csv"
    summary_path = out_dir / "feature_importance_summary.csv"
    native_path = out_dir / "xgb_native_importance.csv"
    json_path = out_dir / "xgb_feature_importance.json"
    txt_path = out_dir / "feature_importance_report.txt"
    ranking_path = out_dir / "feature_ranking_all.txt"
    md_path = out_dir / "feature_importance_report.md"

    perm_full.to_csv(per_model_path, index=False)
    summary.to_csv(summary_path, index=False)
    native_df.to_csv(native_path, index=False)

    perm_ranked = perm_full.sort_values("importance_drop_pr_auc", ascending=False, ignore_index=True)
    json_payload = {
        "model_source": model_source,
        "model_path": model_path_str,
        "baseline_pr_auc_permutation": baseline,
        "baseline_pr_auc_full_val": full_val_baseline,
        "full_val_rows": full_val_rows,
        "permutation_by_feature": perm_ranked[["feature", "importance_drop_pr_auc"]].to_dict(orient="records"),
        "native_by_feature": native_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top_k = min(15, len(summary))
    top_df = summary.head(top_k)
    low_df = summary.tail(top_k).iloc[::-1].reset_index(drop=True)

    lines_short = [
        "Feature importance report — XGBoost (permutation on time-based val sample)",
        f"Model: {model_source}" + (f" ({model_path_str})" if model_path_str else ""),
        f"Rows used: train={len(x_train)}, val={len(x_val)}",
        f"Permutation repeats: {args.repeats}",
        (
            f"Baseline full val (без permutation): {full_val_baseline:.8f} "
            f"(rows={full_val_rows})"
            if full_val_baseline is not None
            else "Baseline full val (без permutation): not computed"
        ),
        "",
        "TOP most important features (permutation):",
    ]
    for i, row in top_df.iterrows():
        lines_short.append(f"{i + 1:2d}. {row['feature']}: {row['mean_importance_drop_pr_auc']:.8f}")
    lines_short.extend(["", "TOP least important features (permutation):"])
    for i, row in low_df.iterrows():
        lines_short.append(f"{i + 1:2d}. {row['feature']}: {row['mean_importance_drop_pr_auc']:.8f}")
    lines_short.extend(
        [
            "",
            f"Полное ранжирование всех признаков: {ranking_path.name}",
            f"Нативные важности (CSV): {native_path.name}",
            f"JSON (permutation + native): {json_path.name}",
        ]
    )
    txt_path.write_text("\n".join(lines_short), encoding="utf-8")

    _write_full_ranking_txt(
        ranking_path,
        perm_ranked,
        native_df,
        baseline,
        len(x_train),
        len(x_val),
        args.repeats,
        model_source,
    )

    top_png, low_png = _plot_summary(summary, out_dir=out_dir, top_k=top_k)
    _write_markdown_report(
        out_path=md_path,
        train_rows=len(x_train),
        val_rows=len(x_val),
        repeats=args.repeats,
        max_memory_gb=args.max_memory_gb,
        model_source=model_source,
        model_path=model_path_str,
        xgb_config=args.xgb_config,
        summary=summary,
        top_png=top_png,
        low_png=low_png,
        native_csv=native_path,
        ranking_txt=ranking_path,
    )

    logger.info("Сохранено: %s", per_model_path)
    logger.info("Сохранено: %s", summary_path)
    logger.info("Сохранено: %s", native_path)
    logger.info("Сохранено: %s", json_path)
    logger.info("Сохранено: %s", txt_path)
    logger.info("Сохранено: %s", ranking_path)
    logger.info("Сохранено: %s", md_path)
    logger.info("Сохранено: %s", top_png)
    logger.info("Сохранено: %s", low_png)
    logger.info("Самая важная фича (permutation): %s", top_df.iloc[0]["feature"])
    logger.info("Самая неважная фича (permutation): %s", low_df.iloc[0]["feature"])


if __name__ == "__main__":
    main()
