"""
Полный прямой forward pass: на каждом шаге перебираются все ещё не вошедшие фичи,
обучается XGBoost на текущем множестве ∪ {кандидат}, на валидации по времени считается PR-AUC;
добавляется фича с максимальной метрикой. История пишется в JSONL + CSV.

Датасет: output/full_dataset.parquet (shared.config.TRAIN_DATASET_PATH).
Признаки: resolve_model_input_columns — тот же порядок, что в training/main.py.

Запуск из корня репозитория:
  PYTHONPATH=. python -m xgb_forward_selection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.main import (
    _apply_memory_budget,
    _estimate_sample_memory_bytes,
    _find_time_cutoff,
    _format_gb,
    _sample_from_stream,
)
from shared.config import OUTPUT_DIR, TRAIN_DATASET_PATH, resolve_model_input_columns
from training.config import VAL_RATIO, XGB_PARAMS
from training.main import _build_xgb_train_params

logger = logging.getLogger(__name__)


def _detect_columns(path: Path) -> tuple[list[str], str]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    feature_cols = resolve_model_input_columns(names)
    if "event_dttm" not in names:
        raise ValueError("В датасете нет колонки event_dttm")
    if "target" not in names or "sample_weight" not in names:
        raise ValueError("В датасете нет target/sample_weight")
    return feature_cols, "event_dttm"


def _constant_baseline_pr_auc(y_val: np.ndarray) -> float:
    p = float(np.mean(y_val)) if len(y_val) else 0.0
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    pred = np.full(shape=y_val.shape, fill_value=p, dtype=np.float64)
    return float(average_precision_score(y_val, pred))


def _fit_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    col_indices: list[int],
    feature_cols: list[str],
    params: dict[str, float | int | str],
    num_boost_round: int,
):
    import xgboost as xgb

    xt = np.ascontiguousarray(x_train[:, col_indices], dtype=np.float32)
    names = [feature_cols[i] for i in col_indices]
    dtrain = xgb.DMatrix(xt, label=y_train, weight=w_train, feature_names=names)
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)


def _pr_auc_val(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    col_indices: list[int],
    feature_cols: list[str],
) -> float:
    import xgboost as xgb

    xv = np.ascontiguousarray(x_val[:, col_indices], dtype=np.float32)
    names = [feature_cols[i] for i in col_indices]
    d = xgb.DMatrix(xv, feature_names=names)
    pred = np.asarray(model.predict(d), dtype=np.float64)
    return float(average_precision_score(y_val, pred))


def _run_forward_selection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    params: dict[str, float | int | str],
    num_boost_round: int,
    name_to_index: dict[str, int],
    feature_subset: list[str] | None,
) -> list[dict]:
    if feature_subset is not None:
        missing = [f for f in feature_subset if f not in name_to_index]
        if missing:
            raise ValueError(f"Неизвестные фичи в --features: {missing}")
        pool = list(feature_subset)
    else:
        pool = list(feature_cols)

    history: list[dict] = []
    baseline = _constant_baseline_pr_auc(y_val)
    history.append(
        {
            "step": 0,
            "added_feature": None,
            "pr_auc_val": baseline,
            "delta_pr_auc": None,
            "selected_features": [],
            "note": "baseline: constant prediction = mean(target) on val",
        }
    )

    selected: list[str] = []
    remaining = list(pool)
    feat_to_idx = name_to_index
    total_inner = len(remaining) * (len(remaining) + 1) // 2
    pbar = tqdm(total=total_inner, desc="forward selection (train+eval per candidate)", unit="fit")

    prev_metric = baseline
    step = 0
    while remaining:
        step += 1
        best_feat: str | None = None
        best_metric = -1.0
        base_indices = [feat_to_idx[f] for f in selected]

        for cand in remaining:
            cand_idx = feat_to_idx[cand]
            cols = base_indices + [cand_idx]
            model = _fit_xgb(
                x_train, y_train, w_train, cols, feature_cols, params, num_boost_round
            )
            m = _pr_auc_val(model, x_val, y_val, cols, feature_cols)
            pbar.update(1)
            if m > best_metric or (np.isclose(m, best_metric) and (best_feat is None or cand < best_feat)):
                best_metric = m
                best_feat = cand

        assert best_feat is not None
        selected.append(best_feat)
        remaining = [f for f in remaining if f != best_feat]
        delta = best_metric - prev_metric
        prev_metric = best_metric
        history.append(
            {
                "step": step,
                "added_feature": best_feat,
                "pr_auc_val": best_metric,
                "delta_pr_auc": delta,
                "selected_features": list(selected),
            }
        )
        logger.info(
            "Шаг %d: +%s → PR-AUC=%.6f (Δ=%+.6f), всего фич: %d",
            step,
            best_feat,
            best_metric,
            delta,
            len(selected),
        )

    pbar.close()
    return history


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="XGBoost forward feature selection по full_dataset.parquet")
    parser.add_argument(
        "--xgb-config",
        choices=("best", "default"),
        default="best",
        help="Гиперпараметры как в training/main.py",
    )
    parser.add_argument("--max-train-rows", type=int, default=400_000)
    parser.add_argument("--max-val-rows", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory-gb", type=float, default=20.0)
    parser.add_argument("--memory-safety-factor", type=float, default=0.55)
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=None,
        help="Число деревьев XGB (по умолчанию XGB_PARAMS['n_estimators'])",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка для history.jsonl и history.csv (по умолчанию output/research)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Опционально: через запятую подмножество фич (порядок перебора = как в списке)",
    )
    args = parser.parse_args()

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. Соберите датасет: ./dataset_cpp/build/build_dataset ."
        )

    if args.max_memory_gb <= 1.0:
        raise ValueError("--max-memory-gb должен быть > 1")
    if not (0.2 <= args.memory_safety_factor <= 0.9):
        raise ValueError("--memory-safety-factor должен быть в [0.2, 0.9]")

    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    name_to_index = {n: i for i, n in enumerate(feature_cols)}
    feature_subset = None
    if args.features:
        feature_subset = [s.strip() for s in args.features.split(",") if s.strip()]

    n_feat_eff = len(feature_subset) if feature_subset is not None else len(feature_cols)
    safe_train, safe_val = _apply_memory_budget(
        requested_train_rows=args.max_train_rows,
        requested_val_rows=args.max_val_rows,
        n_features=len(feature_cols),
        max_memory_gb=args.max_memory_gb,
        safety_factor=args.memory_safety_factor,
    )
    req_bytes = (
        _estimate_sample_memory_bytes(safe_train, len(feature_cols), include_weight=True)
        + _estimate_sample_memory_bytes(safe_val, len(feature_cols), include_weight=False)
    )
    logger.info("Оценка RAM под матрицы: %s", _format_gb(req_bytes))

    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)
    x_train, y_train, w_train, x_val, y_val = _sample_from_stream(
        TRAIN_DATASET_PATH,
        feature_cols=feature_cols,
        dttm_col=dttm_col,
        cutoff_day=cutoff_day,
        max_train_rows=safe_train,
        max_val_rows=safe_val,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    params = _build_xgb_train_params(args.xgb_config)
    num_rounds = args.num_boost_round
    if num_rounds is None:
        num_rounds = int(XGB_PARAMS.get("n_estimators", 600))
    num_rounds = max(50, num_rounds)

    out_dir = args.output_dir or (OUTPUT_DIR / "research")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "xgb_forward_selection_history.jsonl"
    csv_path = out_dir / "xgb_forward_selection_history.csv"

    meta = {
        "dataset": str(TRAIN_DATASET_PATH),
        "xgb_config": args.xgb_config,
        "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in params.items()},
        "num_boost_round": num_rounds,
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "n_features_full": len(feature_cols),
        "feature_subset": feature_subset,
        "val_ratio": VAL_RATIO,
        "seed": args.seed,
    }
    jsonl_path.write_text("", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False, default=str) + "\n")

    logger.info(
        "Старт forward selection: %d фич в пуле, %d раундов XGB, train=%d val=%d",
        n_feat_eff,
        num_rounds,
        x_train.shape[0],
        x_val.shape[0],
    )

    history = _run_forward_selection(
        x_train,
        y_train,
        w_train,
        x_val,
        y_val,
        feature_cols,
        params,
        num_rounds,
        name_to_index,
        feature_subset,
    )

    with jsonl_path.open("a", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows_out = []
    for row in history:
        rows_out.append(
            {
                "step": row["step"],
                "added_feature": row.get("added_feature"),
                "pr_auc_val": row.get("pr_auc_val"),
                "delta_pr_auc": row.get("delta_pr_auc"),
                "selected_features": json.dumps(row.get("selected_features", []), ensure_ascii=False),
            }
        )
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)

    logger.info("История: %s", jsonl_path)
    logger.info("История: %s", csv_path)


if __name__ == "__main__":
    main()
