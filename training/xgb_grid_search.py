"""
Грид-сёрч гиперпараметров XGBoost в том же режиме, что и training/main.py:
ExtMemQuantileDMatrix / QuantileDMatrix + DataIter, time split по event_dttm,
eval_metric aucpr, early stopping, PR-AUC на val со sample_weight (как в DMatrix).

Метрика каждой конфигурации дописывается в JSONL (по умолчанию training/grid_search/xgb_grid_trials.jsonl)
сразу после trial.

Сетка перебора задаётся только в training.config.XGB_PARAM_GRID.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import OUTPUT_DIR, TRAIN_DATASET_PATH
from training.config import (
    VAL_RATIO,
    XGB_EXTERNAL_PARQUET_BATCH_ROWS,
    XGB_PARAM_GRID,
    XGB_PARAMS,
)
from training.main import _detect_columns, _find_time_cutoff, train_xgb_streaming_prauc

logger = logging.getLogger(__name__)

GRID_SEARCH_DIR = Path(__file__).resolve().parent / "grid_search"


def _xgb_params_from_combo(combo: dict[str, Any]) -> dict[str, float | int | str]:
    """
    Те же имена полей, что в training/main._build_xgb_train_params (XGBoost API).
    Ключи из combo дополняются дефолтами из XGB_PARAMS, чтобы неполная сетка оставалась валидной.
    """
    base = {
        "learning_rate": float(XGB_PARAMS["learning_rate"]),
        "max_depth": int(XGB_PARAMS["max_depth"]),
        "subsample": float(XGB_PARAMS["subsample"]),
        "colsample_bytree": float(XGB_PARAMS["colsample_bytree"]),
        "min_child_weight": float(XGB_PARAMS.get("min_child_weight", 1.0)),
        "gamma": float(XGB_PARAMS.get("gamma", 0.0)),
        "reg_alpha": float(XGB_PARAMS.get("reg_alpha", 0.0)),
        "reg_lambda": float(XGB_PARAMS.get("reg_lambda", 1.0)),
    }
    merged = {**base, **combo}
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": float(merged["learning_rate"]),
        "max_depth": int(merged["max_depth"]),
        "subsample": float(merged["subsample"]),
        "colsample_bytree": float(merged["colsample_bytree"]),
        "min_child_weight": float(merged["min_child_weight"]),
        "gamma": float(merged["gamma"]),
        "alpha": float(merged["reg_alpha"]),
        "lambda": float(merged["reg_lambda"]),
        "tree_method": str(XGB_PARAMS.get("tree_method", "hist")),
        "seed": int(XGB_PARAMS.get("random_state", 42)),
    }


def _iter_grid(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def run_grid_search(
    path: Path,
    param_grid: dict[str, list[Any]] | None = None,
    *,
    val_ratio: float = VAL_RATIO,
    batch_rows: int | None = None,
    num_boost_round: int | None = None,
    max_trials: int | None = None,
    trials_log_path: Path | None = None,
    ext_train_cache: Path | None = None,
    ext_val_cache: Path | None = None,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], Any, Callable[[], None]]:
    """
    Возвращает (лучший combo, лучший PR-AUC, все строки результатов, бустер лучшего прогона, cleanup для него).
    После save_model(best) вызовите cleanup(), иначе останутся дисковый кэш и ссылки на DMatrix.
    """
    grid = param_grid if param_grid is not None else XGB_PARAM_GRID
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")

    br = int(batch_rows if batch_rows is not None else XGB_EXTERNAL_PARQUET_BATCH_ROWS)
    rounds = max(1, int(num_boost_round if num_boost_round is not None else XGB_PARAMS.get("n_estimators", 600)))

    feature_cols, dttm_col = _detect_columns(path)
    cutoff_day = _find_time_cutoff(path, val_ratio, batch_size=br)

    combos = list(_iter_grid(grid))
    if not combos:
        raise ValueError("Пустая сетка гиперпараметров.")
    if max_trials is not None:
        combos = combos[: max_trials]

    train_cache = ext_train_cache if ext_train_cache is not None else OUTPUT_DIR / "xgb_grid_extmem_train"
    val_cache = ext_val_cache if ext_val_cache is not None else OUTPUT_DIR / "xgb_grid_extmem_val"

    results: list[dict[str, Any]] = []
    best_combo: dict[str, Any] | None = None
    best_pr = -1.0
    best_booster: Any = None
    best_cleanup: Callable[[], None] | None = None

    trials_log_file = None
    if trials_log_path is not None:
        trials_log_path.parent.mkdir(parents=True, exist_ok=True)
        trials_log_path.unlink(missing_ok=True)
        trials_log_file = trials_log_path.open("w", encoding="utf-8")

    try:
        for i, combo in enumerate(tqdm(combos, desc="XGBoost grid", unit="trial"), start=1):
            params = _xgb_params_from_combo(combo)
            logger.info("Trial %d/%d params (xgb): %s", i, len(combos), params)
            pr, booster, cleanup = train_xgb_streaming_prauc(
                path,
                feature_cols,
                dttm_col,
                cutoff_day,
                params,
                ext_train_cache=train_cache,
                ext_val_cache=val_cache,
                num_boost_round=rounds,
                batch_rows=br,
            )
            row = {**combo, "pr_auc": pr}
            results.append(row)
            logger.info("Trial %d/%d PR-AUC (val): %.6f", i, len(combos), pr)
            if trials_log_file is not None:
                record = {
                    "trial_index": i,
                    "trial_total": len(combos),
                    "hyperparameters": combo,
                    "xgb_train_params": params,
                    "pr_auc_val": pr,
                }
                trials_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                trials_log_file.flush()
            if pr > best_pr:
                best_pr = pr
                best_combo = combo
                if best_cleanup is not None:
                    best_cleanup()
                best_booster = booster
                best_cleanup = cleanup
            else:
                cleanup()

    finally:
        if trials_log_file is not None:
            trials_log_file.close()

    assert best_combo is not None and best_booster is not None and best_cleanup is not None
    return best_combo, best_pr, results, best_booster, best_cleanup


def _write_results_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_best_hyperparams(out_path: Path, combo: dict[str, Any], pr_auc: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hyperparameters": combo, "pr_auc_val": pr_auc}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(description="Грид-сёрч XGBoost (как training/main: QuantileDMatrix, PR-AUC).")
    p.add_argument(
        "--dataset",
        type=Path,
        default=TRAIN_DATASET_PATH,
        help="Путь к train parquet (по умолчанию shared.config.TRAIN_DATASET_PATH).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=GRID_SEARCH_DIR / "xgb_grid_search_results.csv",
        help="Таблица всех комбинаций и PR-AUC (по умолчанию training/grid_search/).",
    )
    p.add_argument(
        "--out-best-params",
        type=Path,
        default=GRID_SEARCH_DIR / "xgb_best_params.json",
        help="JSON с лучшими гиперпараметрами и PR-AUC (по умолчанию training/grid_search/).",
    )
    p.add_argument(
        "--save-best",
        type=Path,
        default=GRID_SEARCH_DIR / "model_xgb_best.json",
        help="Лучший бустер XGBoost (по умолчанию training/grid_search/model_xgb_best.json).",
    )
    p.add_argument(
        "--no-save-model",
        action="store_true",
        help="Не сохранять файл модели (--save-best), только CSV и JSON с гиперпараметрами.",
    )
    p.add_argument(
        "--trials-log",
        type=Path,
        default=GRID_SEARCH_DIR / "xgb_grid_trials.jsonl",
        help="JSONL: одна строка JSON на trial сразу после метрики.",
    )
    p.add_argument(
        "--no-trials-log",
        action="store_true",
        help="Не писать построчный лог trials.",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=None,
        help=f"Батч parquet (по умолчанию training.config.XGB_EXTERNAL_PARQUET_BATCH_ROWS={XGB_EXTERNAL_PARQUET_BATCH_ROWS}).",
    )
    p.add_argument(
        "--num-boost-round",
        type=int,
        default=None,
        help="Число деревьев / раундов (по умолчанию XGB_PARAMS n_estimators).",
    )
    p.add_argument("--max-trials", type=int, default=None, help="Ограничить число первых комбинаций сетки.")
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = p.parse_args()

    trials_log = None if args.no_trials_log else args.trials_log
    best_combo, best_pr, rows, best_booster, best_cleanup = run_grid_search(
        args.dataset,
        param_grid=XGB_PARAM_GRID,
        val_ratio=args.val_ratio,
        batch_rows=args.batch_rows,
        num_boost_round=args.num_boost_round,
        max_trials=args.max_trials,
        trials_log_path=trials_log,
    )
    _write_results_csv(rows, args.out_csv)
    _write_best_hyperparams(args.out_best_params, best_combo, best_pr)
    logger.info("Лучшая комбинация: %s PR-AUC=%.6f", best_combo, best_pr)
    logger.info("Результаты: %s", args.out_csv)
    logger.info("Лучшие гиперпараметры: %s", args.out_best_params)
    if trials_log is not None:
        logger.info("Лог по trial (JSONL): %s", trials_log)

    if not args.no_save_model:
        args.save_best.parent.mkdir(parents=True, exist_ok=True)
        best_booster.save_model(str(args.save_best))
        logger.info("Лучшая модель сохранена: %s", args.save_best)
    best_cleanup()


if __name__ == "__main__":
    main()
