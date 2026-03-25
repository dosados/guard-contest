"""
Грид-сёрч гиперпараметров XGBoost в том же режиме, что и training/main.py:
DataIter + ExtMemQuantileDMatrix / QuantileDMatrix, time split по event_dttm,
train с весами, val без весов, PR-AUC (pos_label=1), num_boost_round и early stopping из training.config.

Метрика каждой конфигурации дописывается в JSONL (по умолчанию training/grid_search/xgb_grid_trials.jsonl)
сразу после trial — удобно для анализа и при обрыве длинного прогона.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import TRAIN_DATASET_PATH
from training.config import (
    VAL_RATIO,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_EXTERNAL_PARQUET_BATCH_ROWS,
    XGB_PARAMS,
)
from training.parquet_io import detect_columns, find_time_cutoff
from training.xgb_iterative_fit import fit_xgb_parquet_iterative

logger = logging.getLogger(__name__)

# Артефакты грид-сёрча (CSV, лучшие гиперпараметры, веса модели) — рядом с этим модулем, не в output/.
GRID_SEARCH_DIR = Path(__file__).resolve().parent / "grid_search"
# Дисковый кэш ExtMemQuantileDMatrix по trial — не пересекается с output/xgb_extmem_*
GRID_EXTMEM_ROOT = GRID_SEARCH_DIR / "xgb_grid_extmem"

# Сетка: ключи — имена в sklearn-стиле (как в XGB_PARAMS), значения — списки кандидатов.
# Полный перебор ниже = 243 полных прохода по датасету; для прогона используйте --max-trials или свою сетку в коде.
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.05],
    "max_depth": [6],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8],
    "min_child_weight": [3, 5, 7],
    "gamma": [0.0, 1.0, 5.0],
    "reg_alpha": [0.1],
    "reg_lambda": [1.0, 5.0],
}


def _xgb_params_from_grid(
    combo: dict[str, Any],
    *,
    tree_method: str | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    required = set(DEFAULT_PARAM_GRID.keys())
    got = set(combo.keys())
    if got != required:
        missing = sorted(required - got)
        extra = sorted(got - required)
        raise ValueError(
            "Combo гиперпараметров не совпадает с DEFAULT_PARAM_GRID. "
            f"missing={missing}, extra={extra}"
        )

    tm = tree_method if tree_method is not None else XGB_PARAMS.get("tree_method", "hist")
    rs = random_state if random_state is not None else XGB_PARAMS.get("random_state", 42)
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": float(combo["learning_rate"]),
        "max_depth": int(combo["max_depth"]),
        "subsample": float(combo["subsample"]),
        "colsample_bytree": float(combo["colsample_bytree"]),
        "min_child_weight": float(combo["min_child_weight"]),
        "gamma": float(combo["gamma"]),
        "alpha": float(combo["reg_alpha"]),
        "lambda": float(combo["reg_lambda"]),
        "tree_method": tm,
        "seed": int(rs),
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
    batch_size: int | None = None,
    max_trials: int | None = None,
    tree_method: str | None = None,
    random_state: int | None = None,
    trials_log_path: Path | None = None,
    extmem_root: Path | None = None,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], Any]:
    """
    Возвращает (лучший combo по PR-AUC, лучший score, все строки результатов, бустер лучшего прогона).
    Каждая строка results: combo + pr_auc.

    Если задан trials_log_path, в начале запуска файл очищается; после каждого trial в него
    дописывается одна строка JSON (JSONL) с индексом, гиперпараметрами, полными params XGBoost и pr_auc_val.
    """
    import xgboost as xgb  # noqa: F401 — проверка, что пакет установлен

    br = int(batch_size if batch_size is not None else XGB_EXTERNAL_PARQUET_BATCH_ROWS)
    rounds = max(1, int(XGB_PARAMS.get("n_estimators", 600)))
    es = int(XGB_EARLY_STOPPING_ROUNDS)
    cache_root = extmem_root if extmem_root is not None else GRID_EXTMEM_ROOT

    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")

    feature_cols, dttm_col = detect_columns(path)
    cutoff_day = find_time_cutoff(path, val_ratio, batch_size=br)

    combos = list(_iter_grid(grid))
    if not combos:
        raise ValueError("Пустая сетка гиперпараметров.")
    if max_trials is not None:
        combos = combos[: max_trials]

    results: list[dict[str, Any]] = []
    best_combo: dict[str, Any] | None = None
    best_pr = -1.0
    best_booster = None

    trials_log_file = None
    if trials_log_path is not None:
        trials_log_path.parent.mkdir(parents=True, exist_ok=True)
        trials_log_path.unlink(missing_ok=True)
        trials_log_file = trials_log_path.open("w", encoding="utf-8")

    use_extmem = hasattr(xgb, "ExtMemQuantileDMatrix")

    try:
        for i, combo in enumerate(tqdm(combos, desc="Grid trials", unit="trial"), start=1):
            params = _xgb_params_from_grid(combo, tree_method=tree_method, random_state=random_state)
            logger.info("Trial %d/%d params (xgb): %s", i, len(combos), params)

            trial_dir = cache_root / f"trial_{i}"
            train_cache = trial_dir / "train"
            val_cache = trial_dir / "val"

            try:
                booster, pr = fit_xgb_parquet_iterative(
                    path,
                    feature_cols,
                    dttm_col,
                    cutoff_day,
                    params,
                    num_boost_round=rounds,
                    batch_rows=br,
                    train_cache_dir=train_cache,
                    val_cache_dir=val_cache,
                    early_stopping_rounds=es,
                    verbose_eval=False,
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
                    if best_booster is not None:
                        del best_booster
                        gc.collect()
                    best_booster = booster
                else:
                    del booster
                    gc.collect()
            finally:
                if use_extmem:
                    shutil.rmtree(trial_dir, ignore_errors=True)
    finally:
        if trials_log_file is not None:
            trials_log_file.close()

    assert best_combo is not None and best_booster is not None
    return best_combo, best_pr, results, best_booster


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
    p = argparse.ArgumentParser(
        description="Грид-сёрч XGBoost (как training/main: ExtMem/Quantile + DataIter, time split, PR-AUC).",
    )
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
        help="JSONL: одна строка JSON на trial сразу после метрики (для анализа и при сбое прогона).",
    )
    p.add_argument(
        "--no-trials-log",
        action="store_true",
        help="Не писать построчный лог trials.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Батч pyarrow (по умолчанию training.config.XGB_EXTERNAL_PARQUET_BATCH_ROWS={XGB_EXTERNAL_PARQUET_BATCH_ROWS}).",
    )
    p.add_argument("--max-trials", type=int, default=None, help="Ограничить число первых комбинаций сетки.")
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = p.parse_args()

    trials_log = None if args.no_trials_log else args.trials_log
    best_combo, best_pr, rows, best_booster = run_grid_search(
        args.dataset,
        param_grid=DEFAULT_PARAM_GRID,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
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


if __name__ == "__main__":
    main()
