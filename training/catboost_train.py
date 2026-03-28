"""
Обучение CatBoost на full_dataset.parquet.
Сплит train/val по времени как в training/main.py.
Один проход по parquet батчами → временные TSV (фичи + target + sample_weight) на диске;
затем Pool(data=path) и fit — без удержания всего датасета в RAM, логически то же, что
один Pool на всех строках и один вызов fit (CatBoost читает обучающий файл с диска).
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:  
    sys.path.insert(0, str(_PROJECT_ROOT))

from catboost import CatBoostClassifier, Pool

from shared.config import MODEL_PATH, OUTPUT_DIR, TRAIN_DATASET_PATH
from training.config import (
    CATBOOST_PARAMS,
    VAL_RATIO,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_EXTERNAL_PARQUET_BATCH_ROWS,
)
from training.catboost_tsv_cpp import export_train_val_tsv
from training.main import _count_val_rows, _detect_columns, _find_time_cutoff, _prepare_batch, _rm_tree

logger = logging.getLogger(__name__)


def _write_column_description(path: Path, feature_cols: list[str]) -> None:
    """Формат CatBoost: индекс, тип, имя; Label и Weight в конце."""
    lines: list[str] = []
    for i, name in enumerate(feature_cols):
        lines.append(f"{i}\tNum\t{name}")
    li = len(feature_cols)
    lines.append(f"{li}\tLabel\ttarget")
    lines.append(f"{li + 1}\tWeight\tsample_weight")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stream_parquet_to_tsv_splits(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    batch_rows: int,
    train_tsv: Path,
    val_tsv: Path,
    *,
    remap_weight2_positive_label_to_zero: bool = False,
) -> None:
    """Один скан parquet: train/val по event_dttm → два TSV с заголовком."""
    for p in (train_tsv, val_tsv):
        if p.exists():
            p.unlink()

    cols = feature_cols + ["target", "sample_weight", dttm_col]
    pf = pq.ParquetFile(parquet_path)
    train_started = False
    val_started = False

    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        m_train = dttm < cutoff_day
        m_val = ~m_train

        if m_train.any():
            x, y, w = _prepare_batch(
                dfb.loc[m_train],
                feature_cols,
                remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero,
            )
            df = pd.DataFrame(x, columns=feature_cols)
            df["target"] = y
            df["sample_weight"] = w
            df.to_csv(
                train_tsv,
                mode="a" if train_started else "w",
                header=not train_started,
                index=False,
                sep="\t",
                float_format="%.9g",
            )
            train_started = True

        if m_val.any():
            x, y, w = _prepare_batch(
                dfb.loc[m_val],
                feature_cols,
                remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero,
            )
            df = pd.DataFrame(x, columns=feature_cols)
            df["target"] = y
            df["sample_weight"] = w
            df.to_csv(
                val_tsv,
                mode="a" if val_started else "w",
                header=not val_started,
                index=False,
                sep="\t",
                float_format="%.9g",
            )
            val_started = True

    if not train_started:
        raise ValueError("После прохода по parquet нет train-строк (проверьте cutoff и данные).")


def _streaming_val_prauc_from_parquet(
    model: CatBoostClassifier,
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    batch_rows: int,
    *,
    remap_weight2_positive_label_to_zero: bool = False,
) -> float:
    """PR-AUC на val без создания большого val Pool в памяти."""
    pf = pq.ParquetFile(parquet_path)
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    cols = feature_cols + ["target", "sample_weight", dttm_col]

    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        m_val = ~(dttm < cutoff_day)
        if not m_val.any():
            continue
        x, y, w = _prepare_batch(
            dfb.loc[m_val],
            feature_cols,
            remap_weight2_positive_label_to_zero=remap_weight2_positive_label_to_zero,
        )
        p = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
        y_all.append(y.astype(np.int32, copy=False))
        p_all.append(p)
        w_all.append(w.astype(np.float64, copy=False))

    if not y_all:
        logger.warning("Нет val-строк — PR-AUC не считается.")
        return 0.0
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    w = np.concatenate(w_all)
    return float(average_precision_score(y, p, sample_weight=w))


def train_catboost_from_tsv_prauc(
    train_tsv: Path,
    val_tsv: Path,
    column_description: Path,
    *,
    catboost_params: dict,
    early_stopping_rounds: int,
    use_eval_pool: bool,
) -> tuple[float, CatBoostClassifier]:
    """Pool из файлов, fit с eval и early stopping; PR-AUC (val) через sklearn с весами из Pool."""
    train_pool = Pool(
        data=str(train_tsv),
        column_description=str(column_description),
        delimiter="\t",
        has_header=True,
    )
    val_pool = None
    if use_eval_pool and val_tsv.exists() and val_tsv.stat().st_size > 0:
        val_pool = Pool(
            data=str(val_tsv),
            column_description=str(column_description),
            delimiter="\t",
            has_header=True,
        )

    params = {k: v for k, v in catboost_params.items() if k != "verbose"}
    verbose = int(catboost_params.get("verbose", 100))

    model = CatBoostClassifier(**params)

    fit_kw: dict = {"verbose": verbose}
    if val_pool is not None:
        fit_kw["eval_set"] = val_pool
        if early_stopping_rounds > 0:
            fit_kw["early_stopping_rounds"] = early_stopping_rounds
            fit_kw["use_best_model"] = True

    model.fit(train_pool, **fit_kw)

    pr = 0.0
    if val_pool is not None:
        p = np.asarray(model.predict_proba(val_pool)[:, 1], dtype=np.float64)
        yva = np.asarray(val_pool.get_label(), dtype=np.float64).astype(np.int32)
        try:
            w_eval = np.asarray(val_pool.get_weight(), dtype=np.float64)
            if w_eval.size == yva.size:
                pr = float(average_precision_score(yva, p, sample_weight=w_eval))
                logger.info("PR-AUC (val): sklearn с sample_weight из Pool")
            else:
                pr = float(average_precision_score(yva, p))
                logger.warning("PR-AUC (val): sklearn без весов (размер weight не совпал)")
        except Exception:
            pr = float(average_precision_score(yva, p))
            logger.warning("PR-AUC (val): sklearn без весов (get_weight недоступен)")
        try:
            logger.info("CatBoost best_iteration: %s", model.get_best_iteration())
        except Exception:
            pass
    else:
        logger.info("Val Pool отключен: PR-AUC посчитаем потоково после fit.")

    return pr, model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="CatBoost: parquet → TSV на диске → Pool → fit")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=None,
        help=f"батч pyarrow при чтении parquet (по умолчанию {XGB_EXTERNAL_PARQUET_BATCH_ROWS})",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help=f"0 — без early stopping (по умолчанию как XGB: {XGB_EARLY_STOPPING_ROUNDS})",
    )
    parser.add_argument(
        "--use-eval-pool",
        action="store_true",
        help="создавать val Pool целиком и использовать eval_set в fit (требует больше RAM)",
    )
    parser.add_argument(
        "--no-cpp",
        action="store_true",
        help="не вызывать C++ parquet_to_catboost_tsv, только Python",
    )
    parser.add_argument(
        "--cpp-threads",
        type=int,
        default=0,
        help="потоки C++-экспортёра (0 — авто)",
    )
    args = parser.parse_args()

    batch_rows = int(args.batch_rows if args.batch_rows is not None else XGB_EXTERNAL_PARQUET_BATCH_ROWS)
    es = int(XGB_EARLY_STOPPING_ROUNDS if args.early_stopping_rounds is None else args.early_stopping_rounds)

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TRAIN_DATASET_PATH}. Сначала соберите датасет: ./dataset_cpp/build/build_dataset ."
        )

    feature_cols, dttm_col = _detect_columns(TRAIN_DATASET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, VAL_RATIO)

    cache_dir = OUTPUT_DIR / "cat_train_tsv_cache"
    _rm_tree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = cache_dir / "train.tsv"
    val_tsv = cache_dir / "val.tsv"
    cd_path = cache_dir / "column_description.cd"

    logger.info("Один проход parquet → TSV, cutoff=%s …", cutoff_day.date())
    _write_column_description(cd_path, feature_cols)
    export_train_val_tsv(
        TRAIN_DATASET_PATH,
        train_tsv,
        val_tsv,
        feature_cols,
        dttm_col,
        cutoff_day,
        batch_rows,
        prefer_cpp=not args.no_cpp,
        cpp_threads=args.cpp_threads,
    )

    logger.info("Подсчёт val-строк по time split (event_dttm, без сегмента по tr_amount) …")
    n_val = _count_val_rows(
        TRAIN_DATASET_PATH,
        dttm_col,
        cutoff_day,
        "tr_amount",
        batch_size=batch_rows,
    )
    logger.info("Val rows (оценка по time split): %d", n_val)

    try:
        pr, model = train_catboost_from_tsv_prauc(
            train_tsv,
            val_tsv,
            cd_path,
            catboost_params=CATBOOST_PARAMS,
            early_stopping_rounds=es,
            use_eval_pool=args.use_eval_pool,
        )
        if not args.use_eval_pool and n_val > 0:
            logger.info("Потоковый расчёт PR-AUC на val (без val Pool в памяти) …")
            pr = _streaming_val_prauc_from_parquet(
                model,
                TRAIN_DATASET_PATH,
                feature_cols,
                dttm_col,
                cutoff_day,
                batch_rows,
            )
            logger.info("PR-AUC (val): sklearn с sample_weight, потоковый расчёт")
        model.save_model(str(MODEL_PATH))
        logger.info("CatBoost PR-AUC (val): %.6f → %s", pr, MODEL_PATH)
    finally:
        gc.collect()
        _rm_tree(cache_dir)
        logger.info("Временный каталог TSV удалён: %s", cache_dir.name)

    logger.info("Готово.")


if __name__ == "__main__":
    main()
