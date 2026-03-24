"""Загрузка обучающего датасета: output/full_dataset.parquet (C++) или legacy-форматы."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from shared.config import OUTPUT_DIR, TRAIN_DATASET_PATH, remap_sample_weight_from_dataset

logger = logging.getLogger(__name__)


def train_dataset_source_paths() -> list[Path]:
    if TRAIN_DATASET_PATH.exists():
        return [TRAIN_DATASET_PATH]
    parts = sorted(OUTPUT_DIR.glob("train_dataset_part_*.parquet"))
    if parts:
        return parts
    legacy_single = OUTPUT_DIR / "train_dataset.parquet"
    if legacy_single.exists():
        return [legacy_single]
    return []


def train_dataset_is_available() -> bool:
    return bool(train_dataset_source_paths())


def load_train_dataframe() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = train_dataset_source_paths()
    if not paths:
        raise FileNotFoundError(
            f"Нет {TRAIN_DATASET_PATH}, output/train_dataset.parquet и output/train_dataset_part_*.parquet. "
            "Соберите датасет: бинарник dataset_cpp/build_dataset из корня репозитория."
        )
    logger.info("Загрузка train_dataset: %d файл(ов)", len(paths))
    if len(paths) == 1:
        df = pd.read_parquet(paths[0])
    else:
        parts = [pd.read_parquet(p) for p in tqdm(paths, desc="train_dataset parquet", unit="file")]
        df = pd.concat(parts, ignore_index=True)
    if "sample_weight" in df.columns:
        df["sample_weight"] = remap_sample_weight_from_dataset(df["sample_weight"].to_numpy(copy=False))
    logger.info("Train dataframe: %d строк, %d колонок", len(df), len(df.columns))
    return df
