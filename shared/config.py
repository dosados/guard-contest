"""
Конфигурация путей и параметров чтения данных.
Используется пакетами dataset, submission. Параметры модели — в training/config.
"""

from pathlib import Path

from shared import PROJECT_ROOT

# Корень данных
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DATA_ROOT = DATA_ROOT / "train"
TEST_DATA_ROOT = DATA_ROOT / "test"

# Pre-train: части по клиентам (агрегируем по порядку)
PRETRAIN_PATHS = [
    TRAIN_DATA_ROOT / "pretrain_part_1.parquet",
    TRAIN_DATA_ROOT / "pretrain_part_2.parquet",
    TRAIN_DATA_ROOT / "pretrain_part_3.parquet",
]

# Train: части, при проходе считаем фичи и сохраняем разметку
TRAIN_PATHS = [
    TRAIN_DATA_ROOT / "train_part_1.parquet",
    TRAIN_DATA_ROOT / "train_part_2.parquet",
    TRAIN_DATA_ROOT / "train_part_3.parquet",
]

# Разметка для train (event_id, target): 0 — жёлтая метка, 1 — целевой класс (красная).
TRAIN_LABELS_PATH = DATA_ROOT / "train_labels.parquet"

# Pre-test и Test
PRETEST_PATH = TEST_DATA_ROOT / "pretest.parquet"
TEST_PATH = TEST_DATA_ROOT / "test.parquet"

# Выходы
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATASET_PATH = OUTPUT_DIR / "train_dataset.parquet"
MODEL_PATH = OUTPUT_DIR / "model.cbm"
MODEL_XGB_PATH = OUTPUT_DIR / "model_xgb.json"
MODEL_LGB_PATH = OUTPUT_DIR / "model_lgb.txt"
MODEL_PYTORCH_PATH = OUTPUT_DIR / "model_pytorch.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "submission.csv"

# Опционально: путь к тестовым меткам для локального расчёта PR-AUC
TEST_LABELS_PATH = None  # e.g. TEST_DATA_ROOT / "test_labels.csv"

# Параметры чтения
BATCH_SIZE = 100_000

# Количество процессов для pretrain (1 = последовательно)
PRETRAIN_N_WORKERS = 4

# Оконный режим: фичи считаются по последним WINDOW_TRANSACTIONS транзакциям клиента
WINDOW_TRANSACTIONS = 150
# Меньший размер батча для режима window_50 (снижает пиковое потребление памяти)
WINDOWED_BATCH_SIZE = 30_000

# Параллельность при создании датасета: 0 = последовательно, 3 = по одному процессу на пару (pretrain_part_i, train_part_i)
DATASET_N_WORKERS = 3
