"""
Конфигурация путей и параметров пайплайна.
Меняй здесь пути и модель для переключения окружения или эксперимента.
"""

from pathlib import Path

# Корень данных (относительно репозитория или абсолютный)
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
TRAIN_DATA_ROOT = DATA_ROOT / "train"
TEST_DATA_ROOT = DATA_ROOT / "test"

# Pre-train: 3 части по третям клиентов (агрегируем по порядку)
PRETRAIN_PATHS = [
    TRAIN_DATA_ROOT / "pretrain_part_0.parquet",
    TRAIN_DATA_ROOT / "pretrain_part_1.parquet",
    TRAIN_DATA_ROOT / "pretrain_part_2.parquet",
]

# Train: 3 части, при проходе считаем фичи и сохраняем разметку
TRAIN_PATHS = [
    TRAIN_DATA_ROOT / "train_part_0.parquet",
    TRAIN_DATA_ROOT / "train_part_1.parquet",
    TRAIN_DATA_ROOT / "train_part_2.parquet",
]

# Разметка для train (event_id, target): 0 — жёлтая метка (только подозрение), 1 — целевой класс (красная).
TRAIN_LABELS_PATH = DATA_ROOT / "train_labels.parquet"

# Pre-test и Test (один файл каждый)
PRETEST_PATH = TEST_DATA_ROOT / "pretest.parquet"
TEST_PATH = TEST_DATA_ROOT / "test.parquet"

# Выходы (можно вынести в отдельную папку артефактов)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FEATURES_PATH = OUTPUT_DIR / "train_features.parquet"
MODEL_PATH = OUTPUT_DIR / "catboost_model.cbm"
PREDICTIONS_PATH = OUTPUT_DIR / "submission.csv"

# Опционально: путь к тестовым меткам для локального расчёта PR-AUC
TEST_LABELS_PATH = None  # e.g. TEST_DATA_ROOT / "test_labels.csv"

# Параметры чтения
BATCH_SIZE = 100_000

# Параметры CatBoost (легко заменить модель — поменять класс и параметры)
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "verbose": 100,
    "random_seed": 42,
}
