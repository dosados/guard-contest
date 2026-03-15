"""
Конфигурация обучения: параметры модели, разбиение train/val.
Меняй здесь для экспериментов с разными моделями и долей валидации.
"""

# Доля данных для валидации (остальное — обучение)
VAL_RATIO = 0.3
RANDOM_SEED = 42

# Параметры CatBoost
# task_type="CPU" и thread_count избегают зависаний после обучения (GPU/многопоточность)
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "verbose": 100,
    "random_seed": 42,
    "task_type": "CPU",
    "thread_count": 4,
}

# Параметры XGBoost
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "use_label_encoder": False,
    "verbosity": 0,
}

# Параметры LightGBM
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "binary",
    "metric": "average_precision",
    "random_state": 42,
    "verbosity": -1,
}

# Параметры PyTorch MLP
PYTORCH_PARAMS = {
    "hidden_sizes": (128, 64),
    "epochs": 50,
    "batch_size": 2048,
    "lr": 1e-3,
    "dropout": 0.2,
    "verbose": 10,
}
