"""
Параметры окна датасета (агрегаты по клиенту).

Вынесены из shared.config, чтобы избежать циклического импорта:
config → features → parquet_batch_aggregates → (раньше) config.
"""

from __future__ import annotations

# Размер скользящего окна транзакций на клиента (как в dataset_cpp_module_spec.txt)
WINDOW_TRANSACTIONS = 150

# Режим датасета: "full" — учитывается вся история в окне; "window_50" — окно WINDOW_TRANSACTIONS_MODE
DATASET_MODE = "full"
WINDOW_TRANSACTIONS_MODE = 50  # используется если DATASET_MODE == "window_50"
