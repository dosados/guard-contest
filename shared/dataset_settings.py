from __future__ import annotations

# Window params (separate from shared.config to avoid import cycles)

# Max transactions in the per-customer sliding window (active window cap).
WINDOW_TRANSACTIONS = 512

# Dataset mode: "full" uses full history in window; "window_50" uses WINDOW_TRANSACTIONS_MODE
DATASET_MODE = "full"
WINDOW_TRANSACTIONS_MODE = 50  # used when DATASET_MODE == "window_50"
