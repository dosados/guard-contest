"""
Запуск полного пайплайна по шагам (опционально).
Обычно лучше запускать модули по отдельности:
  1) python prepare_training_data.py
  2) python train_model.py
  3) python predict_and_evaluate.py
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from prepare_training_data import main as step1
    from train_model import main as step2
    from predict_and_evaluate import main as step3

    logger.info("Step 1: prepare_training_data")
    step1()
    logger.info("Step 2: train_model")
    step2()
    logger.info("Step 3: predict_and_evaluate")
    step3()
    logger.info("Done. Predictions: config.PREDICTIONS_PATH")


if __name__ == "__main__":
    main()
    sys.exit(0)
