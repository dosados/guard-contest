"""
GPU-вариант обучения XGBoost.

Переиспользует training.main без изменений в training/config.py:
добавляет device=cuda в параметры бустинга.

Важно: тот же пайплайн данных, что и в training — ExtMemQuantileDMatrix (дисковый кэш),
иначе XGBoost переходит на QuantileDMatrix и процесс часто убивается по RAM (Killed).

Если в вашей сборке XGBoost нельзя обучать GPU на ExtMemQuantileDMatrix, при первой ошибке
обучение автоматически повторяется на CPU (как обычный training/main.py).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Запуск `python training_gpu/main.py` не кладёт корень репозитория в sys.path — добавляем вручную.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training import main as training_main
from training.config import XGB_PARAMS

logger = logging.getLogger(__name__)

_orig_build_xgb_train_params = training_main._build_xgb_train_params


def _build_xgb_train_params_gpu(config_mode: str) -> dict[str, float | int | str]:
    """Как training.main._build_xgb_train_params, но с device=cuda для xgb.train."""
    params = _orig_build_xgb_train_params(config_mode)
    params["tree_method"] = "hist"
    params["device"] = "cuda"
    params.pop("predictor", None)
    return params


def _strip_gpu_from_params(p: dict) -> dict[str, float | int | str]:
    out = dict(p)
    out.pop("device", None)
    out.pop("predictor", None)
    out.setdefault("tree_method", "hist")
    return out  # type: ignore[return-value]


def _install_xgb_train_fallback() -> tuple[object, object]:
    import xgboost as xgb_mod

    real_train = xgb_mod.train

    def train_with_cpu_fallback(*args: object, **kwargs: object):
        try:
            return real_train(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            gpu_extmem = (
                "extmemquantile" in msg
                or "ext_mem" in msg
                or "cpu data" in msg
                or "cannot be used for gpu" in msg
            )
            if not gpu_extmem:
                raise
            params = kwargs.get("params")
            if not isinstance(params, dict):
                raise
            logger.warning(
                "XGBoost: GPU недоступен для ExtMemQuantileDMatrix в этой сборке; "
                "повторяю xgb.train на CPU (как training/main.py)."
            )
            kwargs = dict(kwargs)
            kwargs["params"] = _strip_gpu_from_params(params)
            return real_train(*args, **kwargs)

    xgb_mod.train = train_with_cpu_fallback  # type: ignore[assignment]
    return xgb_mod, real_train


def _restore_xgb_train(xgb_mod: object | None, real_train: object | None) -> None:
    if xgb_mod is not None and real_train is not None:
        setattr(xgb_mod, "train", real_train)


def _enable_gpu_params() -> tuple[dict[str, object], object, object | None, object | None]:
    snapshot = dict(XGB_PARAMS)
    training_main._build_xgb_train_params = _build_xgb_train_params_gpu

    XGB_PARAMS["tree_method"] = "hist"
    XGB_PARAMS["device"] = "cuda"
    XGB_PARAMS.pop("predictor", None)

    xgb_mod, real_train = _install_xgb_train_fallback()
    return snapshot, _orig_build_xgb_train_params, xgb_mod, real_train


def _restore_params(
    snapshot: dict[str, object],
    orig_builder: object,
    xgb_mod: object | None,
    real_train: object | None,
) -> None:
    training_main._build_xgb_train_params = orig_builder  # type: ignore[assignment]
    _restore_xgb_train(xgb_mod, real_train)
    XGB_PARAMS.clear()
    XGB_PARAMS.update(snapshot)


def main() -> None:
    snapshot, orig_builder, xgb_mod, real_train = _enable_gpu_params()
    logger.info(
        "GPU mode: device=cuda; тот же ExtMemQuantileDMatrix-путь, что и в training/main.py "
        "(без него — QuantileDMatrix и риск Killed по RAM)."
    )
    try:
        training_main.main()
    finally:
        _restore_params(snapshot, orig_builder, xgb_mod, real_train)


if __name__ == "__main__":
    main()
