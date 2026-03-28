from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_TORCH_MLP_PATH, OUTPUT_DIR
from torch_logic.mlp_model import apply_nan_fill, load_mlp_checkpoint

logger = logging.getLogger(__name__)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def predict_mlp_dataframe(
    X: pd.DataFrame,
    model_path: Path,
    device: torch.device,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """
    Логиты для табличной матрицы с колонками как при обучении (MODEL_INPUT_FEATURES).
    NaN/inf обрабатываются так же, как при train: средние из чекпоинта.
    """
    model, fill, feature_names = load_mlp_checkpoint(model_path, device)
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"В DataFrame нет колонок фичей: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    x = X[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x = apply_nan_fill(x, fill)
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xt = torch.from_numpy(x[start:end]).to(device)
        logits = model(xt)
        out[start:end] = logits.cpu().numpy().astype(np.float64)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Инференс табличного MLP по parquet с теми же фичами, что full_dataset.")
    parser.add_argument("--model-path", type=Path, default=MODEL_TORCH_MLP_PATH)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "mlp_predict.csv")
    parser.add_argument("--device", default="cuda", help="cuda | cpu | auto")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--event-id-col",
        default="event_id",
        help="Колонка id строки в выходном CSV (если есть в parquet).",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Нет чекпоинта: {args.model_path}")
    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Нет входного parquet: {args.input_parquet}")

    device = _device_from_arg(args.device)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("Запрошено --device cuda, но CUDA недоступна.")

    logger.info("Чтение %s …", args.input_parquet)
    X = pd.read_parquet(args.input_parquet)
    logits = predict_mlp_dataframe(X, args.model_path, device, batch_size=int(args.batch_size))

    if args.event_id_col in X.columns:
        out = pd.DataFrame({args.event_id_col: X[args.event_id_col].values, "predict": logits})
    else:
        out = pd.DataFrame({"predict": logits})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Сохранено %d строк в %s", len(out), args.output)


if __name__ == "__main__":
    main()
