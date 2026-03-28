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

from shared.config import MODEL_TORCH_PATH, OUTPUT_DIR, TEST_PATH
from torch_logic.data import RawRow, iter_raw_rows, rows_to_tensor
from torch_logic.model import load_checkpoint

logger = logging.getLogger(__name__)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def predict_logits(model_path: Path, test_path: Path, device: torch.device, *, seq_chunk_size: int = 128) -> pd.DataFrame:
    model = load_checkpoint(model_path, device)
    event_ids: list[int] = []
    logits: list[float] = []
    current_customer: str | None = None
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    chunk: list[RawRow] = []
    chunk_sz = max(1, int(seq_chunk_size))

    def flush() -> None:
        nonlocal state
        if not chunk:
            return
        x = rows_to_tensor(chunk, device)
        logit_seq, state = model(x, state)
        for r, lt in zip(chunk, logit_seq.squeeze(0)):
            event_ids.append(r.event_id)
            logits.append(float(lt.item()))
        chunk.clear()

    for row in iter_raw_rows([test_path]):
        if current_customer != row.customer_id:
            flush()
            current_customer = row.customer_id
            state = None
        chunk.append(row)
        if len(chunk) >= chunk_sz:
            flush()
    flush()

    return pd.DataFrame({"event_id": event_ids, "predict": np.asarray(logits, dtype=np.float64)})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_TORCH_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "submission.csv")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument(
        "--seq-chunk-size",
        type=int,
        default=128,
        help="Сколько подряд шагов одного пользователя объединять в один forward.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Не найден torch checkpoint: {args.model_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Не найден test parquet: {args.test_path}")

    device = _device_from_arg(args.device)
    logger.info("Torch inference device: %s", device)
    out = predict_logits(args.model_path, args.test_path, device, seq_chunk_size=args.seq_chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Saved submission rows=%d to %s", len(out), args.output)


if __name__ == "__main__":
    main()
