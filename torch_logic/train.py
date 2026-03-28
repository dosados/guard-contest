from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import MODEL_TORCH_PATH, TRAIN_LABELS_PATH, TRAIN_PATHS
from torch_logic.data import EVENT_DTTM_COL, RawRow, load_positive_event_ids, iter_raw_rows, rows_to_tensor
from torch_logic.model import TorchSequenceModel, TorchSequenceModelConfig, save_checkpoint

logger = logging.getLogger(__name__)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _log_device_info(device: torch.device) -> None:
    """Явный лог: CPU или CUDA, имя GPU и версии — чтобы было видно, куда реально идёт compute."""
    cuda_avail = torch.cuda.is_available()
    logger.info("torch.cuda.is_available(): %s", cuda_avail)
    logger.info("Выбранное устройство: %s (type=%s)", device, device.type)

    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            logger.info(
                "CUDA compute: index=%s name=%s capability=%s total_mem=%.2f GiB",
                idx,
                name,
                cap,
                mem_gb,
            )
            logger.info("CUDA runtime: torch.version.cuda=%s cudnn.is_available=%s cudnn.version=%s", torch.version.cuda, torch.backends.cudnn.is_available(), torch.backends.cudnn.version())
        except Exception as exc:
            logger.warning("Не удалось прочитать детали CUDA: %s", exc)
    else:
        if cuda_avail:
            logger.warning(
                "CUDA доступна в PyTorch, но выбрано устройство %s. "
                "Запустите с --device cuda для GPU.",
                device,
            )
        else:
            logger.info("Режим CPU: GPU не используется (нет CUDA или драйвер/сборка PyTorch без CUDA).")


def train_one_epoch(
    model: TorchSequenceModel,
    device: torch.device,
    positive_event_ids: set[int],
    total_rows: int,
    cutoff_day: datetime,
    *,
    seq_chunk_size: int,
    lr: float,
    log_every: int,
) -> float:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    total_loss = 0.0
    total_steps = 0
    current_customer: str | None = None
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    chunk_rows: list[RawRow] = []
    chunk_is_train: list[bool] = []

    def flush_chunk() -> None:
        nonlocal state, total_loss, total_steps
        if not chunk_rows:
            return
        x = rows_to_tensor(chunk_rows, device)
        y = torch.tensor(
            [1.0 if r.event_id in positive_event_ids else 0.0 for r in chunk_rows],
            dtype=torch.float32,
            device=device,
        )
        train_mask = torch.tensor(chunk_is_train, dtype=torch.bool, device=device)
        has_train = bool(train_mask.any())

        if has_train:
            optimizer.zero_grad(set_to_none=True)
            logits, state = model(x, state)
            logits_1d = logits.squeeze(0)
            per_t = criterion(logits_1d, y)
            loss = per_t[train_mask].mean()
            loss.backward()
            optimizer.step()
            state = (state[0].detach(), state[1].detach())
            total_loss += float(loss.item())
            total_steps += 1
        else:
            with torch.no_grad():
                _, state = model(x, state)
                state = (state[0].detach(), state[1].detach())

        chunk_rows.clear()
        chunk_is_train.clear()

    with tqdm(total=total_rows if total_rows > 0 else None, desc="torch train", unit="rows") as pbar:
        for row in iter_raw_rows(TRAIN_PATHS):
            if current_customer != row.customer_id:
                flush_chunk()
                current_customer = row.customer_id
                state = None

            is_train = row.event_dttm is not None and row.event_dttm < cutoff_day
            chunk_rows.append(row)
            chunk_is_train.append(is_train)
            if len(chunk_rows) >= seq_chunk_size:
                flush_chunk()

            pbar.update(1)
            if log_every > 0 and total_steps > 0 and total_steps % log_every == 0:
                logger.info("train optimizer steps=%d avg_loss=%.6f", total_steps, total_loss / total_steps)

        flush_chunk()

    return total_loss / max(1, total_steps)


@torch.no_grad()
def evaluate_aucpr(
    model: TorchSequenceModel,
    device: torch.device,
    positive_event_ids: set[int],
    total_rows: int,
    cutoff_day: datetime,
    *,
    seq_chunk_size: int,
) -> float:
    model.eval()
    current_customer: str | None = None
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    y_true: list[int] = []
    y_score: list[float] = []
    chunk_rows: list[RawRow] = []

    def flush_chunk_eval() -> None:
        nonlocal state
        if not chunk_rows:
            return
        x = rows_to_tensor(chunk_rows, device)
        logits, state = model(x, state)
        logits_1d = logits.squeeze(0)
        for r, logit_t in zip(chunk_rows, logits_1d):
            is_val = not (r.event_dttm is not None and r.event_dttm < cutoff_day)
            if is_val:
                y_true.append(1 if r.event_id in positive_event_ids else 0)
                y_score.append(float(logit_t.item()))
        chunk_rows.clear()

    with tqdm(total=total_rows if total_rows > 0 else None, desc="torch eval", unit="rows") as pbar:
        for row in iter_raw_rows(TRAIN_PATHS):
            if current_customer != row.customer_id:
                flush_chunk_eval()
                current_customer = row.customer_id
                state = None

            chunk_rows.append(row)
            if len(chunk_rows) >= seq_chunk_size:
                flush_chunk_eval()
            pbar.update(1)

        flush_chunk_eval()

    if not y_true:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _count_total_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        total += int(pf.metadata.num_rows)
    return total


def _find_time_cutoff(paths: list[Path]) -> datetime:
    by_day: Counter[pd.Timestamp] = Counter()
    total = 0
    for path in paths:
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        if EVENT_DTTM_COL not in set(pf.schema_arrow.names):
            continue
        for rb in pf.iter_batches(columns=[EVENT_DTTM_COL], batch_size=2_500_000):
            s = pd.to_datetime(rb.column(0).to_pandas(), errors="coerce").dt.floor("D")
            vc = s.value_counts(dropna=True)
            for day, cnt in vc.items():
                by_day[day] += int(cnt)
                total += int(cnt)
    if total == 0:
        raise ValueError("Не удалось определить cutoff: отсутствуют валидные event_dttm в train_part*.")
    val_target = max(1, int(total * 0.2))
    acc = 0
    cutoff = None
    for day, cnt in sorted(by_day.items(), reverse=True):
        acc += cnt
        cutoff = day
        if acc >= val_target:
            break
    assert cutoff is not None
    return cutoff.to_pydatetime()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--linear-dim", type=int, default=32)
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20000)
    parser.add_argument(
        "--seq-chunk-size",
        type=int,
        default=128,
        help="Сколько подряд идущих шагов одного customer_id склеить в один forward LSTM (меньше вызовов GPU).",
    )
    parser.add_argument("--output", type=Path, default=MODEL_TORCH_PATH)
    args = parser.parse_args()

    for path in TRAIN_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Не найден train parquet: {path}")
    if not TRAIN_LABELS_PATH.exists():
        raise FileNotFoundError(f"Не найден labels parquet: {TRAIN_LABELS_PATH}")

    device = _device_from_arg(args.device)
    _log_device_info(device)

    positive_event_ids = load_positive_event_ids(TRAIN_LABELS_PATH)
    logger.info("Loaded positive event ids: %d", len(positive_event_ids))
    total_rows = _count_total_rows(TRAIN_PATHS)
    logger.info("Train rows total (metadata): %d", total_rows)
    cutoff_day = _find_time_cutoff(TRAIN_PATHS)
    logger.info("Time split cutoff day (val ~= 20%% newest by day): %s", cutoff_day.date())
    seq_chunk = max(1, int(args.seq_chunk_size))
    logger.info("seq_chunk_size=%d (батч по времени внутри одного пользователя)", seq_chunk)

    config = TorchSequenceModelConfig(
        input_dim=4,
        linear_dim=args.linear_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    )
    model = TorchSequenceModel(config).to(device)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            device,
            positive_event_ids,
            total_rows,
            cutoff_day,
            seq_chunk_size=seq_chunk,
            lr=args.lr,
            log_every=args.log_every,
        )
        logger.info("epoch=%d avg_loss=%.6f", epoch, avg_loss)

    aucpr = evaluate_aucpr(
        model,
        device,
        positive_event_ids,
        total_rows,
        cutoff_day,
        seq_chunk_size=seq_chunk,
    )
    logger.info("PR-AUC (val, sklearn average_precision_score): %.6f", aucpr)

    save_checkpoint(args.output, model)
    logger.info("Saved torch checkpoint: %s", args.output)


if __name__ == "__main__":
    main()
