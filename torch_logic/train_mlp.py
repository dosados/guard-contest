from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
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

from shared.config import MODEL_TORCH_MLP_PATH, TRAIN_DATASET_PATH, remap_sample_weight_from_dataset
from torch_logic.mlp_model import TorchMLP, TorchMLPConfig, apply_nan_fill, save_mlp_checkpoint
from training.config import VAL_RATIO
from training.main import _detect_columns, _find_time_cutoff, _prepare_batch

logger = logging.getLogger(__name__)


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _log_device_info(device: torch.device, device_arg: str) -> None:
    cuda_avail = torch.cuda.is_available()
    logger.info("torch.cuda.is_available(): %s", cuda_avail)
    logger.info("Устройство: %s", device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("GPU: %s", torch.cuda.get_device_name(idx))
    elif cuda_avail and device_arg == "cpu":
        logger.warning("CUDA доступна, но выбран CPU.")


def _compute_train_nan_fill(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    batch_rows: int,
) -> np.ndarray:
    """
    Для каждой фичи — среднее по конечным значениям только на train-строках (event_dttm < cutoff).
    NaN/inf в данных заменяются этими средними при обучении и инференсе (как в чекпоинте).
    """
    pf = pq.ParquetFile(parquet_path)
    cols = feature_cols + [dttm_col]
    sums = np.zeros(len(feature_cols), dtype=np.float64)
    counts = np.zeros(len(feature_cols), dtype=np.float64)

    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        mask = dttm < cutoff_day
        if not bool(mask.any()):
            continue
        sub = dfb.loc[mask, feature_cols].apply(pd.to_numeric, errors="coerce")
        arr = sub.to_numpy(dtype=np.float64)
        finite = np.isfinite(arr)
        sums += np.where(finite, arr, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)

    fill = np.zeros(len(feature_cols), dtype=np.float32)
    for i in range(len(feature_cols)):
        fill[i] = float(sums[i] / counts[i]) if counts[i] > 0 else 0.0
    return fill


def _micro_batch_slices(n: int, micro_batch_size: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    m = max(1, int(micro_batch_size))
    return [(i, min(i + m, n)) for i in range(0, n, m)]


def train_mlp_epoch(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    fill: np.ndarray,
    model: TorchMLP,
    optimizer: AdamW,
    device: torch.device,
    *,
    batch_rows: int,
    micro_batch_size: int,
    max_grad_norm: float,
    epoch: int,
    val_ratio: float,
) -> float:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()

    pf = pq.ParquetFile(parquet_path)
    cols = feature_cols + ["target", "sample_weight", dttm_col]
    total_loss = 0.0
    n_steps = 0

    num_rows = int(pf.metadata.num_rows)
    est_train_rows = max(1, int(num_rows * (1.0 - float(val_ratio))))
    est_micro_steps = max(1, (est_train_rows + micro_batch_size - 1) // micro_batch_size)

    pbar = tqdm(
        desc=f"epoch {epoch} train",
        unit="micro",
        total=est_micro_steps,
        mininterval=0.5,
    )

    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        train_mask = dttm < cutoff_day
        if not bool(train_mask.any()):
            continue
        tr = dfb.loc[train_mask].copy()
        x, y, w = _prepare_batch(tr, feature_cols)
        x = apply_nan_fill(x, fill)
        n_loc = x.shape[0]
        for a, b in _micro_batch_slices(n_loc, micro_batch_size):
            xt = torch.from_numpy(x[a:b]).to(device, non_blocking=True)
            yt = torch.from_numpy(y[a:b].astype(np.float32)).to(device, non_blocking=True)
            wt = torch.from_numpy(w[a:b].astype(np.float32)).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xt)
            per = criterion(logits, yt)
            loss = (per * wt).sum() / (wt.sum() + 1e-8)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

    pbar.close()
    return total_loss / max(1, n_steps)


@torch.no_grad()
def evaluate_val_prauc(
    parquet_path: Path,
    feature_cols: list[str],
    dttm_col: str,
    cutoff_day: pd.Timestamp,
    fill: np.ndarray,
    model: TorchMLP,
    device: torch.device,
    *,
    batch_rows: int,
    micro_batch_size: int,
) -> float:
    model.eval()
    pf = pq.ParquetFile(parquet_path)
    cols = feature_cols + ["target", "sample_weight", dttm_col]
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []

    num_rows = int(pf.metadata.num_rows)

    for rb in tqdm(
        pf.iter_batches(columns=cols, batch_size=batch_rows),
        desc="val PR-AUC",
        unit="parquet",
        total=max(1, (num_rows + batch_rows - 1) // batch_rows),
        mininterval=0.5,
    ):
        dfb = rb.to_pandas()
        dttm = pd.to_datetime(dfb[dttm_col], errors="coerce")
        val_mask = ~(dttm < cutoff_day)
        if not bool(val_mask.any()):
            continue
        va = dfb.loc[val_mask].copy()
        x, y, w = _prepare_batch(va, feature_cols)
        x = apply_nan_fill(x, fill)
        probs_chunks: list[np.ndarray] = []
        n_loc = x.shape[0]
        for a, b in _micro_batch_slices(n_loc, micro_batch_size):
            xt = torch.from_numpy(x[a:b]).to(device, non_blocking=True)
            logits = model(xt)
            probs_chunks.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
        prob = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.array([], dtype=np.float64)
        y_parts.append(y.astype(np.int32, copy=False))
        p_parts.append(prob)
        w_parts.append(w.astype(np.float64, copy=False))

    if not y_parts:
        return 0.0
    y_all = np.concatenate(y_parts)
    p_all = np.concatenate(p_parts)
    w_all = np.concatenate(w_parts)
    return float(average_precision_score(y_all, p_all, sample_weight=w_all))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="MLP на full_dataset.parquet (как XGBoost: time split + sample_weight).")
    parser.add_argument("--dataset", type=Path, default=TRAIN_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_TORCH_MLP_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-rows", type=int, default=1_000_000, help="Размер чтения parquet (CPU RAM).")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=16_384,
        help="Сколько строк одновременно на GPU (forward/backward). Меньше — меньше VRAM.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 16],
        help="Размеры скрытых слоёв. Пусто не задаётся: нужен хотя бы один слой.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda | cpu | auto. По умолчанию cuda (как требование GPU).",
    )
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Доля val по времени (как training.config.VAL_RATIO).")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Нет датасета: {args.dataset}")

    device = _device_from_arg(args.device)
    _log_device_info(device, args.device)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("Запрошено --device cuda, но CUDA недоступна. Установите PyTorch с CUDA или укажите --device cpu.")

    feature_cols, dttm_col = _detect_columns(args.dataset)
    cutoff_day = _find_time_cutoff(args.dataset, float(args.val_ratio), batch_size=args.batch_rows)
    logger.info(
        "Сплит как у бустингов: VAL_RATIO=%.4f, cutoff day=%s",
        float(args.val_ratio),
        cutoff_day.date(),
    )

    logger.info("Считаем средние для imputation NaN/inf по train-части …")
    fill = _compute_train_nan_fill(
        args.dataset,
        feature_cols,
        dttm_col,
        cutoff_day,
        args.batch_rows,
    )

    hidden = tuple(int(x) for x in args.hidden_dims)
    if not hidden:
        raise ValueError("Укажите хотя бы один --hidden-dims, например 512 256 128")

    config = TorchMLPConfig(input_dim=len(feature_cols), hidden_dims=hidden, dropout=float(args.dropout))
    model = TorchMLP(config).to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    micro_bs = max(1, int(args.micro_batch_size))
    logger.info(
        "MLP: input_dim=%d hidden=%s dropout=%s | parquet batch_rows=%d micro_batch_size=%d (GPU)",
        config.input_dim,
        hidden,
        config.dropout,
        args.batch_rows,
        micro_bs,
    )

    val_ratio_f = float(args.val_ratio)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_mlp_epoch(
            args.dataset,
            feature_cols,
            dttm_col,
            cutoff_day,
            fill,
            model,
            optimizer,
            device,
            batch_rows=args.batch_rows,
            micro_batch_size=micro_bs,
            max_grad_norm=float(args.max_grad_norm),
            epoch=epoch,
            val_ratio=val_ratio_f,
        )
        pr = evaluate_val_prauc(
            args.dataset,
            feature_cols,
            dttm_col,
            cutoff_day,
            fill,
            model,
            device,
            batch_rows=args.batch_rows,
            micro_batch_size=micro_bs,
        )
        logger.info("epoch=%d train_loss=%.6f PR-AUC(val, weighted)=%.6f", epoch, avg_loss, pr)

    save_mlp_checkpoint(args.output, model, feature_names=feature_cols, nan_fill_values=fill)
    logger.info("Сохранено: %s", args.output)


if __name__ == "__main__":
    main()
