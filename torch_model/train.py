"""
Обучение TabularLSTMClassifier только по output/full_dataset.parquet (shared.config.TRAIN_DATASET_PATH).

Датасет сгруппирован по пользователям: все строки одного customer_id идут подряд (как обычно после
build_dataset, если исходные train parquet уже сгруппированы). Тогда LSTM переносит (hn, cn) от строки
к строке внутри пользователя и обнуляет состояние при смене customer_id.

Чтение — потоковое (pyarrow iter_batches), порядок строк = порядок в файле.

Train / val (как у бустингов в training/main.py): event_dttm < cutoff_day → шаг SGD с sample_weight
(после remap_sample_weight_from_dataset); иначе — только накопление предсказаний для PR-AUC; hidden state
при этом всё равно обновляется по цепочке.

Запуск из корня репозитория:
  PYTHONPATH=. python torch_model/train.py
По умолчанию GPU (CUDA), если доступна; иначе CPU. Принудительно CPU: --device cpu

На CUDA: mixed precision (bf16 при поддержке, иначе fp16 + GradScaler), склейка до seq_chunk подряд
идущих шагов одного пользователя в один вызов LSTM, pinned memory. При seq_chunk>1 градиент идёт по
всему чанку (truncated BPTT); для максимальной близости к построчному SGD: --seq-chunk 1.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from torch_model.model import TabularLSTMClassifier
from shared.config import (
    BATCH_SIZE,
    MODEL_TORCH_PATH,
    TRAIN_DATASET_PATH,
    remap_sample_weight_from_dataset,
    resolve_model_input_columns,
)
from training.config import RANDOM_SEED, VAL_RATIO

logger = logging.getLogger(__name__)

# Чтение parquet небольшими порциями: меньше пиков RAM; прогресс tqdm обновляется каждые PROGRESS_ROW_STRIDE строк.
PARQUET_BATCH_ROWS = min(BATCH_SIZE, 16_384)
PROGRESS_ROW_STRIDE = 256
SCAN_BATCH = 2_500_000
# Подряд идущие шаги одного пользователя (train или val) — один вызов LSTM на GPU.
DEFAULT_SEQ_CHUNK_CUDA = 512
DEFAULT_SEQ_CHUNK_CPU = 64


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_training_device(requested: str) -> torch.device:
    r = requested.strip().lower()
    wants_cuda = r in ("cuda", "gpu") or r.startswith("cuda:")
    if wants_cuda:
        if not torch.cuda.is_available():
            logger.warning("Запрошена CUDA, но torch.cuda.is_available()=False — используется CPU")
            return torch.device("cpu")
        if r in ("cuda", "gpu"):
            return torch.device("cuda:0")
        return torch.device(requested.strip())
    return torch.device(requested.strip())


def _log_device(device: torch.device) -> None:
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        logger.info(
            "Устройство: CUDA — %s, GPU %d: %s, %.2f ГБ VRAM",
            device,
            idx,
            props.name,
            props.total_memory / (1024**3),
        )
    else:
        logger.info("Устройство: CPU (%s)", device)


def _cid_key(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    return s


def _find_time_cutoff(path: Path, val_ratio: float) -> pd.Timestamp:
    """Тот же принцип, что training/main.py: по дням event_dttm, с конца набираем ~val_ratio доли строк."""
    pf = pq.ParquetFile(path)
    by_day: Counter[pd.Timestamp] = Counter()
    total = 0
    for rb in tqdm(
        pf.iter_batches(columns=["event_dttm"], batch_size=SCAN_BATCH),
        desc="Скан event_dttm (full_dataset)",
        leave=False,
        unit="batch",
    ):
        s = pd.to_datetime(rb.column(0).to_pandas(), errors="coerce").dt.floor("D")
        vc = s.value_counts(dropna=True)
        for k, v in vc.items():
            by_day[k] += int(v)
            total += int(v)
    if total == 0:
        raise ValueError("Не удалось прочитать event_dttm из full_dataset.parquet")
    val_target = max(1, int(total * val_ratio))
    acc = 0
    cutoff: pd.Timestamp | None = None
    for day, cnt in sorted(by_day.items(), reverse=True):
        acc += cnt
        cutoff = day
        if acc >= val_target:
            break
    assert cutoff is not None
    logger.info("Time split cutoff day: %s (val target rows ~= %d)", cutoff.date(), val_target)
    return cutoff


def _dataset_columns(path: Path) -> list[str]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    feature_cols = resolve_model_input_columns(names)
    need = ("customer_id", "target", "sample_weight", "event_dttm")
    for c in need:
        if c not in names:
            raise ValueError(f"В {path} нет колонки {c!r}")
    return list(feature_cols) + list(need)


def _record_batch_to_arrays(
    rb: pa.RecordBatch,
    feature_cols: list[str],
    cutoff_day: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Без rb.to_pandas() на всю таблицу: numpy по фичам + только нужные колонки как Series.
    """
    sch = rb.schema

    def col(name: str) -> pa.Array:
        return rb.column(sch.get_field_index(name))

    feat_mats: list[np.ndarray] = []
    for name in feature_cols:
        raw = col(name).to_numpy(zero_copy_only=False)
        feat_mats.append(np.asarray(raw, dtype=np.float32))
    x_blk = np.column_stack(feat_mats)
    x_blk = np.nan_to_num(x_blk, nan=0.0, posinf=0.0, neginf=0.0)

    y_blk = np.asarray(col("target").to_numpy(zero_copy_only=False), dtype=np.float32)
    y_blk = np.where(np.isfinite(y_blk), y_blk, 0.0)
    w_blk = np.asarray(col("sample_weight").to_numpy(zero_copy_only=False), dtype=np.float32)
    w_blk = np.where(np.isfinite(w_blk), w_blk, 1.0)
    w_blk = remap_sample_weight_from_dataset(w_blk)

    dt_series = col("event_dttm").to_pandas()
    is_val_blk = (pd.to_datetime(dt_series, errors="coerce") >= cutoff_day).to_numpy(dtype=np.bool_, copy=False)
    cid_obj = col("customer_id").to_pandas().to_numpy(copy=False)
    return x_blk, y_blk, w_blk, is_val_blk, cid_obj


def _to_device_2d(
    arr: np.ndarray,
    device: torch.device,
    *,
    pin: bool,
) -> torch.Tensor:
    """(L, F) → GPU; pin_memory + non_blocking при CUDA."""
    x = np.ascontiguousarray(arr)
    t = torch.from_numpy(x)
    if pin and device.type == "cuda":
        t = t.pin_memory()
    return t.to(device, non_blocking=pin and device.type == "cuda")


def _amp_context(device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def run_train_epoch(
    model: TabularLSTMClassifier,
    optimizer: torch.optim.Optimizer,
    dataset_path: Path,
    feature_cols: list[str],
    cutoff_day: pd.Timestamp,
    device: torch.device,
    max_grad_norm: float,
    show_progress: bool,
    seq_chunk: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, list[float], list[float]]:
    """
    Один проход по full_dataset.parquet в порядке строк (и батчей pyarrow).
    Ожидается, что строки одного customer_id образуют непрерывные отрезки в этом порядке.
    Возвращает средний взвешенный loss по train-шагам, списки y и prob для val.
    """
    cols = _dataset_columns(dataset_path)
    pf = pq.ParquetFile(dataset_path)
    total_rows = int(pf.metadata.num_rows)
    pin = device.type == "cuda"
    logger.info(
        "Эпоха: %d строк, parquet по %d; tqdm каждые ~%d строк; seq_chunk=%d; AMP=%s.",
        total_rows,
        PARQUET_BATCH_ROWS,
        PROGRESS_ROW_STRIDE,
        seq_chunk,
        f"{amp_dtype}" if use_amp else "off",
    )

    loss_sum = 0.0
    train_steps = 0
    y_val: list[float] = []
    p_val: list[float] = []

    hx: torch.Tensor | None = None
    cx: torch.Tensor | None = None
    prev_cid: str | None = None

    batch_iter = pf.iter_batches(columns=cols, batch_size=PARQUET_BATCH_ROWS)
    bar = tqdm(
        total=total_rows,
        desc="full_dataset (stateful LSTM)",
        unit="row",
        disable=not show_progress,
        mininterval=0.3,
    )

    logger.info("Чтение с диска: ожидание первого батча parquet…")
    t_batch_wait = time.perf_counter()
    first_batch = True
    rows_pending_bar = 0

    def flush_bar(nflush: int) -> None:
        nonlocal rows_pending_bar
        if nflush <= 0:
            return
        bar.update(nflush)
        rows_pending_bar = 0

    for rb in batch_iter:
        if first_batch:
            logger.info(
                "Первый батч из parquet получен за %.1f с (%d строк), подготовка массивов…",
                time.perf_counter() - t_batch_wait,
                rb.num_rows,
            )
            first_batch = False

        t_prep_start = time.perf_counter()
        try:
            x_blk, y_blk, w_blk, is_val_blk, cid_obj = _record_batch_to_arrays(rb, feature_cols, cutoff_day)
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError) as e:
            logger.warning("PyArrow→numpy не удалось (%s), fallback to_pandas() на батч.", e)
            dfb = rb.to_pandas()
            feat_df = dfb[feature_cols]
            try:
                x_blk = feat_df.to_numpy(dtype=np.float32, copy=False)
            except (TypeError, ValueError):
                x_blk = feat_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
            x_blk = np.nan_to_num(x_blk, nan=0.0, posinf=0.0, neginf=0.0)
            y_blk = pd.to_numeric(dfb["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
            w_blk = pd.to_numeric(dfb["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=False)
            w_blk = remap_sample_weight_from_dataset(w_blk)
            dttm = pd.to_datetime(dfb["event_dttm"], errors="coerce")
            is_val_blk = (dttm >= cutoff_day).to_numpy(dtype=np.bool_, copy=False)
            cid_obj = dfb["customer_id"].to_numpy()

        n = int(x_blk.shape[0])
        if n == 0:
            continue

        prep_s = time.perf_counter() - t_prep_start
        if prep_s > 5.0:
            logger.info("Подготовка массивов для батча из %d строк заняла %.1f с", n, prep_s)

        i = 0
        while i < n:
            ck = _cid_key(cid_obj[i])
            if ck is None:
                rows_pending_bar += 1
                if rows_pending_bar >= PROGRESS_ROW_STRIDE:
                    flush_bar(rows_pending_bar)
                i += 1
                continue

            if prev_cid is None or ck != prev_cid:
                hx, cx = None, None
                prev_cid = ck

            is_val_row = bool(is_val_blk[i])
            L = 0
            j = i
            while j < n and L < seq_chunk:
                ckj = _cid_key(cid_obj[j])
                if ckj != ck:
                    break
                if bool(is_val_blk[j]) != is_val_row:
                    break
                L += 1
                j += 1

            rows_pending_bar += L
            if rows_pending_bar >= PROGRESS_ROW_STRIDE:
                flush_bar(rows_pending_bar)

            x_seq = _to_device_2d(x_blk[i : i + L], device, pin=pin).unsqueeze(0)
            y_seq = _to_device_2d(y_blk[i : i + L, np.newaxis], device, pin=pin).squeeze(-1)
            w_seq = _to_device_2d(w_blk[i : i + L, np.newaxis], device, pin=pin).squeeze(-1)

            if is_val_row:
                model.eval()
                with torch.no_grad():
                    with _amp_context(device, use_amp, amp_dtype):
                        logits, hx, cx = model.forward_sequence(x_seq, hx, cx)
                    probs = torch.sigmoid(logits.float()).detach().cpu().numpy().tolist()
                y_val.extend(float(y_blk[i + k]) for k in range(L))
                p_val.extend(float(p) for p in probs)
                model.train()
            else:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with _amp_context(device, use_amp, amp_dtype):
                    logits, hn, cn = model.forward_sequence(x_seq, hx, cx)
                loss_vec = F.binary_cross_entropy_with_logits(
                    logits.float(), y_seq.float(), reduction="none"
                )
                loss = (loss_vec * w_seq.float()).sum()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                loss_sum += float(loss.detach().cpu())
                train_steps += L
                hx, cx = hn.detach(), cn.detach()

            if train_steps and train_steps % 5000 == 0:
                bar.set_postfix(loss=f"{loss_sum / train_steps:.4f}", refresh=False)

            i += L

    flush_bar(rows_pending_bar)
    bar.close()
    avg_loss = loss_sum / max(train_steps, 1)
    return avg_loss, y_val, p_val


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="LSTM по full_dataset.parquet (как бустинги: время + веса).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu (по умолчанию cuda, если доступна)",
    )
    parser.add_argument(
        "--seq-chunk",
        type=int,
        default=None,
        help=f"сколько подряд идущих шагов одного пользователя склеить в один вызов LSTM "
        f"(по умолчанию {DEFAULT_SEQ_CHUNK_CUDA} на CUDA, {DEFAULT_SEQ_CHUNK_CPU} на CPU; 1 = максимально похоже на построчный режим, но медленно)",
    )
    parser.add_argument("--no-amp", action="store_true", help="на CUDA не использовать mixed precision (bf16/fp16)")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    val_ratio = VAL_RATIO if args.val_ratio is None else args.val_ratio
    show_progress = not args.no_progress

    if not TRAIN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Нет {TRAIN_DATASET_PATH}. Соберите датасет: dataset_cpp/build/build_dataset <корень_проекта>"
        )

    logger.info("Источник обучения: %s", TRAIN_DATASET_PATH.resolve())
    logger.info(
        "LSTM stateful: контекст между подряд идущими строками одного customer_id; "
        "при смене пользователя — сброс. Убедитесь, что в parquet блоки по customer_id непрерывны."
    )

    feature_cols = resolve_model_input_columns(pq.ParquetFile(TRAIN_DATASET_PATH).schema_arrow.names)
    cutoff_day = _find_time_cutoff(TRAIN_DATASET_PATH, val_ratio)

    _set_seed(RANDOM_SEED)
    device = _resolve_training_device(args.device)
    _log_device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    n_features = len(feature_cols)
    model = TabularLSTMClassifier(
        n_features=n_features,
        embed_dim=args.embed_dim,
        lstm_hidden=args.lstm_hidden,
        num_lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler: torch.amp.GradScaler | None
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    else:
        amp_dtype = torch.float32
        scaler = None

    if args.seq_chunk is not None:
        seq_chunk = max(1, args.seq_chunk)
    else:
        seq_chunk = DEFAULT_SEQ_CHUNK_CUDA if device.type == "cuda" else DEFAULT_SEQ_CHUNK_CPU

    best_pr = -1.0

    for epoch in range(args.epochs):
        logger.info("Эпоха %d / %d", epoch + 1, args.epochs)
        avg_loss, y_val, p_val = run_train_epoch(
            model,
            optimizer,
            TRAIN_DATASET_PATH,
            feature_cols,
            cutoff_day,
            device,
            args.max_grad_norm,
            show_progress=show_progress,
            seq_chunk=seq_chunk,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        logger.info("Средний train loss (по шагам): %.6f", avg_loss)

        if y_val:
            y_arr = np.asarray(y_val, dtype=np.float64)
            p_arr = np.asarray(p_val, dtype=np.float64)
            if np.unique(y_arr).size >= 2:
                pr = float(average_precision_score(y_arr, p_arr))
            else:
                pr = float("nan")
                logger.warning("Val содержит один класс — PR-AUC не определён")
            if not math.isnan(pr):
                logger.info("PR-AUC (val по времени): %.6f", pr)
            if not math.isnan(pr) and pr > best_pr:
                best_pr = pr
                payload = {
                    "model_state": model.state_dict(),
                    "feature_cols": list(feature_cols),
                    "n_features": n_features,
                    "embed_dim": args.embed_dim,
                    "lstm_hidden": args.lstm_hidden,
                    "num_lstm_layers": args.lstm_layers,
                    "dropout": args.dropout,
                    "stateful_train": True,
                    "dataset": "full_dataset.parquet",
                }
                MODEL_TORCH_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, MODEL_TORCH_PATH)
                logger.info("Сохранено лучшее по PR-AUC: %s", MODEL_TORCH_PATH)
        else:
            logger.warning("Нет val-предсказаний.")

    if best_pr >= 0 and not math.isnan(best_pr):
        logger.info("Лучший PR-AUC на val: %.6f → %s", best_pr, MODEL_TORCH_PATH)


if __name__ == "__main__":
    main()
