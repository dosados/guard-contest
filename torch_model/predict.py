"""
Загрузка TabularLSTMClassifier и stateful-инференс по списку фич (submission).

Порядок строк должен совпадать с порядком обхода test (как при обучении: подряд идущие события
одного customer_id — один рекуррентный контекст; смена customer_id — новый контекст).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from torch_model.model import TabularLSTMClassifier

_INFER_CHUNK_CUDA = 512
_INFER_CHUNK_CPU = 64


def _cid_key(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    return s


def load_tabular_lstm(
    path: Path,
    device: torch.device,
) -> TabularLSTMClassifier:
    payload = torch.load(path, map_location=device)
    model = TabularLSTMClassifier(
        n_features=int(payload["n_features"]),
        embed_dim=int(payload["embed_dim"]),
        lstm_hidden=int(payload["lstm_hidden"]),
        num_lstm_layers=int(payload["num_lstm_layers"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def logits_stateful_sequence(
    model: TabularLSTMClassifier,
    device: torch.device,
    customer_ids: list[Any],
    feature_rows: list[Mapping[str, float]],
    feature_order: list[str],
) -> np.ndarray:
    """
    Одна цепочка в порядке строк: смена customer_id (после нормализации _cid_key) обнуляет LSTM.
    На CUDA подряд идущие строки одного пользователя прогоняются пакетом через forward_sequence.
    """
    n_feat = len(feature_order)
    seq_chunk = _INFER_CHUNK_CUDA if device.type == "cuda" else _INFER_CHUNK_CPU
    pin = device.type == "cuda"
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    hx: torch.Tensor | None = None
    cx: torch.Tensor | None = None
    prev: str | None = None
    out: list[float] = []
    n = len(customer_ids)
    i = 0
    while i < n:
        cid_raw = customer_ids[i]
        feats = feature_rows[i]
        ck = _cid_key(cid_raw)
        if ck is None:
            hx, cx = None, None
            prev = None
            mat = np.zeros((1, n_feat), dtype=np.float32)
            for c, name in enumerate(feature_order):
                mat[0, c] = float(feats.get(name, 0.0))
            np.nan_to_num(mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            x_seq = torch.from_numpy(mat)
            if pin:
                x_seq = x_seq.pin_memory().to(device, non_blocking=True)
            else:
                x_seq = x_seq.to(device)
            x_seq = x_seq.unsqueeze(0)
            with torch.no_grad():
                with amp_ctx:
                    logit, hx, cx = model.forward_sequence(x_seq, hx, cx)
            out.append(float(logit.float().cpu().numpy().reshape(-1)[0]))
            i += 1
            continue

        if prev is None or ck != prev:
            hx, cx = None, None
            prev = ck

        L = 0
        j = i
        while j < n and L < seq_chunk:
            if _cid_key(customer_ids[j]) != ck:
                break
            L += 1
            j += 1

        mat = np.zeros((L, n_feat), dtype=np.float32)
        for k in range(L):
            fr = feature_rows[i + k]
            for c, name in enumerate(feature_order):
                mat[k, c] = float(fr.get(name, 0.0))
        np.nan_to_num(mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x_seq = torch.from_numpy(np.ascontiguousarray(mat))
        if pin:
            x_seq = x_seq.pin_memory().to(device, non_blocking=True)
        else:
            x_seq = x_seq.to(device)
        x_seq = x_seq.unsqueeze(0)
        with torch.no_grad():
            with amp_ctx:
                logits, hx, cx = model.forward_sequence(x_seq, hx, cx)
        out.extend(float(x) for x in logits.float().cpu().numpy().reshape(-1).tolist())
        i += L

    return np.asarray(out, dtype=np.float64)
