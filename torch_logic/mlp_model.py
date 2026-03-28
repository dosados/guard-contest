from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TorchMLPConfig:
    input_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TorchMLPConfig:
        hd = raw.get("hidden_dims", [])
        return cls(
            input_dim=int(raw["input_dim"]),
            hidden_dims=tuple(int(x) for x in hd),
            dropout=float(raw.get("dropout", 0.0)),
        )


class TorchMLP(nn.Module):
    """Полносвязная сеть: вектор фич → один logit (BCEWithLogitsLoss снаружи)."""

    def __init__(self, config: TorchMLPConfig) -> None:
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        d_in = config.input_dim
        for h in config.hidden_dims:
            blocks.append(nn.Linear(d_in, h))
            blocks.append(nn.ReLU())
            if config.dropout > 0:
                blocks.append(nn.Dropout(config.dropout))
            d_in = h
        blocks.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_mlp_checkpoint(
    path: Path,
    model: TorchMLP,
    *,
    feature_names: list[str],
    nan_fill_values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "torch_mlp",
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "feature_names": list(feature_names),
        "nan_fill_values": np.asarray(nan_fill_values, dtype=np.float32).tolist(),
    }
    torch.save(payload, path)


def apply_nan_fill(x: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Заменить NaN и inf в матрице фич (n, f) значениями fill (f,) — средние по train."""
    out = np.array(x, dtype=np.float32, copy=True)
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    out[bad] = np.broadcast_to(fill, out.shape)[bad]
    return out


def load_mlp_checkpoint(path: Path, device: torch.device) -> tuple[TorchMLP, np.ndarray, list[str]]:
    payload = torch.load(path, map_location=device)
    if payload.get("kind") != "torch_mlp":
        raise ValueError(f"Ожидался чекпоинт torch_mlp, получено: {payload.get('kind')!r}")
    config = TorchMLPConfig.from_dict(payload["config"])
    model = TorchMLP(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    fill = np.asarray(payload["nan_fill_values"], dtype=np.float32)
    names = list(payload["feature_names"])
    return model, fill, names
