from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class TorchSequenceModelConfig:
    input_dim: int = 2
    linear_dim: int = 32
    lstm_hidden_dim: int = 64
    lstm_layers: int = 1
    dropout: float = 0.0
    input_scale: float = 1_000_000.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, int | float]) -> "TorchSequenceModelConfig":
        return cls(
            input_dim=int(raw["input_dim"]),
            linear_dim=int(raw["linear_dim"]),
            lstm_hidden_dim=int(raw["lstm_hidden_dim"]),
            lstm_layers=int(raw["lstm_layers"]),
            dropout=float(raw.get("dropout", 0.0)),
            input_scale=float(raw.get("input_scale", 1_000_000.0)),
        )


class TorchSequenceModel(nn.Module):
    """
    Архитектура: linear -> LSTM -> linear(logit).
    Модель работает по последовательности транзакций одного пользователя.
    """

    def __init__(self, config: TorchSequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.input_dim, config.linear_dim)
        self.activation = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=config.linear_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_linear = nn.Linear(config.lstm_hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        scaled = x / self.config.input_scale
        z = self.activation(self.input_linear(scaled))
        y, new_state = self.lstm(z, state)
        logits = self.output_linear(y).squeeze(-1)
        return logits, new_state


def save_checkpoint(path: Path, model: TorchSequenceModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> TorchSequenceModel:
    payload = torch.load(path, map_location=device)
    config = TorchSequenceModelConfig.from_dict(payload["config"])
    model = TorchSequenceModel(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
