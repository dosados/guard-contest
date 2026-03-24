"""
Табличный классификатор: Linear → LSTM (один шаг за вызов) → логит.

Режим stateful: при последовательных вызовах forward_step с тем же пользователем
передаются hn, cn от предыдущего шага; при смене пользователя передавайте hx=cx=None.
"""

from __future__ import annotations

import torch
from torch import nn


class TabularLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        embed_dim: int = 128,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers

        self.input_proj = nn.Linear(n_features, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, 1)

    def forward_step(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        cx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Один временной шаг, batch=1.
        x: (n_features,) или (1, n_features) на device.
        hx, cx: (num_layers, 1, hidden) или None — тогда нулевое начальное состояние.
        Возвращает: logit (), hn, cn.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.input_proj(x).unsqueeze(1)  # (1, 1, embed_dim)
        if hx is None or cx is None:
            out, (hn, cn) = self.lstm(h)
        else:
            out, (hn, cn) = self.lstm(h, (hx, cx))
        logit = self.head(out[:, -1, :]).squeeze(-1).squeeze(0)
        return logit, hn, cn

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        hx: torch.Tensor | None = None,
        cx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Одна последовательность одного пользователя: T шагов за один вызов LSTM (эффективнее на GPU).
        x_seq: (1, T, n_features) или (T, n_features).
        Возвращает logits (T,), hn, cn.
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
        h = self.input_proj(x_seq)
        if hx is None or cx is None:
            out, (hn, cn) = self.lstm(h)
        else:
            out, (hn, cn) = self.lstm(h, (hx, cx))
        logits = self.head(out).squeeze(-1).squeeze(0)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        return logits, hn, cn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Без рекуррентного состояния: каждая строка батча с нулевым h0, c0 (инференс по батчу)."""
        h = self.input_proj(x)
        h = h.unsqueeze(1)
        b = h.size(0)
        h0 = torch.zeros(
            self.num_lstm_layers,
            b,
            self.lstm_hidden,
            device=h.device,
            dtype=h.dtype,
        )
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(h, (h0, c0))
        return self.head(out[:, -1, :]).squeeze(-1)
