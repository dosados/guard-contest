"""
Простейшая полносвязная нейросеть на PyTorch для бинарной классификации.
Используется в training/main.py вместе с CatBoost, XGBoost, LightGBM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Полносвязная сеть из нескольких слоёв: Linear -> ReLU -> ... -> Linear -> выход 1."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    """Преобразует массив в тензор, заменяет NaN на 0."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(X).to(device)


def train_and_validate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_sizes: tuple[int, ...] = (128, 64),
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
    dropout: float = 0.2,
    device: str | None = None,
    verbose: int = 10,
) -> tuple[MLP, float]:
    """
    Обучает MLP и возвращает модель и PR-AUC на валидации.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    input_size = X_train.shape[1]
    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = _to_tensor(X_train, dev)
    y_t = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)).squeeze(-1).to(dev)
    X_v = _to_tensor(X_val, dev)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % verbose == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, loss={total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits_val = model(X_v)
        probs = torch.sigmoid(logits_val).cpu().numpy()
    pr_auc = average_precision_score(y_val, probs)
    return model, float(pr_auc)


def save_model(model: MLP, path: Path) -> None:
    """Сохраняет state_dict модели."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, input_size: int, hidden_sizes: tuple[int, ...] = (128, 64)) -> MLP:
    """Загружает state_dict в модель (для инференса)."""
    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
