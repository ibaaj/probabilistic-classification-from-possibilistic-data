from __future__ import annotations
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from klbox.kl_types import FloatArray


class TorchLinearSoftmax(nn.Module):
    def __init__(self, d: int, C: int, seed: int = 0):
        super().__init__()
        self.linear = nn.Linear(int(d), int(C), bias=True, dtype=torch.float64)

        g = torch.Generator()
        g.manual_seed(int(seed))
        with torch.no_grad():
            self.linear.weight.copy_(0.01 * torch.randn((int(C), int(d)), generator=g, dtype=torch.float64))
            self.linear.bias.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

    @torch.no_grad()
    def predict_proba(self, X: FloatArray) -> FloatArray:
        X_np = np.asarray(X, dtype=np.float64)
        X_t = torch.from_numpy(X_np).to(device="cpu", dtype=torch.float64)
        logits = self.forward(X_t)
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy().astype(np.float64)


class TorchMLPSoftmax(nn.Module):
    """Two-layer MLP: d -> hidden_dim (ReLU + Dropout) -> C, float64."""

    def __init__(
        self,
        d: int,
        C: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(d), int(hidden_dim), dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(C), dtype=torch.float64),
        )
        g = torch.Generator()
        g.manual_seed(int(seed))
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    module.weight.copy_(
                        torch.randn(module.weight.shape, generator=g, dtype=torch.float64) * 0.01
                    )
                    module.bias.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    @torch.no_grad()
    def predict_proba(self, X: FloatArray) -> FloatArray:
        X_t          = torch.from_numpy(np.asarray(X, dtype=np.float64))
        was_training = self.training
        self.eval()
        probs = torch.softmax(self.forward(X_t), dim=1)
        if was_training:
            self.train()
        return probs.cpu().numpy().astype(np.float64)


AnyHead = Union[TorchLinearSoftmax, TorchMLPSoftmax]

def build_head(
    *,
    head: str,
    d: int,
    C: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    seed: int = 0,
) -> AnyHead:
    h = str(head).lower()
    if h == "linear":
        return TorchLinearSoftmax(d=d, C=C, seed=seed)
    if h == "mlp":
        return TorchMLPSoftmax(d=d, C=C, hidden_dim=hidden_dim, dropout=dropout, seed=seed)
    raise ValueError(f"Unknown head {head!r}. Choose 'linear' or 'mlp'.")

