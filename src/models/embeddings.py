"""
Sinusoidal embeddings for continuous scalars (t, lambda).

Same construction as transformer positional encodings (Vaswani et al.)
and DDPM / score-based diffusion time conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Swish(nn.Module):
    """x * sigmoid(x) — smooth ReLU used in the notebook U-Net."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def sinusoidal_embedding(scalars: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Map batch of scalars (B,) to (B, dim) sinusoidal features.

    scalars are scaled by 1000 before frequency mixing (DDPM convention).
    """
    if dim % 2 != 0:
        raise ValueError("embedding dim must be even")

    scalars = scalars.reshape(-1).float() * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=scalars.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = scalars[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ScalarEmbedding(nn.Module):
    """sinusoidal(s) -> small MLP -> conditioning vector."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal_embedding")
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, out_dim),
            Swish(),
            nn.Linear(out_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(s, self.d_model))
