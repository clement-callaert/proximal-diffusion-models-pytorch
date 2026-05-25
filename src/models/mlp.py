"""
Time-conditioned MLP for low-dimensional point clouds (2D experiments).

Residual blocks with (t [, lambda]) embeddings (notebook RobustMLP).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.embeddings import ScalarEmbedding, Swish


class TimeConditionedBlock(nn.Module):
    def __init__(self, dim: int, emb_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(Swish(), nn.Linear(emb_dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x + self.mlp(emb)
        return x + self.net(self.norm(h))


class TimeConditionedMLP(nn.Module):
    """
    MLP epsilon network for 2D data.

    with_lamb=False: (x, t) -> eps  — score matching
    with_lamb=True:  (x, t, lambda) -> eps  — proximal matching (Eq. 9)
    """

    def __init__(
        self,
        in_ch: int = 2,
        ch: int = 512,
        num_blocks: int = 4,
        *,
        with_lamb: bool = False,
    ) -> None:
        super().__init__()
        self.with_lamb = with_lamb
        emb_dim = ch * 2

        if with_lamb:
            half = emb_dim // 2
            self.time_emb = ScalarEmbedding(ch, half)
            self.lamb_emb = ScalarEmbedding(ch, half)
        else:
            self.time_emb = ScalarEmbedding(ch, emb_dim)
            self.lamb_emb = None

        self.in_proj = nn.Linear(in_ch, ch)
        self.blocks = nn.ModuleList(
            [TimeConditionedBlock(ch, emb_dim) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Linear(ch, in_ch)

    def _embedding(self, t: torch.Tensor, lamb: torch.Tensor | None) -> torch.Tensor:
        if self.with_lamb:
            if lamb is None:
                raise ValueError("lamb required when with_lamb=True")
            return torch.cat([self.time_emb(t), self.lamb_emb(lamb)], dim=1)  # type: ignore[union-attr]
        return self.time_emb(t)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        lamb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self._embedding(t, lamb)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out_proj(h)
