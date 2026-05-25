"""
DDPM-style U-Net backbone for images (MNIST-scale).

- UNetTime: conditions on t only (score / epsilon network).
- UNetTimeLambda: conditions on (t, lambda) (proximal network).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings import ScalarEmbedding, Swish


class DownSample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        del emb
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        del emb
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttnBlock(nn.Module):
    """Self-attention over H x W tokens (one head, full spatial)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)

        q = self.q(y).permute(0, 2, 3, 1).reshape(b, h * w, c)
        k = self.k(y).reshape(b, c, h * w)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)

        v = self.v(y).permute(0, 2, 3, 1).reshape(b, h * w, c)
        y = torch.bmm(attn, v).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + self.proj(y)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float,
        *,
        attn: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_proj = nn.Sequential(Swish(), nn.Linear(emb_dim, out_ch))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)  # type: ignore[index]

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, emb, use_reentrant=False
            )
        return self._forward(x, emb)


class UNet(nn.Module):
    """U-Net core: shared by score and prox backbones."""

    def __init__(
        self,
        ch: int,
        ch_mult: Sequence[int],
        attn: Sequence[int],
        num_res_blocks: int,
        dropout: float,
        emb_dim: int,
        in_ch: int,
        *,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if any(i >= len(ch_mult) for i in attn):
            raise ValueError("attn index out of bound for ch_mult")

        self.head = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        now_ch,
                        out_ch,
                        emb_dim,
                        dropout,
                        attn=(i in attn),
                        use_checkpoint=use_checkpoint,
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middle = nn.ModuleList(
            [
                ResBlock(
                    now_ch, now_ch, emb_dim, dropout, attn=True, use_checkpoint=use_checkpoint
                ),
                ResBlock(
                    now_ch, now_ch, emb_dim, dropout, attn=False, use_checkpoint=use_checkpoint
                ),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        chs.pop() + now_ch,
                        out_ch,
                        emb_dim,
                        dropout,
                        attn=(i in attn),
                        use_checkpoint=use_checkpoint,
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, in_ch, 3, padding=1),
        )
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)  # type: ignore[index, union-attr]
        nn.init.zeros_(self.tail[-1].bias)  # type: ignore[index, union-attr]

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, emb)
            hs.append(h)

        for layer in self.middle:
            h = layer(h, emb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, emb)

        return self.tail(h)


class UNetTime(nn.Module):
    """Predict epsilon_hat(x, t) for VP score matching."""

    def __init__(
        self,
        ch: int,
        ch_mult: Sequence[int],
        attn: Sequence[int],
        num_res_blocks: int,
        dropout: float,
        in_ch: int,
        *,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        emb_dim = ch * 4
        self.time_emb = ScalarEmbedding(ch, emb_dim)
        self.unet = UNet(
            ch,
            ch_mult,
            attn,
            num_res_blocks,
            dropout,
            emb_dim,
            in_ch,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x, self.time_emb(t))


class UNetTimeLambda(nn.Module):
    """Predict epsilon_hat(x, t, lambda) for proximal matching (Eq. 9, Fang et al.)."""

    def __init__(
        self,
        ch: int,
        ch_mult: Sequence[int],
        attn: Sequence[int],
        num_res_blocks: int,
        dropout: float,
        in_ch: int,
        *,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        emb_dim = ch * 4
        half = emb_dim // 2
        self.time_emb = ScalarEmbedding(ch, half)
        self.lamb_emb = ScalarEmbedding(ch, half)
        self.unet = UNet(
            ch,
            ch_mult,
            attn,
            num_res_blocks,
            dropout,
            emb_dim,
            in_ch,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, lamb: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.time_emb(t), self.lamb_emb(lamb)], dim=1)
        return self.unet(x, emb)
