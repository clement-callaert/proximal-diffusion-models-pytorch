"""
Build epsilon backbones from Hydra config (model + data groups).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

from src.models.mlp import TimeConditionedMLP
from src.models.unet import UNetTime, UNetTimeLambda


def _as_int_tuple(values: Sequence[int] | Any) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _data_dim(cfg: Any) -> int:
    if hasattr(cfg.data, "dim"):
        return int(cfg.data.dim)
    if hasattr(cfg.data, "in_ch"):
        return int(cfg.data.in_ch)
    raise ValueError("cfg.data must define dim (2D) or in_ch (images)")


def _is_image_data(cfg: Any) -> bool:
    return hasattr(cfg.data, "in_ch") and not hasattr(cfg.data, "dim")


def build_eps_backbone(cfg: Any, *, with_lambda: bool = False) -> nn.Module:
    """
    Raw epsilon network (no ScoreModel / ProxModel wrapper).

    cfg.model.backbone: "mlp" | "unet"
    """
    backbone = str(cfg.model.backbone)

    if backbone == "mlp":
        in_ch = _data_dim(cfg)
        return TimeConditionedMLP(
            in_ch=in_ch,
            ch=int(cfg.model.hidden_dim),
            num_blocks=int(cfg.model.num_blocks),
            with_lamb=with_lambda,
        )

    if backbone == "unet":
        if not _is_image_data(cfg):
            raise ValueError("unet backbone requires cfg.data.in_ch (image data)")
        ch = int(cfg.model.ch)
        ch_mult = _as_int_tuple(cfg.model.ch_mult)
        attn = _as_int_tuple(cfg.model.attn)
        common = dict(
            ch=ch,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=int(cfg.model.num_res_blocks),
            dropout=float(cfg.model.dropout),
            in_ch=int(cfg.data.in_ch),
            use_checkpoint=bool(getattr(cfg.model, "use_checkpoint", False)),
        )
        if with_lambda:
            return UNetTimeLambda(**common)
        return UNetTime(**common)

    raise ValueError(f"Unknown model.backbone: {backbone}")
