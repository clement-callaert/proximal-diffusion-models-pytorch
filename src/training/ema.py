"""
Exponential moving average of model weights (inference checkpoints).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


def ema_update(
    model: nn.Module,
    ema_model: nn.Module,
    decay: float = 0.9999,
) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def clone_ema_model(model: nn.Module) -> nn.Module:
    """Deep copy in eval mode for EMA tracking."""
    ema = copy.deepcopy(model)
    ema.eval()
    return ema


def maybe_init_ema(
    model: nn.Module,
    ema_model: Optional[nn.Module],
) -> nn.Module:
    if ema_model is None:
        return clone_ema_model(model)
    return ema_model
