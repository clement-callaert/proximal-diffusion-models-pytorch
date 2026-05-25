"""
Optimizer helpers (notebook make_adamw / set_lr).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def score_lr_at_iter(it: int, base_lr: float, warmup_iters: int) -> float:
    """Linear warmup: lr = base_lr * min(it, warmup) / warmup."""
    if warmup_iters <= 0:
        return base_lr
    return base_lr * min(it, warmup_iters) / warmup_iters


def make_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    *,
    device: torch.device,
    use_fused: bool = True,
) -> torch.optim.AdamW:
    kwargs: dict[str, Any] = {"lr": lr}
    if device.type == "cuda" and use_fused:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(params, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(params, **kwargs)


def build_score_optimizer(cfg: Any, params: Iterable[torch.nn.Parameter]) -> torch.optim.AdamW:
    device = torch.device(str(cfg.hardware.device))
    return make_adamw(
        params,
        float(cfg.score_training.lr),
        device=device,
        use_fused=bool(getattr(cfg.hardware, "use_fused_adamw", True)),
    )


def build_prox_optimizer(cfg: Any, params: Iterable[torch.nn.Parameter]) -> torch.optim.AdamW:
    device = torch.device(str(cfg.hardware.device))
    return make_adamw(
        params,
        float(cfg.prox_training.l1_lr),
        device=device,
        use_fused=bool(getattr(cfg.hardware, "use_fused_adamw", True)),
    )
