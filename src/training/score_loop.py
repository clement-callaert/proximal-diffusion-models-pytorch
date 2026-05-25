"""
Score matching training loop (VP-SDE, Euler-Maruyama target model).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from src.losses.score_trainer import ScoreTrainerVP, build_score_trainer
from src.training._amp import amp_dtype_from_cfg, autocast_ctx, maybe_cudagraph_mark_step
from src.training.ema import ema_update, maybe_init_ema
from src.training.optim import build_score_optimizer, set_lr
from src.training.schedules import score_training_lr


def train_score_step(
    model: nn.Module,
    trainer: ScoreTrainerVP,
    optimizer: torch.optim.Optimizer,
    batch_fn: Callable[[], torch.Tensor],
    *,
    scaler: Optional[GradScaler] = None,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.9999,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    lr: Optional[float] = None,
) -> float:
    """
    One optimizer step with optional gradient accumulation.

    batch_fn: called grad_accum_steps times per step.
    """
    if lr is not None:
        set_lr(optimizer, lr)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_scaler = scaler is not None and device.type == "cuda"

    accum = max(1, int(grad_accum_steps))
    running = torch.zeros((), device=device)

    for _ in range(accum):
        maybe_cudagraph_mark_step()
        x = batch_fn().to(device, non_blocking=device.type == "cuda")

        with autocast_ctx(device, amp_dtype):
            loss = trainer(x).mean()

        running = running + loss.detach()
        if use_scaler:
            scaler.scale(loss / accum).backward()
        else:
            (loss / accum).backward()

    if grad_clip is not None:
        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    if ema_model is not None:
        ema_update(model, ema_model, ema_decay)

    return (running / accum).item()


def run_score_training(
    cfg: Any,
    model: nn.Module,
    trainer: ScoreTrainerVP,
    batch_fn: Callable[[], torch.Tensor],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    scaler: Optional[GradScaler] = None,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> List[float]:
    """Full score phase; returns per-iter losses."""
    device = torch.device(str(cfg.hardware.device))
    model = model.to(device)
    trainer = trainer.to(device)

    if optimizer is None:
        optimizer = build_score_optimizer(cfg, model.parameters())
    ema_model = maybe_init_ema(model, ema_model)

    amp_dtype = amp_dtype_from_cfg(cfg)
    grad_clip = getattr(cfg.hardware, "grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
    accum = int(getattr(cfg.hardware, "grad_accum_steps", 1))
    ema_decay = float(cfg.score_training.ema_decay)
    total = int(cfg.score_training.total_iters)

    if scaler is None and device.type == "cuda":
        scaler = GradScaler()

    losses: List[float] = []
    for it in range(1, total + 1):
        lr = score_training_lr(cfg, it)
        loss = train_score_step(
            model,
            trainer,
            optimizer,
            batch_fn,
            scaler=scaler,
            grad_clip=grad_clip,
            grad_accum_steps=accum,
            ema_model=ema_model,
            ema_decay=ema_decay,
            device=device,
            amp_dtype=amp_dtype,
            lr=lr,
        )
        losses.append(loss)
        if on_step is not None:
            on_step(it, loss)
    return losses


def build_score_training_bundle(cfg: Any, model: nn.Module) -> dict[str, Any]:
    """Hydra helper: trainer + optimizer + EMA."""
    device = torch.device(str(cfg.hardware.device))
    trainer = build_score_trainer(cfg, model)
    trainer = trainer.to(device)
    optimizer = build_score_optimizer(cfg, model.parameters())
    ema = maybe_init_ema(model, None)
    return {
        "trainer": trainer,
        "optimizer": optimizer,
        "ema_model": ema,
    }
