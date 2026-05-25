"""Smoke tests for one training step."""

from __future__ import annotations

import torch
from types import SimpleNamespace

from src.losses.prox_trainer import build_prox_trainer
from src.losses.score_trainer import build_score_trainer
from src.models.prox_model import build_prox_model
from src.models.score_model import build_score_model
from src.training.ema import ema_update, clone_ema_model
from src.training.prox_loop import train_prox_step
from src.training.schedules import build_pm_schedule
from src.training.score_loop import train_score_step


def _cfg():
    return SimpleNamespace(
        hardware=SimpleNamespace(
            device="cpu",
            grad_clip=1.0,
            grad_accum_steps=1,
            amp_dtype="float32",
            use_fused_adamw=False,
        ),
        sde=SimpleNamespace(beta_min=0.1, beta_max=20.0),
        data=SimpleNamespace(dim=2),
        model=SimpleNamespace(backbone="mlp", hidden_dim=32, num_blocks=2),
        score_training=SimpleNamespace(lr=1e-3, warmup_iters=0, ema_decay=0.99),
        prox_training=SimpleNamespace(
            discretization="hybrid",
            total_iters=100,
            l1_iters=30,
            l1_lr=1e-3,
            pm_lr=1e-4,
            pm_gamma_start=1.0,
            pm_gamma_decay=0.5,
            pm_gamma_stages=2,
            nfe_candidates=[5, 10],
            ema_decay=0.99,
        ),
        sampler=SimpleNamespace(discretization="hybrid"),
    )


def test_train_score_step():
    cfg = _cfg()
    model = build_score_model(cfg)
    trainer = build_score_trainer(cfg, model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = clone_ema_model(model)

    def batch():
        return torch.randn(8, 2)

    loss = train_score_step(
        model,
        trainer,
        opt,
        batch,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        ema_model=ema,
    )
    assert loss >= 0
    ema_update(model, ema, 0.99)


def test_train_prox_step_l1_and_pm():
    cfg = _cfg()
    model = build_prox_model(cfg)
    trainer = build_prox_trainer(cfg, model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_pm_schedule(cfg)

    def batch():
        return torch.randn(8, 2)

    p_l1, lr_l1 = sched.get(1)
    loss_l1 = train_prox_step(
        model,
        trainer,
        opt,
        batch,
        p_l1,
        cfg.prox_training.nfe_candidates,
        lr=lr_l1,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
    )
    p_pm, lr_pm = sched.get(50)
    loss_pm = train_prox_step(
        model,
        trainer,
        opt,
        batch,
        p_pm,
        cfg.prox_training.nfe_candidates,
        lr=lr_pm,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
    )
    assert loss_l1 >= 0 and loss_pm >= 0
