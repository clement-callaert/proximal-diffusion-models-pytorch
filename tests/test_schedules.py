"""Tests for training schedules."""

from __future__ import annotations

from types import SimpleNamespace

from src.training.schedules import PMLossSchedule, build_pm_schedule, score_training_lr


def test_pm_schedule_l1_then_pm():
    sched = PMLossSchedule(
        total_iters=300,
        l1_iters=100,
        gamma_start=1.0,
        gamma_decay=0.5,
        gamma_stages=2,
        l1_lr=1e-4,
        pm_lr=1e-5,
    )
    p0, lr0 = sched.get(50)
    assert p0 == {"type": "l1"}
    assert lr0 == 1e-4

    p1, lr1 = sched.get(150)
    assert p1["type"] == "prox_match"
    assert p1["gamma"] == 1.0
    assert lr1 == 1e-5

    p2, _ = sched.get(250)
    assert p2["gamma"] == 0.5


def test_build_pm_schedule_null_l1():
    cfg = SimpleNamespace(
        prox_training=SimpleNamespace(
            total_iters=90,
            l1_iters=None,
            pm_gamma_start=1.0,
            pm_gamma_decay=0.5,
            pm_gamma_stages=2,
            l1_lr=1e-4,
            pm_lr=1e-5,
        )
    )
    sched = build_pm_schedule(cfg)
    assert sched.l1_iters == 30


def test_score_warmup_lr():
    cfg = SimpleNamespace(
        score_training=SimpleNamespace(lr=0.2, warmup_iters=10),
    )
    assert score_training_lr(cfg, 1) == 0.02
    assert score_training_lr(cfg, 10) == 0.2
    assert score_training_lr(cfg, 100) == 0.2
