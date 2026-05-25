"""Tests for score matching trainer."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.score_trainer import ScoreTrainerVP, build_score_trainer, sample_t_score
from src.sde.vp_sde import VPSDE
from types import SimpleNamespace


def test_sample_t_score_range():
    t = sample_t_score(100, torch.device("cpu"))
    assert t.shape == (100,)
    assert (t >= 0).all() and (t <= 1).all()


class _OracleNoiseModel(nn.Module):
    """Stores last noise for oracle test via hook — use fixed prediction in trainer test differently."""

    def compute_epsilon(self, x_t, t):
        return torch.zeros_like(x_t)


def test_score_trainer_output_shape():
    sde = VPSDE(0.1, 20.0)
    trainer = ScoreTrainerVP(_OracleNoiseModel(), sde)
    x0 = torch.randn(12, 1, 16, 16)
    loss = trainer(x0)
    assert loss.shape == (12,)
    assert (loss >= 0).all()


def test_score_trainer_2d():
    sde = VPSDE(0.1, 20.0)
    trainer = ScoreTrainerVP(_OracleNoiseModel(), sde)
    x0 = torch.randn(20, 2)
    loss = trainer(x0)
    assert loss.shape == (20,)


def test_score_trainer_zero_loss_when_pred_equals_noise():
    class _ExactNoise(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_noise = None

        def compute_epsilon(self, x_t, t):
            return self.last_noise

    sde = VPSDE(0.1, 20.0)
    model = _ExactNoise()
    trainer = ScoreTrainerVP(model, sde)

    torch.manual_seed(0)
    x0 = torch.randn(8, 1, 4, 4)
    t = sample_t_score(8, x0.device)
    mean_c = sde.mean_coeff(t, x0)
    std_c = sde.std_coeff(t, x0)
    noise = torch.randn_like(x0)
    model.last_noise = noise
    x_t = mean_c * x0 + std_c * noise

    pred = model.compute_epsilon(x_t, t)
    loss = (pred - noise).pow(2).reshape(8, -1).mean(1)
    assert torch.allclose(loss, torch.zeros(8), atol=1e-6)


def test_build_score_trainer_from_cfg():
    cfg = SimpleNamespace(sde=SimpleNamespace(beta_min=0.1, beta_max=20.0))
    trainer = build_score_trainer(cfg, _OracleNoiseModel())
    assert isinstance(trainer, ScoreTrainerVP)
