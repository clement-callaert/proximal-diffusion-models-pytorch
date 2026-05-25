"""Tests for sample_t_lamb and prox trainer sampling."""

from __future__ import annotations

import torch

from src.losses.prox_trainer import ProxTrainerVP, sample_t_lamb
from src.losses.proximal_matching import ProximalMatchingLoss
from src.sde.vp_sde import VPSDE


def test_sample_t_lamb_shapes_and_bounds():
    sde = VPSDE(0.1, 20.0)
    device = torch.device("cpu")
    t, lamb, meta = sample_t_lamb(64, sde, device, [5, 10, 20, 50])

    assert t.shape == (64,)
    assert lamb.shape == (64,)
    assert meta["step_num"].shape == (64,)
    assert (t >= 0).all() and (t <= 1).all()
    assert (lamb > 0).all()


def test_hybrid_vs_backward_lambda_ordering():
    sde = VPSDE(0.1, 20.0)
    device = torch.device("cpu")
    torch.manual_seed(0)
    t_h, lamb_h, _ = sample_t_lamb(256, sde, device, [20], discretization="hybrid")
    torch.manual_seed(0)
    t_b, lamb_b, _ = sample_t_lamb(256, sde, device, [20], discretization="backward")

    assert torch.allclose(t_h, t_b)
    assert (lamb_b >= lamb_h).all()


def test_lambda_matches_sde_formulas():
    sde = VPSDE(0.1, 20.0)
    device = torch.device("cpu")
    t, lamb, _ = sample_t_lamb(32, sde, device, [10], discretization="hybrid")
    t_plus = t + 0.1
    expected = sde.lambda_hybrid(t, t_plus)
    assert torch.allclose(lamb, expected, atol=1e-5)


def test_proximal_matching_loss_range():
    loss_fn = ProximalMatchingLoss()
    pred = torch.randn(8, 1, 4, 4)
    target = pred + 0.1 * torch.randn_like(pred)
    out = loss_fn(pred, target, gamma=1.0)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out < 1).all()


class _MockProxModel(torch.nn.Module):
    def compute_epsilon(self, y, t, lamb):
        return torch.zeros_like(y)

    def forward(self, y, t, lamb):
        return y


def test_prox_trainer_forward_shapes():
    sde = VPSDE(0.1, 20.0)
    trainer = ProxTrainerVP(_MockProxModel(), sde)
    x0 = torch.randn(16, 1, 8, 8)

    l1 = trainer(x0, {"type": "l1"}, [5, 10])
    pm = trainer(x0, {"type": "prox_match", "gamma": 1.0}, [5, 10])

    assert l1.shape == (16,)
    assert pm.shape == (16,)


def test_prox_trainer_2d():
    sde = VPSDE(0.1, 20.0)
    trainer = ProxTrainerVP(_MockProxModel(), sde)
    x0 = torch.randn(32, 2)
    loss = trainer(x0, {"type": "l1"}, [10, 20])
    assert loss.shape == (32,)
