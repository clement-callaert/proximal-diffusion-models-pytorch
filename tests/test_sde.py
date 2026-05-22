"""Tests for VP-SDE math and device placement."""

from __future__ import annotations

import torch

from src.sde.vp_sde import SDE, VPSDE, build_vp_sde, expand_time_like
from types import SimpleNamespace


def test_beta_integral_matches_trapezoid():
    sde = VPSDE(0.1, 20.0)
    a = torch.tensor([0.0, 0.2])
    b = torch.tensor([0.5, 0.9])
    trap = (sde.beta(a) + sde.beta(b)) * (b - a) / 2
    exact = sde.beta_integral(a, b)
    assert torch.allclose(trap, exact)


def test_alpha_at_zero_is_one():
    sde = VPSDE(0.1, 20.0)
    assert torch.allclose(sde.alpha(0.0), torch.tensor(1.0))


def test_marginal_std_identity():
    sde = VPSDE(0.1, 20.0)
    t = torch.tensor([0.3, 0.7])
    alpha = sde.alpha(t)
    sigma = sde.sigma(t)
    assert torch.allclose(sigma**2, 1.0 - alpha)


def test_expand_time_like_4d():
    x = torch.zeros(8, 1, 32, 32)
    c = torch.ones(8)
    assert expand_time_like(x, c).shape == (8, 1, 1, 1)


def test_cuda_matches_cpu():
    if not torch.cuda.is_available():
        return
    sde_cpu = VPSDE(0.1, 20.0)
    sde_gpu = VPSDE(0.1, 20.0).cuda()
    t = torch.linspace(0.1, 0.9, 16).cuda()
    assert torch.allclose(sde_cpu.alpha(t.cpu()), sde_gpu.alpha(t).cpu())


def test_build_vp_sde_from_cfg():
    cfg = SimpleNamespace(sde=SimpleNamespace(beta_min=0.1, beta_max=20.0))
    sde = build_vp_sde(cfg)
    assert isinstance(sde, VPSDE)
    assert torch.isclose(sde.beta_min, torch.tensor(0.1))


def test_sde_alias():
    assert SDE is VPSDE
