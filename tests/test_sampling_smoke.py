"""Smoke tests for reverse samplers."""

from __future__ import annotations

import torch
from types import SimpleNamespace

from src.models.prox_model import build_prox_model
from src.models.score_model import build_score_model
from src.sampling.prox_sampler import prox_sample, prox_sample_backward, prox_sample_hybrid
from src.sampling.score_sampler import score_sample_euler_maruyama
from src.sampling.shape import sample_shape_from_cfg
from src.sde.vp_sde import VPSDE


def _cfg_mlp():
    return SimpleNamespace(
        hardware=SimpleNamespace(device="cpu"),
        sde=SimpleNamespace(beta_min=0.1, beta_max=20.0),
        data=SimpleNamespace(dim=2),
        model=SimpleNamespace(backbone="mlp", hidden_dim=32, num_blocks=2),
        sampler=SimpleNamespace(discretization="hybrid"),
    )


def test_sample_shape_from_cfg():
    assert sample_shape_from_cfg(_cfg_mlp()) == (2,)


def test_score_sampler_shape():
    cfg = _cfg_mlp()
    model = build_score_model(cfg)
    sde = VPSDE(0.1, 20.0)
    out = score_sample_euler_maruyama(
        model, sde, 16, 5, (2,), device=torch.device("cpu"), time_eps=1e-3
    )
    assert out.shape == (16, 2)
    assert torch.isfinite(out).all()


def test_prox_hybrid_and_backward_shapes():
    cfg = _cfg_mlp()
    model = build_prox_model(cfg)
    sde = VPSDE(0.1, 20.0)
    for fn in (prox_sample_hybrid, prox_sample_backward):
        out = fn(model, sde, 8, 4, (2,), device=torch.device("cpu"))
        assert out.shape == (8, 2)
        assert torch.isfinite(out).all()


def test_prox_discretizations_differ():
    cfg = _cfg_mlp()
    model = build_prox_model(cfg)
    sde = VPSDE(0.1, 20.0)
    torch.manual_seed(0)
    h = prox_sample(
        model, sde, 4, 3, (2,), device=torch.device("cpu"), discretization="hybrid"
    )
    torch.manual_seed(0)
    b = prox_sample(
        model, sde, 4, 3, (2,), device=torch.device("cpu"), discretization="backward"
    )
    assert not torch.allclose(h, b)
