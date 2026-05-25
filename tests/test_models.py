"""Smoke tests for model backbones and wrappers."""

from __future__ import annotations

import torch
from types import SimpleNamespace

from src.losses.prox_trainer import ProxTrainerVP
from src.losses.score_trainer import ScoreTrainerVP
from src.models.factory import build_eps_backbone
from src.models.prox_model import ProxModel, build_prox_model
from src.models.score_model import ScoreModel, build_score_model
from src.sde.vp_sde import VPSDE


def _cfg_mlp():
    return SimpleNamespace(
        sde=SimpleNamespace(beta_min=0.1, beta_max=20.0),
        data=SimpleNamespace(dim=2, name="dino"),
        model=SimpleNamespace(backbone="mlp", hidden_dim=64, num_blocks=2),
    )


def _cfg_unet():
    return SimpleNamespace(
        sde=SimpleNamespace(beta_min=0.1, beta_max=20.0),
        data=SimpleNamespace(in_ch=1, image_size=32, name="mnist"),
        model=SimpleNamespace(
            backbone="unet",
            ch=32,
            ch_mult=[1, 2],
            attn=[1],
            num_res_blocks=1,
            dropout=0.0,
            use_checkpoint=False,
        ),
    )


def test_mlp_score_forward_shapes():
    cfg = _cfg_mlp()
    model = build_score_model(cfg)
    x = torch.randn(8, 2)
    t = torch.rand(8)
    eps = model.compute_epsilon(x, t)
    score = model(x, t)
    assert eps.shape == x.shape
    assert score.shape == x.shape


def test_mlp_prox_epsilon_param():
    cfg = _cfg_mlp()
    model = build_prox_model(cfg)
    x = torch.randn(8, 2)
    t = torch.rand(8)
    lamb = torch.rand(8) * 0.1
    eps = model.compute_epsilon(x, t, lamb)
    prox = model(x, t, lamb)
    assert eps.shape == x.shape
    assert prox.shape == x.shape
    # f = x - sqrt(l)*eps  =>  prox + sqrt(l)*eps = x
    sqrt_l = torch.sqrt(lamb).view(-1, 1)
    assert torch.allclose(prox + sqrt_l * eps, x, atol=1e-5)


def test_unet_score_forward_shapes():
    cfg = _cfg_unet()
    model = build_score_model(cfg)
    x = torch.randn(4, 1, 32, 32)
    t = torch.rand(4)
    assert model.compute_epsilon(x, t).shape == x.shape
    assert model(x, t).shape == x.shape


def test_unet_prox_forward_shapes():
    cfg = _cfg_unet()
    model = build_prox_model(cfg)
    x = torch.randn(4, 1, 32, 32)
    t = torch.rand(4)
    lamb = torch.rand(4) * 0.05
    assert model.compute_epsilon(x, t, lamb).shape == x.shape
    assert model(x, t, lamb).shape == x.shape


def test_trainers_with_built_models():
    cfg = _cfg_mlp()
    sde = VPSDE(0.1, 20.0)
    score = build_score_model(cfg, sde=sde)
    prox = build_prox_model(cfg)
    x0 = torch.randn(16, 2)
    score_loss = ScoreTrainerVP(score, sde)(x0)
    prox_loss = ProxTrainerVP(prox, sde)(x0, {"type": "l1"}, [5, 10])
    assert score_loss.shape == (16,)
    assert prox_loss.shape == (16,)


def test_build_eps_backbone_lambda_flag():
    cfg = _cfg_mlp()
    score_net = build_eps_backbone(cfg, with_lambda=False)
    prox_net = build_eps_backbone(cfg, with_lambda=True)
    x = torch.randn(4, 2)
    t = torch.rand(4)
    assert score_net(x, t).shape == x.shape
    lamb = torch.rand(4)
    assert prox_net(x, t, lamb).shape == x.shape
