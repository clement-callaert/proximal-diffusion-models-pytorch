"""
ProxDM reverse sampling (Algorithm 1, Fang et al. arXiv:2507.08956).

Hybrid (PDA-hybrid): forward noise + proximal backward step.
Backward (PDA): implicit backward discretization only.

Uses ProxModel.forward(y, t, lambda) = prox^{-lambda ln p_t}(y).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn

from src.sde.vp_sde import VPSDE, build_vp_sde
from src.sampling.shape import sample_shape_from_cfg

Discretization = Literal["hybrid", "backward"]


def _lambda_for_step(
    sde: VPSDE,
    t_new: torch.Tensor,
    t_now: torch.Tensor,
    discretization: Discretization,
) -> torch.Tensor:
    """lambda from eff = integral_{t_new}^{t_now} beta(s) ds (Alg. 1)."""
    if discretization == "hybrid":
        return sde.lambda_hybrid(t_new, t_now)
    if discretization == "backward":
        return sde.lambda_backward(t_new, t_now)
    raise ValueError(f"Unknown discretization: {discretization}")


@torch.no_grad()
def prox_sample(
    model: nn.Module,
    sde: VPSDE,
    n_samples: int,
    n_steps: int,
    sample_shape: Tuple[int, ...],
    *,
    device: torch.device,
    discretization: Discretization = "hybrid",
    time_eps: float = 0.0,
) -> torch.Tensor:
    """
    Generate samples with NFE = n_steps proximal steps.

    Grid: t_0 = time_eps < ... < t_N = 1, integrate backward in time.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    x = torch.randn(n_samples, *sample_shape, device=device)
    times = torch.linspace(time_eps, 1.0, n_steps + 1, device=device)

    for i in range(n_steps - 1, -1, -1):
        t_now = times[i + 1]
        t_new = times[i]
        eff = sde.step_eff(t_new, t_now)
        lamb = _lambda_for_step(sde, t_new, t_now, discretization)

        if discretization == "hybrid":
            # y = (1 + eff/2) x + sqrt(eff) z
            y = (1.0 + 0.5 * eff) * x + torch.sqrt(eff) * torch.randn_like(x)
        else:
            # backward: y = (x + sqrt(eff) z) / (1 - eff/2)
            y = (x + torch.sqrt(eff) * torch.randn_like(x)) / (1.0 - 0.5 * eff)

        vec_t = torch.full((n_samples,), t_new, device=device, dtype=times.dtype)
        vec_l = torch.full((n_samples,), lamb, device=device, dtype=times.dtype)
        x = model(y, vec_t, vec_l)

    return x


@torch.no_grad()
def prox_sample_hybrid(
    model: nn.Module,
    sde: VPSDE,
    n_samples: int,
    n_steps: int,
    sample_shape: Tuple[int, ...],
    *,
    device: torch.device,
    time_eps: float = 0.0,
) -> torch.Tensor:
    return prox_sample(
        model,
        sde,
        n_samples,
        n_steps,
        sample_shape,
        device=device,
        discretization="hybrid",
        time_eps=time_eps,
    )


@torch.no_grad()
def prox_sample_backward(
    model: nn.Module,
    sde: VPSDE,
    n_samples: int,
    n_steps: int,
    sample_shape: Tuple[int, ...],
    *,
    device: torch.device,
    time_eps: float = 0.0,
) -> torch.Tensor:
    return prox_sample(
        model,
        sde,
        n_samples,
        n_steps,
        sample_shape,
        device=device,
        discretization="backward",
        time_eps=time_eps,
    )


def sample_prox(
    model: nn.Module,
    cfg: Any,
    n_samples: int,
    n_steps: int,
    *,
    sde: Optional[VPSDE] = None,
    device: Optional[torch.device] = None,
    discretization: Optional[Discretization] = None,
    time_eps: float = 0.0,
) -> torch.Tensor:
    """Hydra helper; discretization from cfg.sampler if omitted."""
    if sde is None:
        sde = build_vp_sde(cfg)
    if device is None:
        device = torch.device(str(cfg.hardware.device))
    if discretization is None:
        discretization = str(cfg.sampler.discretization)  # type: ignore[assignment]
    model.eval()
    shape = sample_shape_from_cfg(cfg)
    return prox_sample(
        model,
        sde,
        n_samples,
        n_steps,
        shape,
        device=device,
        discretization=discretization,
        time_eps=time_eps,
    )
