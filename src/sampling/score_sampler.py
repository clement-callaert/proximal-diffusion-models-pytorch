"""
Score-based reverse sampling: forward Euler–Maruyama on the VP-SDE.

Uses ScoreModel.forward(x, t) = score (not epsilon).
Discretization matches the notebook score_sample_euler_maruyama_eps.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from src.sde.vp_sde import VPSDE, build_vp_sde
from src.sampling.shape import sample_shape_from_cfg


@torch.no_grad()
def score_sample_euler_maruyama(
    model: nn.Module,
    sde: VPSDE,
    n_samples: int,
    n_steps: int,
    sample_shape: Tuple[int, ...],
    *,
    device: torch.device,
    time_eps: float = 1e-3,
) -> torch.Tensor:
    """
    Reverse VP-SDE from t=1 toward t=time_eps in n_steps.

    Update (notebook, forward EM):
        eff = beta(t) * dt
        x <- (1 + eff/2) x + eff * score(x,t) + sqrt(eff) * z,  z=0 on last step
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if time_eps <= 0 or time_eps >= 1:
        raise ValueError("time_eps must be in (0, 1)")

    x = torch.randn(n_samples, *sample_shape, device=device)
    times = torch.linspace(1.0, time_eps, n_steps, device=device)

    for i in range(n_steps):
        t_now = times[i]
        if i == n_steps - 1:
            dt = torch.tensor(time_eps, device=device, dtype=times.dtype)
        else:
            dt = (1.0 - time_eps) / (n_steps - 1)
            dt = torch.tensor(dt, device=device, dtype=times.dtype)

        eff = sde.beta(t_now) * dt
        vec_t = torch.full((n_samples,), t_now, device=device, dtype=times.dtype)
        score = model(x, vec_t)

        if i == n_steps - 1:
            noise = 0.0
        else:
            noise = torch.sqrt(eff) * torch.randn_like(x)

        x = (1.0 + 0.5 * eff) * x + eff * score + noise

    return x


def sample_score(
    model: nn.Module,
    cfg: Any,
    n_samples: int,
    n_steps: int,
    *,
    sde: Optional[VPSDE] = None,
    device: Optional[torch.device] = None,
    time_eps: float = 1e-3,
) -> torch.Tensor:
    """Hydra helper."""
    if sde is None:
        sde = build_vp_sde(cfg)
    if device is None:
        device = torch.device(str(cfg.hardware.device))
    model.eval()
    shape = sample_shape_from_cfg(cfg)
    return score_sample_euler_maruyama(
        model,
        sde,
        n_samples,
        n_steps,
        shape,
        device=device,
        time_eps=time_eps,
    )
