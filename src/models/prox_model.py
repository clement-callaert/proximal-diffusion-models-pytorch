"""
Proximal model wrapper (Fang et al., arXiv:2507.08956, Sec. 3.3).

Epsilon parameterization (default):
    f_theta(x; t, lambda) = x - sqrt(lambda) * eps_theta(x; t, lambda)

Direct prox parameterization:
    f_theta = net output; eps recovered as (x - f) / sqrt(lambda).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from src.sde.vp_sde import expand_time_like

ProxParam = Literal["epsilon", "prox"]


class ProxModel(nn.Module):
    """
    Wraps a (t, lambda)-conditioned epsilon or prox backbone.

    Trainers call compute_epsilon(y, t, lamb) with target eps_pm ~ N(0, I).
    Samplers call forward(y, t, lamb) for the MAP proximal step.
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        model_type: ProxParam = "epsilon",
    ) -> None:
        super().__init__()
        if model_type not in ("epsilon", "prox"):
            raise ValueError("model_type must be 'epsilon' or 'prox'")
        self.net = net
        self.model_type = model_type

    def _sqrt_lamb(self, x: torch.Tensor, lamb: torch.Tensor) -> torch.Tensor:
        return expand_time_like(x, torch.sqrt(lamb.clamp_min(0.0)))

    def forward(self, x: torch.Tensor, t: torch.Tensor, lamb: torch.Tensor) -> torch.Tensor:
        """Approximate prox^{-lambda ln p_t}(x)."""
        if self.model_type == "prox":
            return self.net(x, t, lamb)
        sqrt_l = self._sqrt_lamb(x, lamb)
        return x - self.net(x, t, lamb) * sqrt_l

    def compute_epsilon(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        lamb: torch.Tensor,
    ) -> torch.Tensor:
        """Residual eps in Eq. (9); target is standard Gaussian."""
        if self.model_type == "epsilon":
            return self.net(x, t, lamb)
        prox = self.net(x, t, lamb)
        sqrt_l = self._sqrt_lamb(x, lamb).clamp_min(1e-12)
        return (x - prox) / sqrt_l


def build_prox_model(
    cfg: Any,
    *,
    model_type: ProxParam = "epsilon",
) -> ProxModel:
    """Hydra helper: cfg.model + cfg.data -> ProxModel."""
    from src.models.factory import build_eps_backbone

    net = build_eps_backbone(cfg, with_lambda=True)
    return ProxModel(net, model_type=model_type)
