"""
Score matching training (denoising score matching on VP-SDE).

Objective (Song et al.): predict noise eps in
    X_t = sqrt(alpha_t) * X_0 + sqrt(1 - alpha_t) * eps.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sde.vp_sde import VPSDE, build_vp_sde


def sample_t_score(
    batch_size: int,
    device: torch.device,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    """Uniform continuous t in [t_min, t_max] — standard score matching."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if t_min >= t_max:
        raise ValueError("t_min must be < t_max")

    return torch.empty(batch_size, device=device).uniform_(t_min, t_max)


def _per_sample_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all dims except batch."""
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1)


class ScoreTrainerVP(nn.Module):
    """
    VP score matching: MSE between predicted and actual noise eps_t.

    model must implement compute_epsilon(x_t, t) -> same shape as x_t.
    """

    def __init__(self, model: nn.Module, sde: VPSDE) -> None:
        super().__init__()
        self.model = model
        self.sde = sde

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: clean batch (B, C, H, W) or (B, D)

        Returns:
            Per-sample MSE loss (B,)
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = sample_t_score(batch_size, device)

        mean_coeff = self.sde.mean_coeff(t, x0)
        std_coeff = self.sde.std_coeff(t, x0)
        noise = torch.randn_like(x0)
        x_t = mean_coeff * x0 + std_coeff * noise

        pred = self.model.compute_epsilon(x_t, t)
        return _per_sample_mean(F.mse_loss(pred, noise, reduction="none"))


def build_score_trainer(
    cfg: Any,
    model: nn.Module,
    *,
    sde: Optional[VPSDE] = None,
) -> ScoreTrainerVP:
    """Hydra helper."""
    if sde is None:
        sde = build_vp_sde(cfg)
    return ScoreTrainerVP(model, sde)
