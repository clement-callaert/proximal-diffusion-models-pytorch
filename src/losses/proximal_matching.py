"""Proximal matching loss l_PM from Fang et al. (arXiv:2507.08956)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProximalMatchingLoss(nn.Module):
    """
    l_PM(pred, target; zeta) = 1 - exp(-MSE / zeta^2)  per sample.

    zeta is cfg.prox_training.pm_gamma_start with decay during training.
    Bounded in [0, 1); smoother than raw MSE for the prox phase.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        if gamma <= 0:
            raise ValueError("gamma (zeta) must be positive")

        batch = pred.shape[0]
        mse = (pred - target).pow(2).reshape(batch, -1).mean(dim=1)
        return 1.0 - torch.exp(-mse / (gamma * gamma))
