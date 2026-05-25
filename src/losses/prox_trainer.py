"""
ProxDM training objective (proximal matching + L1 warm-up).

Matches notebook ProxTrainerVP / sample_t_lamb, using VPSDE from src.sde.vp_sde.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.proximal_matching import ProximalMatchingLoss
from src.sde.vp_sde import VPSDE, build_vp_sde, expand_time_like

Discretization = Literal["hybrid", "backward"]
LossOn = Literal["epsilon", "prox"]


def sample_t_lamb(
    batch_size: int,
    sde: VPSDE,
    device: torch.device,
    candidates: Sequence[int],
    *,
    discretization: Discretization = "hybrid",
    weights_type: Literal["log", "uniform"] = "log",
    n_steps_candidates: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Sample (t, lambda) for one prox training step.

    1) Pick NFE (number of reverse steps) from candidates with log10 weights.
    2) Align t to the uniform grid with step size delta_t = 1 / NFE.
    3) lambda = eff (hybrid) or eff/(1-eff/2) (backward), eff = ∫_t^{t+} beta.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if n_steps_candidates is None:
        n_steps_candidates = torch.tensor(
            list(candidates),
            device=device,
            dtype=torch.float32,
        )
    else:
        n_steps_candidates = n_steps_candidates.to(device=device, dtype=torch.float32)

    if weights is None:
        if weights_type == "log":
            weights = torch.log10(n_steps_candidates.clamp_min(1.0))
        else:
            weights = torch.ones_like(n_steps_candidates)

    idx = torch.multinomial(weights, batch_size, replacement=True)
    n_steps = n_steps_candidates[idx]
    delta_t = 1.0 / n_steps

    t = torch.rand(batch_size, device=device)
    t = torch.floor(t / delta_t) * delta_t
    t = torch.clamp(t, max=1.0 - delta_t)
    t_plus = t + delta_t

    if discretization == "hybrid":
        lamb = sde.lambda_hybrid(t, t_plus)
    elif discretization == "backward":
        lamb = sde.lambda_backward(t, t_plus)
    else:
        raise ValueError(f"Unknown discretization: {discretization}")

    return t, lamb, {"step_num": n_steps}


def _per_sample_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean loss over all dims except batch."""
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1)


class ProxTrainerVP(nn.Module):
    """
    Train prox network on corrupted targets y = x_t + sqrt(lambda) * eps_pm.

    Phase 1: L1 on epsilon (or prox) target.
    Phase 2: proximal matching loss with gamma = zeta.
    """

    def __init__(
        self,
        model: nn.Module,
        sde: VPSDE,
        *,
        loss_on: LossOn = "epsilon",
    ) -> None:
        super().__init__()
        self.model = model
        self.sde = sde
        self.loss_on = loss_on
        self.pm_loss = ProximalMatchingLoss()

    def forward(
        self,
        x0: torch.Tensor,
        loss_params: dict[str, Any],
        candidates: Sequence[int],
        discretization: Discretization = "hybrid",
    ) -> torch.Tensor:
        """
        Args:
            x0: clean data (B, C, H, W) or (B, D)
            loss_params: {"type": "l1"} or {"type": "prox_match", "gamma": zeta}
            candidates: NFE list from cfg.prox_training.nfe_candidates

        Returns:
            Per-sample loss (B,)
        """
        device = x0.device
        batch_size = x0.shape[0]

        n_steps_candidates = torch.tensor(
            list(candidates),
            device=device,
            dtype=torch.float32,
        )
        weights = torch.log10(n_steps_candidates.clamp_min(1.0))

        t, lamb, _meta = sample_t_lamb(
            batch_size,
            self.sde,
            device,
            candidates,
            discretization=discretization,
            n_steps_candidates=n_steps_candidates,
            weights=weights,
        )

        # Forward VP marginal: x_t
        mean_coeff = self.sde.mean_coeff(t, x0)
        std_coeff = self.sde.std_coeff(t, x0)
        eps_t = torch.randn_like(x0)
        x_t = mean_coeff * x0 + std_coeff * eps_t

        # Proximal corruption: y = x_t + sqrt(lambda) * eps_pm
        sqrt_lamb = torch.sqrt(lamb.clamp_min(0.0))
        sqrt_lamb = expand_time_like(x0, sqrt_lamb)
        eps_pm = torch.randn_like(x0)
        y = x_t + sqrt_lamb * eps_pm

        if self.loss_on == "prox":
            pred = self.model(y, t, lamb)
            target = x_t
        else:
            pred = self.model.compute_epsilon(y, t, lamb)
            target = eps_pm

        loss_type = loss_params["type"]
        if loss_type == "l1":
            return _per_sample_mean(F.l1_loss(pred, target, reduction="none"))
        if loss_type == "prox_match":
            gamma = float(loss_params["gamma"])
            return self.pm_loss(pred, target, gamma)
        if loss_type == "mse":
            return _per_sample_mean(F.mse_loss(pred, target, reduction="none"))

        raise ValueError(f"Unknown loss type: {loss_type}")


def build_prox_trainer(
    cfg: Any,
    model: nn.Module,
    *,
    loss_on: LossOn = "epsilon",
    sde: Optional[VPSDE] = None,
) -> ProxTrainerVP:
    """Hydra helper."""
    if sde is None:
        sde = build_vp_sde(cfg)
    return ProxTrainerVP(model, sde, loss_on=loss_on)
