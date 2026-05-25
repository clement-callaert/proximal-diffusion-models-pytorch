"""
Score model wrapper: epsilon network -> VP-SDE score.

VP marginal: X_t = sqrt(alpha_t) X_0 + sqrt(1 - alpha_t) eps.
Tweedie / Ho et al.:  score(x, t) = -eps_theta(x, t) / sqrt(1 - alpha_t).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.sde.vp_sde import VPSDE, build_vp_sde, expand_time_like


class ScoreModel(nn.Module):
    """
    Wraps an epsilon-predicting backbone (UNetTime or TimeConditionedMLP).

    compute_epsilon(x, t): denoising target for score matching.
    forward(x, t): score s_theta(x, t) for Euler-Maruyama sampling.
    """

    def __init__(
        self,
        eps_net: nn.Module,
        sde: VPSDE,
    ) -> None:
        super().__init__()
        self.eps_net = eps_net
        self.sde = sde

    def compute_epsilon(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.eps_net(x, t)

    def _sqrt_one_minus_alpha(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """sqrt(1 - alpha_t), broadcast to x."""
        log_alpha = self.sde.log_mean_coeff(t)
        # 1 - alpha_t = -expm1(log alpha_t)  (numerically stable)
        var = (-torch.expm1(log_alpha)).clamp_min(1e-12)
        return expand_time_like(x, torch.sqrt(var))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = self.compute_epsilon(x, t)
        return -eps / self._sqrt_one_minus_alpha(x, t)


def build_score_model(
    cfg: Any,
    *,
    sde: Optional[VPSDE] = None,
) -> ScoreModel:
    """Hydra helper: cfg.model + cfg.data -> ScoreModel."""
    from src.models.factory import build_eps_backbone

    if sde is None:
        sde = build_vp_sde(cfg)
    eps_net = build_eps_backbone(cfg, with_lambda=False)
    return ScoreModel(eps_net, sde)
