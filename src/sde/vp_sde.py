"""
Variance-preserving (VP) SDE from Song et al. / ProxDM (Fang et al., arXiv:2507.08956).

Forward process (t in [0, 1]):
    dX_t = -1/2 * beta(t) * X_t dt + sqrt(beta(t)) dW_t

Linear noise schedule:
    beta(t) = beta_min + (beta_max - beta_min) * t

Marginal at time t (X_0 ~ data, eps ~ N(0, I)):
    X_t = sqrt(alpha_t) * X_0 + sqrt(1 - alpha_t) * eps
    alpha_t = exp(-∫_0^t beta(s) ds)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch.nn as nn

TimeInput = Union[float, torch.Tensor]


class VPSDE(nn.Module):
    """VP-SDE with linear beta; all ops are vectorized for batch + CUDA."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        super().__init__()
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")
        if beta_min > beta_max:
            raise ValueError("beta_min must be <= beta_max")

        # Buffers move with .to(device) and work in mixed precision training.
        self.register_buffer("beta_min", torch.tensor(float(beta_min)))
        self.register_buffer("beta_max", torch.tensor(float(beta_max)))
        self.register_buffer(
            "beta_slope",
            torch.tensor(float(beta_max - beta_min)),
        )

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _as_tensor(self, t: TimeInput, *, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cast time to tensor on the right device/dtype."""
        if isinstance(t, torch.Tensor):
            out = t
        else:
            out = torch.tensor(t, dtype=torch.float32)
        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)
        return out

    def beta(self, t: TimeInput) -> torch.Tensor:
        """Instantaneous noise rate beta(t)."""
        t = self._as_tensor(t)
        return self.beta_min + self.beta_slope * t

    def beta_integral(self, a: TimeInput, b: TimeInput) -> torch.Tensor:
        """
        ∫_a^b beta(s) ds  (exact for linear beta).

        Equal to the trapezoid rule used in the notebook:
        (beta(a) + beta(b)) * (b - a) / 2.
        """
        a = self._as_tensor(a)
        b = self._as_tensor(b)
        return self.beta_min * (b - a) + 0.5 * self.beta_slope * (b * b - a * a)

    def log_mean_coeff(self, t: TimeInput) -> torch.Tensor:
        """log(alpha_t) = -∫_0^t beta(s) ds."""
        t = self._as_tensor(t)
        return -(self.beta_min * t + 0.5 * self.beta_slope * t * t)

    def alpha(self, t: TimeInput) -> torch.Tensor:
        """alpha_t = exp(-∫_0^t beta(s) ds), shape follows t."""
        return torch.exp(self.log_mean_coeff(t))

    def sigma(self, t: TimeInput) -> torch.Tensor:
        """Std of the noise part: sqrt(1 - alpha_t)."""
        log_alpha = self.log_mean_coeff(t)
        return torch.sqrt((-torch.expm1(log_alpha)).clamp_min(0.0))

    # ------------------------------------------------------------------
    # Marginal q(x_t | x_0) — used in score / prox training
    # ------------------------------------------------------------------

    def marginal_prob(
        self,
        x0: torch.Tensor,
        t: TimeInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample X_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * eps.

        Returns:
            x_t, mean_coeff (sqrt(alpha)), std (sqrt(1-alpha))
        """
        t = self._as_tensor(t, ref=x0)
        log_alpha = self.log_mean_coeff(t)
        mean_coeff = torch.exp(0.5 * log_alpha)
        std = torch.sqrt((-torch.expm1(log_alpha)).clamp_min(0.0))

        mean_coeff = expand_time_like(x0, mean_coeff)
        std = expand_time_like(x0, std)

        eps = torch.randn_like(x0)
        x_t = mean_coeff * x0 + std * eps
        return x_t, mean_coeff, std

    def mean_coeff(self, t: TimeInput, x: torch.Tensor) -> torch.Tensor:
        """sqrt(alpha_t) broadcast to x shape — for closed-form score paths."""
        t = self._as_tensor(t, ref=x)
        coeff = torch.exp(0.5 * self.log_mean_coeff(t))
        return expand_time_like(x, coeff)

    def std_coeff(self, t: TimeInput, x: torch.Tensor) -> torch.Tensor:
        """sqrt(1 - alpha_t) broadcast to x shape."""
        t = self._as_tensor(t, ref=x)
        log_alpha = self.log_mean_coeff(t)
        std = torch.sqrt((-torch.expm1(log_alpha)).clamp_min(0.0))
        return expand_time_like(x, std)

    # ------------------------------------------------------------------
    # Discrete step sizes (ProxDM Algorithm 1)
    # ------------------------------------------------------------------

    def step_eff(self, t_lo: TimeInput, t_hi: TimeInput) -> torch.Tensor:
        """Effective noise ∫_{t_lo}^{t_hi} beta(s) ds between two grid times."""
        return self.beta_integral(t_lo, t_hi)

    def lambda_hybrid(self, t_lo: TimeInput, t_hi: TimeInput) -> torch.Tensor:
        """Hybrid backward step: lambda = eff."""
        return self.step_eff(t_lo, t_hi)

    def lambda_backward(self, t_lo: TimeInput, t_hi: TimeInput) -> torch.Tensor:
        """Full backward step: lambda = eff / (1 - eff/2)."""
        eff = self.step_eff(t_lo, t_hi)
        return eff / (1.0 - 0.5 * eff)


# Notebook alias
SDE = VPSDE


def expand_time_like(x: torch.Tensor, t_coeff: torch.Tensor) -> torch.Tensor:
    """
    Broadcast per-batch time coeffs to x.

    t_coeff: (B,) -> view (B, 1, 1, 1) for images or (B, 1) for 2D.
    """
    if t_coeff.ndim != 1:
        return t_coeff

    if x.ndim == 4:
        return t_coeff.view(-1, 1, 1, 1)
    if x.ndim == 2:
        return t_coeff.view(-1, 1)
    if x.ndim == 3:
        return t_coeff.view(-1, 1, 1)
    return t_coeff.view(-1, *([1] * (x.ndim - 1)))


def build_vp_sde(cfg: Any) -> VPSDE:
    """Hydra helper: cfg.sde.beta_min / beta_max."""
    return VPSDE(
        beta_min=float(cfg.sde.beta_min),
        beta_max=float(cfg.sde.beta_max),
    )
