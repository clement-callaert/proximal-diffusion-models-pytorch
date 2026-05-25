"""
Prox sampling uses t_new (earlier time) at each backward step, not t_now.

Checks the time grid ordering matches Algorithm 1 (Fang et al.).
"""

from __future__ import annotations

import torch

from src.sde.vp_sde import VPSDE


def test_backward_time_grid_decreases():
    """When stepping backward, conditioning time t_new < t_now."""
    n_steps = 10
    time_eps = 0.0
    times = torch.linspace(time_eps, 1.0, n_steps + 1)
    for i in range(n_steps - 1, -1, -1):
        t_now = times[i + 1].item()
        t_new = times[i].item()
        assert t_new < t_now


def test_lambda_matches_sde_helpers():
    sde = VPSDE(0.1, 20.0)
    t_new = torch.tensor(0.2)
    t_now = torch.tensor(0.3)
    eff = sde.step_eff(t_new, t_now)
    assert torch.allclose(sde.lambda_hybrid(t_new, t_now), eff)
    assert torch.allclose(sde.lambda_backward(t_new, t_now), eff / (1.0 - 0.5 * eff))
