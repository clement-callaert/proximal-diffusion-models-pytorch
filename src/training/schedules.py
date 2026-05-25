"""
ProxDM training schedule: L1 warm-up then proximal matching with decaying zeta.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from src.training.optim import score_lr_at_iter


class PMLossSchedule:
    """
    Iteration schedule for prox training (notebook PMLossSchedule).

    it <= l1_iters:  L1 loss, l1_lr
    else:           prox_match with gamma = gamma_start * gamma_decay^stage, pm_lr
    """

    def __init__(
        self,
        total_iters: int,
        l1_iters: int,
        *,
        gamma_start: float = 1.0,
        gamma_decay: float = 0.5,
        gamma_stages: int = 2,
        l1_lr: float = 1e-4,
        pm_lr: float = 1e-5,
    ) -> None:
        if l1_iters < 0 or total_iters < l1_iters:
            raise ValueError("invalid l1_iters / total_iters")
        if gamma_stages < 1:
            raise ValueError("gamma_stages must be >= 1")

        self.total_iters = total_iters
        self.l1_iters = l1_iters
        self.pm_iters = total_iters - l1_iters
        self.decay_every = max(1, self.pm_iters // gamma_stages)

        self.gamma_start = gamma_start
        self.gamma_decay = gamma_decay
        self.gamma_stages = gamma_stages
        self.l1_lr = l1_lr
        self.pm_lr = pm_lr

    def get(self, it: int) -> Tuple[dict[str, Any], float]:
        if it <= self.l1_iters:
            return {"type": "l1"}, self.l1_lr

        it_pm = it - self.l1_iters - 1
        decay_id = min(it_pm // self.decay_every, self.gamma_stages - 1)
        gamma = self.gamma_start * (self.gamma_decay ** decay_id)
        return {"type": "prox_match", "gamma": gamma}, self.pm_lr


def build_pm_schedule(cfg: Any) -> PMLossSchedule:
    pt = cfg.prox_training
    total_iters = int(pt.total_iters)
    l1_iters = pt.l1_iters
    if l1_iters is None:
        l1_iters = total_iters // 3
    else:
        l1_iters = int(l1_iters)
    # Short Hydra overrides (e.g. total_iters=60) may leave a large experiment l1_iters.
    l1_iters = max(1, min(l1_iters, total_iters - 1))
    return PMLossSchedule(
        total_iters=total_iters,
        l1_iters=l1_iters,
        gamma_start=float(pt.pm_gamma_start),
        gamma_decay=float(pt.pm_gamma_decay),
        gamma_stages=int(pt.pm_gamma_stages),
        l1_lr=float(pt.l1_lr),
        pm_lr=float(pt.pm_lr),
    )


def score_training_lr(cfg: Any, it: int) -> float:
    return score_lr_at_iter(
        it,
        float(cfg.score_training.lr),
        int(getattr(cfg.score_training, "warmup_iters", 0)),
    )
