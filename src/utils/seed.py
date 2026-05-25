"""
Reproducible RNG state (notebook: seed=0 for random / NumPy / PyTorch).
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Optional

import numpy as np
import torch


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
) -> int:
    """
    Seed Python, NumPy, and PyTorch (including all CUDA devices).

    ``deterministic=True`` trades speed for stricter reproducibility on GPU.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # May error on some ops; caller opts in explicitly.
            torch.use_deterministic_algorithms(True, warn_only=True)

    return seed


def seed_from_cfg(cfg: Any, *, deterministic: bool = False) -> int:
    """Read ``cfg.seed`` and apply ``set_seed``."""
    return set_seed(int(cfg.seed), deterministic=deterministic)


def worker_seed_fn(base_seed: int) -> Callable[[int], None]:
    """DataLoader ``worker_init_fn`` with per-worker derived seeds."""
    def init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return init_fn
