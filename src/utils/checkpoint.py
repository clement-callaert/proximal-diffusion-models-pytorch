"""
Training checkpoints: model, EMA, optimizer (notebook ``save_ckpt``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def checkpoint_path(ckpt_dir: Union[str, Path], iteration: int) -> Path:
    """Standard filename ``ckpt_{it}.pt``."""
    return Path(ckpt_dir) / f"ckpt_{int(iteration)}.pt"


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save training state (notebook-compatible keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "it": int(iteration),
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "opt": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    use_ema_for_model: bool = False,
) -> int:
    """
    Restore weights (and optionally optimizer).

    Returns saved iteration ``it``.

    If ``use_ema_for_model``, loads EMA weights into ``model`` (typical for sampling).
    """
    path = Path(path)
    state = torch.load(path, map_location=map_location, weights_only=False)

    if use_ema_for_model:
        model.load_state_dict(state["ema"], strict=strict)
    else:
        model.load_state_dict(state["model"], strict=strict)

    if ema_model is not None and "ema" in state:
        ema_model.load_state_dict(state["ema"], strict=strict)

    if optimizer is not None and "opt" in state:
        optimizer.load_state_dict(state["opt"])

    return int(state.get("it", 0))


def load_ema_weights(
    path: Union[str, Path],
    model: nn.Module,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> int:
    """Load only EMA weights into ``model`` for inference."""
    return load_checkpoint(
        path,
        model,
        map_location=map_location,
        strict=strict,
        use_ema_for_model=True,
    )


_CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def list_checkpoints(ckpt_dir: Union[str, Path]) -> list[Path]:
    """All ``ckpt_*.pt`` files in a directory, sorted by iteration."""
    root = Path(ckpt_dir)
    if not root.is_dir():
        return []
    paths = [p for p in root.iterdir() if _CKPT_RE.match(p.name)]
    return sorted(paths, key=lambda p: int(_CKPT_RE.match(p.name).group(1)))  # type: ignore[union-attr]


def latest_checkpoint(ckpt_dir: Union[str, Path]) -> Optional[Path]:
    """Highest-iteration checkpoint, or ``None``."""
    paths = list_checkpoints(ckpt_dir)
    return paths[-1] if paths else None


def save_if_due(
    iteration: int,
    save_every: int,
    total_iters: int,
    ckpt_dir: Union[str, Path],
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Save when ``it % save_every == 0`` or at the final step."""
    if save_every <= 0:
        return None
    if iteration % save_every != 0 and iteration != total_iters:
        return None
    path = checkpoint_path(ckpt_dir, iteration)
    return save_checkpoint(
        path, model, ema_model, optimizer, iteration, extra=extra
    )
