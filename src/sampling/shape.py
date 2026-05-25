"""Infer sample tensor shape from Hydra data config."""

from __future__ import annotations

from typing import Any, Tuple


def sample_shape_from_cfg(cfg: Any) -> Tuple[int, ...]:
    """
    Shape suffix after batch dim for torch.randn(n, *shape).

    Images: (C, H, W).  Point clouds: (D,).
    """
    data = cfg.data
    if hasattr(data, "dim"):
        return (int(data.dim),)
    if hasattr(data, "in_ch") and hasattr(data, "image_size"):
        return (int(data.in_ch), int(data.image_size), int(data.image_size))
    if hasattr(data, "in_ch"):
        size = int(getattr(data, "image_size", 32))
        return (int(data.in_ch), size, size)
    raise ValueError("cfg.data must define dim or (in_ch, image_size)")
