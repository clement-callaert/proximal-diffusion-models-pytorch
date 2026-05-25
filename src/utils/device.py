"""
Device resolution and CUDA performance knobs (notebook cell 1).
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch.cuda.amp import GradScaler

from src.training._amp import amp_dtype_from_cfg, parse_amp_dtype


def resolve_device(
    device: Union[str, torch.device, None] = None,
    *,
    fallback: str = "cpu",
) -> torch.device:
    """
    Parse Hydra-style device strings.

    ``cuda`` maps to the default GPU when available, else ``fallback``.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device(fallback)

    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device(fallback)
    return dev


def configure_cuda_backends(
    *,
    allow_tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
) -> None:
    """RTX-class defaults from the course notebook."""
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)


def device_from_cfg(cfg: Any) -> torch.device:
    """``cfg.hardware.device`` with CUDA fallback."""
    return resolve_device(getattr(cfg.hardware, "device", None))


def configure_hardware(cfg: Any) -> torch.device:
    """CUDA backends + resolved training device."""
    configure_cuda_backends()
    return device_from_cfg(cfg)


def amp_dtype_for_device(
    cfg: Any,
    device: Optional[torch.device] = None,
) -> torch.dtype:
    """AMP dtype from config; CPU runs in float32."""
    if device is None:
        device = device_from_cfg(cfg)
    if device.type != "cuda":
        return torch.float32
    return amp_dtype_from_cfg(cfg)


def build_grad_scaler(
    device: torch.device,
    *,
    enabled: Optional[bool] = None,
) -> Optional[GradScaler]:
    """FP16/BF16 training scaler; ``None`` on CPU."""
    if enabled is None:
        enabled = device.type == "cuda"
    if not enabled:
        return None
    return GradScaler()


def describe_hardware(cfg: Any) -> str:
    """One-line summary for logs."""
    device = device_from_cfg(cfg)
    parts = [f"device={device}"]
    if device.type == "cuda":
        parts.append(f"gpu={torch.cuda.get_device_name(device)}")
    parts.append(f"amp={parse_amp_dtype(getattr(cfg.hardware, 'amp_dtype', 'bfloat16'))}")
    return "  ".join(parts)
