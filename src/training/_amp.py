"""AMP / compile step helpers shared by training loops."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def parse_amp_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def amp_dtype_from_cfg(cfg: Any) -> torch.dtype:
    return parse_amp_dtype(getattr(cfg.hardware, "amp_dtype", "bfloat16"))


def autocast_ctx(device: torch.device, amp_dtype: torch.dtype):
    enabled = device.type == "cuda"
    return torch.autocast(
        device_type="cuda" if enabled else "cpu",
        dtype=amp_dtype,
        enabled=enabled,
    )


def maybe_cudagraph_mark_step() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
