"""
Fréchet Inception Distance for MNIST (32×32, [-1, 1] grayscale).

Matches the notebook: Inception features, RGB replication, map to [0, 1].
Paper Fig. 2 reports FID vs NFE on MNIST.
"""

from __future__ import annotations

from typing import Optional

import torch


def mnist_to_fid_rgb(images: torch.Tensor) -> torch.Tensor:
    """
    Convert model outputs to torchmetrics FID input.

    Args:
        images: (N, 1, H, W) in [-1, 1].

    Returns:
        (N, 3, H, W) in [0, 1].
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"Expected (N, 1, H, W), got {tuple(images.shape)}")
    rgb = images.repeat(1, 3, 1, 1)
    return ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.inference_mode()
def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
) -> float:
    """
    FID between two sets of MNIST-like images (same preprocessing for both).

    Uses Inception-v3 pool features (2048-d) with ``normalize=True`` ([0, 1] floats).
    """
    if real_images.shape[1:] != fake_images.shape[1:]:
        raise ValueError(
            f"Spatial/channel shape mismatch: real {tuple(real_images.shape)} "
            f"vs fake {tuple(fake_images.shape)}"
        )

    if device is None:
        device = real_images.device

    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real_rgb = mnist_to_fid_rgb(real_images)
    fake_rgb = mnist_to_fid_rgb(fake_images)

    for start in range(0, real_rgb.shape[0], batch_size):
        fid.update(real_rgb[start : start + batch_size].to(device), real=True)
    for start in range(0, fake_rgb.shape[0], batch_size):
        fid.update(fake_rgb[start : start + batch_size].to(device), real=False)

    return float(fid.compute().item())
