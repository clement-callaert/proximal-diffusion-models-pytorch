"""
Hydra-facing data factory: MNIST images or 2D point clouds.
"""

from __future__ import annotations

from typing import Any, Union

from src.data.mnist import CyclicDataLoader, build_mnist_datamodule
from src.data.point_cloud import PointCloudCyclicLoader, build_point_cloud_datamodule

BatchStream = Union[CyclicDataLoader, PointCloudCyclicLoader]


def build_datamodule(cfg: Any) -> BatchStream:
    """Return an infinite training batch stream for the configured experiment."""
    name = str(cfg.data.name)
    if name == "mnist":
        return build_mnist_datamodule(cfg)
    if name in ("dino", "batman"):
        return build_point_cloud_datamodule(cfg)
    raise ValueError(f"Unsupported cfg.data.name={name!r}")
