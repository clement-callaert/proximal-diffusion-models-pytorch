"""
2D point-cloud datasets for ProxDM experiments (Datasaurus dino, Batman outline).

Training draws i.i.d. minibatches with replacement from a fixed empirical cloud
(zero mean, unit std per axis), matching the course notebook.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch

DEFAULT_DINOSAURUS_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-10-13/datasaurus.csv"
)


def normalize_point_cloud(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-axis zero mean and unit variance (PyTorch std, ddof=1)."""
    mean = points.mean(dim=0, keepdim=True)
    std = points.std(dim=0, unbiased=True, keepdim=True).clamp_min(eps)
    return (points - mean) / std


def load_dino_points(csv_path: str | Path) -> torch.Tensor:
    """Load the 'dino' subset from a Datasaurus CSV file."""
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns:
        raise ValueError(f"Expected 'dataset' column in {csv_path}")
    dino = df[df["dataset"] == "dino"]
    if len(dino) == 0:
        raise ValueError(f"No rows with dataset=='dino' in {csv_path}")
    points = torch.tensor(dino[["x", "y"]].values, dtype=torch.float32)
    return normalize_point_cloud(points)


def ensure_datasaurus_csv(
    csv_path: str | Path,
    source_url: str = DEFAULT_DINOSAURUS_URL,
) -> Path:
    """Download Datasaurus CSV if missing."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urlretrieve(source_url, path)  # noqa: S310 — trusted project default URL
    return path


def sample_batman_outline(num_points: int, *, seed: Optional[int] = None) -> np.ndarray:
    """
    Parametric Batman logo outline (notebook / web formula).

    Returns (N, 2) float64 before tensor conversion; NaNs removed, subsampled to num_points.
    """
    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    rng = np.random.default_rng(seed)
    n = int(num_points * 0.6)
    x_u = rng.uniform(-7.0, 7.0, n)
    x_l = rng.uniform(-7.0, 7.0, n)
    ax_u = np.abs(x_u)
    ax_l = np.abs(x_l)
    upper = np.zeros(n)
    lower = np.zeros(n)

    upper[ax_u < 0.5] = 2.25

    m1 = (ax_u >= 0.5) & (ax_u < 0.75)
    upper[m1] = 3.0 * ax_u[m1] + 0.75

    m2 = (ax_u >= 0.75) & (ax_u < 1.0)
    upper[m2] = 9.0 - 8.0 * ax_u[m2]

    m3 = (ax_u >= 1.0) & (ax_u < 3.0)
    upper[m3] = (
        1.5
        - 0.5 * ax_u[m3]
        - (3.0 * np.sqrt(10.0) / 7.0)
        * (np.sqrt(np.clip(3.0 - ax_u[m3] ** 2 + 2.0 * ax_u[m3], 0.0, None)) - 2.0)
    )

    m4 = (ax_u >= 3.0) & (ax_u <= 7.0)
    upper[m4] = 3.0 * np.sqrt(np.clip(1.0 - (ax_u[m4] / 7.0) ** 2, 0.0, None))

    m5 = ax_l < 4.0
    lower[m5] = (
        ax_l[m5] / 2.0
        - ((3.0 * np.sqrt(33.0) - 7.0) / 112.0) * ax_l[m5] ** 2
        - 3.0
        + np.sqrt(np.clip(1.0 - (np.abs(ax_l[m5] - 2.0) - 1.0) ** 2, 0.0, None))
    )

    m6 = (ax_l >= 4.0) & (ax_l <= 7.0)
    lower[m6] = -3.0 * np.sqrt(np.clip(1.0 - (ax_l[m6] / 7.0) ** 2, 0.0, None))

    pts = np.vstack((np.column_stack((x_u, upper)), np.column_stack((x_l, lower))))
    pts = pts[~np.isnan(pts).any(axis=1)]

    if len(pts) > num_points:
        pick = rng.choice(len(pts), num_points, replace=False)
        pts = pts[pick]
    return pts


def load_batman_points(num_points: int = 8000, *, seed: Optional[int] = None) -> torch.Tensor:
    """Generate and normalize the Batman outline point cloud."""
    pts = sample_batman_outline(num_points, seed=seed)
    points = torch.tensor(pts, dtype=torch.float32)
    return normalize_point_cloud(points)


class PointCloudCyclicLoader:
    """
    Infinite stream of random minibatches (with replacement).

    Same interface as ``CyclicDataLoader`` in ``mnist.py``.
    """

    def __init__(
        self,
        points: torch.Tensor,
        batch_size: int,
        device: Optional[torch.device] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        if points.ndim != 2:
            raise ValueError(f"points must be [N, D], got shape {tuple(points.shape)}")
        self._points = points
        self._batch_size = int(batch_size)
        self._device = device
        self._batch_shape: Optional[tuple[int, ...]] = None
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @property
    def batch_shape(self) -> Optional[tuple[int, ...]]:
        return self._batch_shape

    def __iter__(self) -> PointCloudCyclicLoader:
        return self

    def __next__(self) -> torch.Tensor:
        n = self._points.shape[0]
        idx = torch.randint(
            0,
            n,
            (self._batch_size,),
            generator=self._generator,
        )
        batch = self._points[idx]
        if self._batch_shape is None:
            self._batch_shape = tuple(batch.shape)

        if self._device is not None:
            batch = batch.to(self._device, non_blocking=self._device.type == "cuda")
        return batch


def build_dino_points(cfg: Any) -> torch.Tensor:
    data_dir = Path(cfg.paths.data_dir) / "datasaurus"
    csv_path = data_dir / "datasaurus.csv"
    url = str(getattr(cfg.data, "source_url", DEFAULT_DINOSAURUS_URL))
    ensure_datasaurus_csv(csv_path, source_url=url)
    return load_dino_points(csv_path)


def build_batman_points(cfg: Any) -> torch.Tensor:
    num_points = int(getattr(cfg.data, "num_points", 8000))
    seed = int(cfg.seed) if hasattr(cfg, "seed") else None
    return load_batman_points(num_points=num_points, seed=seed)


def build_point_cloud_datamodule(cfg: Any) -> PointCloudCyclicLoader:
    """Cyclic 2D batch stream for dino or batman experiments."""
    name = str(cfg.data.name)
    if name == "dino":
        points = build_dino_points(cfg)
    elif name == "batman":
        points = build_batman_points(cfg)
    else:
        raise ValueError(f"Unknown point-cloud dataset: {name}")

    use_cuda = str(cfg.hardware.device).startswith("cuda")
    device = torch.device(cfg.hardware.device) if use_cuda else None
    seed = int(cfg.seed) if hasattr(cfg, "seed") else None

    return PointCloudCyclicLoader(
        points,
        batch_size=int(cfg.data.batch_size),
        device=device,
        seed=seed,
    )
