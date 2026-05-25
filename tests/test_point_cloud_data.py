"""Tests for 2D point-cloud data loading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.data.datamodule import build_datamodule
from src.data.point_cloud import (
    PointCloudCyclicLoader,
    load_batman_points,
    load_dino_points,
    normalize_point_cloud,
    sample_batman_outline,
)


@pytest.fixture
def tiny_datasaurus_csv(tmp_path: Path) -> Path:
    rows = {
        "dataset": ["dino", "dino", "away"],
        "x": [10.0, 12.0, 0.0],
        "y": [20.0, 22.0, 0.0],
    }
    path = tmp_path / "datasaurus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_normalize_zero_mean_unit_std():
    pts = torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    out = normalize_point_cloud(pts)
    assert out.mean(dim=0).abs().max() < 1e-5
    assert (out.std(dim=0, unbiased=True) - 1.0).abs().max() < 1e-4


def test_load_dino_points(tiny_datasaurus_csv: Path):
    pts = load_dino_points(tiny_datasaurus_csv)
    assert pts.shape == (2, 2)
    assert pts.dtype == torch.float32


def test_batman_outline_finite():
    pts = sample_batman_outline(500, seed=0)
    assert pts.shape[1] == 2
    assert pts.shape[0] <= 500
    assert not (pts != pts).any()  # no NaN


def test_load_batman_points_shape():
    pts = load_batman_points(num_points=1000, seed=1)
    assert pts.shape[0] <= 1000
    assert pts.shape[1] == 2


def test_point_cloud_cyclic_batch_shape():
    pts = load_batman_points(num_points=200, seed=0)
    cyclic = PointCloudCyclicLoader(pts, batch_size=64, seed=0)
    b = next(cyclic)
    assert b.shape == (64, 2)
    assert cyclic.batch_shape == (64, 2)


def test_build_datamodule_batman(tmp_path: Path):
    cfg = SimpleNamespace(
        data=SimpleNamespace(name="batman", batch_size=32, num_points=400),
        hardware=SimpleNamespace(device="cpu"),
        paths=SimpleNamespace(data_dir=str(tmp_path)),
        seed=42,
    )
    stream = build_datamodule(cfg)
    batch = next(stream)
    assert batch.shape == (32, 2)
