"""Tests for evaluation helpers (no Inception download)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.evaluation.fid import mnist_to_fid_rgb
from src.evaluation.metrics_io import FidSweepResult, load_fid_sweep, parse_nfe_list, save_fid_sweep


def test_mnist_to_fid_rgb_range_and_channels():
    x = torch.zeros(4, 1, 32, 32)  # [-1, 1] black
    rgb = mnist_to_fid_rgb(x)
    assert rgb.shape == (4, 3, 32, 32)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0

    x[0] = 1.0
    rgb2 = mnist_to_fid_rgb(x)
    assert rgb2[0, 0, 0, 0].item() == 1.0


def test_fid_sweep_json_roundtrip(tmp_path: Path):
    result = FidSweepResult(
        nfe_list=[5, 10],
        fid_score=[1.2, 1.0],
        fid_prox=[0.9, 0.8],
        prox_discretization="hybrid",
        n_eval_samples=128,
    )
    path = tmp_path / "fid.json"
    save_fid_sweep(path, result)
    loaded = load_fid_sweep(path)
    assert loaded.nfe_list == [5, 10]
    assert loaded.fid_prox == [0.9, 0.8]

    with path.open() as f:
        raw = json.load(f)
    assert raw["n_eval_samples"] == 128


def test_parse_nfe_list_string():
    assert parse_nfe_list("[5, 10, 20]") == [5, 10, 20]
    assert parse_nfe_list([1, 2]) == [1, 2]
