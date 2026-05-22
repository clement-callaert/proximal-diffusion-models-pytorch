"""
Tests for src/data/mnist.py — your "submission judge".

Run: pytest tests/test_mnist_data.py -v
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from src.data.mnist import (
    CyclicDataLoader,
    MNISTImagesDataset,
    build_mnist_datamodule,
    build_mnist_transform,
    build_train_dataloader,
)


@pytest.fixture(scope="module")
def mnist_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    return str(root)


# --- Part 1: transform -------------------------------------------------------

def test_transform_output_shape_and_range():
    tfm = build_mnist_transform(image_size=32)
    from torchvision.datasets import MNIST
    from PIL import Image
    import numpy as np

    raw = Image.fromarray((np.random.rand(28, 28) * 255).astype("uint8"))
    x = tfm(raw)
    assert x.shape == (1, 32, 32)
    assert x.dtype == torch.float32
    assert x.min() >= -1.0 - 1e-5
    assert x.max() <= 1.0 + 1e-5


def test_transform_rejects_wrong_size():
    with pytest.raises(ValueError):
        build_mnist_transform(image_size=28)


# --- Part 2: dataset ---------------------------------------------------------

def test_dataset_len_and_item(mnist_root):
    ds = MNISTImagesDataset(mnist_root, train=True, download=True, image_size=32)
    assert len(ds) == 60_000
    x = ds[0]
    assert x.shape == (1, 32, 32)
    assert x.dtype == torch.float32


def test_dataset_no_label_tuple(mnist_root):
    ds = MNISTImagesDataset(mnist_root, train=True, download=True)
    item = ds[42]
    assert not isinstance(item, (tuple, list))


# --- Part 3: dataloader ------------------------------------------------------

def test_dataloader_batch(mnist_root):
    loader = build_train_dataloader(
        mnist_root,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    assert batch.shape == (64, 1, 32, 32)
    assert batch.dtype == torch.float32


@pytest.mark.skipif(os.name == "nt", reason="persistent_workers flaky on some Windows CI")
def test_dataloader_worker_kwargs(mnist_root):
    loader = build_train_dataloader(
        mnist_root,
        batch_size=32,
        num_workers=2,
        pin_memory=False,
    )
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 2


def test_dataloader_num_workers_zero_no_extra_kwargs(mnist_root):
    loader = build_train_dataloader(
        mnist_root,
        batch_size=16,
        num_workers=0,
    )
    # DataLoader stores None for unused optional kwargs
    assert getattr(loader, "persistent_workers", None) in (None, False)


# --- Part 4: cyclic iterator -------------------------------------------------

def test_cyclic_loader_restarts(mnist_root):
    loader = build_train_dataloader(mnist_root, batch_size=128, num_workers=0)
    cyclic = CyclicDataLoader(loader)
    b1 = next(cyclic)
    b2 = next(cyclic)
    assert b1.shape == b2.shape == (128, 1, 32, 32)
    # burn one epoch worth of steps without crashing
    steps_per_epoch = len(loader)
    for _ in range(steps_per_epoch + 3):
        next(cyclic)


def test_cyclic_loader_device_move(mnist_root):
    loader = build_train_dataloader(mnist_root, batch_size=8, num_workers=0)
    dev = torch.device("cpu")
    cyclic = CyclicDataLoader(loader, device=dev)
    batch = next(cyclic)
    assert batch.device.type == "cpu"


# --- Part 5: hydra wiring ----------------------------------------------------

def test_build_mnist_datamodule(mnist_root):
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            name="mnist",
            image_size=32,
            in_ch=1,
            batch_size=32,
        ),
        hardware=SimpleNamespace(
            num_workers=0,
            device="cpu",
        ),
        paths=SimpleNamespace(data_dir=mnist_root),
    )
    cyclic = build_mnist_datamodule(cfg)
    batch = next(cyclic)
    assert batch.shape == (32, 1, 32, 32)
