"""MNIST loading for ProxDM training (32x32, [-1, 1] normalization)."""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional, Protocol

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DataConfig(Protocol):
    """Shape of Hydra config fields consumed by this module."""

    name: str
    image_size: int
    in_ch: int
    batch_size: int


def _pixels_to_minus_one_one(t: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] tensor pixels to [-1, 1] (top-level for DataLoader worker pickling)."""
    return t * 2.0 - 1.0


def build_mnist_transform(image_size: int = 32) -> transforms.Compose:
    """Pad 28x28 MNIST to 32x32 and scale to [-1, 1] as in the paper / official ProxDM repo."""
    if image_size != 32:
        raise ValueError(f"Only image_size=32 is supported, got {image_size}")

    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Lambda(_pixels_to_minus_one_one),
        ]
    )


class MNISTImagesDataset(Dataset):
    """MNIST images only (labels stripped)."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        image_size: int = 32,
    ) -> None:
        self._dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=build_mnist_transform(image_size),
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _label = self._dataset[index]
        return image


def build_train_dataloader(
    root: str,
    batch_size: int,
    num_workers: int,
    *,
    pin_memory: bool = False,
    drop_last: bool = True,
    image_size: int = 32,
) -> DataLoader:
    """Training DataLoader with notebook-equivalent performance options."""
    dataset = MNISTImagesDataset(root=root, train=True, download=True, image_size=image_size)

    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


class CyclicDataLoader:
    """Infinite batch stream; restarts each epoch without global state."""

    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        self._loader = loader
        self._device = device
        self._iterator: Optional[Iterator[torch.Tensor]] = None
        self._batch_shape: Optional[tuple[int, ...]] = None

    @property
    def batch_shape(self) -> Optional[tuple[int, ...]]:
        return self._batch_shape

    def __iter__(self) -> CyclicDataLoader:
        self._iterator = iter(self._loader)
        return self

    def __next__(self) -> torch.Tensor:
        if self._iterator is None:
            self._iterator = iter(self._loader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            batch = next(self._iterator)

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        if self._batch_shape is None:
            self._batch_shape = tuple(batch.shape)

        if self._device is not None:
            batch = batch.to(self._device, non_blocking=self._device.type == "cuda")

        return batch


def build_mnist_datamodule(cfg: Any) -> CyclicDataLoader:
    """Build a cyclic MNIST stream from the merged Hydra config."""
    data_root = os.path.join(cfg.paths.data_dir, "mnist")

    use_cuda = str(cfg.hardware.device).startswith("cuda")
    pin_memory = use_cuda

    loader = build_train_dataloader(
        root=data_root,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.hardware.num_workers),
        pin_memory=pin_memory,
        image_size=int(cfg.data.image_size),
    )

    device = torch.device(cfg.hardware.device) if use_cuda else None
    return CyclicDataLoader(loader, device=device)
