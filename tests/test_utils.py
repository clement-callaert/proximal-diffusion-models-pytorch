"""Tests for src/utils."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.utils.checkpoint import (
    checkpoint_path,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    load_ema_weights,
    save_checkpoint,
    save_if_due,
)
from src.utils.device import (
    amp_dtype_for_device,
    configure_hardware,
    device_from_cfg,
    resolve_device,
)
from src.utils.logging import format_loss_step, should_log
from src.utils.seed import seed_from_cfg, set_seed


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_set_seed_reproducible():
    set_seed(123)
    a = torch.randn(3)
    set_seed(123)
    b = torch.randn(3)
    assert torch.allclose(a, b)


def test_seed_from_cfg():
    cfg = SimpleNamespace(seed=7)
    assert seed_from_cfg(cfg) == 7


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_device_from_cfg():
    cfg = SimpleNamespace(hardware=SimpleNamespace(device="cpu", amp_dtype="bfloat16"))
    dev = device_from_cfg(cfg)
    assert dev.type == "cpu"
    assert amp_dtype_for_device(cfg, dev) == torch.float32


def test_configure_hardware_no_crash():
    cfg = SimpleNamespace(
        seed=0,
        hardware=SimpleNamespace(device="cpu", amp_dtype="bfloat16"),
    )
    dev = configure_hardware(cfg)
    assert dev.type == "cpu"


def test_checkpoint_roundtrip(tmp_path: Path):
    model = TinyNet()
    ema = TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = save_checkpoint(tmp_path / "ckpt_10.pt", model, ema, opt, 10)
    assert path.is_file()

    model2 = TinyNet()
    ema2 = TinyNet()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    it = load_checkpoint(path, model2, ema_model=ema2, optimizer=opt2)
    assert it == 10

    model3 = TinyNet()
    it_ema = load_ema_weights(path, model3)
    assert it_ema == 10
    for p, p_ema in zip(model3.parameters(), ema.parameters()):
        assert torch.allclose(p, p_ema)


def test_list_and_latest_checkpoint(tmp_path: Path):
    model = TinyNet()
    ema = TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(tmp_path / "ckpt_5.pt", model, ema, opt, 5)
    save_checkpoint(tmp_path / "ckpt_20.pt", model, ema, opt, 20)

    paths = list_checkpoints(tmp_path)
    assert [p.name for p in paths] == ["ckpt_5.pt", "ckpt_20.pt"]
    assert latest_checkpoint(tmp_path).name == "ckpt_20.pt"
    assert checkpoint_path(tmp_path, 20).name == "ckpt_20.pt"


def test_save_if_due(tmp_path: Path):
    model = TinyNet()
    ema = TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    assert save_if_due(3, 5, 10, tmp_path, model, ema, opt) is None
    path = save_if_due(5, 5, 10, tmp_path, model, ema, opt)
    assert path is not None
    path_final = save_if_due(10, 5, 10, tmp_path, model, ema, opt)
    assert path_final is not None


def test_logging_helpers():
    assert should_log(100, 100) is True
    assert should_log(99, 100) is False
    s = format_loss_step("score", 100, 1000, 0.5, extras={"lr": 1e-4})
    assert "[score]" in s and "lr=0.0001" in s
