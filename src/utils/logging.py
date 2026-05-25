"""
Lightweight training logs (notebook-style ``print`` every 100 steps).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional


def setup_logging(
    level: int = logging.INFO,
    *,
    log_file: Optional[str] = None,
    name: str = "proxdm",
) -> logging.Logger:
    """Root logger for the pipeline; safe to call once at startup."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def should_log(iteration: int, every: int = 100) -> bool:
    return every > 0 and iteration % every == 0


def format_loss_step(
    tag: str,
    iteration: int,
    total: int,
    loss: float,
    *,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """``[score] it=100/20000 loss=0.1234`` style line."""
    msg = f"[{tag}] it={iteration}/{total} loss={loss:.4f}"
    if extras:
        parts = " ".join(f"{k}={v}" for k, v in extras.items())
        msg = f"{msg} {parts}"
    return msg


def log_training_step(
    logger: logging.Logger,
    tag: str,
    iteration: int,
    total: int,
    loss: float,
    *,
    every: int = 100,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    if should_log(iteration, every):
        logger.info(format_loss_step(tag, iteration, total, loss, extras=extras))


def log_message(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)
