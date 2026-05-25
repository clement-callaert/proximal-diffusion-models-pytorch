"""
FID vs NFE evaluation (paper Fig. 2 on MNIST).

Each NFE value is one reverse trajectory with that many network evaluations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.fid import compute_fid
from src.evaluation.metrics_io import FidSweepResult, parse_nfe_list, save_fid_sweep, save_samples
from src.sampling.prox_sampler import sample_prox
from src.sampling.score_sampler import sample_score
from src.sampling.shape import sample_shape_from_cfg
from src.sde.vp_sde import build_vp_sde


def collect_reference_images(
    datamodule: Any,
    n_samples: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Stack ``n_samples`` real training images from a cyclic MNIST loader.

    Uses the same distribution as training (notebook convention).
    """
    batches: List[torch.Tensor] = []
    count = 0
    while count < n_samples:
        batch = next(datamodule)
        if device is not None:
            batch = batch.to(device, non_blocking=device.type == "cuda")
        batches.append(batch)
        count += batch.shape[0]
    return torch.cat(batches, dim=0)[:n_samples]


@torch.inference_mode()
def run_fid_nfe_sweep(
    score_model: nn.Module,
    prox_model: nn.Module,
    cfg: Any,
    real_images: torch.Tensor,
    *,
    nfe_list: Optional[List[int]] = None,
    show_progress: bool = True,
    save_dir: Optional[str | Path] = None,
) -> FidSweepResult:
    """
    For each NFE, sample with score EM and ProxDM, then compute FID vs ``real_images``.

    Models should already be EMA weights in eval mode.
    """
    device = torch.device(str(cfg.hardware.device))
    score_model.eval()
    prox_model.eval()

    if nfe_list is None:
        nfe_list = parse_nfe_list(cfg.evaluation.nfe_list)

    n_eval = int(real_images.shape[0])
    fid_score: List[float] = []
    fid_prox: List[float] = []
    discretization = str(cfg.sampler.discretization)

    iterator = nfe_list
    if show_progress:
        iterator = tqdm(nfe_list, desc="FID vs NFE")

    for nfe in iterator:
        fake_score = sample_score(score_model, cfg, n_eval, int(nfe), device=device)
        fake_prox = sample_prox(prox_model, cfg, n_eval, int(nfe), device=device)

        fid_score.append(
            compute_fid(real_images, fake_score, device=device)
        )
        fid_prox.append(
            compute_fid(real_images, fake_prox, device=device)
        )

        if save_dir is not None and bool(getattr(cfg.evaluation, "save_samples", False)):
            root = Path(save_dir)
            save_samples(root / f"score_nfe{nfe}.pt", fake_score, nfe=nfe)
            save_samples(root / f"prox_{discretization}_nfe{nfe}.pt", fake_prox, nfe=nfe)

    result = FidSweepResult(
        nfe_list=list(nfe_list),
        fid_score=fid_score,
        fid_prox=fid_prox,
        prox_discretization=discretization,
        n_eval_samples=n_eval,
    )

    if save_dir is not None:
        save_fid_sweep(Path(save_dir) / "fid_sweep.json", result)

    return result


@torch.inference_mode()
def evaluate_mnist_fid(
    score_model: nn.Module,
    prox_model: nn.Module,
    cfg: Any,
    datamodule: Any,
) -> FidSweepResult:
    """
    End-to-end MNIST FID sweep: reference batch + NFE loop.

    Skips work if ``cfg.evaluation.compute_fid`` is false.
    """
    if not bool(getattr(cfg.evaluation, "compute_fid", True)):
        return FidSweepResult(nfe_list=[], fid_score=[], fid_prox=[])

    device = torch.device(str(cfg.hardware.device))
    n_eval = int(cfg.evaluation.n_eval_samples)
    real_images = collect_reference_images(datamodule, n_eval, device=device)

    save_dir = None
    if getattr(cfg.evaluation, "save_samples", False):
        save_dir = Path(cfg.paths.output_dir) / "eval_samples"

    return run_fid_nfe_sweep(
        score_model,
        prox_model,
        cfg,
        real_images,
        save_dir=save_dir,
    )


def verify_sample_shape(cfg: Any) -> None:
    """Sanity check that cfg.data matches sampling helpers."""
    _ = sample_shape_from_cfg(cfg)
    _ = build_vp_sde(cfg)
