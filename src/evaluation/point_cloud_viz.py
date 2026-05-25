"""
2D scatter plots: score SDE vs ProxDM across NFE values (dino / batman).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.evaluation.metrics_io import parse_nfe_list
from src.sampling.prox_sampler import sample_prox
from src.sampling.score_sampler import sample_score


def _axis_limits(
    reference: Optional[torch.Tensor],
    samples_list: Sequence[torch.Tensor],
    margin: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Symmetric limits around pooled 2D points."""
    chunks = []
    if reference is not None:
        chunks.append(reference.detach().cpu())
    for s in samples_list:
        chunks.append(s.detach().cpu())
    if not chunks:
        return (-2.5, 2.5, -2.5, 2.5)

    pts = torch.cat(chunks, dim=0)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = max(xmax - xmin, ymax - ymin) * 0.5 + margin
    return (cx - half, cx + half, cy - half, cy + half)


@torch.inference_mode()
def sample_point_clouds_at_nfe(
    score_model: nn.Module,
    prox_model: nn.Module,
    cfg: Any,
    nfe_list: Sequence[int],
    n_samples: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return (score_samples_per_nfe, prox_samples_per_nfe)."""
    device = torch.device(str(cfg.hardware.device))
    score_model.eval()
    prox_model.eval()

    score_out: List[torch.Tensor] = []
    prox_out: List[torch.Tensor] = []
    for nfe in nfe_list:
        score_out.append(sample_score(score_model, cfg, n_samples, int(nfe), device=device))
        prox_out.append(sample_prox(prox_model, cfg, n_samples, int(nfe), device=device))
    return score_out, prox_out


def plot_point_cloud_comparison(
    score_samples: Sequence[torch.Tensor],
    prox_samples: Sequence[torch.Tensor],
    nfe_list: Sequence[int],
    *,
    reference_points: Optional[torch.Tensor] = None,
    title: str = "Score SDE vs ProxDM",
    prox_label: str = "Prox",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grid: row 0 = score, row 1 = prox; one column per NFE (notebook layout).
    """
    n_cols = len(nfe_list)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
    x0, x1, y0, y1 = _axis_limits(
        reference_points,
        list(score_samples) + list(prox_samples),
    )
    for col, nfe in enumerate(nfe_list):
        s = score_samples[col].cpu().numpy()
        p = prox_samples[col].cpu().numpy()

        axes[0, col].scatter(s[:, 0], s[:, 1], alpha=0.5, s=5, color="#1f77b4")
        axes[0, col].set_title(f"Score SDE (NFE={nfe})")
        axes[0, col].set_xlim(x0, x1)
        axes[0, col].set_ylim(y0, y1)
        axes[0, col].set_aspect("equal", adjustable="box")

        axes[1, col].scatter(p[:, 0], p[:, 1], alpha=0.5, s=5, color="#ff7f0e")
        axes[1, col].set_title(f"{prox_label} (NFE={nfe})")
        axes[1, col].set_xlim(x0, x1)
        axes[1, col].set_ylim(y0, y1)
        axes[1, col].set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return fig


@torch.inference_mode()
def compare_methods(
    score_model: nn.Module,
    prox_model: nn.Module,
    cfg: Any,
    *,
    nfe_list: Optional[Sequence[int]] = None,
    n_samples: int = 500,
    reference_points: Optional[torch.Tensor] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Sample and plot score vs prox across NFE (notebook ``compare_methods``)."""
    if nfe_list is None:
        nfe_list = parse_nfe_list(cfg.evaluation.nfe_list)

    score_batches, prox_batches = sample_point_clouds_at_nfe(
        score_model, prox_model, cfg, nfe_list, n_samples
    )
    disc = str(cfg.sampler.discretization)
    return plot_point_cloud_comparison(
        score_batches,
        prox_batches,
        nfe_list,
        reference_points=reference_points,
        title=f"Score SDE vs Prox ({disc})",
        prox_label=f"Prox ({disc})",
        save_path=save_path,
        show=show,
    )


def plot_fid_vs_nfe(
    result: Any,
    *,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    log_x: bool = True,
    log_y: bool = False,
) -> plt.Figure:
    """Paper-style FID curve (Fig. 2 uses log axes in places)."""
    from src.evaluation.metrics_io import FidSweepResult

    if isinstance(result, dict):
        result = FidSweepResult.from_dict(result)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        result.nfe_list,
        result.fid_score,
        marker="o",
        label="Score SDE",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        result.nfe_list,
        result.fid_prox,
        marker="s",
        label=f"Prox ({result.prox_discretization})",
        color="#ff7f0e",
        linewidth=2,
    )
    ax.set_title("FID vs NFE (MNIST 32×32)")
    ax.set_xlabel("NFE")
    ax.set_ylabel("FID (lower is better)")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
