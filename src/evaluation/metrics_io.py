"""
Save / load evaluation artifacts (FID sweeps, sample tensors).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class FidSweepResult:
    """FID vs NFE for score SDE and ProxDM (one discretization label)."""

    nfe_list: List[int]
    fid_score: List[float]
    fid_prox: List[float]
    prox_discretization: str = "hybrid"
    n_eval_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FidSweepResult:
        return cls(
            nfe_list=list(data["nfe_list"]),
            fid_score=list(data["fid_score"]),
            fid_prox=list(data["fid_prox"]),
            prox_discretization=str(data.get("prox_discretization", "hybrid")),
            n_eval_samples=int(data.get("n_eval_samples", 0)),
        )


def save_fid_sweep(path: str | Path, result: FidSweepResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_fid_sweep(path: str | Path) -> FidSweepResult:
    with Path(path).open(encoding="utf-8") as f:
        return FidSweepResult.from_dict(json.load(f))


def save_samples(path: str | Path, samples: torch.Tensor, *, nfe: Optional[int] = None) -> None:
    """Persist generated tensors as ``.pt`` (shape documented in sidecar JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"samples": samples.cpu()}
    if nfe is not None:
        payload["nfe"] = int(nfe)
    torch.save(payload, path)


def load_samples(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    raise ValueError(f"Unrecognized sample checkpoint format: {path}")


def write_metrics_summary(
    path: str | Path,
    *,
    fid_sweep: Optional[FidSweepResult] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Single JSON summary for a run directory."""
    out: Dict[str, Any] = {}
    if fid_sweep is not None:
        out["fid_sweep"] = fid_sweep.to_dict()
    if extra:
        out.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def parse_nfe_list(nfe_list: Sequence[int] | str) -> List[int]:
    """Hydra may pass a list or a string like ``[5,10,20]``."""
    if isinstance(nfe_list, str):
        cleaned = nfe_list.strip().strip("[]")
        return [int(x.strip()) for x in cleaned.split(",") if x.strip()]
    return [int(x) for x in nfe_list]
