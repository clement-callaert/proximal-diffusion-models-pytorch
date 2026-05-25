"""
Hydra entry point: train score + prox, evaluate, optional sampling.

Quick dev run (2D, ~1 min on CPU):
  python main.py experiment=dino model=mlp_2d training=quick \\
    score_training.total_iters=100 prox_training.total_iters=300 \\
    pipeline.run_evaluate=false hardware.device=cpu

Quick MNIST (GPU recommended, tens of minutes with training=quick):
  python main.py training=quick
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.data.datamodule import build_datamodule
from src.evaluation.nfe_sweep import evaluate_mnist_fid
from src.evaluation.point_cloud_viz import compare_methods
from src.models.prox_model import build_prox_model
from src.models.score_model import build_score_model
from src.training.prox_loop import build_prox_training_bundle, run_prox_training
from src.training.score_loop import build_score_training_bundle, run_score_training
from src.utils.checkpoint import latest_checkpoint, load_checkpoint, save_if_due
from src.utils.device import configure_hardware, describe_hardware
from src.utils.logging import log_message, log_training_step, setup_logging
from src.utils.seed import seed_from_cfg

log = logging.getLogger("proxdm")


def _maybe_compile(model: nn.Module, cfg: Any) -> nn.Module:
    if not bool(getattr(cfg.hardware, "compile_model", False)):
        return model
    device = torch.device(str(cfg.hardware.device))
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="max-autotune-no-cudagraphs")
    except Exception as exc:
        log.warning("torch.compile skipped: %s", exc)
        return model


def _batch_fn(datamodule: Any) -> Callable[[], torch.Tensor]:
    def fn() -> torch.Tensor:
        return next(datamodule)

    return fn


def _channels_last_if_image(model: nn.Module, cfg: Any) -> nn.Module:
    if getattr(cfg.data, "name", "") == "mnist" and torch.cuda.is_available():
        return model.to(memory_format=torch.channels_last)
    return model


@torch.inference_mode()
def _load_ema_for_eval(
    model: nn.Module,
    ckpt_dir: Path,
    device: torch.device,
) -> nn.Module:
    path = latest_checkpoint(ckpt_dir)
    if path is None:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    model = model.to(device)
    load_checkpoint(path, model, map_location=device, use_ema_for_model=True)
    model.eval()
    return model


def run_pipeline(cfg: DictConfig) -> None:
    seed_from_cfg(cfg)
    device = configure_hardware(cfg)
    log_message(log, describe_hardware(cfg))

    datamodule = build_datamodule(cfg)
    batch_fn = _batch_fn(datamodule)

    # Sanity: one batch
    if bool(cfg.pipeline.run_data):
        b = batch_fn()
        log_message(log, f"data batch shape={tuple(b.shape)} dtype={b.dtype}")

    score_model = _channels_last_if_image(_maybe_compile(build_score_model(cfg), cfg), cfg)
    prox_model = _channels_last_if_image(_maybe_compile(build_prox_model(cfg), cfg), cfg)

    score_ckpt_dir = Path(cfg.paths.score_ckpt_dir)
    prox_ckpt_dir = Path(cfg.paths.prox_ckpt_dir)
    score_ckpt_dir.mkdir(parents=True, exist_ok=True)
    prox_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.pipeline.run_train_score):
        log_message(log, "=== score training ===")
        bundle = build_score_training_bundle(cfg, score_model)
        total = int(cfg.score_training.total_iters)
        save_every = int(cfg.score_training.save_every)

        def on_score_step(it: int, loss: float) -> None:
            log_training_step(log, "score", it, total, loss)
            save_if_due(
                it,
                save_every,
                total,
                score_ckpt_dir,
                score_model,
                bundle["ema_model"],
                bundle["optimizer"],
            )

        run_score_training(
            cfg,
            score_model,
            bundle["trainer"],
            batch_fn,
            optimizer=bundle["optimizer"],
            ema_model=bundle["ema_model"],
            on_step=on_score_step,
        )

    if bool(cfg.pipeline.run_train_prox):
        log_message(log, "=== prox training ===")
        bundle = build_prox_training_bundle(cfg, prox_model)
        total = int(cfg.prox_training.total_iters)
        save_every = int(cfg.prox_training.save_every)

        def on_prox_step(it: int, loss: float, params: dict) -> None:
            log_training_step(log, f"prox-{cfg.sampler.discretization}", it, total, loss, extras=params)
            save_if_due(
                it,
                save_every,
                total,
                prox_ckpt_dir,
                prox_model,
                bundle["ema_model"],
                bundle["optimizer"],
            )

        run_prox_training(
            cfg,
            prox_model,
            bundle["trainer"],
            batch_fn,
            schedule=bundle["schedule"],
            optimizer=bundle["optimizer"],
            ema_model=bundle["ema_model"],
            on_step=on_prox_step,
        )

    if bool(cfg.pipeline.run_evaluate):
        log_message(log, "=== evaluation ===")
        score_eval = build_score_model(cfg).to(device)
        prox_eval = build_prox_model(cfg).to(device)
        score_eval = _load_ema_for_eval(score_eval, score_ckpt_dir, device)
        prox_eval = _load_ema_for_eval(prox_eval, prox_ckpt_dir, device)

        if str(cfg.data.name) == "mnist":
            result = evaluate_mnist_fid(score_eval, prox_eval, cfg, datamodule)
            log_message(log, f"FID sweep: nfe={result.nfe_list} score={result.fid_score} prox={result.fid_prox}")
        else:
            plot_dir = Path(cfg.paths.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            ref = getattr(datamodule, "points", None)
            out = plot_dir / f"{cfg.data.name}_comparison.png"
            compare_methods(
                score_eval,
                prox_eval,
                cfg,
                reference_points=ref,
                save_path=out,
                show=False,
            )
            log_message(log, f"saved 2D comparison to {out}")

    if bool(cfg.pipeline.run_sample):
        log_message(log, "=== sample-only (reuse eval checkpoints) ===")
        # Same as evaluate path; set pipeline.run_evaluate=true for metrics/plots.
        log.warning("pipeline.run_sample=true: use run_evaluate for generation in this repo.")

    log_message(log, "done.")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    OmegaConf.resolve(cfg)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
