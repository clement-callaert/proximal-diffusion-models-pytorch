# Proximal Diffusion Models (ProxDM) — PyTorch

**Author:** Clément Callaert  
**Institution:** CentraleSupélec — M2 Math & AI (Optimization for Computer Vision)  
**Reference:** [Beyond Scores: Proximal Diffusion Models](https://arxiv.org/abs/2507.08956) (Fang, Díaz, Buchanan, Sulam, 2025)

Structured PyTorch implementation of **Proximal Diffusion Models (ProxDM)**: score-based VP-SDE sampling (forward discretization) vs proximal, backward-discretized samplers on **MNIST (32×32)** and **2D point clouds** (Datasaurus dino, Batman outline).

> **Notebook:** exploratory reference — `notebooks/ovo_project_clément_callaert.ipynb`  
> **Pipeline:** `python main.py` + Hydra configs under `configs/`

---

## Overview

| Approach | Discretization | Network conditions on |
|----------|----------------|------------------------|
| Score SDE | Forward (Euler–Maruyama) | time \(t\) |
| ProxDM (hybrid / backward) | Backward + optional forward term | time \(t\) and step size \(\lambda\) |

---

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .                    # editable install (imports ``src.*``)
pip install -e ".[dev]"             # optional: pytest, ruff
```

Or without editable install: `pip install -r requirements.txt`

**Colab / cloned repo:** `cd` into the folder that contains this `setup.py`, then run `pip install -e .` (not a parent directory).

**GPU (recommended for MNIST):** CUDA PyTorch. The default config uses `hardware.device=cuda`, TF32, and `bfloat16` AMP.

### 2. Sanity check

```bash
pytest -q
```

### 3. Smoke run (~1–2 min, CPU, verifies the full pipeline)

Trains score + prox on **dino** with tiny iteration counts (no evaluation):

```bash
python main.py experiment=dino training=quick \
  score_training.total_iters=30 prox_training.total_iters=60 \
  pipeline.run_evaluate=false hardware.device=cpu hardware.num_workers=0
```

Checkpoints land under `outputs/<date>/<time>/outputs/score/` and `.../prox_hybrid/` (Hydra run directory).

### 4. Quick experiment (shortened budget)

**2D dino** (~tens of minutes on GPU with `training=quick` defaults):

```bash
python main.py experiment=dino training=quick hardware.device=cuda
```

**MNIST** (20k score + 20k prox iters — still hours on a good GPU; use smoke overrides below for a faster dry run):

```bash
python main.py training=quick hardware.device=cuda
```

**Faster MNIST dry run** (train + FID sweep, reduced iters — quality will be poor):

```bash
python main.py training=quick \
  score_training.total_iters=500 prox_training.total_iters=1500 \
  score_training.save_every=500 prox_training.save_every=1500 \
  evaluation.nfe_list=[5,10,20] hardware.device=cuda
```

### 5. Paper-scale MNIST (multi-hour / overnight on GPU)

```bash
python main.py training=paper_mnist hardware.device=cuda
```

### 6. Evaluation only (needs checkpoints from a prior run)

```bash
python main.py pipeline.run_train_score=false pipeline.run_train_prox=false \
  pipeline.run_evaluate=true hardware.device=cuda
```

Point the run at existing checkpoints by overriding paths, e.g.  
`paths.score_ckpt_dir=/path/to/score paths.prox_ckpt_dir=/path/to/prox_hybrid`.

### 7. Replot FID curve from saved JSON

```bash
python scripts/plot_fid_curve.py outputs/<date>/<time>/outputs/eval_samples/fid_sweep.json -o fid.png
```

---

## Other useful commands

| Goal | Command |
|------|---------|
| Batman 2D | `python main.py experiment=batman training=quick` |
| Backward prox sampler | `python main.py sampler=backward` |
| Skip training, data check only | `python main.py pipeline.run_train_score=false pipeline.run_train_prox=false pipeline.run_evaluate=false` |
| Override NFE for FID | `python main.py evaluation.nfe_list=[5,10,20,50,100]` |
| CPU fallback | append `hardware.device=cpu` |

Hydra stores each run under `outputs/<YYYY-MM-DD>/<HH-MM-SS>/` (config snapshot + logs + checkpoints).

---

## Repository layout

```
configs/          Hydra YAML (experiments, models, training budgets, samplers)
src/
  sde/            VP-SDE (linear β schedule)
  data/           MNIST, dino, batman point clouds
  models/         U-Net (images), MLP (2D), score & prox wrappers
  losses/         Score matching, proximal matching, λ sampling
  sampling/       Score & prox samplers (hybrid / backward)
  training/       Loops, EMA, PM schedule, optimizers
  evaluation/     FID vs NFE, 2D visual comparison
  utils/          seed, device, checkpoints, logging
scripts/          plot_fid_curve.py, …
tests/            Unit & smoke tests
main.py           Hydra entry point
notebooks/        Course notebook (reference)
papers/           Reference PDFs
figures/          README figures
```

---

## Experiments

| Experiment | Data | Backbone | Main metric |
|------------|------|----------|-------------|
| MNIST | 32×32 grayscale, \([-1,1]\) | U-Net + attention | FID vs NFE (paper Fig. 2) |
| Dino | Datasaurus 2D cloud | Time-conditioned MLP | Scatter samples vs NFE |
| Batman | Parametric outline | Same MLP | Qualitative + loss curves |

**ProxDM schedule:** L1 warm-up (~⅓ of prox iters) → proximal matching with \(\zeta\) decay (1.0 → 0.5). See `configs/training/paper_mnist.yaml`.

**Not in scope:** CIFAR-10, CelebA-HQ, score ODE sampler, precision/recall metrics from the paper.

---

## Configuration

| Config group | Role |
|--------------|------|
| `experiment/` | `mnist`, `dino`, `batman` — data + model choice |
| `model/` | `unet_mnist`, `mlp_2d` |
| `training/` | `paper_mnist` (225k prox), `quick` (20k prox) |
| `sampler/` | `hybrid`, `backward` |

Root defaults: `configs/config.yaml`.

---

## Tests

```bash
pytest
```

---

## References

1. Fang, Z., et al. (2025). **Beyond Scores: Proximal Diffusion Models.** arXiv:2507.08956.  
2. Official code: [ZhenghanFang/ProxDM](https://github.com/ZhenghanFang/ProxDM)  
3. Song, Y., et al. (2020). **Score-Based Generative Modeling through Stochastic Differential Equations.**

---

## Related files

- `README_ovo_project.md` — short course summary  
- `notebooks/ovo_project_clément_callaert.ipynb` — from-scratch reference  
- `report.pdf` — project report  
- `papers/arXiv 2507.08956.pdf` — ProxDM paper  

---

## Figures

<p align="center">
  <img src="./figures/dino.png" alt="Dino samples" width="480">
  <img src="./figures/batman.png" alt="Batman samples" width="480">
</p>

*Example 2D samples (ProxDM vs score SDE). Re-run the pipeline or notebook to regenerate.*
