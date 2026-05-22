# Proximal Diffusion Models (ProxDM) — PyTorch

**Author:** Clément Callaert  
**Institution:** CentraleSupélec — M2 Math & AI (Optimization for Computer Vision)  
**Reference:** [Beyond Scores: Proximal Diffusion Models](https://arxiv.org/abs/2507.08956) (Fang, Díaz, Buchanan, Sulam, 2025)

This repository is a structured, reproducible research implementation of **Proximal Diffusion Models (ProxDM)**. It compares score-based VP-SDE sampling (forward discretization) with proximal, backward-discretized samplers on **MNIST (32×32)** and **2D point-cloud** experiments (Datasaurus dino, custom Batman outline).

> **Course notebook:** exploratory work lives in `ovo_project_clément_callaert.ipynb`.  
> **Production pipeline:** training, evaluation, and sampling are intended to run via `main.py` and Hydra configs under `configs/`.

---

## Overview

Diffusion models typically learn the **score** \(\nabla_x \log p_t(x)\) and simulate the reverse SDE with **forward** schemes (e.g. Euler–Maruyama). That often requires many **function evaluations (NFE)** per sample.

**ProxDM** learns a **proximal operator** of the log-density and uses **backward** (implicit) discretization of the reverse process. In practice this allows **fewer sampling steps** while keeping competitive sample quality—especially on low-dimensional and MNIST-scale setups explored here.

| Approach | Discretization | Network conditions on |
|----------|----------------|------------------------|
| Score SDE | Forward (Euler–Maruyama) | time \(t\) |
| ProxDM (hybrid / backward) | Backward + optional forward term | time \(t\) and step size \(\lambda\) |

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
scripts/          Standalone utilities (sample, replot figures)
tests/            Unit & smoke tests
notebooks/        Optional exploratory notebooks
papers/           Reference PDFs
figures/          README figures (dino, batman, …)
```

Module stubs are in place; implement logic by migrating from `ovo_project_clément_callaert.ipynb`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional dev tools
pip install -r requirements-dev.txt
```

**GPU (recommended):** CUDA-capable PyTorch; tested with RTX 5090-class hardware (TF32, `bfloat16` AMP, large batch sizes).

---

## Usage (after implementing `src/`)

Default run (MNIST, paper training budget, hybrid sampler):

```bash
python main.py
```

**Quick budget** (matches shortened notebook runs):

```bash
python main.py training=quick
```

**2D experiments:**

```bash
python main.py experiment=dino model=mlp_2d training=quick
python main.py experiment=batman model=mlp_2d
```

**Backward discretization ablation:**

```bash
python main.py sampler=backward
```

**Run only evaluation** (requires trained checkpoints):

```bash
python main.py pipeline.run_data=false pipeline.run_train_score=false \
  pipeline.run_train_prox=false pipeline.run_evaluate=true
```

**Hydra overrides** (examples):

```bash
python main.py prox_training.total_iters=50000 score_training.total_iters=20000
python main.py evaluation.nfe_list=[5,10,20,50,100] hardware.compile_model=true
```

Hydra writes run configs under `outputs/<date>/<time>/`.

---

## Experiments (project scope)

| Experiment | Data | Backbone | Main metric |
|------------|------|----------|-------------|
| MNIST | 32×32 grayscale, \([-1,1]\) | U-Net + attention | FID vs NFE (paper Fig. 2) |
| Dino | Datasaurus 2D cloud | Time-conditioned MLP | Scatter samples vs NFE |
| Batman | Parametric outline | Same MLP | Qualitative + loss curves |

**Training schedule (ProxDM, MNIST):** L1 warm-up (~⅓ of prox iters) → proximal matching with \(\zeta\) decay (1.0 → 0.5). See `configs/training/paper_mnist.yaml`.

**Not in scope (future work):** CIFAR-10, CelebA-HQ, score **ODE** sampler, precision/recall metrics from the paper.

---

## Configuration

| Config group | Role |
|--------------|------|
| `experiment/` | `mnist`, `dino`, `batman` — data + default model sizes |
| `model/` | `unet_mnist`, `mlp_2d` |
| `training/` | `paper_mnist` (225k prox), `quick` (20k prox) |
| `sampler/` | `hybrid`, `backward` — \(\lambda\) and sampling discretization |

Root defaults: `configs/config.yaml`.

---

## Tests

```bash
pytest
```

Implement tests in `tests/` as you port code from the notebook.

---

## References

1. Fang, Z., et al. (2025). **Beyond Scores: Proximal Diffusion Models.** arXiv:2507.08956.  
2. Official code: [ZhenghanFang/ProxDM](https://github.com/ZhenghanFang/ProxDM)  
3. Song, Y., et al. (2020). **Score-Based Generative Modeling through Stochastic Differential Equations.**  

---

## Related files

- `README_ovo_project.md` — short course-project summary (legacy)  
- `ovo_project_clément_callaert.ipynb` — full from-scratch implementation notebook  
- `report.pdf` — project report  
- `papers/arXiv 2507.08956.pdf` — ProxDM paper  

---

## Figures

<p align="center">
  <img src="./figures/dino.png" alt="Dino samples" width="480">
  <img src="./figures/batman.png" alt="Batman samples" width="480">
</p>

*2D generative samples (ProxDM vs score SDE) — see notebook and `figures/` after re-running the pipeline.*
