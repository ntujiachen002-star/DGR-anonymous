# DiffGeoReward

Training-free inference-time mesh refinement via three differentiable geometric rewards (symmetry, Huber normal consistency, compactness).

> Anonymous code release accompanying the paper **"DiffGeoReward: Differentiable Geometric Rewards for Language-Guided 3D Shape Refinement"** (under review).

## Overview

Given a text-to-3D baseline mesh (Shap-E / TripoSR / InstantMesh), DiffGeoReward runs 50 steps of Adam on vertex positions to maximise a weighted combination of three closed-form geometric rewards:

- **Symmetry** — bilateral Chamfer to reflected mesh across an estimated plane
- **HNC** — Huberised dihedral-angle consistency across adjacent faces (mesh-adaptive delta)
- **Compactness** — surface-area / volume^(2/3) isoperimetric ratio

All rewards are differentiable w.r.t. vertex positions, so the whole refinement is one AutoDiff loop. Typical overhead: 5-8 s per mesh on a single GPU.

## Repository structure

```
src/                  Core implementation
  geo_reward.py         Three rewards + Huber delta + plane estimator
  shape_gen.py          Shap-E wrapper + refine_with_geo_reward()
  lang2comp.py          Optional language-conditioned weight predictor
  spectral_weighting.py Laplacian-spectrum reward decomposition
  prompts_gpteval3d.py  GPTEval3D 110-prompt list
  evaluate.py / evaluate_full.py  Scoring harness
  run_experiment.py     End-to-end runner
  train_lang2comp.py    Lang2Comp offline training
  test_geo_reward.py    Unit tests

tools/                 Experiment scripts (one per paper subsection)
  exp_k_full_mesh_validity.py     Main benchmark (n=97 prompts, 3 seeds, 6 methods)
  exp_classical_baselines_matched.py  Classical-operator sweep
  exp_pareto_distortion_sweep.py  Matched-shape-CD Pareto
  exp_pareto_component_ablation.py Single/2-reward ablation
  exp_two_reward_combo.py         Generate 2-reward combo meshes
  exp_psr_selfconsistency.py      PSR round-trip downstream test
  exp_pca_stability.py            PCA axis-stability test
  exp_instantmesh_backbone.py     InstantMesh cross-backbone
  exp_adversarial_asymmetric.py   20 adversarial prompts
  exp_scale_controlled_ablation.py Gradient-normalisation ablation
  exp_anticollapse.py             Anti-collapse penalty ablation
  exp_plane_selection_ablation.py Multi-start vs fixed-axis plane
  exp_lang2comp_generalization.py OOD prompts for Lang2Comp
  ... (~100 more diagnostic scripts)

requirements.txt       Minimal dependencies (torch, trimesh, scipy, ...)
```

## Installation

```bash
conda create -n dgr python=3.10 -y
conda activate dgr
pip install -r requirements.txt

# For Shap-E backbone (main benchmark):
pip install git+https://github.com/openai/shap-e.git

# For TripoSR / InstantMesh backbones (cross-backbone only):
#   see upstream repos; expect models cached under $TRIPOSR_PATH / $INSTANTMESH_PATH
```

GPU required for backbone generation; V100 32 GB is sufficient for the full benchmark.

## Quick start

Minimal example: refine a single mesh.

```python
import torch, trimesh
from src.geo_reward import estimate_symmetry_plane
from src.shape_gen import refine_with_geo_reward

mesh = trimesh.load("your_mesh.obj", force="mesh", process=False)
V = torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda")
F = torch.as_tensor(mesh.faces, dtype=torch.long, device="cuda")

# Estimate bilateral plane once (multi-start, ~200 ms)
sym_n, sym_d = estimate_symmetry_plane(V)

# 50 Adam steps, equal weights
weights = torch.tensor([1/3, 1/3, 1/3], device="cuda")
refined_V, history = refine_with_geo_reward(
    V, F, weights, steps=50, lr=0.005,
    sym_normal=sym_n, sym_offset=sym_d,
)
```

## Reproducing paper results

Each experiment script is self-contained and writes JSON + OBJ outputs under `results/` and `analysis_results/`.

```bash
# (1) Main benchmark — 110 prompts × 3 seeds × 6 methods (~4 h on V100)
python tools/exp_k_full_mesh_validity.py

# (2) Classical baselines at matched runtime
python tools/exp_classical_baselines_matched.py

# (3) Matched-shape-CD Pareto sweep (iters 1,2,3,5,10,20,40,80)
python tools/exp_pareto_distortion_sweep.py

# (4) Single-reward + 2-reward ablation (needs baseline meshes from (1))
python tools/exp_two_reward_combo.py
python tools/exp_pareto_component_ablation.py

# (5) Non-circular downstream tests
python tools/exp_psr_selfconsistency.py
python tools/exp_pca_stability.py

# (6) Cross-backbone — TripoSR / InstantMesh (require those repos on disk)
TRIPOSR_PATH=/path/to/TripoSR python tools/exp_instantmesh_backbone.py
```

The `_plane_protocol.py` helper builds a per-(prompt, seed) plane cache that is shared across all methods to keep the comparison paired.

## Key hyperparameters (paper defaults)

| Parameter | Value | Notes |
|---|---|---|
| Steps | 50 | Adam, full-batch vertex optimisation |
| Learning rate | 5e-3 | Adam default betas |
| Equal weights | (1/3, 1/3, 1/3) | The recommended default |
| Huber δ | mesh-adaptive (median dihedral at step 0) | Fixed for the run |
| Plane estimator | 3 PCA + 3 coord + 16 Fibonacci + top-3 refined | Multi-start search |

## Checkpoints

The optional Lang2Comp MLP (~600 KB) trained on 10K synthetic templates is not shipped in this repo (under 50 MB GitHub limit, but kept separate for clarity). It can be reproduced end-to-end with `src/train_lang2comp.py` in under 2 minutes on CPU.

## Data

Prompts: `src/prompts_gpteval3d.py` reproduces the 110 GPTEval3D prompts (symmetry / smoothness / compactness categories) used throughout the paper. OOD generalisation prompts for Lang2Comp are embedded in `tools/exp_lang2comp_generalization.py`.

Baseline meshes are regenerated from Shap-E with seeds {42, 123, 456} — the full benchmark takes ~3-4 h on a single V100 (dominated by Shap-E sampling, not refinement).

## License

Released under the MIT License; see `LICENSE`.

Third-party components retain their upstream licenses: Shap-E (MIT), TripoSR (MIT), InstantMesh (Apache-2.0), sentence-transformers (Apache-2.0), Open3D (MIT), trimesh (MIT), PyMeshLab (GPL-3.0), PyTorch (BSD-3-Clause).

## Citation

Under anonymous review. Will be added upon acceptance.
