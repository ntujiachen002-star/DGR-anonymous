# DiffGeoReward

Training-free inference-time mesh refinement via three differentiable geometric rewards (symmetry, Huber normal consistency, compactness).

> Anonymous code release accompanying the paper **"DiffGeoReward: Differentiable Geometric Rewards for Language-Guided 3D Shape Refinement"** (under review).

## Overview

Given a text-to-3D baseline mesh (Shap-E / TripoSR / InstantMesh / TRELLIS), DiffGeoReward runs 50 steps of Adam on vertex positions to maximise a weighted combination of three closed-form geometric rewards:

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
  exp_resolution_robustness.py    Resolution-stratified analysis (Appendix Table 4)
  nips_push/                      NeurIPS rebuttal-package experiments (see REPRODUCE.md)
    exp_causal_plane.py             4-estimator causal test: plane quality -> PCGrad benefit
    analyze_causal_plane.py         Figures + correlations for the 4-estimator experiment
    exp_causal_plane_perturbation.py  Controlled plane-perturbation causal experiment (Rodrigues rotation, isolates angle from estimator confounds)
    analyze_causal_plane_perturbation.py  Pooled Spearman + per-mesh Kendall tau + mixed-effects regression
    exp_adversarial_triposr.py      50-prompt adversarial stress test on TripoSR+SDXL
    exp_sota_baselines.py           HC Laplacian / ARAP / two-step normal denoising
    exp_trellis_v2.py               TRELLIS-text-large backbone benchmark (180 runs)
    exp_voxel_symmetry_downstream.py Volumetric V-IoU downstream test (third non-circular signal)
    analyze_voxel_symmetry.py       Paired t-test / Wilcoxon + LaTeX for V-IoU
    exp_rignet_downstream.py        Application-level downstream: run pretrained RigNet on baseline vs DGR meshes, compute skeleton-level metrics (JSE / angular / root offset / coverage)
    analyze_rignet.py               Paired stats + per-category breakdown + LaTeX for RigNet
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

## Quick start — single-mesh demo

```bash
# Unpack the included baseline meshes first
tar xzf data/baseline_meshes.tar.gz -C data/

# Refine one mesh in ~5 s on a GPU
python demo.py --input data/baseline/symmetry/a_symmetric_vase_seed42.obj \
               --output refined_vase.obj
```

`demo.py` prints the three baseline rewards, runs 50 Adam steps, and prints the refined values. For a symmetric-vase input you should see all three rewards increase (move toward zero).

### Library usage

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

For a table-by-table walkthrough (which script produces which number, expected runtime, expected output, sanity check commands), see **[REPRODUCE.md](REPRODUCE.md)**.

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

# (6) Cross-backbone — TripoSR / InstantMesh / TRELLIS (require those repos on disk)
TRIPOSR_PATH=$HOME/TripoSR python tools/exp_instantmesh_backbone.py
ATTN_BACKEND=xformers TRELLIS_PATH=$HOME/TRELLIS \
    python tools/nips_push/exp_trellis_v2.py

# (7) NeurIPS rebuttal-package experiments (see REPRODUCE.md for expected numbers):
python tools/nips_push/exp_causal_plane.py    # 4-estimator causal test (~8 min, n=528 mesh-plane pairs)
python tools/nips_push/analyze_causal_plane.py
python tools/nips_push/exp_causal_plane_perturbation.py \
    --n-meshes 100 --angles 0 5 10 20 45 90 --n-dirs 2   # controlled perturbation (~10 min, n=715 obs)
python tools/nips_push/analyze_causal_plane_perturbation.py
python tools/nips_push/exp_adversarial_triposr.py   # adversarial stress test (~3 h)
python tools/nips_push/exp_sota_baselines.py        # extra classical baselines (CPU, ~45 min)

# (8) Application-level downstream: RigNet skeletal rigging (n=76, ~1 h on V100)
#     Requires: https://github.com/zhan-xu/RigNet cloned at $RIGNET_PATH with
#     its pretrained checkpoints/ folder (see upstream README). Run under
#     xvfb-run for headless display; CPU/open3d fallbacks are documented inline.
RIGNET_PATH=$HOME/RigNet xvfb-run -a \
    python tools/nips_push/exp_rignet_downstream.py
python tools/nips_push/analyze_rignet.py
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

## Included assets

| Path | Size | What it is |
|---|---|---|
| `checkpoints/lang2comp_v2_lam050.pt` | 88 MB | Trained Lang2Comp MLP used for the paper's **targeted** Lang2Comp rows (per-category diagonal: +98.1% sym, +24.8% HNC, +60.8% com). Load with `Lang2Comp()` in `src/lang2comp.py`. |
| `checkpoints/lang2comp_best.pt` | 88 MB | Earlier Lang2Comp checkpoint used by `tools/exp_lang2comp_generalization.py` (OOD dominant-property accuracy: 85.6%) and `tools/generate_qualitative_meshes.py`. |
| `data/plane_cache.json` | 65 KB | Pre-estimated bilateral-symmetry plane for every `(prompt, seed)` pair (330 entries). The multi-start plane estimator has non-deterministic randomness, so **using this cache is required for bit-exact reproduction** of the paper's numerical tables. |
| `data/baseline_meshes.tar.gz` | 4.2 MB | The 330 Shap-E baseline OBJs (110 prompts × 3 seeds) used across all paper experiments. Unpack with `tar xzf data/baseline_meshes.tar.gz -C data/` to skip Shap-E setup entirely; every refinement experiment reads from `data/baseline/{symmetry,smoothness,compactness}/`. |

Third-party backbone weights (Shap-E / TripoSR / InstantMesh) are **not included** — reviewers who want to regenerate meshes from scratch should download them following those projects' upstream instructions. Everything else in this repo works without those backbones.

Lang2Comp can be retrained end-to-end with `src/train_lang2comp.py` in under 2 minutes on CPU if you want to reproduce the training pipeline.

## Expected results

Running the main benchmark on the provided baseline meshes with the default `lang2comp_v2_lam050.pt` checkpoint should reproduce the following aggregate numbers (prompt-level means over 97 prompts × 3 seeds, same paired protocol as Table 1):

| Method | R_sym improvement | R_HNC improvement | R_compact improvement |
|---|---|---|---|
| Baseline (Shap-E) | — | — | — |
| **DGR (Equal Wt., 0.33/0.33/0.34)** | **+85.8%** | **+19.7%** | **+49.4%** |
| DGR (Equal Wt., exact 1/3/1/3/1/3) | +91.9% | +19.6% | +49.2% |
| DGR (PCGrad, 1/3/1/3/1/3) | +92.4% | +18.6% | +52.3% |
| Sym-Only | +92.8% | −7.4% | −58.9% |
| HNC-Only | −72.5% | +30.0% | −1.5% |
| Compact-Only | −100.7% | +1.7% | +67.8% |

If your numbers differ by more than ±0.5 pp, check: (a) you are using `data/plane_cache.json` (not re-running the plane estimator), (b) the Lang2Comp checkpoint is `lang2comp_v2_lam050.pt`, (c) random seeds are {42, 123, 456}.

### NeurIPS rebuttal-package results (from `tools/nips_push/`, see REPRODUCE.md for the full tables)

| Experiment | Key number |
|---|---|
| **Causal plane-quality, 4 estimators (`exp_causal_plane.py`)** | Spearman $\rho=+0.25$, $p=8.6\times 10^{-9}$ across $n=528$ mesh-plane pairs ($132$ meshes $\times$ $4$ plane estimators) — worse plane $\to$ larger PCGrad benefit on $R_\mathrm{sym}$ |
| **Causal plane-perturbation, controlled (`exp_causal_plane_perturbation.py`)** | Per-mesh Kendall $\tau$ median ${+}0.30$, $81\%$ of meshes ($53/65$) positive, Wilcoxon $p=2.4\times 10^{-7}$; pooled Spearman $\rho{=}{+}0.24$ across $n_\text{obs}{=}715$ — **Rodrigues rotation isolates angular error from estimator-specific confounds** |
| **Adversarial TripoSR (`exp_adversarial_triposr.py`)** | $150/150$ valid runs; $\Delta$CLIP $=-0.002$ overall; shape-CD $\approx 0.015$ (4$\times$ smaller than main benchmark) |
| **Extra classical baselines (`exp_sota_baselines.py`)** | HC Lap $+41/-8/-12$%; ARAP-analog $-4/0/-2$%; Normal-denoise catastrophic. DGR wins all three axes. |
| **TRELLIS backbone (`exp_trellis_v2.py`)** | $180/180$ valid; $R_\mathrm{sym}$ $+85.1$% ($d=0.74$), $R_\mathrm{smooth}$ $+24.5$%, $R_\mathrm{compact}$ $+64.8$% ($d=1.74$) |
| **Resolution-stratified (`exp_resolution_robustness.py`)** | Improvements stable across $|V|<30$ to $|V|\geq 1000$; dense subset still $+77/+23/+50$% |
| **Volumetric V-IoU (`exp_voxel_symmetry_downstream.py`)** | $n=80$ paired symmetry prompts; V-IoU $0.456 \to 0.634$ ($+39$%, $d=+1.55$, $96$% win, $p=8.8\times 10^{-23}$) |
| **RigNet skeletal rigging (`exp_rignet_downstream.py`)** | $n=76$ mutually-valid pairs; JSE $0.046 \to 0.031$ ($-33$%, paired $t$ $p{=}0.011$, $d{=}-0.30$, $68$% win); root-joint plane offset $-20$% ($70$% win); coverage preserved at $96.2$% vs $95.0$% (McNemar $p{=}1.0$). First application-level (not just geometric-proxy) downstream signal. |

## Data

Prompts: `src/prompts_gpteval3d.py` reproduces the 110 GPTEval3D prompts (symmetry / smoothness / compactness categories) used throughout the paper. OOD generalisation prompts for Lang2Comp are embedded in `tools/exp_lang2comp_generalization.py`.

Baseline meshes (regenerating them from scratch): run `tools/exp_k_full_mesh_validity.py` — this takes ~3-4 h on a single V100 and requires Shap-E installed. The `data/baseline_meshes.tar.gz` in this repo already contains the exact outputs, so most users will not need to do this.

## License

Released under the MIT License; see `LICENSE`.

Third-party components retain their upstream licenses: Shap-E (MIT), TripoSR (MIT), InstantMesh (Apache-2.0), sentence-transformers (Apache-2.0), Open3D (MIT), trimesh (MIT), PyMeshLab (GPL-3.0), PyTorch (BSD-3-Clause).

## Citation

Under anonymous review. Will be added upon acceptance.
