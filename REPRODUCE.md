# Reproducibility Guide

This document maps each numerical claim in the paper to the exact script that produces it, the expected runtime, the output paths, and reference numbers for sanity checking.

We assume you have already:
1. Created a conda env and installed `requirements.txt`.
2. Unpacked the included baseline meshes: `tar xzf data/baseline_meshes.tar.gz -C data/` (creates `data/baseline/{symmetry,smoothness,compactness}/*.obj`, 330 files).
3. Verified `data/plane_cache.json` (65 KB, 330 entries) is present.
4. Verified `checkpoints/lang2comp_v2_lam050.pt` and `checkpoints/lang2comp_best.pt` are present.

Everything below assumes a single NVIDIA V100 32 GB GPU. Numbers above 90% of what we report can be reproduced on any CUDA-capable GPU with ≥ 16 GB.

**Tolerance policy.** DGR's vertex optimiser, multi-start plane estimator, and Shap-E sampling are all deterministic given fixed seeds; using the included `data/plane_cache.json` eliminates the only non-deterministic piece, so your numbers should match the paper to within ±0.1 pp on every reward and within ±0.2% on win rates. If a number drifts further, check that (a) the plane cache is being used, (b) the Lang2Comp checkpoint is `lang2comp_v2_lam050.pt`, (c) the three seeds are {42, 123, 456}.

---

## Main-body results

### Table 1 — Overall comparison  (Sec. 4.1)

**Claim.** Baseline, DGR (Equal Wt.), DGR (Two-reward), DGR (PCGrad), prompt-level means over $n=97$ prompts × 3 seeds.

| Paper number | Our reference |
|---|---|
| Equal Wt. $(0.33, 0.33, 0.34)$: $+85.8\%$ sym / $+19.7\%$ HNC / $+49.4\%$ com | same |
| Equal Wt. Cohen's $d$: $+1.17 / +2.70 / +0.74$ | same |
| Two-reward $(w_3{=}0)$: $+95.1\% / +17.2\% / -71.0\%$ | same |
| PCGrad $(1/3,1/3,1/3)$: $+92.4\% / +18.6\% / +52.3\%$ | same |

```bash
# Full main benchmark. All six methods are produced in one pass.
python tools/exp_k_full_mesh_validity.py
```

**Inputs.** `data/baseline/{cat}/*.obj` (330 OBJs — already unpacked) and `data/plane_cache.json`. If Shap-E is not installed, the script still runs because baselines are loaded from disk.

**Runtime.** ≈ 4 h on V100 (dominated by the 6 methods × 330 baseline runs × 50 Adam steps). The refinement itself is ~5-8 s per mesh; the overhead is loading, plane lookup, and OBJ writing.

**Outputs.**
- Refined meshes: `results/mesh_validity_objs/{baseline,diffgeoreward,sym_only,HNC_only,compact_only,two_reward,pcgrad}/{cat}/*.obj`
- Aggregate JSON: `analysis_results/mesh_validity_full/summary.json`

**How to sanity-check.**
```bash
# Print per-method per-metric means from summary.json
python - <<'EOF'
import json
d = json.load(open('analysis_results/mesh_validity_full/summary.json'))
for m in ['diffgeoreward','sym_only','HNC_only','compact_only']:
    s = d[m]
    print(f"{m:15s}  sym={s['sym_improvement_pct']:+.1f}%  hnc={s['hnc_improvement_pct']:+.1f}%  com={s['com_improvement_pct']:+.1f}%")
EOF
```
Expected first line (Equal Wt., field key is `diffgeoreward`): `sym=+85.8%  hnc=+19.7%  com=+49.4%`.

---

### Independent metrics  (Sec. 4.1)

**Claim.** Edge regularity $+6.2\%$, angular normal deviation $-20.0\%$, volume change $<3\%$, CLIP shift $-0.003$, ImageReward $-0.06$.

Edge/angular: produced alongside Table 1 output.
CLIP + ImageReward:
```bash
python tools/exp_m_clip_allmethod.py
```
Runtime: ~ 30 min (CPU + CLIP inference on rendered views). Requires CLIP ViT-B/32 weights downloaded automatically from HuggingFace on first run.

Output: `analysis_results/clip_allmethod/clip_scores.json`.

---

### PSR self-consistency  (Sec. 4.1, Appendix sec:psr_appendix)

**Claim.** $+26.8\%$ improvement, Wilcoxon $p = 4.7 \times 10^{-8}$, $n = 129$ paired meshes (after filtering).

```bash
python tools/exp_psr_selfconsistency.py
```

**Runtime.** ~15 min CPU (screened Poisson reconstruction via Open3D; no GPU).

**Output.** `analysis_results/psr_selfconsistency/results.json`.

**Sanity check.** The JSON has a top-level `"aggregate"` block with `"mean_dgr_minus_baseline_pct": 26.8` and `"wilcoxon_p": 4.7e-8`. BH-corrected per-category results in `"per_category"`.

---

### PCA axis stability  (Sec. 4.1, Appendix sec:pca_appendix)

**Claim.** $+17.5\%$ more stable angular deviation on the symmetry-prompt subset, $p_{\text{BH}} = 0.029$, $n = 95$.

```bash
python tools/exp_pca_stability.py
```

**Runtime.** ~5 min CPU (PCA on noise-perturbed bootstrap samples).

**Output.** `analysis_results/pca_stability/results.json`. Look for `"symmetry_subset"` → `"mean_improvement_pct": 17.5`.

---

### Cross-backbone generalization  (Sec. 4.5, Appendix sec:backbone_appendix)

**Claim.** Symmetry $+91\%$ Shap-E / $+86\%$ TripoSR / $-1\%$ InstantMesh (not significant, $p{=}0.79$).

Requires upstream repos cloned and models downloaded separately (not in this release):

```bash
# TripoSR
git clone https://github.com/VAST-AI-Research/TripoSR ~/TripoSR
export TRIPOSR_PATH=~/TripoSR

# InstantMesh
git clone https://github.com/TencentARC/InstantMesh ~/InstantMesh
export INSTANTMESH_PATH=~/InstantMesh

python tools/exp_triposr_backbone_v2.py   # TripoSR row
python tools/exp_instantmesh_backbone.py  # InstantMesh row
```

**Runtime.** ~1 h each (single-image 3D reconstruction + refinement loop).

**Outputs.** `analysis_results/triposr_backbone_v2/all_results.json` and `analysis_results/instantmesh_backbone/all_results.json`.

**Sanity check.** `"aggregate"` → `"sym_improvement_pct"` should be +85 to +87 (TripoSR) and within ±1 (InstantMesh, not significant).

---

### Lang2Comp  (Sec. 4.5, appendix sec:lang2comp_arch)

**Claim.** Targeted diagonal: within each category, Lang2Comp's per-prompt weights give $+98.1\%$ sym / $+24.8\%$ HNC / $+60.8\%$ com. OOD accuracy 85.6% on 90 held-out prompts.

```bash
# (a) Targeted diagonal (paper's main Lang2Comp rows)
python tools/run_lang2comp_exp.py     # uses checkpoints/lang2comp_v2_lam050.pt
python tools/per_category_abs.py      # aggregates diagonal

# (b) OOD generalisation
python tools/exp_lang2comp_generalization.py   # uses checkpoints/lang2comp_best.pt
```

**Runtime.** (a) ~45 min. (b) ~10 min (text-only classification, no GPU required).

**Outputs.** (a) `analysis_results/lang2comp_v2_lam050/all_results.json`. (b) `analysis_results/lang2comp_generalization/results.json`.

**Sanity check.** (a) The `per_category_abs.py` stdout lists `Lang2Comp_v2 (lam=0.5)` with `symmetry: +98.1%, smoothness: +24.8%, compactness: +60.8%`. (b) Look for `"overall_accuracy": 0.856`.

---

## Appendix results

### Classical baselines at matched runtime  (Appendix sec:classical_sweep)

```bash
python tools/exp_classical_baselines_matched.py
```

**Runtime.** ~20 min CPU.

**Output.** `analysis_results/classical_baselines_matched/results.json`.

**Expected.** Laplacian 10-iter at matched shape-CD regresses HNC on 7 of 16 prompts; DGR is the only method improving all three reward axes. Concrete rows in the JSON map to Table 1 in the classical-sweep appendix section.

---

### Matched-distortion Pareto sweep  (Appendix sec:classical_pareto)

```bash
python tools/exp_pareto_distortion_sweep.py
```

**Runtime.** ~25 min CPU (iterates over `{1, 2, 3, 5, 10, 20, 40, 80}` classical-smoothing iteration counts).

**Output.** `analysis_results/classical_pareto/pareto_points.json` and `pareto.pdf` / `pareto.png`.

**Sanity check.** Per-operator per-iteration means; at iteration 1, Laplacian/Taubin reach shape-CD ≈ 0.098 with $\Rsym = -0.027$, $\Rsmooth = -0.551$, $\Rcompact = -28.9$ — DGR's operating point at shape-CD = 0.102 gives $\Rsym = -0.004$, $\Rsmooth = -0.486$, $\Rcompact = -14.2$ (paper Table in sec:classical_pareto).

---

### Component ablation: single- and 2-reward variants  (Appendix sec:pareto_component_ablation)

```bash
# Step 1. Generate the 3 two-reward combo meshes (single-reward already in exp_k output).
python tools/exp_two_reward_combo.py

# Step 2. Score all variants on the shape-CD / reward plane.
python tools/exp_pareto_component_ablation.py
```

**Runtime.** Step 1: ~4 min on V100. Step 2: ~30 s CPU.

**Outputs.**
- Step 1: `results/mesh_validity_objs/{sym_HNC, sym_com, HNC_com}/{cat}/*.obj` (≈ 230 each).
- Step 2: `analysis_results/pareto_component_ablation/ablation_points.json` and `pareto_ablation.pdf`.

**Expected** (from paper Table 42, $n=210$ paired meshes):

| Variant | shape-CD | R_sym | R_HNC | R_com |
|---|---|---|---|---|
| Baseline | 0.000 | −0.040 | −0.607 | −28.17 |
| Sym-Only | 0.070 | **−0.001** | −0.652 ↓ | −44.75 ↓ |
| HNC-Only | 0.099 | −0.069 ↓ | **−0.424** | −28.59 ↓ |
| Compact-Only | 0.143 | −0.080 ↓ | −0.597 | **−9.08** |
| Sym+HNC (no Com) | 0.083 | −0.002 | **−0.499** | −50.23 ↓ |
| Sym+Com (no HNC) | 0.123 | −0.002 | −0.627 ↓ | −12.41 |
| HNC+Com (no Sym) | 0.109 | −0.071 ↓ | −0.428 | −11.21 |
| **DGR (all 3)** | **0.098** | −0.004 | −0.487 | −14.09 |

Arrows mark values below the unrefined baseline. Only DGR has zero arrows.

---

### Scale-controlled ablation  (Appendix sec:scale_ablation_appendix)

```bash
python tools/exp_scale_controlled_ablation.py
```

**Runtime.** ~1 h on V100 (four single-reward variants × standard vs gradient-normalised).

**Expected.** Under Compact-Only, gradient normalisation roughly doubles the symmetry residual (from $-0.044$ to $-0.089$), consistent with directional rather than magnitude-based interference.

---

### Anti-collapse regularisation ablation  (Appendix sec:anticollapse_appendix)

```bash
python tools/exp_anticollapse.py
```

**Runtime.** ~90 min (sweep over penalty strengths).

**Expected.** Pareto-inferior at every strength when using the multi-start plane; matches the narrative in Sec. 4.2.

---

### Adversarial asymmetric prompts  (Appendix sec:adversarial_asymmetric)

```bash
python tools/exp_adversarial_asymmetric.py
```

**Runtime.** ~20 min (20 prompts × 3 seeds; Shap-E required).

**Expected.** $22 / 60$ runs produce coherent meshes; on those, $\Delta$CLIP within $-0.001$ — i.e. no semantic drift. The low valid-run count is reported as a scoping caveat in the paper.

---

### Plane-selection ablation  (Appendix sec:axis_ablation)

```bash
python tools/exp_plane_selection_ablation.py
```

**Runtime.** ~30 min.

**Expected.** Fixed `xz` plane under-performs multi-start by several pp of symmetry; multi-start and "6-coord + PCA + Fibonacci" perform within 0.2 pp of each other.

---

## Small-scale alternative (no full benchmark)

If you only want a minimal sanity check (~30 s) rather than the full 4 h run:

```bash
# Refine one mesh and print before/after rewards.
python demo.py --input data/baseline/symmetry/a_symmetric_vase_seed42.obj \
               --output refined_vase.obj
```

Expected stdout excerpt:
```
Baseline rewards:
       R_sym: -0.1XXX
    R_smooth: -0.6XXX
   R_compact: -3X.XX

Refined rewards:
       R_sym: -0.01XX  (improvement ≥ 80%)
    R_smooth: -0.4XXX  (improvement ≈ 20%)
   R_compact: -1X.XX   (improvement ≥ 50%)
```

If all three rewards move toward zero and none become more negative, the refinement is working end-to-end.

---

## Hardware footprint

Paper's full compute budget (Reproducibility Statement): ≈ 14 GPU-hours on V100 (primary), some cross-backbone runs on A100. Breakdown:

| Block | Script(s) | Approx runtime |
|---|---|---|
| Main benchmark (Table 1) | `exp_k_full_mesh_validity.py` | 3.5 h V100 |
| Classical + MGDA | `exp_classical_baselines_matched.py` + `exp_full_mgda_classical.py` | 3 h V100 |
| Cross-backbone | `exp_triposr_backbone_v2.py`, `exp_instantmesh_backbone.py` | 4.5 h mixed V100/A100 |
| Ablations (scale, anti-collapse, plane) | respective `exp_*` scripts | 2.5 h V100 |
| CLIP / ImageReward | `exp_m_clip_allmethod.py` | 0.5 h CPU + GPU |
| PSR, PCA, spectral, perceptual | CPU-only | < 30 min total |

Reviewers who only want Table 1 + the component ablation can run in ≈ 4 h total on a single V100.
