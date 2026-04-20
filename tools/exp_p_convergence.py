"""
Exp P: DGR Optimization Convergence + LR Sensitivity.
Justifies hyperparameter choices (steps=50, lr=0.005) for the paper.

Part A — Convergence curve:
  30 prompts (10 per category, strict balanced) × 3 seeds = 90 (prompt, seed) pairs
  Record all 3 metrics at: [0, 5, 10, 20, 30, 40, 50, 75, 100]
  Report: mean ± std across 90 pairs at each step.
  Claims supported: DGR improves monotonically; plateau by ~50 steps.

Part B — LR sensitivity:
  15 prompts (5 per category, balanced) × 1 seed × lr in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
  Steps fixed at DEFAULT_STEPS=50. Final metrics + std reported.
  Claim supported: lr=0.005 is in the robust flat region.

Sampling is strictly balanced per category (not stride-based, which gave 8/7/5).
Both parts reuse Shap-E baseline meshes from exp_k (fallback to Shap-E generation).

~30min on V100.
Output: analysis_results/convergence/all_results.json
"""

import os, sys, json, time, re
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS   = [42, 123, 456]        # 3 seeds for variance estimate
DGR_W   = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)

# Part A: convergence
N_CONV_PER_CAT = 10             # 10 per category = 30 total, strictly balanced
MAX_STEPS      = 100
RECORD_STEPS   = [0, 5, 10, 20, 30, 40, 50, 75, 100]
DEFAULT_LR     = 0.005

# Part B: LR sensitivity
N_LR_PER_CAT  = 5               # 5 per category = 15 total, balanced
LR_VALUES     = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
LR_SEED       = 42              # single seed (LR effect is the variable of interest)
DEFAULT_STEPS = 50

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CAT  = {}
for p in SYMMETRY_PROMPTS:   PROMPT_CAT[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS: PROMPT_CAT[p] = "smoothness"
for p in COMPACTNESS_PROMPTS:PROMPT_CAT[p] = "compactness"

# Strictly balanced prompt subsets (N per category, evenly spaced within each list)
def _balanced_subset(lst, n):
    """Pick n prompts evenly spaced from lst."""
    if len(lst) <= n:
        return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]

CONV_PROMPTS = (_balanced_subset(SYMMETRY_PROMPTS,  N_CONV_PER_CAT) +
                _balanced_subset(SMOOTHNESS_PROMPTS, N_CONV_PER_CAT) +
                _balanced_subset(COMPACTNESS_PROMPTS,N_CONV_PER_CAT))
LR_PROMPTS   = (_balanced_subset(SYMMETRY_PROMPTS,  N_LR_PER_CAT) +
                _balanced_subset(SMOOTHNESS_PROMPTS, N_LR_PER_CAT) +
                _balanced_subset(COMPACTNESS_PROMPTS,N_LR_PER_CAT))

OBJ_DIR = Path("results/mesh_validity_objs/baseline")
OUT_DIR = Path("analysis_results/convergence")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ── Mesh I/O ──────────────────────────────────────────────────────────────────

def load_obj(path):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                idx = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    if not verts or not faces:
        return None, None  # degenerate mesh: caller falls back to Shap-E
    v = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
    f = torch.tensor(faces, dtype=torch.long,    device=DEVICE)
    if v.dim() != 2 or f.dim() != 2:
        return None, None
    return v, f


def get_baseline_mesh(prompt, seed):
    """Load baseline OBJ from exp_k if available, else generate via Shap-E."""
    cat  = PROMPT_CAT[prompt]
    path = OBJ_DIR / cat / f"{slug(prompt)}_seed{seed}.obj"
    if path.exists():
        return load_obj(path)

    # Fallback: generate via Shap-E
    try:
        from shape_gen import load_shap_e, generate_mesh
        print(f"  Generating Shap-E mesh for '{prompt[:40]}'...")
        xm, model, diffusion = load_shap_e(device=DEVICE)
        torch.manual_seed(seed)
        mesh_list = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        verts, faces, _ = mesh_list[0]
        del xm, model, diffusion
        torch.cuda.empty_cache()
        return verts, faces
    except Exception as e:
        print(f"  Shap-E fallback failed: {e}")
        return None, None


# ── Differentiable optimization with metric recording ─────────────────────────

def run_dgr_with_checkpoints(verts, faces, weights, record_steps, lr):
    """
    Run Adam optimizer on vertices, record metrics at each step in record_steps.
    Returns list of {step: int, symmetry: float, smoothness: float, compactness: float}.
    """
    v = torch.tensor(verts.cpu().numpy(), dtype=torch.float32,
                     device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=lr)

    # Initial normalization values
    # symmetry_reward(vertices, axis=1) — no faces argument
    with torch.no_grad():
        sym0   = max(abs(symmetry_reward(v).item()),          1e-6)
        smo0   = max(abs(smoothness_reward(v, faces).item()), 1e-6)
        com0   = max(abs(compactness_reward(v, faces).item()),1e-6)

    max_step  = max(record_steps)
    record_set = set(record_steps)
    records   = []

    for step in range(max_step + 1):
        if step in record_set:
            with torch.no_grad():
                sym = symmetry_reward(v).item()          # no faces arg
                smo = smoothness_reward(v, faces).item()
                com = compactness_reward(v, faces).item()
            records.append(dict(step=step,
                                symmetry=sym, smoothness=smo, compactness=com))

        if step < max_step:
            optimizer.zero_grad()
            sym_r = symmetry_reward(v)                   # no faces arg
            smo_r = smoothness_reward(v, faces)
            com_r = compactness_reward(v, faces)
            loss  = -(weights[0]*sym_r/sym0 + weights[1]*smo_r/smo0 + weights[2]*com_r/com0)
            loss.backward()
            optimizer.step()

    return records


# ── Part A: Convergence ───────────────────────────────────────────────────────

def run_convergence():
    print("\n" + "="*60)
    print("PART A — Convergence Curve")
    print(f"  Prompts: {len(CONV_PROMPTS)}  |  Seeds: {SEEDS}  |  "
          f"Record steps: {RECORD_STEPS}  |  LR: {DEFAULT_LR}")
    print("="*60)

    ckpt = OUT_DIR / "convergence_checkpoint.json"
    results = []
    done_keys = set()
    if ckpt.exists():
        with open(ckpt) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["seed"]) for r in results
                     if r.get("experiment") == "convergence"}
        print(f"  Resuming: {len(done_keys)} (prompt,seed) pairs done")

    weights = DGR_W.to(DEVICE)
    t0 = time.time()
    n_total = len(CONV_PROMPTS) * len(SEEDS)

    for i, prompt in enumerate(CONV_PROMPTS):
        cat = PROMPT_CAT[prompt]
        for seed in SEEDS:
            key = (prompt, seed)
            if key in done_keys:
                continue

            verts, faces = get_baseline_mesh(prompt, seed)
            if verts is None:
                print(f"  No mesh: {prompt[:40]} seed={seed}")
                continue

            try:
                checkpoints = run_dgr_with_checkpoints(
                    verts, faces, weights, RECORD_STEPS, lr=DEFAULT_LR
                )
                for ck in checkpoints:
                    results.append(dict(prompt=prompt, seed=seed, category=cat,
                                        lr=DEFAULT_LR, experiment="convergence", **ck))
                done_keys.add(key)
            except Exception as e:
                print(f"  [{i+1}] Error seed={seed}: {e}")
                continue

        # checkpoint after each prompt (all its seeds)
        with open(ckpt, 'w') as f:
            json.dump(results, f)
        print(f"  [{i+1}/{len(CONV_PROMPTS)}] {prompt[:40]}  "
              f"{len(done_keys)}/{n_total} pairs  {time.time()-t0:.0f}s")

    n_pairs = len({(r["prompt"], r["seed"]) for r in results
                   if r.get("experiment") == "convergence"})
    print(f"  Convergence done: {n_pairs} (prompt,seed) pairs, {len(results)} records")
    return results


# ── Part B: LR Sensitivity ────────────────────────────────────────────────────

def run_lr_sensitivity():
    print("\n" + "="*60)
    print("PART B — LR Sensitivity")
    print(f"  Prompts: {len(LR_PROMPTS)} ({N_LR_PER_CAT}/category)  |  "
          f"Seed: {LR_SEED}  |  Steps: {DEFAULT_STEPS}  |  LRs: {LR_VALUES}")
    print("="*60)

    ckpt = OUT_DIR / "lr_sensitivity_checkpoint.json"
    results = []
    done_keys = set()
    if ckpt.exists():
        with open(ckpt) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["lr"]) for r in results}
        print(f"  Resuming: {len(done_keys)} (prompt, lr) done")

    weights = DGR_W.to(DEVICE)
    t0 = time.time()

    for prompt in LR_PROMPTS:
        cat = PROMPT_CAT[prompt]
        verts, faces = get_baseline_mesh(prompt, LR_SEED)
        if verts is None:
            continue

        # baseline (step=0)
        if (prompt, 0.0) not in done_keys:
            try:
                sym = symmetry_reward(verts).item()      # no faces arg
                smo = smoothness_reward(verts, faces).item()
                com = compactness_reward(verts, faces).item()
                results.append(dict(prompt=prompt, seed=LR_SEED, category=cat,
                                    lr=0.0, step=0, experiment="lr_sensitivity",
                                    symmetry=sym, smoothness=smo, compactness=com))
                done_keys.add((prompt, 0.0))
            except Exception:
                pass

        for lr_val in LR_VALUES:
            if (prompt, lr_val) in done_keys:
                continue
            try:
                # run DEFAULT_STEPS, record only final step
                cks = run_dgr_with_checkpoints(
                    verts, faces, weights,
                    record_steps=[0, DEFAULT_STEPS], lr=lr_val
                )
                # take final record
                final = cks[-1]
                results.append(dict(prompt=prompt, seed=LR_SEED, category=cat,
                                    lr=lr_val, step=DEFAULT_STEPS,
                                    experiment="lr_sensitivity",
                                    symmetry=final["symmetry"],
                                    smoothness=final["smoothness"],
                                    compactness=final["compactness"]))
                done_keys.add((prompt, lr_val))
            except Exception as e:
                print(f"  LR={lr_val} {prompt[:30]}: {e}")

        with open(ckpt, 'w') as f:
            json.dump(results, f)

    print(f"  LR sensitivity done: {len(results)} records  {time.time()-t0:.0f}s")
    return results


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_convergence(results):
    conv = [r for r in results if r.get("experiment") == "convergence"]
    if not conv:
        return

    n_pairs = len({(r["prompt"], r["seed"]) for r in conv})
    print(f"\n--- Convergence (mean ± std, n={n_pairs} prompt×seed pairs) ---")
    print(f"{'Step':>6} | {'Sym mean±std':>16} | {'Smo mean±std':>16} | {'Com mean±std':>16} | N")
    print("-" * 75)

    for step in RECORD_STEPS:
        recs = [r for r in conv if r["step"] == step]
        if not recs:
            continue
        sym_v = [r["symmetry"]    for r in recs if r.get("symmetry")    is not None]
        smo_v = [r["smoothness"]  for r in recs if r.get("smoothness")  is not None]
        com_v = [r["compactness"] for r in recs if r.get("compactness") is not None]
        n = min(len(sym_v), len(smo_v), len(com_v))
        if n == 0:
            continue
        step_lbl = "0(BL)" if step == 0 else str(step)
        print(f"{step_lbl:>6} | "
              f"{np.mean(sym_v):>+8.5f}±{np.std(sym_v):.4f} | "
              f"{np.mean(smo_v):>+8.5f}±{np.std(smo_v):.4f} | "
              f"{np.mean(com_v):>+8.5f}±{np.std(com_v):.4f} | {n}")

    # Δ% vs step=0 with bootstrap SE
    base = [r for r in conv if r["step"] == 0]
    if len(base) < 5:
        return
    rng = np.random.default_rng(0)
    print(f"\n  Δ% vs step=0 (mean ± bootstrap SE, n_boot=500):")
    bl = {m: np.array([r[m] for r in base if r.get(m) is not None])
          for m in ["symmetry", "smoothness", "compactness"]}
    for step in [10, 20, 30, 50, 75, 100]:
        recs = [r for r in conv if r["step"] == step]
        if not recs:
            continue
        row = []
        for m in ["symmetry", "smoothness", "compactness"]:
            vals = np.array([r[m] for r in recs if r.get(m) is not None])
            if len(vals) == 0 or abs(bl[m].mean()) < 1e-12:
                row.append("    --  ")
                continue
            delta_mean = (vals.mean() - bl[m].mean()) / abs(bl[m].mean()) * 100
            # bootstrap SE of the delta %
            boot = []
            for _ in range(500):
                idx_a = rng.integers(0, len(bl[m]), len(bl[m]))
                idx_b = rng.integers(0, len(vals),  len(vals))
                d = (vals[idx_b].mean() - bl[m][idx_a].mean()) / abs(bl[m][idx_a].mean()) * 100
                boot.append(d)
            se = np.std(boot)
            row.append(f"{delta_mean:>+5.1f}%±{se:.1f}")
        print(f"    step {step:>3}: sym {row[0]}  smo {row[1]}  com {row[2]}")


def analyze_lr(results):
    lr_data = [r for r in results if r.get("experiment") == "lr_sensitivity"]
    if not lr_data:
        return

    n_prompts = len(set(r["prompt"] for r in lr_data))
    print(f"\n--- LR Sensitivity (mean ± std, n={n_prompts} prompts, step={DEFAULT_STEPS}) ---")
    print(f"{'LR':>8} | {'Sym mean±std':>16} | {'Smo mean±std':>16} | {'Com mean±std':>16} | N")
    print("-" * 75)

    # baseline (lr=0.0, step=0)
    bl_recs = [r for r in lr_data if r["lr"] == 0.0]
    if bl_recs:
        sym_v = [r["symmetry"]    for r in bl_recs]
        smo_v = [r["smoothness"]  for r in bl_recs]
        com_v = [r["compactness"] for r in bl_recs]
        print(f"{'baseline':>8} | "
              f"{np.mean(sym_v):>+8.5f}±{np.std(sym_v):.4f} | "
              f"{np.mean(smo_v):>+8.5f}±{np.std(smo_v):.4f} | "
              f"{np.mean(com_v):>+8.5f}±{np.std(com_v):.4f} | {len(bl_recs)}")

    for lr_val in LR_VALUES:
        recs = [r for r in lr_data if r["lr"] == lr_val]
        if not recs:
            continue
        sym_v = [r["symmetry"]    for r in recs]
        smo_v = [r["smoothness"]  for r in recs]
        com_v = [r["compactness"] for r in recs]
        print(f"{lr_val:>8.4f} | "
              f"{np.mean(sym_v):>+8.5f}±{np.std(sym_v):.4f} | "
              f"{np.mean(smo_v):>+8.5f}±{np.std(smo_v):.4f} | "
              f"{np.mean(com_v):>+8.5f}±{np.std(com_v):.4f} | {len(recs)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Part A
    conv_results = run_convergence()

    # Part B
    lr_results = run_lr_sensitivity()

    # Merge and save
    all_results = conv_results + lr_results
    with open(OUT_DIR / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Analysis
    analyze_convergence(all_results)
    analyze_lr(all_results)

    print(f"\nTotal: {len(all_results)} records  {time.time()-t0:.0f}s")
    print(f"Output: {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
