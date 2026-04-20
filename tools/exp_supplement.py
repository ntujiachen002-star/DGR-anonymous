"""
Supplement Experiment: Missing methods for paper

Adds the following to full_mgda_classical results:
  1. two_reward      — sym + creg only (w3=0), weights [0.5, 0.5, 0]
  2. sym_only        — single-reward ablation [1, 0, 0]
  3. smooth_only     — single-reward ablation [0, 1, 0]
  4. compact_only    — single-reward ablation [0, 0, 1]
  5. diffgeoreward   — Lang2Comp adaptive weights

Also reruns scale_controlled_ablation with current Huber NC code:
  6. sym_only_gradnorm
  7. smooth_only_gradnorm
  8. compact_only_gradnorm

Reuses baseline OBJs from full_mgda_classical (no GPU needed for refinement).

Usage:
    PYTHONPATH=src python tools/exp_supplement.py
"""

import os, sys, json, time, re
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import (symmetry_reward, smoothness_reward, compactness_reward,
                        compute_face_normals, _build_face_adjacency,
                        compute_initial_huber_delta)
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

SEEDS = [42, 123, 456]
STEPS = 50
LR = 0.005

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CAT = {}
for p in SYMMETRY_PROMPTS:   PROMPT_CAT[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS: PROMPT_CAT[p] = "smoothness"
for p in COMPACTNESS_PROMPTS: PROMPT_CAT[p] = "compactness"

OBJ_DIR = Path("results/full_mgda_classical_objs")
OUT_DIR = Path("analysis_results/supplement")
METRICS = ["symmetry", "smoothness", "compactness"]
INDEP_METRICS = ["edge_regularity", "normal_consistency_deg"]


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:60]


def load_obj(path, device='cpu'):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                face_idx = [int(x.split('/')[0])-1 for x in line.split()[1:4]]
                if len(face_idx) == 3:
                    faces.append(face_idx)
    v = torch.tensor(verts, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device).reshape(-1, 3)
    return v, f


def edge_length_regularity(vertices, faces):
    edges = torch.cat([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]], dim=0)
    edges_sorted = torch.sort(edges, dim=1).values
    edges_unique = torch.unique(edges_sorted, dim=0)
    lengths = (vertices[edges_unique[:,1]] - vertices[edges_unique[:,0]]).norm(dim=1)
    return (lengths.std() / (lengths.mean() + 1e-10)).item()


def normal_consistency_deg(vertices, faces):
    face_normals = compute_face_normals(vertices, faces)
    edge_dict = defaultdict(list)
    for fi in range(faces.shape[0]):
        for local in range(3):
            v0 = faces[fi, local].item()
            v1 = faces[fi, (local + 1) % 3].item()
            edge_dict[(min(v0,v1), max(v0,v1))].append(fi)
    angles = []
    for fids in edge_dict.values():
        if len(fids) == 2:
            cos_val = (face_normals[fids[0]] * face_normals[fids[1]]).sum().clamp(-1,1)
            angles.append(torch.acos(cos_val).item() * 180 / np.pi)
    return float(np.mean(angles)) if angles else 0.0


def compute_all_metrics(vertices, faces, sym_axis=1):
    with torch.no_grad():
        return {
            'symmetry': symmetry_reward(vertices, axis=sym_axis).item(),
            'smoothness': smoothness_reward(vertices, faces).item(),
            'compactness': compactness_reward(vertices, faces).item(),
            'edge_regularity': edge_length_regularity(vertices, faces),
            'normal_consistency_deg': normal_consistency_deg(vertices, faces),
        }


def refine_single_reward(vertices, faces, reward_name, steps=50, lr=0.005,
                         gradient_normalize=False, target_grad_norm=0.01):
    """Single-reward refinement with optional gradient normalization."""
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    # Precompute for smoothness
    face_adj = _build_face_adjacency(faces)
    huber_delta = compute_initial_huber_delta(v_opt, faces)

    # Initial scale
    with torch.no_grad():
        if reward_name == 'symmetry':
            r0 = abs(symmetry_reward(v_opt, axis=1).item()) + 1e-6
        elif reward_name == 'smoothness':
            r0 = abs(smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj).item()) + 1e-6
        else:
            r0 = abs(compactness_reward(v_opt, faces).item()) + 1e-6

    for step in range(steps):
        optimizer.zero_grad()
        if reward_name == 'symmetry':
            reward = symmetry_reward(v_opt, axis=1)
        elif reward_name == 'smoothness':
            reward = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj)
        else:
            reward = compactness_reward(v_opt, faces)

        loss = -(reward / r0)
        loss.backward()

        if gradient_normalize:
            grad_norm = v_opt.grad.norm().item()
            if grad_norm > 1e-10:
                v_opt.grad.data = v_opt.grad.data * (target_grad_norm / grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_([v_opt], 1.0)

        optimizer.step()

    return v_opt.detach()


def run_task(args):
    """Worker function for parallel refinement."""
    prompt, seed, method, cat, obj_path = args
    try:
        v_bl, f_bl = load_obj(obj_path, device='cpu')
        if f_bl.shape[0] < 4:
            return None

        W_TWO = torch.tensor([0.5, 0.5, 0.0])
        W_SYM = torch.tensor([1.0, 0.0, 0.0])
        W_SMO = torch.tensor([0.0, 1.0, 0.0])
        W_COM = torch.tensor([0.0, 0.0, 1.0])

        if method == 'two_reward':
            v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=W_TWO, steps=STEPS, lr=LR)
        elif method == 'sym_only':
            v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=W_SYM, steps=STEPS, lr=LR)
        elif method == 'smooth_only':
            v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=W_SMO, steps=STEPS, lr=LR)
        elif method == 'compact_only':
            v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=W_COM, steps=STEPS, lr=LR)
        elif method == 'sym_only_gradnorm':
            v_ref = refine_single_reward(v_bl.clone(), f_bl, 'symmetry', steps=STEPS, lr=LR, gradient_normalize=True)
        elif method == 'smooth_only_gradnorm':
            v_ref = refine_single_reward(v_bl.clone(), f_bl, 'smoothness', steps=STEPS, lr=LR, gradient_normalize=True)
        elif method == 'compact_only_gradnorm':
            v_ref = refine_single_reward(v_bl.clone(), f_bl, 'compactness', steps=STEPS, lr=LR, gradient_normalize=True)
        elif method == 'diffgeoreward':
            from lang2comp import Lang2Comp; import torch as _torch
            lc = Lang2Comp(); lc.load_state_dict(_torch.load('checkpoints/lang2comp_best.pt', map_location='cpu')); lc.eval()
            pred = lc.predict(prompt); weights = _torch.tensor([pred['weights']['symmetry'], pred['weights']['smoothness'], pred['weights']['compactness']])
            v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=weights, steps=STEPS, lr=LR)
        else:
            return None

        ref_metrics = compute_all_metrics(v_ref, f_bl)
        return {
            'prompt': prompt, 'seed': seed, 'method': method,
            'category': cat, **ref_metrics,
        }
    except Exception as e:
        print(f"  [ERR] {method} {prompt[:30]} s={seed}: {e}", flush=True)
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results to get baseline data
    existing_path = Path("analysis_results/full_mgda_classical/all_results.json")
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing records")
    else:
        existing = []

    # Collect all baseline OBJ paths
    baseline_objs = {}
    for obj_path in (OBJ_DIR / "baseline").rglob("*.obj"):
        parts = obj_path.stem.rsplit("_s", 1)
        if len(parts) == 2:
            ps_name, seed_str = parts
            for prompt in ALL_PROMPTS:
                if slug(prompt) == ps_name:
                    try:
                        baseline_objs[(prompt, int(seed_str))] = str(obj_path)
                    except ValueError:
                        pass
                    break

    print(f"Found {len(baseline_objs)} baseline OBJs")

    # Methods to run
    supplement_methods = [
        'two_reward',
        'sym_only', 'smooth_only', 'compact_only',
        'sym_only_gradnorm', 'smooth_only_gradnorm', 'compact_only_gradnorm',
        'diffgeoreward',
    ]

    # Check what's already done
    ckpt_path = OUT_DIR / "checkpoint.json"
    results = []
    done_keys = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        done_keys = {(r['prompt'], r['seed'], r['method']) for r in results}
        print(f"Resuming: {len(results)} supplement records done")

    # Build task list
    tasks = []
    for (prompt, seed), obj_path in baseline_objs.items():
        cat = PROMPT_CAT.get(prompt, 'unknown')
        for method in supplement_methods:
            if (prompt, seed, method) not in done_keys:
                tasks.append((prompt, seed, method, cat, obj_path))

    print(f"\n{len(tasks)} tasks to run ({len(supplement_methods)} methods x {len(baseline_objs)} meshes)")

    if not tasks:
        print("All tasks done!")
    else:
        import multiprocessing as mp
        N_WORKERS = min(5, max(1, mp.cpu_count() - 2))
        print(f"Using {N_WORKERS} CPU workers\n")

        t0 = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result is not None:
                    results.append(result)
                    done_keys.add((result['prompt'], result['seed'], result['method']))

                if completed % 100 == 0:
                    with open(ckpt_path, 'w') as f:
                        json.dump(results, f)
                    elapsed = time.time() - t0
                    print(f"  [checkpoint] {completed}/{len(tasks)} tasks, "
                          f"{len(results)} records, {elapsed/60:.1f}min", flush=True)

        elapsed = time.time() - t0
        print(f"\nDone: {len(results)} supplement records, {elapsed/60:.1f}min")

    # Save results
    with open(ckpt_path, 'w') as f:
        json.dump(results, f)
    with open(OUT_DIR / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Also copy baseline records from existing for unified stats
    baseline_recs = [r for r in existing if r['method'] == 'baseline']
    combined = baseline_recs + results
    with open(OUT_DIR / "combined_results.json", 'w') as f:
        json.dump(combined, f, indent=2)

    # Print summary stats
    from scipy import stats as sp_stats

    print(f"\n{'='*80}")
    print("PAIRED STATISTICS vs BASELINE")
    print(f"{'='*80}")

    bl_dict = {(r['prompt'], r['seed']): r for r in baseline_recs}

    for method in supplement_methods:
        mt_recs = [r for r in results if r['method'] == method]
        mt_dict = {(r['prompt'], r['seed']): r for r in mt_recs}
        common = sorted(set(bl_dict) & set(mt_dict))
        if len(common) < 10:
            print(f"\n--- {method}: only {len(common)} pairs, skipping ---")
            continue

        print(f"\n--- {method} (n={len(common)}) ---")
        for metric in METRICS:
            a = [bl_dict[k][metric] for k in common]
            b = [mt_dict[k][metric] for k in common]
            t_val, p_raw = sp_stats.ttest_rel(b, a)
            diff = np.array(b) - np.array(a)
            d = diff.mean() / (diff.std(ddof=1) + 1e-10)
            delta_pct = (np.mean(b) - np.mean(a)) / (abs(np.mean(a)) + 1e-10) * 100
            wr = np.mean(np.array(b) > np.array(a)) * 100
            sig = "*" if p_raw < 0.05 else ""
            print(f"  {metric:15s}: {delta_pct:+8.1f}% (p={p_raw:.2e}, d={d:+.3f}){sig} wr={wr:.1f}%")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
