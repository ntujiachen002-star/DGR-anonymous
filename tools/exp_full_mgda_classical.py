"""
Full-Scale Experiment: MGDA + Classical Baselines

Runs all methods on 110 GPTEval3D prompts × 3 seeds = 330 runs per method.
Generates meshes via Shap-E, then applies each refinement method.

Methods:
  1. baseline          — raw Shap-E mesh (no refinement)
  2. handcrafted       — weighted sum, equal weights [1/3, 1/3, 1/3]
  3. mgda_0.05         — constrained MGDA with w_min=0.05
  4. laplacian         — Laplacian smoothing (classical baseline)
  5. normal_consist    — Normal consistency (classical baseline)
  6. classical_combined — Laplacian + NormalConsist + EdgeReg (combined classical)

Also computes independent evaluation metrics (not optimized by any method):
  - edge_regularity: coefficient of variation of edge lengths
  - normal_consistency_deg: mean angular deviation between adjacent face normals

Output: analysis_results/full_mgda_classical/
  - all_results.json         (all per-run records)
  - stats.json               (paired statistics vs baseline)
  - checkpoint.json          (for resuming interrupted runs)

Usage:
  python tools/exp_full_mgda_classical.py --device cuda:0
  python tools/exp_full_mgda_classical.py --device cuda:0 --resume  # resume from checkpoint

Estimated time: ~3-4h on A100 (330 Shap-E generations + 330×5 refinements)
"""

import os, sys, json, time, gc, re, argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / 'models_cache'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import (symmetry_reward, smoothness_reward, compactness_reward,
                        compute_face_normals)
from shape_gen import load_shap_e, refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456]
STEPS = 50
LR = 0.005
W_EQUAL = torch.tensor([1/3, 1/3, 1/3])
MGDA_WMIN = 0.05

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CAT = {}
for p in SYMMETRY_PROMPTS:   PROMPT_CAT[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS: PROMPT_CAT[p] = "smoothness"
for p in COMPACTNESS_PROMPTS: PROMPT_CAT[p] = "compactness"

OUT_DIR = Path("analysis_results/full_mgda_classical")
OBJ_DIR = Path("results/full_mgda_classical_objs")

METRICS = ["symmetry", "smoothness", "compactness"]
INDEP_METRICS = ["edge_regularity", "normal_consistency_deg"]


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:60]


# ── Independent Metrics ──────────────────────────────────────────────────────

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


# ── OBJ I/O ──────────────────────────────────────────────────────────────────

def save_obj(verts, faces, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    v, f = verts.cpu().numpy(), faces.cpu().numpy()
    with open(path, 'w') as fp:
        for vi in v: fp.write(f"v {vi[0]:.6f} {vi[1]:.6f} {vi[2]:.6f}\n")
        for fi in f: fp.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")


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
    v = torch.tensor(verts, dtype=torch.float32, device=device) if verts else torch.zeros(0, 3, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device).reshape(-1, 3) if faces else torch.zeros(0, 3, dtype=torch.long, device=device)
    return v, f


# ── MGDA Solver ──────────────────────────────────────────────────────────────

def mgda_solve_constrained(grads, w_min=0.05):
    n = len(grads)
    flat = [g.flatten() for g in grads]
    w = torch.ones(n) / n

    for _ in range(25):
        gw = sum(w[i] * flat[i] for i in range(n))
        dots = torch.tensor([torch.dot(flat[i], gw).item() for i in range(n)])
        t = dots.argmin().item()

        corner = torch.full((n,), w_min)
        corner[t] = 1.0 - w_min * (n - 1)

        d = corner - w
        gd = sum(d[i] * flat[i] for i in range(n))

        a = torch.dot(gd, gd)
        b = 2 * torch.dot(gw, gd)
        if a.abs() < 1e-10:
            gamma = 0.0
        else:
            gamma = max(0.0, min(1.0, (-b / (2*a)).item()))

        w = w + gamma * d
        w = w.clamp(min=w_min)
        w = w / w.sum()

    return w.tolist()


# ── Refinement Methods ────────────────────────────────────────────────────────

def refine_mgda(vertices, faces, w_min=0.05, steps=50, lr=0.005, sym_axis=1):
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    with torch.no_grad():
        s0 = abs(symmetry_reward(v_opt, axis=sym_axis).item()) + 1e-6
        m0 = abs(smoothness_reward(v_opt, faces).item()) + 1e-6
        c0 = abs(compactness_reward(v_opt, faces).item()) + 1e-6

    for step in range(steps):
        optimizer.zero_grad()
        reward_fns = [
            lambda v: symmetry_reward(v, axis=sym_axis) / s0,
            lambda v: smoothness_reward(v, faces) / m0,
            lambda v: compactness_reward(v, faces) / c0,
        ]
        grads = []
        for fn in reward_fns:
            v_opt.grad = None
            r = fn(v_opt)
            (-r).backward(retain_graph=True)
            grads.append(v_opt.grad.clone())

        w = mgda_solve_constrained(grads, w_min=w_min)
        combined = sum(wi * gi for wi, gi in zip(w, grads))
        v_opt.grad = combined
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()

    return v_opt.detach()


def laplacian_loss(vertices, faces):
    V = vertices.shape[0]
    edges = torch.cat([
        faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]],
        faces[:,[1,0]], faces[:,[2,1]], faces[:,[0,2]]
    ], dim=0)
    src, dst = edges[:,0], edges[:,1]
    ns = torch.zeros_like(vertices)
    nc = torch.zeros(V, 1, device=vertices.device)
    ns.scatter_add_(0, dst.unsqueeze(1).expand(-1,3), vertices[src])
    nc.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0],1, device=vertices.device))
    nc = nc.clamp(min=1)
    lap = vertices - ns / nc
    return (lap**2).sum(dim=1).mean()


def normal_consistency_loss(vertices, faces):
    face_normals = compute_face_normals(vertices, faces)
    edge_dict = defaultdict(list)
    for fi in range(faces.shape[0]):
        for local in range(3):
            v0 = faces[fi, local].item()
            v1 = faces[fi, (local+1)%3].item()
            edge_dict[(min(v0,v1), max(v0,v1))].append(fi)
    cos_vals = []
    for fids in edge_dict.values():
        if len(fids) == 2:
            cos_vals.append((face_normals[fids[0]] * face_normals[fids[1]]).sum())
    if not cos_vals:
        return torch.tensor(0.0, device=vertices.device)
    return (1.0 - torch.stack(cos_vals)).mean()


def edge_length_loss(vertices, faces):
    edges = torch.cat([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]], dim=0)
    lengths = (vertices[edges[:,1]] - vertices[edges[:,0]]).norm(dim=1)
    return lengths.var()


def refine_laplacian(vertices, faces, steps=50, lr=0.005):
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = laplacian_loss(v_opt, faces)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()
    return v_opt.detach()


def refine_normal_consist(vertices, faces, steps=50, lr=0.005):
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = normal_consistency_loss(v_opt, faces)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()
    return v_opt.detach()


def refine_classical_combined(vertices, faces, steps=50, lr=0.005):
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)
    with torch.no_grad():
        l0 = abs(laplacian_loss(v_opt, faces).item()) + 1e-6
        n0 = abs(normal_consistency_loss(v_opt, faces).item()) + 1e-6
        e0 = abs(edge_length_loss(v_opt, faces).item()) + 1e-6
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (laplacian_loss(v_opt, faces)/l0
                + normal_consistency_loss(v_opt, faces)/n0
                + edge_length_loss(v_opt, faces)/e0) / 3.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()
    return v_opt.detach()


# ── Statistics ────────────────────────────────────────────────────────────────

def bh_correct(p_values):
    n = len(p_values)
    order = np.argsort(p_values)
    p_adj = np.empty(n)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(float(p_values[idx]) * n / (rank + 1), 1.0)
    for i in range(n-2, -1, -1):
        if p_adj[order[i]] > p_adj[order[i+1]]:
            p_adj[order[i]] = p_adj[order[i+1]]
    return p_adj.tolist()


def compute_stats(results, method_name):
    """Compute paired statistics: method vs baseline."""
    bl = {(r['prompt'], r['seed']): r for r in results if r['method'] == 'baseline'}
    mt = {(r['prompt'], r['seed']): r for r in results if r['method'] == method_name}
    common = sorted(set(bl) & set(mt))
    if len(common) < 10:
        return None

    out = {'n_pairs': len(common), 'method': method_name}
    all_metrics = METRICS + INDEP_METRICS
    p_raws = []
    rows = []

    for metric in all_metrics:
        a = [bl[k][metric] for k in common if bl[k].get(metric) is not None and mt[k].get(metric) is not None]
        b = [mt[k][metric] for k in common if bl[k].get(metric) is not None and mt[k].get(metric) is not None]
        if len(a) < 10:
            continue
        t_val, p_raw = stats.ttest_rel(b, a)
        diff = np.array(b) - np.array(a)
        d = diff.mean() / (diff.std(ddof=1) + 1e-10)
        delta_pct = (np.mean(b) - np.mean(a)) / (abs(np.mean(a)) + 1e-10) * 100
        win_rate = np.mean(np.array(b) > np.array(a)) * 100 if metric in METRICS else None
        p_raws.append(p_raw)
        rows.append((metric, np.mean(a), np.mean(b), delta_pct, t_val, p_raw, d, win_rate))

    if not rows:
        return None

    p_adjs = bh_correct([r[5] for r in rows])
    for i, (metric, bl_m, mt_m, dpct, t_val, p_raw, d, wr) in enumerate(rows):
        out[metric] = {
            'baseline_mean': float(bl_m), 'method_mean': float(mt_m),
            'delta_pct': float(dpct), 't': float(t_val),
            'p_raw': float(p_raw), 'p_adj': float(p_adjs[i]),
            'cohens_d': float(d), 'significant': p_adjs[i] < 0.05,
        }
        if wr is not None:
            out[metric]['win_rate'] = float(wr)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OBJ_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    ckpt_path = OUT_DIR / "checkpoint.json"
    results = []
    done_keys = set()
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        done_keys = {(r['prompt'], r['seed'], r['method']) for r in results}
        print(f"Resuming: {len(results)} records done")

    methods_to_run = [
        'baseline', 'handcrafted', 'mgda_0.05',
        'laplacian', 'normal_consist', 'classical_combined'
    ]

    total = len(ALL_PROMPTS) * len(SEEDS) * len(methods_to_run)
    print(f"Total planned: {len(ALL_PROMPTS)} prompts x {len(SEEDS)} seeds x "
          f"{len(methods_to_run)} methods = {total}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # ── Phase 1: Generate all baseline meshes (GPU, serial) ──────────
    print(f"\n{'='*60}")
    print("Phase 1: Generate baseline meshes (GPU)")
    print(f"{'='*60}\n")

    print(f"Loading Shap-E on {DEVICE}...")
    xm, model, diffusion = load_shap_e(device=DEVICE)
    print("Shap-E loaded.\n")

    t0 = time.time()
    baseline_objs = {}  # (prompt, seed) -> obj_path

    for pi, prompt in enumerate(ALL_PROMPTS):
        cat = PROMPT_CAT[prompt]
        ps = slug(prompt)

        for seed in SEEDS:
            bl_key = (prompt, seed, 'baseline')
            bl_obj = OBJ_DIR / "baseline" / cat / f"{ps}_s{seed}.obj"

            if bl_obj.exists():
                v_bl, f_bl = load_obj(str(bl_obj), device='cpu')
                if f_bl.shape[0] < 4:
                    print(f"  [SKIP] {ps[:30]} s={seed}: OBJ has {f_bl.shape[0]} faces", flush=True)
                    continue
                baseline_objs[(prompt, seed)] = str(bl_obj)
                if bl_key not in done_keys:
                    bl_metrics = compute_all_metrics(v_bl, f_bl)
                    results.append({
                        'prompt': prompt, 'seed': seed, 'method': 'baseline',
                        'category': cat, **bl_metrics,
                    })
                    done_keys.add(bl_key)
                continue

            if bl_key in done_keys:
                continue

            print(f"  [{pi+1}/110] Generating: {ps[:40]} seed={seed}", flush=True)
            from shape_gen import run_single_experiment
            try:
                metrics = run_single_experiment(
                    prompt=prompt, method='baseline', seed=seed,
                    weights=W_EQUAL, xm=xm, model=model, diffusion=diffusion,
                    output_dir=str(OBJ_DIR / "baseline" / cat), device=DEVICE,
                )
                v_bl, f_bl = load_obj(metrics['mesh_path'], device='cpu')
                if f_bl.shape[0] < 4:
                    print(f"    [SKIP] Empty mesh ({f_bl.shape[0]} faces)", flush=True)
                    continue
                save_obj(v_bl, f_bl, bl_obj)
                baseline_objs[(prompt, seed)] = str(bl_obj)

                bl_metrics = compute_all_metrics(v_bl, f_bl)
                results.append({
                    'prompt': prompt, 'seed': seed, 'method': 'baseline',
                    'category': cat, **bl_metrics,
                })
                done_keys.add(bl_key)
            except Exception as e:
                print(f"    [SKIP] Generation failed: {e}", flush=True)

    # Free GPU for refinement
    del xm, model, diffusion
    torch.cuda.empty_cache()
    gc.collect()

    # Checkpoint after Phase 1
    with open(ckpt_path, 'w') as f:
        json.dump(results, f)
    elapsed1 = time.time() - t0
    n_baselines = len(baseline_objs)
    print(f"\nPhase 1 done: {n_baselines} baseline meshes, {elapsed1/60:.1f}min")

    # Collect all existing baseline OBJs
    for obj_path in (OBJ_DIR / "baseline").rglob("*.obj"):
        parts = obj_path.stem.rsplit("_s", 1)
        if len(parts) == 2:
            ps_name, seed_str = parts
            for prompt in ALL_PROMPTS:
                if slug(prompt) == ps_name:
                    baseline_objs[(prompt, int(seed_str))] = str(obj_path)
                    break

    # ── Phase 2: Run refinements (CPU parallel) ──────────────────────
    print(f"\n{'='*60}")
    print("Phase 2: Run refinements (CPU parallel)")
    print(f"{'='*60}\n")

    # Build task list
    refine_tasks = []
    for (prompt, seed), obj_path in baseline_objs.items():
        cat = PROMPT_CAT.get(prompt, 'unknown')
        for method in methods_to_run:
            if method == 'baseline':
                continue
            mk = (prompt, seed, method)
            if mk in done_keys:
                continue
            refine_tasks.append((prompt, seed, method, cat, obj_path))

    print(f"  {len(refine_tasks)} refinement tasks to run")

    def run_single_refine(args):
        """Worker function for parallel refinement (CPU only)."""
        prompt, seed, method, cat, obj_path = args
        try:
            v_bl, f_bl = load_obj(obj_path, device='cpu')

            if method == 'handcrafted':
                v_ref, _ = refine_with_geo_reward(
                    v_bl.clone(), f_bl, weights=W_EQUAL, steps=STEPS, lr=LR)
            elif method == 'mgda_0.05':
                v_ref = refine_mgda(v_bl.clone(), f_bl, w_min=MGDA_WMIN,
                                   steps=STEPS, lr=LR)
            elif method == 'laplacian':
                v_ref = refine_laplacian(v_bl.clone(), f_bl, steps=STEPS, lr=LR)
            elif method == 'normal_consist':
                v_ref = refine_normal_consist(v_bl.clone(), f_bl, steps=STEPS, lr=LR)
            elif method == 'classical_combined':
                v_ref = refine_classical_combined(v_bl.clone(), f_bl, steps=STEPS, lr=LR)
            else:
                return None

            ref_metrics = compute_all_metrics(v_ref, f_bl)
            return {
                'prompt': prompt, 'seed': seed, 'method': method,
                'category': cat, **ref_metrics,
            }
        except Exception as e:
            return None

    # Run in parallel with N_WORKERS
    N_WORKERS = min(5, max(1, mp.cpu_count() - 2))
    print(f"  Using {N_WORKERS} CPU workers\n")

    completed = 0
    # Use ThreadPoolExecutor (shares memory, no pickle issues with torch)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_single_refine, task): task for task in refine_tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result is not None:
                results.append(result)
                done_keys.add((result['prompt'], result['seed'], result['method']))

            if completed % 50 == 0:
                with open(ckpt_path, 'w') as f:
                    json.dump(results, f)
                elapsed = time.time() - t0
                print(f"  [checkpoint] {completed}/{len(refine_tasks)} tasks, "
                      f"{len(results)} records, {elapsed/60:.1f}min", flush=True)

    # Final save
    with open(ckpt_path, 'w') as f:
        json.dump(results, f)
    with open(OUT_DIR / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} records, {elapsed/60:.1f}min")

    # Statistics
    print(f"\n{'='*90}")
    print("PAIRED STATISTICS vs BASELINE")
    print(f"{'='*90}")

    all_stats = {}
    for method in methods_to_run:
        if method == 'baseline':
            continue
        s = compute_stats(results, method)
        if s:
            all_stats[method] = s
            print(f"\n--- {method} (n={s['n_pairs']}) ---")
            for metric in METRICS + INDEP_METRICS:
                if metric in s:
                    m = s[metric]
                    sig = "*" if m['significant'] else ""
                    wr = f" wr={m.get('win_rate',0):.1f}%" if 'win_rate' in m else ""
                    print(f"  {metric:25s}: {m['delta_pct']:+.1f}% "
                          f"(p={m['p_adj']:.2e}, d={m['cohens_d']:+.3f}){sig}{wr}")

    with open(OUT_DIR / "stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nStats saved to {OUT_DIR}/stats.json")


if __name__ == '__main__':
    main()
