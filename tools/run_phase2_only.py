"""
Phase 2 Only: Skip Shap-E generation, directly run refinements on existing OBJ files.
Uses 16 CPU workers for parallel refinement.
"""
import os, sys, json, time, re
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats as sp_stats

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward, compute_face_normals
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS
from collections import defaultdict

STEPS = 50
LR = 0.005
W_EQUAL = torch.tensor([1/3, 1/3, 1/3])
MGDA_WMIN = 0.05
N_WORKERS = 16

OUT_DIR = Path("analysis_results/full_mgda_classical")
OBJ_DIR = Path("results/full_mgda_classical_objs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CAT = {}
for p in SYMMETRY_PROMPTS: PROMPT_CAT[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS: PROMPT_CAT[p] = "smoothness"
for p in COMPACTNESS_PROMPTS: PROMPT_CAT[p] = "compactness"
SEEDS = [42, 123, 456]


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:60]


def load_obj(path, device='cpu'):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                fi = [int(x.split('/')[0])-1 for x in line.split()[1:4]]
                if len(fi) == 3:
                    faces.append(fi)
    v = torch.tensor(verts, dtype=torch.float32, device=device) if verts else torch.zeros(0, 3, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device).reshape(-1, 3) if faces else torch.zeros(0, 3, dtype=torch.long, device=device)
    return v, f


# ── Import refinement functions from main script ──
def edge_length_regularity(vertices, faces):
    edges = torch.cat([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]], dim=0)
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
            v1 = faces[fi, (local+1)%3].item()
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
        gamma = max(0.0, min(1.0, (-b / (2*a)).item())) if a.abs() > 1e-10 else 0.0
        w = w + gamma * d
        w = w.clamp(min=w_min)
        w = w / w.sum()
    return w.tolist()


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
    edges = torch.cat([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]],
                       faces[:,[1,0]], faces[:,[2,1]], faces[:,[0,2]]], dim=0)
    src, dst = edges[:,0], edges[:,1]
    ns = torch.zeros_like(vertices)
    nc = torch.zeros(V, 1, device=vertices.device)
    ns.scatter_add_(0, dst.unsqueeze(1).expand(-1,3), vertices[src])
    nc.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0],1, device=vertices.device))
    nc = nc.clamp(min=1)
    return ((vertices - ns / nc)**2).sum(dim=1).mean()


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
        laplacian_loss(v_opt, faces).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()
    return v_opt.detach()


def refine_normal_consist(vertices, faces, steps=50, lr=0.005):
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        normal_consistency_loss(v_opt, faces).backward()
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


def run_single_refine(args):
    prompt, seed, method, cat, obj_path = args
    try:
        v_bl, f_bl = load_obj(obj_path, device='cpu')
        if method == 'handcrafted':
            v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=W_EQUAL, steps=STEPS, lr=LR)
        elif method == 'mgda_0.05':
            v_ref = refine_mgda(v_bl.clone(), f_bl, w_min=MGDA_WMIN, steps=STEPS, lr=LR)
        elif method == 'laplacian':
            v_ref = refine_laplacian(v_bl.clone(), f_bl, steps=STEPS, lr=LR)
        elif method == 'normal_consist':
            v_ref = refine_normal_consist(v_bl.clone(), f_bl, steps=STEPS, lr=LR)
        elif method == 'classical_combined':
            v_ref = refine_classical_combined(v_bl.clone(), f_bl, steps=STEPS, lr=LR)
        else:
            return None
        ref_metrics = compute_all_metrics(v_ref, f_bl)
        return {'prompt': prompt, 'seed': seed, 'method': method, 'category': cat, **ref_metrics}
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("Phase 2 Only: Parallel Refinement (16 workers)")
    print("=" * 60, flush=True)

    # Load checkpoint
    ckpt_path = OUT_DIR / "checkpoint.json"
    results = []
    done_keys = set()
    if ckpt_path.exists():
        results = json.load(open(ckpt_path))
        done_keys = {(r['prompt'], r['seed'], r['method']) for r in results}
        print(f"Checkpoint: {len(results)} records, {len(done_keys)} done keys")

    # Collect valid baseline OBJs
    baseline_objs = {}
    for obj_path in (OBJ_DIR / "baseline").rglob("*.obj"):
        v, f = load_obj(str(obj_path), device='cpu')
        if f.shape[0] < 4:
            continue
        stem = obj_path.stem
        parts = stem.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        ps_name, seed_str = parts
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        for prompt in ALL_PROMPTS:
            if slug(prompt) == ps_name:
                baseline_objs[(prompt, seed)] = str(obj_path)
                bl_key = (prompt, seed, 'baseline')
                if bl_key not in done_keys:
                    cat = PROMPT_CAT.get(prompt, 'unknown')
                    bl_metrics = compute_all_metrics(v, f)
                    results.append({'prompt': prompt, 'seed': seed, 'method': 'baseline',
                                    'category': cat, **bl_metrics})
                    done_keys.add(bl_key)
                break

    print(f"Valid baseline OBJs: {len(baseline_objs)}")

    # Build task list
    methods = ['handcrafted', 'mgda_0.05', 'laplacian', 'normal_consist', 'classical_combined']
    refine_tasks = []
    for (prompt, seed), obj_path in baseline_objs.items():
        cat = PROMPT_CAT.get(prompt, 'unknown')
        for method in methods:
            if (prompt, seed, method) not in done_keys:
                refine_tasks.append((prompt, seed, method, cat, obj_path))

    print(f"Refinement tasks: {len(refine_tasks)}")
    print(f"Workers: {N_WORKERS}", flush=True)

    # Run
    t0 = time.time()
    completed = 0
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
                print(f"  [{completed}/{len(refine_tasks)}] {len(results)} records, "
                      f"{elapsed/60:.1f}min", flush=True)

    # Final save
    with open(ckpt_path, 'w') as f:
        json.dump(results, f)
    with open(OUT_DIR / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} records, {elapsed/60:.1f}min", flush=True)

    # Stats
    print(f"\n{'='*70}")
    print("PAIRED STATISTICS vs BASELINE")
    print(f"{'='*70}")
    bl = {(r['prompt'], r['seed']): r for r in results if r['method'] == 'baseline'}
    all_stats = {}
    for method in methods:
        mt = {(r['prompt'], r['seed']): r for r in results if r['method'] == method}
        common = sorted(set(bl) & set(mt))
        if len(common) < 10:
            continue
        print(f"\n--- {method} (n={len(common)}) ---")
        all_metrics = ['symmetry', 'smoothness', 'compactness', 'edge_regularity', 'normal_consistency_deg']
        p_raws, rows = [], []
        for metric in all_metrics:
            a = [bl[k][metric] for k in common if bl[k].get(metric) is not None and mt[k].get(metric) is not None]
            b = [mt[k][metric] for k in common if bl[k].get(metric) is not None and mt[k].get(metric) is not None]
            if len(a) < 10:
                continue
            t_val, p_raw = sp_stats.ttest_rel(b, a)
            diff = np.array(b) - np.array(a)
            d = diff.mean() / (diff.std(ddof=1) + 1e-10)
            delta_pct = (np.mean(b) - np.mean(a)) / (abs(np.mean(a)) + 1e-10) * 100
            p_raws.append(p_raw)
            rows.append((metric, delta_pct, t_val, p_raw, d))
        if not rows:
            continue
        n = len(p_raws)
        order = np.argsort(p_raws)
        p_adj = np.empty(n)
        for rank, idx in enumerate(order):
            p_adj[idx] = min(float(p_raws[idx]) * n / (rank + 1), 1.0)
        for i in range(n-2, -1, -1):
            if p_adj[order[i]] > p_adj[order[i+1]]:
                p_adj[order[i]] = p_adj[order[i+1]]
        method_stats = {}
        for i, (metric, dpct, t_val, p_raw, d) in enumerate(rows):
            sig = "*" if p_adj[i] < 0.05 else ""
            print(f"  {metric:25s}: {dpct:+.1f}% (p={p_adj[i]:.2e}, d={d:+.3f}){sig}")
            method_stats[metric] = {'delta_pct': float(dpct), 'p_adj': float(p_adj[i]),
                                    'cohens_d': float(d), 'significant': bool(p_adj[i] < 0.05)}
        all_stats[method] = method_stats

    with open(OUT_DIR / "stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {OUT_DIR}/stats.json")
    print("ALL DONE.", flush=True)


if __name__ == '__main__':
    main()
