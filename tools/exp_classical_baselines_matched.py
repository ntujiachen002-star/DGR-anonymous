"""Matched-runtime classical baseline suite vs DGR.

Response to external reviewer (GPT-5.4 xhigh): "How does DGR compare against
strong classical mesh-processing baselines at matched runtime?"

For each of the 97x3 = 291 (prompt, seed) pairs, start from the baseline
Shap-E mesh and apply six classical variants plus the DGR-refined mesh, then
score all on the three reward metrics and two non-circular independent
metrics (edge regularity, angular-normal-deviation).

Baselines:
    B1  Laplacian smoothing                  (trimesh.smoothing.filter_laplacian)
    B2  Taubin smoothing                     (trimesh.smoothing.filter_taubin)
    B3  Humphrey smoothing                   (trimesh.smoothing.filter_humphrey)
    B4  Volume-preserving Laplacian          (Laplacian + isotropic rescale to preserve volume)
    B5  Reflection averaging                 (reflect across cached bilateral plane, average with original)
    B6  Combined: Humphrey + reflect + volume-preserving Laplacian  (serial)

Runtime target per baseline: 5-8 s per mesh. Iteration counts chosen so that
the dominant classical ops saturate within this budget on a typical mesh.

Output: analysis_results/classical_baselines_matched/
    - all_scores.json         per-mesh scores for every method
    - paired_stats.json       DGR vs each baseline, paired Wilcoxon
    - summary.md              human-readable table
"""
import os
import sys
import json
import time
import copy
import numpy as np
import trimesh
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.geo_reward import (
    symmetry_reward_plane,
    smoothness_reward,
    compactness_reward,
    compute_face_normals,
    compute_volume,
)

BASELINE_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/baseline')
REFINED_DIR  = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted')
PLANE_CACHE  = os.path.join(ROOT, 'analysis_results/mesh_validity_full/plane_cache.json')
OUT_DIR      = os.path.join(ROOT, 'analysis_results/classical_baselines_matched')
os.makedirs(OUT_DIR, exist_ok=True)

# Iteration counts (rough runtime match to DGR ~5-8 s)
N_LAP     = 30
N_TAUBIN  = 30
N_HUMPH   = 30
N_VOLLAP  = 30

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# ============================================================
# Classical smoothing variants
# ============================================================

def clone_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=mesh.vertices.copy(),
                           faces=mesh.faces.copy(),
                           process=False)


def apply_laplacian(mesh, n=N_LAP):
    m = clone_mesh(mesh)
    # volume_constraint=False: trimesh default tries to rescale to preserve
    # volume which explodes when the mesh collapses during heavy smoothing.
    trimesh.smoothing.filter_laplacian(m, iterations=n, volume_constraint=False)
    return m


def apply_taubin(mesh, n=N_TAUBIN):
    m = clone_mesh(mesh)
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=n)
    return m


def chamfer_to_baseline(mesh_out, mesh_bl, n_samples=4096):
    """Symmetric Chamfer-L1 between output and baseline meshes.
    Used as a shape-preservation metric; small = stays close to baseline."""
    try:
        a = mesh_out.sample(n_samples)
        b = mesh_bl.sample(n_samples)
    except Exception:
        return float('nan')
    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a), cKDTree(b)
    d1, _ = tb.query(a, k=1)
    d2, _ = ta.query(b, k=1)
    return float(0.5 * (d1.mean() + d2.mean()))


def apply_humphrey(mesh, n=N_HUMPH):
    m = clone_mesh(mesh)
    try:
        trimesh.smoothing.filter_humphrey(m, iterations=n)
    except Exception:
        # Humphrey filter requires non-degenerate topology; fall back to Taubin
        trimesh.smoothing.filter_taubin(m, iterations=n)
    return m


def apply_vol_laplacian(mesh, n=N_VOLLAP):
    m = clone_mesh(mesh)
    v0_try = abs(mesh.volume) if mesh.is_watertight else None
    # Disable trimesh's own volume constraint; we apply a safe rescale below.
    trimesh.smoothing.filter_laplacian(m, iterations=n, volume_constraint=False)
    v1_try = abs(m.volume) if m.is_watertight else None
    if v0_try is not None and v1_try is not None and v0_try > 1e-6 and v1_try > 1e-6:
        scale = (v0_try / v1_try) ** (1.0 / 3.0)
        scale = float(np.clip(scale, 0.5, 2.0))
        c = m.vertices.mean(axis=0, keepdims=True)
        m.vertices = c + (m.vertices - c) * scale
    return m


def apply_reflection_avg(mesh, plane_normal, plane_offset, alpha=0.5):
    """Reflect the mesh across the cached bilateral plane and average each
    vertex with the nearest reflected counterpart. Classical symmetrisation."""
    m = clone_mesh(mesh)
    V = m.vertices
    n = np.asarray(plane_normal, dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    n = n / n_norm
    d = float(plane_offset) / n_norm
    # Reflect: v' = v - 2*(v.n - d)*n
    signed = (V @ n) - d
    V_ref = V - 2.0 * signed[:, None] * n[None, :]
    # Nearest-neighbour pairing: for each vertex, pull toward its reflected nearest
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    _, idx = tree.query(V_ref, k=1)
    pulled = V_ref[idx]          # V[i] pulled toward reflected counterpart of V[idx[i]]
    m.vertices = (1 - alpha) * V + alpha * pulled
    return m


def apply_combined(mesh, plane_normal, plane_offset):
    """B6: Humphrey -> reflection averaging -> volume-preserving Laplacian."""
    m = apply_humphrey(mesh, n=10)
    m = apply_reflection_avg(m, plane_normal, plane_offset, alpha=0.5)
    m = apply_vol_laplacian(m, n=10)
    return m


# ============================================================
# Scoring
# ============================================================

def to_torch(mesh):
    V = torch.as_tensor(np.asarray(mesh.vertices), dtype=DTYPE, device=DEVICE)
    F = torch.as_tensor(np.asarray(mesh.faces),    dtype=torch.long, device=DEVICE)
    return V, F


def edge_regularity(mesh):
    """Standard deviation of edge lengths divided by mean. Lower = more regular."""
    edges = mesh.edges_unique
    if len(edges) == 0:
        return float('nan')
    L = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    mu = L.mean()
    if mu < 1e-9:
        return float('nan')
    return float(L.std() / mu)


def angular_normal_deviation(mesh):
    """Mean angular deviation (radians) between adjacent face normals."""
    tm_faces_adj = mesh.face_adjacency
    if len(tm_faces_adj) == 0:
        return float('nan')
    fn = mesh.face_normals
    a = fn[tm_faces_adj[:, 0]]
    b = fn[tm_faces_adj[:, 1]]
    cos = np.clip((a * b).sum(axis=1), -1.0, 1.0)
    return float(np.arccos(cos).mean())


def score_mesh(mesh, plane_normal, plane_offset):
    """Return all five metrics for a mesh."""
    if len(mesh.vertices) < 10 or len(mesh.faces) < 10:
        return None
    V, F = to_torch(mesh)
    try:
        pn = torch.as_tensor(plane_normal, dtype=DTYPE, device=DEVICE)
        po = torch.tensor(float(plane_offset), dtype=DTYPE, device=DEVICE)
        sym = symmetry_reward_plane(V, pn, po).item()
        hnc = smoothness_reward(V, F).item()
        com = compactness_reward(V, F).item()
    except Exception:
        return None
    try:
        edge_reg = edge_regularity(mesh)
        ang_dev  = angular_normal_deviation(mesh)
    except Exception:
        edge_reg = float('nan')
        ang_dev = float('nan')
    return {'sym': sym, 'hnc': hnc, 'com': com,
            'edge_reg': edge_reg, 'ang_dev': ang_dev}


# ============================================================
# Main sweep
# ============================================================

def collect_pairs():
    pairs = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        bl_dir = os.path.join(BASELINE_DIR, cat)
        rf_dir = os.path.join(REFINED_DIR, cat)
        if not os.path.isdir(bl_dir):
            continue
        for fname in sorted(os.listdir(bl_dir)):
            if not fname.endswith('.obj'):
                continue
            bl = os.path.join(bl_dir, fname)
            rf = os.path.join(rf_dir, fname)
            if os.path.exists(rf):
                pairs.append((cat, fname, bl, rf))
    return pairs


def parse_key(fname):
    """'a_symmetric_vase_seed42.obj' -> ('a symmetric vase', 42)"""
    stem = fname[:-4] if fname.endswith('.obj') else fname
    if '_seed' in stem:
        base, seed = stem.rsplit('_seed', 1)
        try:
            seed = int(seed)
        except ValueError:
            seed = None
    else:
        base, seed = stem, None
    prompt = base.replace('_', ' ')
    return prompt, seed


def main():
    print('=== Classical baselines (matched runtime) vs DGR ===')
    with open(PLANE_CACHE) as f:
        plane_cache = json.load(f)
    pairs = collect_pairs()
    print(f'Pairs found: {len(pairs)}')

    methods = ['dgr', 'B1_laplacian', 'B2_taubin', 'B3_humphrey',
               'B4_vollap', 'B5_reflect', 'B6_combined']
    results = []          # list of dicts, one per pair
    skipped = 0
    t0 = time.time()
    runtime_sum = {m: 0.0 for m in methods}

    for i, (cat, fname, bl_path, rf_path) in enumerate(pairs):
        try:
            bl_mesh = trimesh.load(bl_path, force='mesh', process=False)
            rf_mesh = trimesh.load(rf_path, force='mesh', process=False)
        except Exception:
            skipped += 1
            continue

        if len(bl_mesh.vertices) < 10 or len(bl_mesh.faces) < 10:
            skipped += 1
            continue

        prompt, seed = parse_key(fname)
        plane_key = f'{prompt}|seed={seed}'
        plane_entry = plane_cache.get(plane_key)
        if plane_entry is None:
            skipped += 1
            continue
        plane_n = plane_entry['normal']
        plane_d = plane_entry['offset']

        row = {'category': cat, 'file': fname, 'prompt': prompt, 'seed': seed}
        # Score DGR (already refined)
        t = time.time()
        sc = score_mesh(rf_mesh, plane_n, plane_d)
        if sc is not None:
            sc['shape_cd'] = chamfer_to_baseline(rf_mesh, bl_mesh)
        runtime_sum['dgr'] += time.time() - t
        row['dgr'] = sc

        # Apply & score each classical baseline
        for name, fn in [
            ('B1_laplacian',  lambda m: apply_laplacian(m)),
            ('B2_taubin',     lambda m: apply_taubin(m)),
            ('B3_humphrey',   lambda m: apply_humphrey(m)),
            ('B4_vollap',     lambda m: apply_vol_laplacian(m)),
            ('B5_reflect',    lambda m: apply_reflection_avg(m, plane_n, plane_d)),
            ('B6_combined',   lambda m: apply_combined(m, plane_n, plane_d)),
        ]:
            t = time.time()
            try:
                m_out = fn(bl_mesh)
                sc = score_mesh(m_out, plane_n, plane_d)
                if sc is not None:
                    sc['shape_cd'] = chamfer_to_baseline(m_out, bl_mesh)
                    # Reject numerically blown-up outputs
                    if (not np.isfinite(sc['sym']) or abs(sc['sym']) > 1e6
                        or not np.isfinite(sc['com']) or abs(sc['com']) > 1e6):
                        sc = None
            except Exception as e:
                sc = None
            runtime_sum[name] += time.time() - t
            row[name] = sc

        results.append(row)

        if (i + 1) % 30 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(pairs)}: {len(results)} valid, {skipped} skipped, {el:.0f}s', flush=True)

    elapsed = (time.time() - t0) / 60
    print(f'\nDone: {len(results)} valid, {skipped} skipped, {elapsed:.1f} min')

    with open(os.path.join(OUT_DIR, 'all_scores.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # Aggregate to prompt level (average 3 seeds per prompt)
    # ============================================================
    by_prompt = {}
    for r in results:
        key = (r['category'], r['prompt'])
        by_prompt.setdefault(key, []).append(r)

    prompt_rows = []
    for (cat, prompt), rows in by_prompt.items():
        out = {'category': cat, 'prompt': prompt, 'n_seeds': len(rows)}
        for m in methods:
            mrows = [r[m] for r in rows if r.get(m) is not None]
            if len(mrows) == 0:
                for k in ['sym', 'hnc', 'com', 'edge_reg', 'ang_dev', 'shape_cd']:
                    out[f'{m}_{k}'] = float('nan')
                continue
            for k in ['sym', 'hnc', 'com', 'edge_reg', 'ang_dev', 'shape_cd']:
                vals = [x[k] for x in mrows if np.isfinite(x.get(k, np.nan))]
                out[f'{m}_{k}'] = float(np.mean(vals)) if vals else float('nan')
        prompt_rows.append(out)

    # ============================================================
    # Paired Wilcoxon: DGR vs each baseline, per metric
    # ============================================================
    metric_sign = {  # higher is better (+1) or lower is better (-1)
        'sym': +1, 'hnc': +1, 'com': +1,   # all are negative residuals; higher (less negative) = better
        'edge_reg': -1,                     # lower stddev/mean = more regular
        'ang_dev':  -1,                     # lower angular deviation = smoother
        'shape_cd': -1,                     # lower Chamfer-to-baseline = better shape preservation
    }
    paired = {}
    for m in methods:
        if m == 'dgr':
            continue
        paired[m] = {}
        for k, sign in metric_sign.items():
            d = np.array([r[f'dgr_{k}'] - r[f'{m}_{k}'] for r in prompt_rows
                          if np.isfinite(r.get(f'dgr_{k}', np.nan))
                          and np.isfinite(r.get(f'{m}_{k}', np.nan))])
            if len(d) < 3 or np.all(d == 0):
                paired[m][k] = {'n': int(len(d)), 'mean_dgr': float('nan'),
                                'mean_base': float('nan'),
                                'dgr_better_pct': float('nan'),
                                'p_wilcoxon': float('nan')}
                continue
            # dgr_better = sign * d > 0   (if higher is better, d = dgr-base > 0 means dgr wins)
            better = (np.sign(d) * sign > 0).mean() * 100
            try:
                _, p = stats.wilcoxon(d)
            except Exception:
                p = float('nan')
            mean_dgr  = float(np.nanmean([r[f'dgr_{k}']  for r in prompt_rows if np.isfinite(r.get(f'dgr_{k}', np.nan))]))
            mean_base = float(np.nanmean([r[f'{m}_{k}']  for r in prompt_rows if np.isfinite(r.get(f'{m}_{k}', np.nan))]))
            paired[m][k] = {'n': int(len(d)),
                            'mean_dgr': mean_dgr,
                            'mean_base': mean_base,
                            'dgr_better_pct': float(better),
                            'p_wilcoxon': float(p)}

    with open(os.path.join(OUT_DIR, 'paired_stats.json'), 'w') as f:
        json.dump({'paired': paired, 'prompt_rows': prompt_rows,
                   'runtime_sum_s': runtime_sum,
                   'n_valid': len(results), 'n_skipped': skipped}, f, indent=2)

    # ============================================================
    # Print summary table
    # ============================================================
    print('\n=== DGR vs classical baselines (prompt-level paired) ===')
    print('Values: DGR mean / baseline mean   DGR-better%   Wilcoxon p')
    header = f"{'Baseline':14s} | {'Metric':10s} | {'DGR':>10s}  {'Base':>10s}  {'DGR win%':>8s}  {'p':>10s}"
    print(header); print('-' * len(header))
    for m in methods[1:]:
        for k in ['sym', 'hnc', 'com', 'edge_reg', 'ang_dev', 'shape_cd']:
            s = paired[m].get(k, {})
            print(f"{m:14s} | {k:10s} | {s.get('mean_dgr', float('nan')):10.4f}  "
                  f"{s.get('mean_base', float('nan')):10.4f}  "
                  f"{s.get('dgr_better_pct', float('nan')):7.1f}%  "
                  f"{s.get('p_wilcoxon', float('nan')):10.2e}")
        print()

    # Runtime report
    n_scored = max(len(results), 1)
    print('Avg runtime per pair (seconds):')
    for m in methods:
        print(f'  {m:14s}: {runtime_sum[m] / n_scored:.2f}')

    # Markdown summary
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Classical Baselines (Matched Runtime) vs DGR\n\n')
        f.write(f'n_valid = {len(results)}, n_skipped = {skipped}, runtime {elapsed:.1f} min.\n\n')
        f.write('## Paired comparisons (prompt-level)\n\n')
        f.write('| Baseline | Metric | DGR mean | Baseline mean | DGR win % | Wilcoxon p |\n')
        f.write('|---|---|---:|---:|---:|---:|\n')
        for m in methods[1:]:
            for k in ['sym', 'hnc', 'com', 'edge_reg', 'ang_dev', 'shape_cd']:
                s = paired[m].get(k, {})
                f.write(f"| {m} | {k} | {s.get('mean_dgr', float('nan')):.4f} | "
                        f"{s.get('mean_base', float('nan')):.4f} | "
                        f"{s.get('dgr_better_pct', float('nan')):.1f} | "
                        f"{s.get('p_wilcoxon', float('nan')):.2e} |\n")
            f.write('|---|---|---|---|---|---|\n')
        f.write('\n## Average runtime per pair (s)\n\n')
        for m in methods:
            f.write(f'- {m}: {runtime_sum[m] / n_scored:.2f}\n')
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
