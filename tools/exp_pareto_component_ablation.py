"""Pareto frontier component ablation.

Overlays single-reward variants (Sym-Only, HNC-Only, Compact-Only) and the
full DGR point on the shape-CD vs reward Pareto plane to show that no single
reward reproduces DGR's operating point.

Inputs (pre-computed refined meshes per variant):
    ablation_meshes/baseline/{category}/*.obj
    ablation_meshes/diffgeoreward/{category}/*.obj      <-- DGR full
    ablation_meshes/sym_only/{category}/*.obj
    ablation_meshes/HNC_only/{category}/*.obj
    ablation_meshes/compact_only/{category}/*.obj

Outputs:
    analysis_results/pareto_component_ablation/ablation_points.json
    analysis_results/pareto_component_ablation/pareto_ablation.pdf / .png
"""
import os
import sys
import json
import time
import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.geo_reward import symmetry_reward_plane, smoothness_reward, compactness_reward

ABL_DIR = os.path.join(ROOT, 'ablation_meshes')
PLANE_CACHE = os.path.join(ROOT, 'analysis_results/mesh_validity_full/plane_cache.json')
OUT_DIR = os.path.join(ROOT, 'analysis_results/pareto_component_ablation')
PARETO_JSON = os.path.join(ROOT, 'analysis_results/classical_pareto/pareto_points.json')
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = [
    'diffgeoreward',
    # single-reward
    'sym_only', 'HNC_only', 'compact_only',
    # 2-reward combos (may be absent — script skips silently if missing)
    'sym_HNC', 'sym_com', 'HNC_com',
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


def chamfer(a, b, n=4096):
    if len(a) > n:
        a = a[np.random.default_rng(0).choice(len(a), n, replace=False)]
    if len(b) > n:
        b = b[np.random.default_rng(0).choice(len(b), n, replace=False)]
    ta, tb = cKDTree(a), cKDTree(b)
    d1, _ = tb.query(a)
    d2, _ = ta.query(b)
    return float(0.5 * (d1.mean() + d2.mean()))


def edge_reg(m):
    edges = m.edges_unique
    if len(edges) == 0:
        return float('nan')
    L = np.linalg.norm(m.vertices[edges[:, 0]] - m.vertices[edges[:, 1]], axis=1)
    mu = L.mean()
    return float('nan') if mu < 1e-9 else float(L.std() / mu)


def ang_dev(m):
    adj = m.face_adjacency
    if len(adj) == 0:
        return float('nan')
    fn = m.face_normals
    c = np.clip((fn[adj[:, 0]] * fn[adj[:, 1]]).sum(axis=1), -1.0, 1.0)
    return float(np.arccos(c).mean())


def score(m, plane_n, plane_d, baseline_verts):
    if len(m.vertices) < 10 or len(m.faces) < 10:
        return None
    V = torch.as_tensor(m.vertices, dtype=DTYPE, device=DEVICE)
    F = torch.as_tensor(m.faces, dtype=torch.long, device=DEVICE)
    try:
        pn = torch.as_tensor(plane_n, dtype=DTYPE, device=DEVICE)
        po = torch.tensor(float(plane_d), dtype=DTYPE, device=DEVICE)
        sym = symmetry_reward_plane(V, pn, po).item()
        hnc = smoothness_reward(V, F).item()
        com = compactness_reward(V, F).item()
        if not (np.isfinite(sym) and np.isfinite(hnc) and np.isfinite(com)):
            return None
        if abs(sym) > 1e4 or abs(com) > 1e4:
            return None
    except Exception:
        return None
    try:
        er = edge_reg(m)
        ad = ang_dev(m)
    except Exception:
        er, ad = float('nan'), float('nan')
    return {
        'sym': sym, 'hnc': hnc, 'com': com,
        'edge_reg': er, 'ang_dev': ad,
        'shape_cd': chamfer(np.asarray(m.vertices), baseline_verts),
    }


def parse_key(fname):
    stem = fname[:-4]
    if '_seed' in stem:
        base, seed = stem.rsplit('_seed', 1)
        try:
            seed = int(seed)
        except ValueError:
            seed = None
    else:
        base, seed = stem, None
    return base.replace('_', ' '), seed


def collect_pairs(variant):
    out = []
    vd = os.path.join(ABL_DIR, variant)
    bd = os.path.join(ABL_DIR, 'baseline')
    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_v = os.path.join(vd, cat)
        cat_b = os.path.join(bd, cat)
        if not (os.path.isdir(cat_v) and os.path.isdir(cat_b)):
            continue
        for fn in sorted(os.listdir(cat_v)):
            if not fn.endswith('.obj'):
                continue
            pv = os.path.join(cat_v, fn)
            pb = os.path.join(cat_b, fn)
            if os.path.exists(pb):
                out.append((cat, fn, pb, pv))
    return out


def main():
    print('=== Pareto component ablation ===')
    with open(PLANE_CACHE) as f:
        plane_cache = json.load(f)

    per_variant = {}
    # Silently drop variants with no meshes on disk (e.g. 2-reward combos not yet run)
    present = []
    keys_by_variant = {}
    for v in VARIANTS:
        pairs = collect_pairs(v)
        if not pairs:
            print(f'  {v}: skipped (no meshes found)')
            continue
        present.append(v)
        keys = {(c, n) for (c, n, _, _) in pairs}
        keys_by_variant[v] = keys
        print(f'  {v}: {len(pairs)} baseline-refined pairs')
    if not present:
        raise SystemExit('No variant meshes found; nothing to score.')
    common = set.intersection(*keys_by_variant.values())
    print(f'  intersection over {len(present)} variants: {len(common)}')
    active_variants = present

    t0 = time.time()
    for v in active_variants:
        points = []
        skipped = 0
        pairs = [p for p in collect_pairs(v) if (p[0], p[1]) in common]
        for (cat, fname, bl_path, rf_path) in pairs:
            try:
                bl = trimesh.load(bl_path, force='mesh', process=False)
                rf = trimesh.load(rf_path, force='mesh', process=False)
            except Exception:
                skipped += 1
                continue
            if len(bl.vertices) < 10 or len(bl.faces) < 10:
                skipped += 1
                continue
            prompt, seed = parse_key(fname)
            pkey = f'{prompt}|seed={seed}'
            p = plane_cache.get(pkey)
            if p is None:
                skipped += 1
                continue
            sc = score(rf, p['normal'], p['offset'], np.asarray(bl.vertices))
            if sc is not None:
                sc['cat'] = cat
                sc['fname'] = fname
                points.append(sc)
        per_variant[v] = points
        print(f'  {v}: kept {len(points)}, skipped {skipped}  ({time.time()-t0:.0f}s)')

    # Baseline point = rewards evaluated on the unrefined baseline mesh itself
    # (shape-CD = 0 by definition)
    baseline_points = []
    for (cat, fname, bl_path, _) in collect_pairs('diffgeoreward'):
        if (cat, fname) not in common:
            continue
        try:
            bl = trimesh.load(bl_path, force='mesh', process=False)
        except Exception:
            continue
        if len(bl.vertices) < 10 or len(bl.faces) < 10:
            continue
        prompt, seed = parse_key(fname)
        p = plane_cache.get(f'{prompt}|seed={seed}')
        if p is None:
            continue
        sc = score(bl, p['normal'], p['offset'], np.asarray(bl.vertices))
        if sc is not None:
            sc['cat'] = cat
            sc['fname'] = fname
            baseline_points.append(sc)
    per_variant['baseline'] = baseline_points

    # Aggregate
    def mean_of(ps, k):
        vals = [x[k] for x in ps if np.isfinite(x.get(k, np.nan))]
        return float(np.mean(vals)) if vals else float('nan')

    def std_of(ps, k):
        vals = [x[k] for x in ps if np.isfinite(x.get(k, np.nan))]
        return float(np.std(vals)) if vals else float('nan')

    agg = {}
    for v, pts in per_variant.items():
        agg[v] = {
            'n': len(pts),
            **{k: mean_of(pts, k) for k in ['sym', 'hnc', 'com',
                                             'edge_reg', 'ang_dev', 'shape_cd']},
            **{f'{k}_std': std_of(pts, k) for k in ['sym', 'hnc', 'com',
                                                      'edge_reg', 'ang_dev', 'shape_cd']},
        }

    with open(os.path.join(OUT_DIR, 'ablation_points.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    # ============================================================
    # Pareto overlay plot (single panel per metric)
    # ============================================================
    # Load existing classical Pareto curve for context
    classical = None
    if os.path.exists(PARETO_JSON):
        with open(PARETO_JSON) as f:
            classical = json.load(f)

    metrics_to_plot = [
        ('sym', r'$\mathcal{R}_{\mathrm{sym}}$  (higher = better)'),
        ('hnc', r'$\mathcal{R}_{\mathrm{HNC}}$  (higher = better)'),
        ('com', r'$\mathcal{R}_{\mathrm{com}}$  (higher = better)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams.update({'font.size': 13})

    colors = {
        'diffgeoreward': 'black',
        'sym_only':     '#1f77b4',
        'HNC_only':     '#2ca02c',
        'compact_only': '#ff7f0e',
        'sym_HNC':      '#9467bd',
        'sym_com':      '#8c564b',
        'HNC_com':      '#e377c2',
        'baseline':     '#888888',
    }
    labels = {
        'diffgeoreward': 'DGR (all 3 rewards)',
        'sym_only':     'Sym-Only',
        'HNC_only':     'HNC-Only',
        'compact_only': 'Compact-Only',
        'sym_HNC':      'Sym+HNC (no Com)',
        'sym_com':      'Sym+Com (no HNC)',
        'HNC_com':      'HNC+Com (no Sym)',
        'baseline':     'Baseline (unrefined)',
    }
    markers = {
        'diffgeoreward': '*',
        'sym_only':     's',
        'HNC_only':     'D',
        'compact_only': '^',
        'sym_HNC':      'P',
        'sym_com':      'X',
        'HNC_com':      'h',
        'baseline':     'o',
    }

    for ax, (key, label) in zip(axes, metrics_to_plot):
        if classical is not None:
            # Laplacian-family merged curve
            iters_keys = sorted(
                [int(k) for k in classical['Laplacian'].keys() if k.isdigit()]
            )
            xs, ys = [], []
            for n in iters_keys:
                a = classical['Laplacian'][str(n)]
                b = classical['Taubin'][str(n)]
                if (np.isfinite(a.get('shape_cd', np.nan)) and np.isfinite(a.get(key, np.nan))
                        and np.isfinite(b.get('shape_cd', np.nan)) and np.isfinite(b.get(key, np.nan))):
                    xs.append(0.5 * (a['shape_cd'] + b['shape_cd']))
                    ys.append(0.5 * (a[key] + b[key]))
            if xs:
                ax.plot(xs, ys, '-', color='#1f77b4', linewidth=1.6, alpha=0.55,
                        label='Laplacian-family (iters 1-80)')
            xs, ys = [], []
            for n in iters_keys:
                r = classical['Humphrey'][str(n)]
                if np.isfinite(r.get('shape_cd', np.nan)) and np.isfinite(r.get(key, np.nan)):
                    xs.append(r['shape_cd'])
                    ys.append(r[key])
            if xs:
                ax.plot(xs, ys, '-', color='#d62728', linewidth=1.6, alpha=0.55,
                        label='Humphrey (iters 1-80)')

        # Overlay DGR variants (order: baseline, 1-reward, 2-reward, full)
        plot_order = ['baseline', 'sym_only', 'HNC_only', 'compact_only',
                      'sym_HNC', 'sym_com', 'HNC_com', 'diffgeoreward']
        for v in plot_order:
            if v not in agg:
                continue
            x = agg[v]['shape_cd']
            y = agg[v][key]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ms = 28 if v == 'diffgeoreward' else 14
            ax.plot(x, y, markers[v], color=colors[v], markersize=ms,
                    markeredgecolor='white', markeredgewidth=1.2,
                    label=labels[v], zorder=5 if v == 'diffgeoreward' else 4)

        ax.set_xlabel('shape-CD  (lower = closer to baseline)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.tick_params(axis='both', labelsize=11)

    fig.suptitle('Pareto component ablation: single-reward variants vs full DGR',
                 fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pareto_ablation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'pareto_ablation.png'), dpi=130, bbox_inches='tight')

    # ============================================================
    # Console summary
    # ============================================================
    print('\n=== aggregate means ===')
    print(f"{'variant':20s} {'n':>4s} {'shape-CD':>10s} {'R_sym':>10s} {'R_hnc':>10s} {'R_com':>10s} {'edge_reg':>10s} {'ang_dev':>10s}")
    summary_order = ['baseline', 'sym_only', 'HNC_only', 'compact_only',
                     'sym_HNC', 'sym_com', 'HNC_com', 'diffgeoreward']
    for v in summary_order:
        if v not in agg:
            continue
        r = agg[v]
        print(f"{v:20s} {r['n']:>4d} "
              f"{r['shape_cd']:>10.4f} {r['sym']:>10.4f} "
              f"{r['hnc']:>10.4f} {r['com']:>10.4f} "
              f"{r['edge_reg']:>10.4f} {r['ang_dev']:>10.4f}")
    print(f'\nSaved to {OUT_DIR}/ (ablation_points.json, pareto_ablation.pdf, pareto_ablation.png)')


if __name__ == '__main__':
    main()
