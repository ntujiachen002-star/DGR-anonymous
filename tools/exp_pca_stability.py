"""Principal Axis Stability Experiment.

Downstream claim: bilaterally symmetric meshes have well-defined principal
directions (symmetry-normal = clear eigenvector). Asymmetric meshes have
PCA axes that wobble under noise/subsampling.

Non-circular with our symmetry reward:
- Our multi-start symmetry plane estimator is OPTIMIZATION-driven (searches
  over candidate planes + iterative scoring), not PCA.
- PCA uses eigendecomposition of the covariance matrix, a structurally
  different operation.
- The metric measures STABILITY of principal axes under bootstrap
  subsampling, not the symmetry residual itself.

Procedure per mesh:
1. Sample N_POINTS = 10000 surface points (trimesh area-weighted).
2. Compute PCA eigenvectors on full point set -> V_full (3x3, cols = axes).
3. Bootstrap K = 20 iterations: randomly draw 50% of points, recompute PCA.
4. For each bootstrap, match each full axis to its best bootstrap axis
   (max |cos|) and record the angular deviation in degrees.
5. Aggregate: mean angular deviation across 3 axes x K bootstraps.

Lower mean angular deviation = more stable principal axes = better canonical
alignment quality.
"""
import os
import sys
import json
import time
import numpy as np
import trimesh
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/baseline')
REFINED_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted')
OUT_DIR = os.path.join(ROOT, 'analysis_results/pca_stability')
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS = 10000
N_BOOT = 30
BOOT_FRAC = 0.2
NOISE_SIGMA = 0.03  # Gaussian noise sigma (fraction of unit-normalized scale)
SEED = 0


def pca_axes(points):
    """Return (eigvals, eigvecs) of sample covariance. eigvecs columns are axes,
    sorted by descending eigenvalue."""
    c = points - points.mean(axis=0, keepdims=True)
    cov = (c.T @ c) / max(len(c) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def angular_deviation_deg(V_ref, V_boot):
    """For each column of V_ref, find best match in V_boot (max |cos|)
    and return angular deviation in degrees. Returns 3 values."""
    devs = np.zeros(3)
    used = [False, False, False]
    for i in range(3):
        best_j, best_cos = -1, -1.0
        for j in range(3):
            if used[j]:
                continue
            c = abs(float(V_ref[:, i] @ V_boot[:, j]))
            c = min(c, 1.0)
            if c > best_cos:
                best_cos = c
                best_j = j
        used[best_j] = True
        devs[i] = np.degrees(np.arccos(best_cos))
    return devs


def pca_stability(mesh_path, rng):
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if len(m.vertices) < 10 or len(m.faces) < 10:
        return None
    try:
        pts = m.sample(N_POINTS)
    except Exception:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 100:
        return None
    # Normalize scale so results are scale-invariant
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-9:
        return None
    pts = pts / scale

    vals_full, V_full = pca_axes(pts)
    n_boot = int(len(pts) * BOOT_FRAC)
    devs_all = np.zeros((N_BOOT, 3))
    for b in range(N_BOOT):
        idx = rng.choice(len(pts), size=n_boot, replace=False)
        sub = pts[idx] + rng.normal(0.0, NOISE_SIGMA, size=(n_boot, 3))
        _, V_b = pca_axes(sub)
        devs_all[b] = angular_deviation_deg(V_full, V_b)

    # Eigenvalue ratios describe axis separability (big gap -> stable axis)
    # lambda1 >= lambda2 >= lambda3
    gap12 = float((vals_full[0] - vals_full[1]) / (vals_full[0] + 1e-12))
    gap23 = float((vals_full[1] - vals_full[2]) / (vals_full[1] + 1e-12))

    return {
        'mean_dev_deg': float(devs_all.mean()),
        'dev_axis1_deg': float(devs_all[:, 0].mean()),
        'dev_axis2_deg': float(devs_all[:, 1].mean()),
        'dev_axis3_deg': float(devs_all[:, 2].mean()),
        'gap12': gap12,
        'gap23': gap23,
    }


def main():
    print('=== PCA Axis-Stability Downstream Experiment ===')
    print(f'N_POINTS={N_POINTS}  N_BOOT={N_BOOT}  BOOT_FRAC={BOOT_FRAC}')

    pairs = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        bl_dir = os.path.join(BASELINE_DIR, cat)
        rf_dir = os.path.join(REFINED_DIR, cat)
        if not os.path.isdir(bl_dir):
            continue
        for fname in sorted(os.listdir(bl_dir)):
            if not fname.endswith('.obj'):
                continue
            bl_path = os.path.join(bl_dir, fname)
            rf_path = os.path.join(rf_dir, fname)
            if os.path.exists(rf_path):
                pairs.append((cat, fname, bl_path, rf_path))
    print(f'Found {len(pairs)} pairs')

    rng = np.random.default_rng(SEED)
    results = []
    skip = 0
    t0 = time.time()

    for i, (cat, fname, bl_path, rf_path) in enumerate(pairs):
        bl_s = pca_stability(bl_path, rng)
        rf_s = pca_stability(rf_path, rng)
        if bl_s is None or rf_s is None:
            skip += 1
            continue
        results.append({
            'category': cat, 'file': fname,
            'bl_mean_dev': bl_s['mean_dev_deg'],
            'rf_mean_dev': rf_s['mean_dev_deg'],
            'bl_dev_axes': [bl_s['dev_axis1_deg'], bl_s['dev_axis2_deg'], bl_s['dev_axis3_deg']],
            'rf_dev_axes': [rf_s['dev_axis1_deg'], rf_s['dev_axis2_deg'], rf_s['dev_axis3_deg']],
            'bl_gap12': bl_s['gap12'], 'rf_gap12': rf_s['gap12'],
            'bl_gap23': bl_s['gap23'], 'rf_gap23': rf_s['gap23'],
        })
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pairs)}: {len(results)} valid, {skip} skipped, {elapsed:.0f}s', flush=True)

    elapsed_min = (time.time() - t0) / 60
    print(f'\nDone: {len(results)} valid, {skip} skipped, {elapsed_min:.1f} min')

    with open(os.path.join(OUT_DIR, 'pca_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    if not results:
        print('No valid results!')
        return

    bl = np.array([r['bl_mean_dev'] for r in results])
    rf = np.array([r['rf_mean_dev'] for r in results])

    print(f'\n=== AGGREGATE (n={len(results)}) ===')
    imp = (1 - rf.mean() / bl.mean()) * 100
    wr = (rf < bl).mean() * 100
    diff = bl - rf
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    try:
        _, p = stats.wilcoxon(bl, rf)
    except Exception:
        p = 1.0
    print(f'  Mean axis deviation: bl={bl.mean():.3f} deg  rf={rf.mean():.3f} deg')
    print(f'  Improvement: {imp:+.1f}%  win rate={wr:.1f}%  Cohen d={d:+.3f}  p={p:.2e}')

    # Per-axis (axis 1 = largest eigenvalue = most stable in principle)
    for ax in [1, 2, 3]:
        bl_ax = np.array([r['bl_dev_axes'][ax - 1] for r in results])
        rf_ax = np.array([r['rf_dev_axes'][ax - 1] for r in results])
        imp_ax = (1 - rf_ax.mean() / bl_ax.mean()) * 100
        try:
            _, p_ax = stats.wilcoxon(bl_ax, rf_ax)
        except Exception:
            p_ax = 1.0
        print(f'  Axis {ax}: bl={bl_ax.mean():.3f}  rf={rf_ax.mean():.3f}  imp={imp_ax:+.1f}%  p={p_ax:.2e}')

    # Per-category
    print(f'\n=== PER-CATEGORY ===')
    by_cat = {}
    for cat in ['symmetry', 'smoothness', 'compactness']:
        mask = [r['category'] == cat for r in results]
        if sum(mask) < 3:
            continue
        cb = bl[mask]; cr = rf[mask]
        imp_c = (1 - cr.mean() / cb.mean()) * 100
        wr_c = (cr < cb).mean() * 100
        try:
            _, p_c = stats.wilcoxon(cb, cr)
        except Exception:
            p_c = 1.0
        print(f'  [{cat:12s}] n={sum(mask):3d}  bl={cb.mean():.3f}  rf={cr.mean():.3f}  '
              f'imp={imp_c:+.1f}%  wr={wr_c:.0f}%  p={p_c:.2e}')
        by_cat[cat] = {'n': sum(mask), 'bl': float(cb.mean()), 'rf': float(cr.mean()),
                       'imp_pct': float(imp_c), 'wr': float(wr_c), 'p': float(p_c)}

    # Eigenvalue gap analysis (symmetric -> larger gap on axis perpendicular to sym plane)
    bl_gap = np.array([r['bl_gap12'] for r in results])
    rf_gap = np.array([r['rf_gap12'] for r in results])
    print(f'\n  Eigval gap (1-2): bl={bl_gap.mean():.3f}  rf={rf_gap.mean():.3f}  '
          f'delta={rf_gap.mean() - bl_gap.mean():+.3f}')

    summary = {
        'n': len(results), 'n_skip': skip,
        'mean_dev': {'bl': float(bl.mean()), 'rf': float(rf.mean()),
                     'imp_pct': float(imp), 'wr': float(wr),
                     'cohens_d': float(d), 'p_wilcoxon': float(p)},
        'by_category': by_cat,
        'n_points': N_POINTS, 'n_boot': N_BOOT, 'boot_frac': BOOT_FRAC,
    }
    with open(os.path.join(OUT_DIR, 'pca_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
