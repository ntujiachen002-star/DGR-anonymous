"""PSR Self-Consistency Experiment: does the mesh represent a coherent surface?

Key insight (vs failed pilot): We do NOT compare refined to baseline as a
reference. Instead, each mesh is measured against ITSELF:
  mesh M -> sample oriented points P -> reconstruct M' via PSR -> compare M' to M

A mesh with inconsistent normals or fragmented surface will reconstruct
poorly (holes, ghosts). A coherent mesh reconstructs faithfully.

Hypothesis: DGR's HNC reward produces more consistent normals, so refined
meshes are MORE self-consistent (lower reconstruction error).

This is NON-CIRCULAR: each mesh is scored individually against its own
surface, and we compare the scores between baseline and refined. The
metric (self-Chamfer) is orthogonal to sym/HNC/compactness.
"""
import os, sys, numpy as np, trimesh, open3d as o3d, time, json
from scipy.spatial import cKDTree
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/baseline')
REFINED_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted')
OUT_DIR = os.path.join(ROOT, 'analysis_results/psr_selfconsistency')
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLE = 8000
N_REF = 30000
DEPTH = 8
MIN_VERTS = 50


def self_consistency_psr(mesh_path):
    """Returns (self_cd, f1, n_components, watertight_out) or None."""
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if len(m.vertices) < MIN_VERTS or len(m.faces) < 20:
        return None

    # Normalize
    m.vertices = m.vertices - m.centroid
    extent = np.max(np.abs(m.vertices))
    if extent < 1e-6:
        return None
    m.vertices = m.vertices / extent

    # Dense reference sample from THIS mesh (not some other target)
    try:
        ref = m.sample(N_REF)
    except Exception:
        return None
    bbox_diag = np.linalg.norm(m.bounds[1] - m.bounds[0])
    tau = 0.01 * bbox_diag

    # Convert to o3d
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(m.vertices)
    om.triangles = o3d.utility.Vector3iVector(m.faces)
    om.compute_vertex_normals()

    try:
        pcd = om.sample_points_poisson_disk(N_SAMPLE)
    except Exception:
        return None
    if len(pcd.points) < 100:
        return None
    if not pcd.has_normals():
        pcd.estimate_normals()

    try:
        rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=DEPTH, scale=1.1, linear_fit=False
        )
    except Exception:
        return None

    verts = np.asarray(rec.vertices)
    faces = np.asarray(rec.triangles)
    if len(verts) < 10 or len(faces) < 10:
        return None

    rec_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        rec_sample = rec_tm.sample(N_REF // 2)
    except Exception:
        return None

    # Self-Chamfer: how well does PSR reconstruct THIS mesh's surface?
    tree_ref, tree_rec = cKDTree(ref), cKDTree(rec_sample)
    d1, _ = tree_rec.query(ref, k=1)
    d2, _ = tree_ref.query(rec_sample, k=1)
    self_cd = 0.5 * (d1.mean() + d2.mean())

    p = (d1 < tau).mean(); r = (d2 < tau).mean()
    f1 = 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0

    try:
        n_comp = len(rec_tm.split(only_watertight=False))
        wt = rec_tm.is_watertight
    except Exception:
        n_comp = -1
        wt = False

    return {
        'self_cd': float(self_cd),
        'f1': float(f1),
        'n_components': int(n_comp),
        'watertight': bool(wt),
    }


def main():
    print('=== PSR Self-Consistency Experiment ===')
    print(f'Samples: {N_SAMPLE}, Reference: {N_REF}, PSR depth: {DEPTH}')

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

    print(f'Found {len(pairs)} pairs\n')

    results = []
    skip = 0
    t0 = time.time()

    for i, (cat, fname, bl_path, rf_path) in enumerate(pairs):
        bl_stats = self_consistency_psr(bl_path)
        rf_stats = self_consistency_psr(rf_path)
        if bl_stats is None or rf_stats is None:
            skip += 1
            continue

        results.append({
            'category': cat, 'file': fname,
            'bl_self_cd': bl_stats['self_cd'], 'rf_self_cd': rf_stats['self_cd'],
            'bl_f1': bl_stats['f1'], 'rf_f1': rf_stats['f1'],
            'bl_n_comp': bl_stats['n_components'], 'rf_n_comp': rf_stats['n_components'],
            'bl_watertight': bl_stats['watertight'], 'rf_watertight': rf_stats['watertight'],
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pairs)}: {len(results)} valid, {skip} skipped, {elapsed:.0f}s')

    elapsed = (time.time() - t0) / 60
    print(f'\nDone: {len(results)} valid, {skip} skipped, {elapsed:.1f} min')

    with open(os.path.join(OUT_DIR, 'psr_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if not results:
        print('No valid results!')
        return

    bl_cd = np.array([r['bl_self_cd'] for r in results])
    rf_cd = np.array([r['rf_self_cd'] for r in results])
    bl_f1 = np.array([r['bl_f1'] for r in results])
    rf_f1 = np.array([r['rf_f1'] for r in results])
    bl_nc = np.array([r['bl_n_comp'] for r in results])
    rf_nc = np.array([r['rf_n_comp'] for r in results])
    bl_wt = np.array([r['bl_watertight'] for r in results])
    rf_wt = np.array([r['rf_watertight'] for r in results])

    print(f'\n=== AGGREGATE (n={len(results)}) ===')

    # Self-Chamfer
    t_cd, p_cd = stats.wilcoxon(bl_cd, rf_cd)
    imp_cd = (1 - rf_cd.mean() / bl_cd.mean()) * 100
    wr_cd = (rf_cd < bl_cd).mean() * 100
    diff_cd = bl_cd - rf_cd
    d_cd = diff_cd.mean() / (diff_cd.std(ddof=1) + 1e-12)
    print(f'  Self-Chamfer:  bl={bl_cd.mean():.5f}  rf={rf_cd.mean():.5f}  '
          f'imp={imp_cd:+.1f}%  wr={wr_cd:.1f}%  d={d_cd:+.3f}  Wilcoxon p={p_cd:.2e}')

    # F1
    t_f1, p_f1 = stats.wilcoxon(bl_f1, rf_f1)
    imp_f1 = (rf_f1.mean() - bl_f1.mean())
    wr_f1 = (rf_f1 > bl_f1).mean() * 100
    print(f'  F1@1% delta:   bl={bl_f1.mean():.3f}    rf={rf_f1.mean():.3f}    '
          f'delta={imp_f1:+.3f}  wr={wr_f1:.1f}%  Wilcoxon p={p_f1:.2e}')

    # Component count (lower = less fragmented)
    imp_nc = bl_nc.mean() - rf_nc.mean()
    wr_nc = (rf_nc < bl_nc).mean() * 100
    print(f'  Components:    bl={bl_nc.mean():.1f}    rf={rf_nc.mean():.1f}    '
          f'delta={imp_nc:+.2f}  wr={wr_nc:.1f}%  (lower=less fragmented)')

    # Watertightness
    print(f'  Watertight out: bl={bl_wt.mean()*100:.1f}%  rf={rf_wt.mean()*100:.1f}%')

    print('\n=== PER-CATEGORY (Self-Chamfer) ===')
    by_cat = {}
    for cat in ['symmetry', 'smoothness', 'compactness']:
        mask = [r['category'] == cat for r in results]
        if sum(mask) < 3:
            continue
        cat_bl = bl_cd[mask]
        cat_rf = rf_cd[mask]
        imp = (1 - cat_rf.mean() / cat_bl.mean()) * 100
        wr = (cat_rf < cat_bl).mean() * 100
        try:
            _, p = stats.wilcoxon(cat_bl, cat_rf)
        except Exception:
            p = 1.0
        print(f'  [{cat:12s}] n={sum(mask):3d}  imp={imp:+.1f}%  wr={wr:.0f}%  p={p:.2e}')
        by_cat[cat] = {'n': sum(mask), 'imp_pct': float(imp), 'wr': float(wr), 'p': float(p)}

    summary = {
        'n': len(results), 'n_skip': skip,
        'self_cd': {'bl': float(bl_cd.mean()), 'rf': float(rf_cd.mean()),
                    'imp_pct': float(imp_cd), 'wr': float(wr_cd),
                    'cohens_d': float(d_cd), 'p_wilcoxon': float(p_cd)},
        'f1': {'bl': float(bl_f1.mean()), 'rf': float(rf_f1.mean()),
               'delta': float(imp_f1), 'wr': float(wr_f1), 'p_wilcoxon': float(p_f1)},
        'components': {'bl': float(bl_nc.mean()), 'rf': float(rf_nc.mean()),
                       'wr_lower': float(wr_nc)},
        'watertight': {'bl_rate': float(bl_wt.mean() * 100), 'rf_rate': float(rf_wt.mean() * 100)},
        'by_category': by_cat,
    }
    with open(os.path.join(OUT_DIR, 'psr_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
