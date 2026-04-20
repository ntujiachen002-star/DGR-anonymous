"""Remeshing Robustness Experiment: downstream validation that DGR-refined
meshes are more amenable to standard geometry processing operations.

For each (prompt, seed) pair, we:
1. Load baseline and refined (handcrafted) OBJ
2. Decimate both to a target face count via Quadric Edge Collapse
3. Measure the Hausdorff distance between original and decimated mesh
4. Compare: if refined mesh has LOWER Hausdorff distance, it means the
   geometry is more regular and loses less information under decimation.

This is a NON-CIRCULAR downstream evaluation — Hausdorff distance is not
any of the three optimized rewards. It measures geometric fidelity under
a standard mesh processing operation.

CPU only. ~10-30 minutes depending on mesh count.
"""
import os
import sys
import json
import numpy as np
import trimesh
import pymeshlab
from collections import defaultdict
from scipy import stats
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/baseline')
REFINED_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted')
OUT_DIR = os.path.join(ROOT, 'analysis_results/remeshing_robustness')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FACE_RATIO = 0.1  # Decimate to 10% of original faces
MIN_FACES = 20  # Skip meshes with fewer faces


def hausdorff_distance(mesh_a, mesh_b):
    """Compute symmetric Hausdorff distance between two meshes using pymeshlab."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh_a.vertices, mesh_a.faces))
    ms.add_mesh(pymeshlab.Mesh(mesh_b.vertices, mesh_b.faces))
    result = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1)
    return result['max']


def decimate_mesh(mesh, target_faces):
    """Decimate mesh using Quadric Edge Collapse."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int32)
    ))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    m = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=m.vertex_matrix(),
        faces=m.face_matrix(),
        process=False
    )


def chamfer_distance(mesh_a, mesh_b, n_samples=5000):
    """Approximate Chamfer distance via point sampling."""
    pts_a = mesh_a.sample(min(n_samples, max(100, len(mesh_a.vertices))))
    pts_b = mesh_b.sample(min(n_samples, max(100, len(mesh_b.vertices))))

    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    d_a2b, _ = tree_b.query(pts_a)
    d_b2a, _ = tree_a.query(pts_b)

    return (d_a2b.mean() + d_b2a.mean()) / 2


def main():
    print('=== Remeshing Robustness Experiment ===')
    print(f'Baseline dir: {BASELINE_DIR}')
    print(f'Refined dir:  {REFINED_DIR}')
    print(f'Target face ratio: {TARGET_FACE_RATIO}')

    # Collect pairs
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

    print(f'Found {len(pairs)} paired meshes')

    results = []
    n_skip = 0
    t0 = time.time()

    for i, (cat, fname, bl_path, rf_path) in enumerate(pairs):
        try:
            bl = trimesh.load(bl_path, process=False, force='mesh')
            rf = trimesh.load(rf_path, process=False, force='mesh')
        except Exception:
            n_skip += 1
            continue

        if len(bl.faces) < MIN_FACES or len(rf.faces) < MIN_FACES:
            n_skip += 1
            continue

        # Target face count
        target = max(10, int(len(bl.faces) * TARGET_FACE_RATIO))

        try:
            # Decimate both
            bl_dec = decimate_mesh(bl, target)
            rf_dec = decimate_mesh(rf, target)

            # Chamfer distance: original vs decimated
            bl_chamfer = chamfer_distance(bl, bl_dec)
            rf_chamfer = chamfer_distance(rf, rf_dec)

            results.append({
                'category': cat,
                'file': fname,
                'bl_faces': len(bl.faces),
                'rf_faces': len(rf.faces),
                'target_faces': target,
                'bl_dec_faces': len(bl_dec.faces),
                'rf_dec_faces': len(rf_dec.faces),
                'bl_chamfer': float(bl_chamfer),
                'rf_chamfer': float(rf_chamfer),
                'improvement': float((bl_chamfer - rf_chamfer) / (bl_chamfer + 1e-10) * 100),
            })
        except Exception as e:
            n_skip += 1
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pairs)} done, {n_skip} skipped, {elapsed:.0f}s')

    elapsed = (time.time() - t0) / 60
    print(f'\nDone: {len(results)} valid, {n_skip} skipped, {elapsed:.1f} min')

    # Save raw results
    with open(os.path.join(OUT_DIR, 'remeshing_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # === Aggregate statistics ===
    bl_chamfers = np.array([r['bl_chamfer'] for r in results])
    rf_chamfers = np.array([r['rf_chamfer'] for r in results])

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(bl_chamfers, rf_chamfers)
    diff = bl_chamfers - rf_chamfers  # positive = baseline worse = DGR better
    d_cohen = diff.mean() / (diff.std(ddof=1) + 1e-12)
    win_rate = (rf_chamfers < bl_chamfers).mean() * 100

    print(f'\n=== RESULTS (n={len(results)}) ===')
    print(f'  Baseline Chamfer (mean): {bl_chamfers.mean():.6f}')
    print(f'  Refined  Chamfer (mean): {rf_chamfers.mean():.6f}')
    print(f'  Improvement: {(1 - rf_chamfers.mean()/bl_chamfers.mean())*100:+.1f}%')
    print(f'  Win rate (refined < baseline): {win_rate:.1f}%')
    print(f'  Paired t: t={t_stat:.3f}, p={p_val:.2e}')
    print(f'  Cohen\'s d: {d_cohen:+.3f}')

    # Per-category
    print(f'\n=== PER-CATEGORY ===')
    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_bl = np.array([r['bl_chamfer'] for r in results if r['category'] == cat])
        cat_rf = np.array([r['rf_chamfer'] for r in results if r['category'] == cat])
        if len(cat_bl) < 3:
            continue
        t2, p2 = stats.ttest_rel(cat_bl, cat_rf)
        wr = (cat_rf < cat_bl).mean() * 100
        print(f'  {cat:12s} n={len(cat_bl):3d}  '
              f'bl={cat_bl.mean():.6f}  rf={cat_rf.mean():.6f}  '
              f'imp={(1-cat_rf.mean()/cat_bl.mean())*100:+.1f}%  '
              f'wr={wr:.0f}%  p={p2:.2e}')

    # Save summary
    summary = {
        'n': len(results),
        'n_skip': n_skip,
        'target_face_ratio': TARGET_FACE_RATIO,
        'bl_chamfer_mean': float(bl_chamfers.mean()),
        'rf_chamfer_mean': float(rf_chamfers.mean()),
        'improvement_pct': float((1 - rf_chamfers.mean()/bl_chamfers.mean())*100),
        'win_rate': float(win_rate),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'cohens_d': float(d_cohen),
    }
    with open(os.path.join(OUT_DIR, 'remeshing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
