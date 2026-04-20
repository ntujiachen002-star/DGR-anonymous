"""Pilot: Poisson Surface Reconstruction round-trip on 10 meshes.

Mechanism: DGR's HNC reward produces more consistent face normals, which
PSR (Kazhdan 2006) directly integrates. Better normals -> cleaner
reconstruction. This is NON-CIRCULAR because the evaluation metric is
Chamfer distance to a reference cloud, not dihedral consistency.

Pilot goal: verify the sign of the effect on 10 meshes before scaling.
"""
import os, sys, numpy as np, trimesh, open3d as o3d, time
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/baseline')
REFINED_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted')


def psr_reconstruct(mesh_path, n_points=10000, depth=8):
    """Load mesh -> sample oriented points -> PSR -> new mesh."""
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if len(m.vertices) < 50 or len(m.faces) < 20:
        return None, None

    # Normalize to unit bbox (same scale for baseline + refined)
    m.vertices -= m.centroid
    extent = np.max(np.abs(m.vertices))
    if extent < 1e-6:
        return None, None
    m.vertices /= extent

    # Convert to o3d mesh
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(m.vertices)
    om.triangles = o3d.utility.Vector3iVector(m.faces)
    om.compute_vertex_normals()

    # Sample oriented points
    try:
        pcd = om.sample_points_poisson_disk(n_points)
    except Exception:
        return None, None

    if len(pcd.points) < 100:
        return None, None

    # Ensure normals
    if not pcd.has_normals():
        pcd.estimate_normals()

    # Screened Poisson reconstruction
    rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False
    )

    verts = np.asarray(rec.vertices)
    faces = np.asarray(rec.triangles)
    if len(verts) < 10 or len(faces) < 10:
        return None, None

    return m, trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def chamfer_l1(pcd_a, pcd_b):
    """Symmetric Chamfer-L1."""
    tree_a, tree_b = cKDTree(pcd_a), cKDTree(pcd_b)
    d_ab, _ = tree_b.query(pcd_a, k=1)
    d_ba, _ = tree_a.query(pcd_b, k=1)
    return 0.5 * (d_ab.mean() + d_ba.mean())


def f_score(pcd_a, pcd_b, tau):
    """F-score at threshold tau."""
    tree_a, tree_b = cKDTree(pcd_a), cKDTree(pcd_b)
    d_ab, _ = tree_b.query(pcd_a, k=1)
    d_ba, _ = tree_a.query(pcd_b, k=1)
    p = (d_ab < tau).mean()
    r = (d_ba < tau).mean()
    return 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0


# === Pilot ===
print('=== PSR Pilot (10 meshes) ===')
pairs = []
for cat in ['symmetry', 'smoothness', 'compactness']:
    bl_dir = os.path.join(BASELINE_DIR, cat)
    rf_dir = os.path.join(REFINED_DIR, cat)
    for fname in sorted(os.listdir(bl_dir))[:5]:  # first 5 per category
        if not fname.endswith('.obj'): continue
        rf_path = os.path.join(rf_dir, fname)
        bl_path = os.path.join(bl_dir, fname)
        if os.path.exists(rf_path):
            pairs.append((cat, fname, bl_path, rf_path))
        if len(pairs) >= 10:
            break
    if len(pairs) >= 10:
        break

print(f'Testing {len(pairs)} meshes\n')

results = []
t0 = time.time()
for cat, fname, bl_path, rf_path in pairs:
    print(f'[{cat}] {fname}')
    try:
        bl_orig, bl_rec = psr_reconstruct(bl_path)
        rf_orig, rf_rec = psr_reconstruct(rf_path)
    except Exception as e:
        print(f'  error: {e}')
        continue

    if bl_rec is None or rf_rec is None:
        print('  skip: reconstruction failed')
        continue

    # Reference = dense sample of BASELINE original (fair to DGR)
    ref = bl_orig.sample(30000)
    bbox_diag = np.linalg.norm(bl_orig.bounds[1] - bl_orig.bounds[0])
    tau = 0.01 * bbox_diag

    # Sample from reconstructions
    try:
        s_bl = bl_rec.sample(20000)
        s_rf = rf_rec.sample(20000)
    except Exception:
        print('  skip: sample failed')
        continue

    cd_bl = chamfer_l1(s_bl, ref)
    cd_rf = chamfer_l1(s_rf, ref)
    f1_bl = f_score(s_bl, ref, tau)
    f1_rf = f_score(s_rf, ref, tau)

    improvement = (cd_bl - cd_rf) / cd_bl * 100
    print(f'  CD: bl={cd_bl:.5f}  rf={cd_rf:.5f}  ({improvement:+.1f}%)')
    print(f'  F1: bl={f1_bl:.3f}  rf={f1_rf:.3f}')

    results.append({
        'cat': cat, 'file': fname,
        'cd_bl': cd_bl, 'cd_rf': cd_rf,
        'f1_bl': f1_bl, 'f1_rf': f1_rf,
    })

elapsed = time.time() - t0
print(f'\n=== PILOT SUMMARY (n={len(results)}, {elapsed:.0f}s) ===')
if results:
    import numpy as np
    cd_bls = np.array([r['cd_bl'] for r in results])
    cd_rfs = np.array([r['cd_rf'] for r in results])
    f1_bls = np.array([r['f1_bl'] for r in results])
    f1_rfs = np.array([r['f1_rf'] for r in results])

    cd_imp = (1 - cd_rfs.mean() / cd_bls.mean()) * 100
    f1_imp = (f1_rfs.mean() - f1_bls.mean())
    wr_cd = (cd_rfs < cd_bls).mean() * 100
    wr_f1 = (f1_rfs > f1_bls).mean() * 100

    print(f'  Chamfer: bl={cd_bls.mean():.5f}  rf={cd_rfs.mean():.5f}  imp={cd_imp:+.1f}%  wr={wr_cd:.0f}%')
    print(f'  F1@1%:   bl={f1_bls.mean():.3f}    rf={f1_rfs.mean():.3f}    delta={f1_imp:+.3f}  wr={wr_f1:.0f}%')

    print(f'\n  Sign of Chamfer improvement: {"POSITIVE (DGR wins)" if cd_imp > 0 else "NEGATIVE (DGR loses)"}')
    if cd_imp > 5 and wr_cd > 60:
        print('  => Recommended to scale to full N=241')
    else:
        print('  => Weak signal; may need larger N or different depth')
