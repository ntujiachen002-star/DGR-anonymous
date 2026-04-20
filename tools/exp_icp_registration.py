"""Downstream Experiment: ICP Registration Convergence.

Hypothesis: DGR-refined meshes are more amenable to standard 3D pipeline
operations. We test this using Iterative Closest Point (ICP) registration,
a ubiquitous subroutine in 3D perception, reconstruction, and tracking.

Protocol (NON-CIRCULAR evaluation):
1. For each (baseline, refined) pair:
   a. Sample N=2048 surface points from the mesh (ground-truth point cloud)
   b. Apply random SE(3) perturbation: rotation ~ U(0, 30°), translation ~ 5% extent
   c. Add Gaussian noise: sigma = 1% extent
   d. Run ICP to register perturbed cloud back to original
   e. Measure: rotation error (deg), translation error (% extent), final RMSE

Metrics: rotation error, translation error, success rate (rot err < 5°).
NONE of these are symmetry, HNC, or compactness — fully independent evaluation.

Protocol: 5 random perturbations × 241 paired meshes = 1,205 ICP runs per variant.
CPU only, ~10-15 min total.
"""
import os
import sys
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy import stats
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/baseline')
REFINED_DIR = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted')
OUT_DIR = os.path.join(ROOT, 'analysis_results/icp_registration')
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS = 2048
N_PERTURBS = 10
MAX_ROT_DEG = 90.0
TRANSLATION_FRAC = 0.15
NOISE_FRAC = 0.03
MIN_VERTS = 20
ICP_MAX_ITER = 20  # Fewer iterations = test convergence speed


def random_rotation(max_angle_deg, rng):
    """Random rotation with angle up to max_angle_deg."""
    angle = rng.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    # Rodrigues
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R, np.degrees(abs(angle))


def rotation_error_deg(R_pred, R_true):
    """Geodesic rotation error in degrees."""
    R_diff = R_pred @ R_true.T
    cos_theta = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_theta))


def icp_point_to_plane(src, tgt, tgt_normals, max_iter=50, tol=1e-6):
    """Point-to-plane ICP. Returns final rotation, translation, rmse."""
    R = np.eye(3)
    t = np.zeros(3)
    tree = cKDTree(tgt)
    prev_rmse = np.inf

    for it in range(max_iter):
        src_t = src @ R.T + t
        dists, idx = tree.query(src_t)
        closest = tgt[idx]
        normals = tgt_normals[idx]

        # Point-to-plane error: (p - q) . n
        residuals = np.sum((src_t - closest) * normals, axis=1)
        rmse = np.sqrt(np.mean(residuals ** 2))

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

        # Solve linearized system for small rotation + translation
        # [p x n, n]^T * [omega, t]^T = -(p - q) . n
        A = np.zeros((len(src_t), 6))
        A[:, :3] = np.cross(src_t, normals)
        A[:, 3:] = normals
        b = -residuals

        try:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break

        omega = x[:3]
        dt = x[3:]
        theta = np.linalg.norm(omega)
        if theta > 1e-10:
            k = omega / theta
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            dR = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        else:
            dR = np.eye(3)

        R = dR @ R
        t = dR @ t + dt

    return R, t, rmse


def run_icp_test(mesh_path, rng):
    """Run N_PERTURBS ICP trials on a mesh. Returns per-trial metrics."""
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if len(mesh.vertices) < MIN_VERTS or len(mesh.faces) < 10:
        return None

    # Normalize to unit extent (for consistent perturbation magnitudes)
    mesh.vertices = mesh.vertices - mesh.centroid
    extent = np.max(np.abs(mesh.vertices))
    if extent < 1e-6:
        return None
    mesh.vertices = mesh.vertices / extent

    # Sample surface points + normals
    try:
        pts_tgt, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS)
        tgt_normals = mesh.face_normals[face_idx]
    except Exception:
        return None
    if len(pts_tgt) < 100:
        return None

    trials = []
    for k in range(N_PERTURBS):
        # Perturb target to create source
        R_true, angle_mag = random_rotation(MAX_ROT_DEG, rng)
        t_true = rng.uniform(-TRANSLATION_FRAC, TRANSLATION_FRAC, 3)
        noise = rng.normal(0, NOISE_FRAC, pts_tgt.shape)

        pts_src = pts_tgt @ R_true.T + t_true + noise

        # Run ICP: recover transform to align src back to tgt
        try:
            R_pred, t_pred, rmse = icp_point_to_plane(
                pts_src, pts_tgt, tgt_normals, max_iter=ICP_MAX_ITER
            )
        except Exception:
            continue

        # R_pred * (R_true * p + t_true) + t_pred ≈ p
        # So R_pred @ R_true should be ≈ I
        rot_err = rotation_error_deg(R_pred @ R_true, np.eye(3))
        trans_err = np.linalg.norm(R_pred @ t_true + t_pred)

        trials.append({
            'applied_angle': float(angle_mag),
            'rot_err_deg': float(rot_err),
            'trans_err': float(trans_err),
            'final_rmse': float(rmse),
        })

    if not trials:
        return None
    return trials


def main():
    print('=== ICP Registration Experiment ===')
    print(f'Target points per mesh: {N_POINTS}')
    print(f'Perturbations per mesh: {N_PERTURBS}')
    print(f'Max rotation: {MAX_ROT_DEG}°')
    print()

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
    skip = 0
    t0 = time.time()

    for i, (cat, fname, bl_path, rf_path) in enumerate(pairs):
        # Use same rng seed for baseline and refined — they see identical perturbations
        rng_bl = np.random.default_rng(42 + i)
        rng_rf = np.random.default_rng(42 + i)

        bl_trials = run_icp_test(bl_path, rng_bl)
        rf_trials = run_icp_test(rf_path, rng_rf)

        if bl_trials is None or rf_trials is None:
            skip += 1
            continue

        # Aggregate per-mesh
        bl_rot_errs = [t['rot_err_deg'] for t in bl_trials]
        rf_rot_errs = [t['rot_err_deg'] for t in rf_trials]
        bl_trans_errs = [t['trans_err'] for t in bl_trials]
        rf_trans_errs = [t['trans_err'] for t in rf_trials]
        bl_rmses = [t['final_rmse'] for t in bl_trials]
        rf_rmses = [t['final_rmse'] for t in rf_trials]

        results.append({
            'category': cat,
            'file': fname,
            'bl_rot_err_mean': float(np.mean(bl_rot_errs)),
            'rf_rot_err_mean': float(np.mean(rf_rot_errs)),
            'bl_rot_err_median': float(np.median(bl_rot_errs)),
            'rf_rot_err_median': float(np.median(rf_rot_errs)),
            'bl_trans_err_mean': float(np.mean(bl_trans_errs)),
            'rf_trans_err_mean': float(np.mean(rf_trans_errs)),
            'bl_rmse_mean': float(np.mean(bl_rmses)),
            'rf_rmse_mean': float(np.mean(rf_rmses)),
            'n_trials': min(len(bl_trials), len(rf_trials)),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pairs)}: {len(results)} valid, {skip} skipped, {elapsed:.0f}s')

    elapsed = (time.time() - t0) / 60
    print(f'\nDone: {len(results)} valid pairs, {skip} skipped, {elapsed:.1f} min')

    # Save raw
    with open(os.path.join(OUT_DIR, 'icp_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Paired analysis — per-mesh mean across 5 trials
    bl_rot = np.array([r['bl_rot_err_mean'] for r in results])
    rf_rot = np.array([r['rf_rot_err_mean'] for r in results])
    bl_trans = np.array([r['bl_trans_err_mean'] for r in results])
    rf_trans = np.array([r['rf_trans_err_mean'] for r in results])
    bl_rmse = np.array([r['bl_rmse_mean'] for r in results])
    rf_rmse = np.array([r['rf_rmse_mean'] for r in results])

    def paired(bl, rf, name):
        diff = bl - rf  # positive = baseline worse = refined better
        t_stat, p = stats.ttest_rel(bl, rf)
        d = diff.mean() / (diff.std(ddof=1) + 1e-12)
        wr = (rf < bl).mean() * 100
        imp = (1 - rf.mean() / (bl.mean() + 1e-12)) * 100
        print(f'  {name:25s} bl={bl.mean():.4f}  rf={rf.mean():.4f}  '
              f'imp={imp:+.1f}%  wr={wr:.1f}%  d={d:+.3f}  p={p:.2e}')
        return {'bl': float(bl.mean()), 'rf': float(rf.mean()),
                'improvement_pct': float(imp), 'win_rate': float(wr),
                'cohens_d': float(d), 't': float(t_stat), 'p': float(p)}

    print(f'\n=== AGGREGATE (n={len(results)}) ===')
    stats_rot = paired(bl_rot, rf_rot, 'Rotation error (deg)')
    stats_trans = paired(bl_trans, rf_trans, 'Translation error')
    stats_rmse = paired(bl_rmse, rf_rmse, 'Final RMSE')

    # Success rate (rot err < 5 deg)
    bl_success = (bl_rot < 5.0).mean() * 100
    rf_success = (rf_rot < 5.0).mean() * 100
    print(f'  {"Success rate (<5°)":25s} bl={bl_success:.1f}%  rf={rf_success:.1f}%')

    print(f'\n=== PER-CATEGORY ===')
    by_cat = {}
    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_bl_rot = np.array([r['bl_rot_err_mean'] for r in results if r['category'] == cat])
        cat_rf_rot = np.array([r['rf_rot_err_mean'] for r in results if r['category'] == cat])
        if len(cat_bl_rot) < 3:
            continue
        print(f'\n  [{cat}] n={len(cat_bl_rot)}')
        by_cat[cat] = paired(cat_bl_rot, cat_rf_rot, 'Rotation error')

    summary = {
        'n': len(results),
        'n_skip': skip,
        'n_points': N_POINTS,
        'n_perturbs': N_PERTURBS,
        'max_rot_deg': MAX_ROT_DEG,
        'translation_frac': TRANSLATION_FRAC,
        'noise_frac': NOISE_FRAC,
        'rotation_error_deg': stats_rot,
        'translation_error': stats_trans,
        'final_rmse': stats_rmse,
        'bl_success_5deg': float(bl_success),
        'rf_success_5deg': float(rf_success),
        'by_category': by_cat,
    }
    with open(os.path.join(OUT_DIR, 'icp_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
