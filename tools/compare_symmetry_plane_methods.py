"""Visual comparison: baseline vs old (PCA-only) vs new (multi-start) symmetry plane.

For each test mesh:
  1. Estimate plane via the legacy single-PCA method.
  2. Estimate plane via the new multi-start (PCA + Fibonacci sphere) method.
  3. Refine the mesh with each plane using DGR.
  4. Render baseline / old-refined / new-refined as 3 columns.
  5. Annotate each cell with the symmetry score and plane normal.

Run:
    python tools/compare_symmetry_plane_methods.py \
        --output figures/symmetry_plane_comparison.png

The comparison is offline — it consumes pre-existing baseline OBJ files,
no Shap-E generation required.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geo_reward import (
    estimate_symmetry_plane,
    estimate_symmetry_plane_pca,
    symmetry_reward_plane,
    smoothness_reward,
    compactness_reward,
    compute_initial_huber_delta,
    _build_face_adjacency,
)


DEFAULT_MESHES = [
    ('results/qualitative_meshes/a_mirror-symmetric_face_mask_seed456_baseline.obj', 'face mask'),
    ('results/qualitative_meshes/a_symmetric_butterfly_sculpture_seed123_baseline.obj', 'butterfly'),
    ('results/qualitative_meshes/a_symmetric_vase_seed123_baseline.obj', 'vase'),
    ('results/qualitative_meshes/a_compact_treasure_chest_seed42_baseline.obj', 'treasure chest'),
    ('results/full/baseline/symmetry/baseline_seed42.obj', 'window frame'),
    ('results/full/baseline/symmetry/baseline_seed123.obj', 'torii gate'),
    ('results/baseline/symmetry/baseline_seed42.obj', 'hourglass'),
]


def load_mesh(path, device):
    m = trimesh.load(path, process=False)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    f = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=device)
    return v, f


def refine_mesh(verts, faces, sym_normal, sym_offset, weights, steps=50, lr=0.005):
    """Inlined DGR refinement with a fixed pre-estimated symmetry plane.

    Replicates `shape_gen.refine_with_geo_reward` but imports only from
    `geo_reward` so we don't depend on Shap-E being installed.
    """
    v_opt = verts.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)
    face_adj = _build_face_adjacency(faces)
    huber_delta = compute_initial_huber_delta(v_opt, faces)

    with torch.no_grad():
        sym0 = symmetry_reward_plane(v_opt, sym_normal, sym_offset).item()
        smo0 = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj).item()
        com0 = compactness_reward(v_opt, faces).item()
    sym_s = max(abs(sym0), 1e-6)
    smo_s = max(abs(smo0), 1e-6)
    com_s = max(abs(com0), 1e-6)

    for _ in range(steps):
        optimizer.zero_grad()
        reward = torch.tensor(0.0, device=verts.device)
        if weights[0] > 0:
            reward = reward + weights[0] * symmetry_reward_plane(v_opt, sym_normal, sym_offset) / sym_s
        if weights[1] > 0:
            reward = reward + weights[1] * smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj) / smo_s
        if weights[2] > 0:
            reward = reward + weights[2] * compactness_reward(v_opt, faces) / com_s
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()

    return v_opt.detach()


def render_mesh_with_plane(ax, verts, faces, sym_normal, sym_offset,
                            title, score, color='#4a90d9'):
    """Render a triangle mesh on a 3D axis with the symmetry plane overlaid."""
    v_np = verts.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()

    tris = v_np[f_np]
    coll = Poly3DCollection(tris, alpha=0.55, facecolor=color,
                             edgecolor='#1a1a1a', linewidth=0.05)
    ax.add_collection3d(coll)

    bbox_min = v_np.min(axis=0)
    bbox_max = v_np.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = (bbox_max - bbox_min).max() * 0.6

    n = sym_normal.detach().cpu().numpy()
    d = float(sym_offset.detach().cpu().item())
    if abs(n[0]) < 0.9:
        u = np.cross(n, np.array([1.0, 0.0, 0.0]))
    else:
        u = np.cross(n, np.array([0.0, 1.0, 0.0]))
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-8)

    p0 = d * n
    s = extent
    corners = np.array([
        p0 + s * (+u + v),
        p0 + s * (+u - v),
        p0 + s * (-u - v),
        p0 + s * (-u + v),
    ])
    plane_poly = Poly3DCollection([corners], alpha=0.25, facecolor='#ff6b6b',
                                    edgecolor='#aa0000', linewidth=1.2)
    ax.add_collection3d(plane_poly)

    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    n_str = f'n=[{n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}]'
    ax.set_title(f'{title}\nsym={score:+.5f}\n{n_str}', fontsize=8)


def evaluate_symmetry(verts, n, d):
    with torch.no_grad():
        return symmetry_reward_plane(verts, n, d).item()


def make_comparison(mesh_paths_with_labels, output_path, device='cpu',
                    refine_steps=50):
    n_rows = len(mesh_paths_with_labels)
    n_cols = 3
    fig = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))

    sym_weight = torch.tensor([0.7, 0.15, 0.15], device=device)

    headers = ['Baseline (no refinement)',
               'Old: single PCA + refine',
               'New: multi-start + refine']

    summary = []

    for row_idx, (path, label) in enumerate(mesh_paths_with_labels):
        if not os.path.exists(path):
            print(f'  [skip] {path} not found')
            continue
        verts, faces = load_mesh(path, device=device)
        print(f'\n[{row_idx+1}/{n_rows}] {label} — {verts.shape[0]} verts')

        t = time.time()
        n_old, d_old = estimate_symmetry_plane_pca(verts.detach())
        old_est_t = time.time() - t

        t = time.time()
        n_new, d_new = estimate_symmetry_plane(verts.detach())
        new_est_t = time.time() - t

        baseline_score_old = evaluate_symmetry(verts, n_old, d_old)
        baseline_score_new = evaluate_symmetry(verts, n_new, d_new)
        print(f'  plane est: old={old_est_t*1000:.0f}ms  new={new_est_t*1000:.0f}ms')
        print(f'  init score: old={baseline_score_old:+.5f}  new={baseline_score_new:+.5f}')

        verts_old = refine_mesh(verts, faces, n_old, d_old, sym_weight, steps=refine_steps)
        verts_new = refine_mesh(verts, faces, n_new, d_new, sym_weight, steps=refine_steps)

        score_baseline_under_new = evaluate_symmetry(verts, n_new, d_new)
        score_old = evaluate_symmetry(verts_old, n_old, d_old)
        score_new = evaluate_symmetry(verts_new, n_new, d_new)
        print(f'  after 50 DGR steps: old={score_old:+.5f}  new={score_new:+.5f}')
        summary.append({
            'mesh': label,
            'init_old': baseline_score_old,
            'init_new': baseline_score_new,
            'refined_old': score_old,
            'refined_new': score_new,
        })

        ax_b = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + 1, projection='3d')
        ax_o = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + 2, projection='3d')
        ax_n = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + 3, projection='3d')

        render_mesh_with_plane(ax_b, verts, faces, n_new, d_new,
                                f'{label}\nbaseline mesh\n(plane = new estimate)',
                                score_baseline_under_new, color='#9aa0a6')
        render_mesh_with_plane(ax_o, verts_old, faces, n_old, d_old,
                                f'{label}\nrefined w/ old plane',
                                score_old, color='#4a90d9')
        render_mesh_with_plane(ax_n, verts_new, faces, n_new, d_new,
                                f'{label}\nrefined w/ new plane',
                                score_new, color='#3cb371')

        if row_idx == 0:
            for col_idx, h in enumerate(headers):
                ax = fig.axes[col_idx]
                ax.text2D(0.5, 1.18, h, transform=ax.transAxes,
                          ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f'\nSaved comparison figure to: {output_path}')

    print('\n=== Summary (symmetry score; closer to 0 = more symmetric) ===')
    print(f'{"mesh":<22} {"init(old)":>12} {"init(new)":>12} {"refined(old)":>14} {"refined(new)":>14}')
    for s in summary:
        print(f'{s["mesh"]:<22} {s["init_old"]:>12.5f} {s["init_new"]:>12.5f} '
              f'{s["refined_old"]:>14.5f} {s["refined_new"]:>14.5f}')

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='figures/symmetry_plane_comparison.png')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--refine-steps', type=int, default=50)
    args = parser.parse_args()

    mesh_list = [(p, lbl) for p, lbl in DEFAULT_MESHES if os.path.exists(p)]
    if not mesh_list:
        print('No test meshes found. Edit DEFAULT_MESHES.')
        return

    make_comparison(mesh_list, args.output,
                     device=args.device,
                     refine_steps=args.refine_steps)


if __name__ == '__main__':
    main()
