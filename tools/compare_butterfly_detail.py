"""Focused render of the butterfly case: baseline vs old vs new algorithm.

The butterfly is the mesh where multi-start plane estimation most dramatically
beats single-PCA (7.4x better init score). This script renders each mesh from
3 viewing angles with the estimated symmetry plane overlaid, so the plane
orientation difference between the old and new algorithm is visually obvious.
"""
import os
import sys
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
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
    compute_face_normals,
)


BUTTERFLY = 'results/qualitative_meshes/a_symmetric_butterfly_sculpture_seed123_baseline.obj'

# (elev, azim, label)
VIEWS = [
    (15, -70, 'front-left'),
    (15,  20, 'front-right'),
    (85,   0, 'top-down'),
]


def load(path):
    m = trimesh.load(path, process=False)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32)
    f = torch.tensor(np.asarray(m.faces), dtype=torch.long)
    return v, f


def refine(verts, faces, n, d, steps=50):
    v_opt = verts.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.005)
    adj = _build_face_adjacency(faces)
    delta = compute_initial_huber_delta(v_opt, faces)
    with torch.no_grad():
        s0 = symmetry_reward_plane(v_opt, n, d).item()
        sm0 = smoothness_reward(v_opt, faces, delta=delta, _adj=adj).item()
        cp0 = compactness_reward(v_opt, faces).item()
    ss = max(abs(s0), 1e-6); sms = max(abs(sm0), 1e-6); cps = max(abs(cp0), 1e-6)
    for _ in range(steps):
        opt.zero_grad()
        r = 0.7 * symmetry_reward_plane(v_opt, n, d) / ss \
            + 0.15 * smoothness_reward(v_opt, faces, delta=delta, _adj=adj) / sms \
            + 0.15 * compactness_reward(v_opt, faces) / cps
        (-r).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


def shaded_face_colors(verts_np, faces_np, base_rgb, light_dir=(0.3, -0.5, 0.8)):
    from matplotlib.colors import to_rgb
    if isinstance(base_rgb, str):
        base_rgb = to_rgb(base_rgb)
    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
    L = np.asarray(light_dir, dtype=np.float32)
    L = L / (np.linalg.norm(L) + 1e-8)
    intensity = np.clip(np.abs(nrm @ L), 0.0, 1.0)
    intensity = 0.35 + 0.65 * intensity
    base = np.asarray(base_rgb, dtype=np.float32)
    return np.stack([intensity * base[c] for c in range(3)], axis=1)


def render_cell(ax, verts, faces, sym_n, sym_d, elev, azim, base_rgb,
                title='', show_plane=True):
    v_np = verts.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()

    tris = v_np[f_np]
    face_rgb = shaded_face_colors(v_np, f_np, base_rgb)
    face_rgba = np.concatenate([face_rgb, np.full((face_rgb.shape[0], 1), 1.0)], axis=1)

    coll = Poly3DCollection(tris, facecolors=face_rgba,
                             edgecolor=(0, 0, 0, 0.15), linewidth=0.1)
    ax.add_collection3d(coll)

    bmin = v_np.min(0); bmax = v_np.max(0)
    center = (bmin + bmax) / 2.0
    extent = (bmax - bmin).max() * 0.55

    if show_plane:
        n = sym_n.detach().cpu().numpy()
        d = float(sym_d.detach().cpu().item())
        if abs(n[0]) < 0.9:
            u = np.cross(n, np.array([1.0, 0.0, 0.0]))
        else:
            u = np.cross(n, np.array([0.0, 1.0, 0.0]))
        u /= (np.linalg.norm(u) + 1e-8)
        w = np.cross(n, u); w /= (np.linalg.norm(w) + 1e-8)
        p0 = d * n
        s = extent * 1.05
        corners = np.array([
            p0 + s * (+u + w),
            p0 + s * (+u - w),
            p0 + s * (-u - w),
            p0 + s * (-u + w),
        ])
        plane = Poly3DCollection([corners], facecolor=(1.0, 0.35, 0.35, 0.22),
                                  edgecolor=(0.7, 0.0, 0.0, 0.95), linewidth=1.8)
        ax.add_collection3d(plane)

    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_axis_off()
    except Exception:
        pass
    if title:
        ax.set_title(title, fontsize=10)


def main():
    print(f'Loading {BUTTERFLY}')
    verts, faces = load(BUTTERFLY)
    print(f'  {verts.shape[0]} verts, {faces.shape[0]} faces')

    print('Estimating planes...')
    n_old, d_old = estimate_symmetry_plane_pca(verts.detach())
    n_new, d_new = estimate_symmetry_plane(verts.detach())

    score_old_init = symmetry_reward_plane(verts, n_old, d_old).item()
    score_new_init = symmetry_reward_plane(verts, n_new, d_new).item()
    print(f'  init scores: old={score_old_init:+.5f}  new={score_new_init:+.5f}')
    print(f'  old normal:  {n_old.tolist()}  d={d_old.item():+.4f}')
    print(f'  new normal:  {n_new.tolist()}  d={d_new.item():+.4f}')

    print('Refining meshes (50 DGR steps each)...')
    verts_old = refine(verts, faces, n_old, d_old)
    verts_new = refine(verts, faces, n_new, d_new)

    score_old_ref = symmetry_reward_plane(verts_old, n_old, d_old).item()
    score_new_ref = symmetry_reward_plane(verts_new, n_new, d_new).item()
    print(f'  refined scores: old={score_old_ref:+.5f}  new={score_new_ref:+.5f}')

    cases = [
        ('Baseline mesh\n(plane = old single-PCA estimate)',
         verts, n_old, d_old, score_old_init, '#9aa0a6'),
        ('Refined with OLD plane\n(single-PCA, stuck at local min)',
         verts_old, n_old, d_old, score_old_ref, '#4a90d9'),
        ('Refined with NEW plane\n(multi-start, global best)',
         verts_new, n_new, d_new, score_new_ref, '#3cb371'),
    ]

    n_rows = len(cases)
    n_cols = len(VIEWS)
    fig = plt.figure(figsize=(4.5 * n_cols, 4.5 * n_rows))

    for r, (case_title, v, nn, dd, sc, color) in enumerate(cases):
        for c, (elev, azim, vlabel) in enumerate(VIEWS):
            idx = r * n_cols + c + 1
            ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
            title = f'{vlabel}' if r == 0 else ''
            render_cell(ax, v, faces, nn, dd, elev, azim, color, title=title)
            if c == 0:
                ax.text2D(-0.15, 0.5,
                           f'{case_title}\nsym = {sc:+.5f}',
                           transform=ax.transAxes, rotation=90,
                           ha='center', va='center', fontsize=11,
                           fontweight='bold')

    fig.suptitle('Butterfly — symmetry plane estimation: old vs new algorithm\n'
                  f'(old init {score_old_init:+.5f} \u2192 new init {score_new_init:+.5f}'
                  f' = {abs(score_old_init/score_new_init):.1f}x better plane before refinement)',
                  fontsize=13, y=1.00)
    plt.tight_layout()

    out = 'figures/butterfly_plane_detail.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
