"""Render vase baseline + refined (new protocol) in user-study style.
Same rendering as exp_o_userstudy_gen.py: blue faces, gray edges, 3 angles.
Outputs individual PNGs for use in pipeline figure.
"""
import os, sys, numpy as np, trimesh, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

BASELINE_OBJ = os.path.join(ROOT, 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj')
ANGLES = [0, 45, 135]  # Front, Front-Right, Back-Left (3 views like reference)
ANGLE_NAMES = ['front', 'front_right', 'back_left']


def render_single(mesh_path, angle, out_path, resolution=512):
    """Render one view — same style as user study."""
    mesh = trimesh.load(str(mesh_path), force='mesh')
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Center + normalize
    verts = verts - verts.mean(axis=0)
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale

    # Subsample if needed
    if len(faces) > 6000:
        idx = np.random.RandomState(42).choice(len(faces), 6000, replace=False)
        faces = faces[idx]

    fig = plt.figure(figsize=(resolution/72, resolution/72), dpi=72)
    ax = fig.add_subplot(111, projection='3d')

    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=0.85, facecolor='#87CEEB',
                          edgecolor='#666666', linewidth=0.1)
    ax.add_collection3d(pc)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=angle)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02,
                facecolor='white')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)}')


def refine_with_new_protocol(baseline_path):
    """Run 50-step refinement with multi-start plane on CPU."""
    import torch
    from geo_reward import (symmetry_reward_plane, smoothness_reward, compactness_reward,
                            estimate_symmetry_plane, compute_initial_huber_delta,
                            _build_face_adjacency)

    mesh = trimesh.load(baseline_path, process=False)
    v = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    f = torch.tensor(np.array(mesh.faces), dtype=torch.long)

    sym_n, sym_d = estimate_symmetry_plane(v.detach())
    print(f'  Plane: n=[{sym_n[0]:+.3f},{sym_n[1]:+.3f},{sym_n[2]:+.3f}]')

    weights = torch.tensor([1/3, 1/3, 1/3])
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.005)
    adj = _build_face_adjacency(f)
    hd = compute_initial_huber_delta(v_opt, f)

    with torch.no_grad():
        s0 = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        h0 = smoothness_reward(v_opt, f, delta=hd, _adj=adj).item()
        c0 = compactness_reward(v_opt, f).item()
    ss, hs, cs = max(abs(s0), 1e-6), max(abs(h0), 1e-6), max(abs(c0), 1e-6)

    for step in range(50):
        opt.zero_grad()
        sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
        smo = smoothness_reward(v_opt, f, delta=hd, _adj=adj)
        com = compactness_reward(v_opt, f)
        reward = weights[0]*sym/ss + weights[1]*smo/hs + weights[2]*com/cs
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()

    with torch.no_grad():
        sf = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        hf = smoothness_reward(v_opt, f, delta=hd, _adj=adj).item()
    print(f'  sym: {s0:+.5f} -> {sf:+.5f} ({(sf-s0)/abs(s0)*100:+.1f}%)')
    print(f'  hnc: {h0:+.4f} -> {hf:+.4f} ({(hf-h0)/abs(h0)*100:+.1f}%)')

    refined_path = os.path.join(OUT, 'vase_refined_newproto.obj')
    refined = trimesh.Trimesh(vertices=v_opt.detach().numpy(),
                              faces=mesh.faces.copy(), process=False)
    refined.export(refined_path)
    return refined_path


if __name__ == '__main__':
    print('=== Refining with new protocol ===')
    refined_path = refine_with_new_protocol(BASELINE_OBJ)

    print('\n=== Rendering baseline (3 views) ===')
    for angle, name in zip(ANGLES, ANGLE_NAMES):
        render_single(BASELINE_OBJ, angle, os.path.join(OUT, f'baseline_{name}_study.png'))

    print('\n=== Rendering refined (3 views) ===')
    for angle, name in zip(ANGLES, ANGLE_NAMES):
        render_single(refined_path, angle, os.path.join(OUT, f'refined_{name}_study.png'))

    print('\nDone! All images in:', OUT)
