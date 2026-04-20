"""Generate pipeline figure assets for the DiffGeoReward paper.

Produces:
1. Baseline vase rendered from 2 views (front + 3/4)
2. Refined vase rendered from same views
3. Dihedral angle heatmaps for both
4. Symmetry plane visualization
5. Intermediate steps (step 0, 10, 25, 50) if CPU refinement is feasible

All output to: figures/pipeline_assets/
"""
import os
import sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

BASELINE = os.path.join(ROOT, 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj')
REFINED  = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted/symmetry/a_symmetric_vase_seed42.obj')


def compute_dihedral_angles(mesh):
    """Compute dihedral angles between adjacent faces."""
    face_normals = mesh.face_normals
    face_adjacency = mesh.face_adjacency
    n0 = face_normals[face_adjacency[:, 0]]
    n1 = face_normals[face_adjacency[:, 1]]
    cos_angles = np.clip(np.sum(n0 * n1, axis=1), -1, 1)
    angles = np.arccos(cos_angles)  # in radians
    # Map to per-face: average the dihedral angles of all edges of each face
    face_angles = np.zeros(len(mesh.faces))
    face_counts = np.zeros(len(mesh.faces))
    for i, (f0, f1) in enumerate(face_adjacency):
        face_angles[f0] += angles[i]
        face_counts[f0] += 1
        face_angles[f1] += angles[i]
        face_counts[f1] += 1
    face_counts[face_counts == 0] = 1
    return face_angles / face_counts


def render_mesh_matplotlib(mesh, face_values, title, out_path,
                           elev=25, azim=-60, cmap='coolwarm',
                           vmin=None, vmax=None, figsize=(6, 5)):
    """Render a mesh with per-face coloring using matplotlib."""
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    verts = mesh.vertices
    faces = mesh.faces

    # Subsample faces if too many for matplotlib
    max_faces = 5000
    if len(faces) > max_faces:
        idx = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
        faces_sub = faces[idx]
        values_sub = face_values[idx]
    else:
        faces_sub = faces
        values_sub = face_values

    # Create polygon collection
    polygons = verts[faces_sub]

    if vmin is None:
        vmin = np.percentile(values_sub, 5)
    if vmax is None:
        vmax = np.percentile(values_sub, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cm.get_cmap(cmap)(norm(values_sub))

    poly = Poly3DCollection(polygons, alpha=0.95)
    poly.set_facecolor(colors)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    # Set axis limits
    center = verts.mean(axis=0)
    scale = max(verts.max(axis=0) - verts.min(axis=0)) * 0.6
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold', pad=0)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', transparent=False)
    plt.close()
    print(f'  Saved: {out_path}')


def render_mesh_solid(mesh, title, out_path, elev=25, azim=-60,
                      color='#6CA6CD', figsize=(6, 5)):
    """Render a mesh with solid shading (Phong-like via face normals)."""
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    verts = mesh.vertices
    faces = mesh.faces

    max_faces = 5000
    if len(faces) > max_faces:
        idx = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
        faces_sub = faces[idx]
    else:
        faces_sub = faces

    polygons = verts[faces_sub]

    # Compute simple lighting
    light_dir = np.array([0.5, 0.8, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    face_normals = mesh.face_normals[idx if len(faces) > max_faces else slice(None)]
    intensity = np.clip(np.dot(face_normals, light_dir), 0.15, 1.0)

    base = np.array(matplotlib.colors.to_rgb(color))
    colors = np.outer(intensity, base)
    colors = np.clip(colors, 0, 1)
    colors_rgba = np.column_stack([colors, np.ones(len(colors))])

    poly = Poly3DCollection(polygons, alpha=1.0)
    poly.set_facecolor(colors_rgba)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    center = verts.mean(axis=0)
    scale = max(verts.max(axis=0) - verts.min(axis=0)) * 0.6
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold', pad=0)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', transparent=False)
    plt.close()
    print(f'  Saved: {out_path}')


def render_symmetry_plane(mesh, out_path, elev=25, azim=-60):
    """Render mesh with symmetry plane overlay."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import torch
    from geo_reward import estimate_symmetry_plane

    v = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    n, d = estimate_symmetry_plane(v)
    n_np = n.numpy()
    d_val = d.item()

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    verts = mesh.vertices
    faces = mesh.faces
    max_faces = 3000
    if len(faces) > max_faces:
        idx = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
        faces_sub = faces[idx]
    else:
        faces_sub = faces

    polygons = verts[faces_sub]
    poly = Poly3DCollection(polygons, alpha=0.4)
    poly.set_facecolor('#B0C4DE')
    poly.set_edgecolor('#888888')
    poly.set_linewidth(0.1)
    ax.add_collection3d(poly)

    # Draw symmetry plane
    center = verts.mean(axis=0)
    scale = max(verts.max(axis=0) - verts.min(axis=0)) * 0.7

    # Create plane vertices
    if abs(n_np[2]) < 0.9:
        u = np.cross(n_np, [0, 0, 1])
    else:
        u = np.cross(n_np, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v_dir = np.cross(n_np, u)

    plane_center = center + n_np * d_val
    corners = []
    for su, sv in [(-1,-1), (1,-1), (1,1), (-1,1)]:
        corners.append(plane_center + su * scale * u + sv * scale * v_dir)
    corners = np.array(corners)

    plane_poly = Poly3DCollection([corners], alpha=0.25)
    plane_poly.set_facecolor('#FF6B6B')
    plane_poly.set_edgecolor('#CC0000')
    plane_poly.set_linewidth(1.5)
    ax.add_collection3d(plane_poly)

    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title('Symmetry Plane\nEstimation', fontsize=13, fontweight='bold', pad=0)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', transparent=False)
    plt.close()
    print(f'  Saved: {out_path}')
    return n_np, d_val


def run_refinement_snapshots(mesh_path, sym_n, sym_d, out_dir, steps=[0, 10, 25, 50]):
    """Run CPU refinement and save mesh snapshots at specified steps."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import torch
    from geo_reward import (symmetry_reward_plane, smoothness_reward,
                           compactness_reward, compute_initial_huber_delta,
                           _build_face_adjacency)

    mesh = trimesh.load(mesh_path, process=False)
    v = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    f = torch.tensor(np.array(mesh.faces), dtype=torch.long)
    sym_n_t = torch.tensor(sym_n, dtype=torch.float32)
    sym_d_t = torch.tensor(sym_d, dtype=torch.float32)

    weights = torch.tensor([1/3, 1/3, 1/3])
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.005)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)

    with torch.no_grad():
        sym_init = symmetry_reward_plane(v_opt, sym_n_t, sym_d_t).item()
        smo_init = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item()
        com_init = compactness_reward(v_opt, f).item()
    sym_scale = max(abs(sym_init), 1e-6)
    smo_scale = max(abs(smo_init), 1e-6)
    com_scale = max(abs(com_init), 1e-6)

    max_step = max(steps)
    snapshots = {}

    for step in range(max_step + 1):
        if step in steps:
            with torch.no_grad():
                sym_val = symmetry_reward_plane(v_opt, sym_n_t, sym_d_t).item()
                smo_val = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item()
                com_val = compactness_reward(v_opt, f).item()
            snapshot_mesh = trimesh.Trimesh(
                vertices=v_opt.detach().numpy(),
                faces=mesh.faces.copy(),
                process=False
            )
            snapshots[step] = (snapshot_mesh, sym_val, smo_val, com_val)
            print(f'  Step {step:3d}: sym={sym_val:+.5f} hnc={smo_val:+.4f} com={com_val:+.1f}')

        if step < max_step:
            opt.zero_grad()
            sym = symmetry_reward_plane(v_opt, sym_n_t, sym_d_t)
            smo = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj)
            com = compactness_reward(v_opt, f)
            reward = (weights[0] * sym / sym_scale +
                      weights[1] * smo / smo_scale +
                      weights[2] * com / com_scale)
            (-reward).backward()
            torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
            opt.step()

    return snapshots


def main():
    print('Loading meshes...')
    baseline = trimesh.load(BASELINE, process=False)
    refined  = trimesh.load(REFINED, process=False)
    print(f'  Baseline: {len(baseline.vertices)}v, {len(baseline.faces)}f')
    print(f'  Refined:  {len(refined.vertices)}v, {len(refined.faces)}f')

    # Compute dihedral angles
    print('\nComputing dihedral angles...')
    bl_angles = compute_dihedral_angles(baseline)
    rf_angles = compute_dihedral_angles(refined)
    vmin = min(np.percentile(bl_angles, 5), np.percentile(rf_angles, 5))
    vmax = max(np.percentile(bl_angles, 95), np.percentile(rf_angles, 95))
    print(f'  Baseline: mean={np.degrees(bl_angles.mean()):.1f}°, std={np.degrees(bl_angles.std()):.1f}°')
    print(f'  Refined:  mean={np.degrees(rf_angles.mean()):.1f}°, std={np.degrees(rf_angles.std()):.1f}°')

    # Render views
    for azim, suffix in [(-60, 'front'), (-150, 'back')]:
        print(f'\nRendering {suffix} view...')
        # Solid renders
        render_mesh_solid(baseline, 'Baseline',
                         os.path.join(OUT, f'vase_baseline_solid_{suffix}.png'),
                         azim=azim, color='#8FB8DE')
        render_mesh_solid(refined, 'Refined (+DGR)',
                         os.path.join(OUT, f'vase_refined_solid_{suffix}.png'),
                         azim=azim, color='#7BC67E')

        # Dihedral heatmaps
        render_mesh_matplotlib(baseline, bl_angles, 'Baseline\n(dihedral angles)',
                              os.path.join(OUT, f'vase_baseline_heatmap_{suffix}.png'),
                              azim=azim, vmin=vmin, vmax=vmax)
        render_mesh_matplotlib(refined, rf_angles, 'Refined\n(dihedral angles)',
                              os.path.join(OUT, f'vase_refined_heatmap_{suffix}.png'),
                              azim=azim, vmin=vmin, vmax=vmax)

    # Symmetry plane
    print('\nRendering symmetry plane...')
    sym_n, sym_d = render_symmetry_plane(baseline,
                                          os.path.join(OUT, 'vase_symmetry_plane.png'))

    # Run refinement with snapshots
    print('\nRunning CPU refinement (50 steps)...')
    snapshots = run_refinement_snapshots(BASELINE, sym_n, sym_d, OUT)

    for step, (snap_mesh, sv, hv, cv) in snapshots.items():
        angles = compute_dihedral_angles(snap_mesh)
        render_mesh_solid(snap_mesh, f'Step {step}',
                         os.path.join(OUT, f'vase_step{step:03d}_solid.png'),
                         azim=-60, color='#8FB8DE' if step == 0 else '#A8D8A8')
        render_mesh_matplotlib(snap_mesh, angles, f'Step {step}',
                              os.path.join(OUT, f'vase_step{step:03d}_heatmap.png'),
                              azim=-60, vmin=vmin, vmax=vmax)

    # Summary
    print('\n=== Pipeline assets generated ===')
    for f_name in sorted(os.listdir(OUT)):
        if f_name.endswith('.png'):
            size = os.path.getsize(os.path.join(OUT, f_name))
            print(f'  {f_name:45s} {size:>10,} bytes')


if __name__ == '__main__':
    main()
