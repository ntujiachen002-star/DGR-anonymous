"""Clean mesh rendering using trimesh scene with proper camera angles."""
import os
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

MESHES = {
    'baseline': 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj',
    'refined':  'results/mesh_validity_objs/handcrafted/symmetry/a_symmetric_vase_seed42.obj',
}


def render_mesh(mesh_path, out_path, angle_deg=30):
    """Render a mesh from a 3/4 view with good lighting."""
    mesh = trimesh.load(mesh_path, process=True)

    # Center and normalize
    mesh.vertices -= mesh.centroid
    scale = mesh.bounding_box.extents.max()
    mesh.vertices /= scale

    # Create scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    # Set camera: rotate around Y axis
    angle_rad = np.radians(angle_deg)
    dist = 2.5
    camera_pos = np.array([
        dist * np.sin(angle_rad),
        dist * 0.3,  # slightly above
        dist * np.cos(angle_rad)
    ])

    # Camera transform: look at origin
    forward = -camera_pos / np.linalg.norm(camera_pos)
    right = np.cross(forward, [0, 1, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    cam_tf = np.eye(4)
    cam_tf[:3, 0] = right
    cam_tf[:3, 1] = up
    cam_tf[:3, 2] = -forward
    cam_tf[:3, 3] = camera_pos

    scene.camera_transform = cam_tf

    png = scene.save_image(resolution=[800, 800])
    with open(out_path, 'wb') as f:
        f.write(png)
    print(f'Saved: {out_path} ({len(png):,} bytes)')


def render_with_heatmap(mesh_path, out_path, angle_deg=30):
    """Render mesh with dihedral-angle face colors."""
    mesh = trimesh.load(mesh_path, process=True)

    # Center and normalize
    mesh.vertices -= mesh.centroid
    scale = mesh.bounding_box.extents.max()
    mesh.vertices /= scale

    # Compute dihedral angles per face
    face_normals = mesh.face_normals
    adj = mesh.face_adjacency
    n0 = face_normals[adj[:, 0]]
    n1 = face_normals[adj[:, 1]]
    cos_a = np.clip(np.sum(n0 * n1, axis=1), -1, 1)
    angles = np.arccos(cos_a)

    face_angle = np.zeros(len(mesh.faces))
    face_count = np.zeros(len(mesh.faces))
    for i, (f0, f1) in enumerate(adj):
        face_angle[f0] += angles[i]
        face_count[f0] += 1
        face_angle[f1] += angles[i]
        face_count[f1] += 1
    face_count[face_count == 0] = 1
    face_angle /= face_count

    # Color: blue (smooth, low angle) -> red (rough, high angle)
    vmin, vmax = np.percentile(face_angle, [5, 95])
    norm = np.clip((face_angle - vmin) / (vmax - vmin + 1e-10), 0, 1)

    # Blue to red colormap
    colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    colors[:, 0] = (norm * 255).astype(np.uint8)      # R
    colors[:, 1] = ((1 - 2*np.abs(norm - 0.5)) * 80).astype(np.uint8)  # G
    colors[:, 2] = ((1 - norm) * 255).astype(np.uint8) # B
    colors[:, 3] = 255

    mesh.visual = trimesh.visual.ColorVisual(
        face_colors=colors
    )

    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    angle_rad = np.radians(angle_deg)
    dist = 2.5
    camera_pos = np.array([
        dist * np.sin(angle_rad),
        dist * 0.3,
        dist * np.cos(angle_rad)
    ])
    forward = -camera_pos / np.linalg.norm(camera_pos)
    right = np.cross(forward, [0, 1, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    cam_tf = np.eye(4)
    cam_tf[:3, 0] = right
    cam_tf[:3, 1] = up
    cam_tf[:3, 2] = -forward
    cam_tf[:3, 3] = camera_pos
    scene.camera_transform = cam_tf

    png = scene.save_image(resolution=[800, 800])
    with open(out_path, 'wb') as f:
        f.write(png)
    print(f'Saved: {out_path} ({len(png):,} bytes)')


def main():
    for name, rel_path in MESHES.items():
        full = os.path.join(ROOT, rel_path)
        print(f'\n=== {name} ===')

        # Solid render from two angles
        for angle, suffix in [(30, 'front'), (150, 'side')]:
            render_mesh(full, os.path.join(OUT, f'vase_{name}_{suffix}.png'), angle)

        # Heatmap render
        render_with_heatmap(full, os.path.join(OUT, f'vase_{name}_heatmap.png'), 30)

    # Also run quick CPU refinement and render intermediate steps
    print('\n=== CPU refinement snapshots ===')
    import sys, torch
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from geo_reward import (symmetry_reward_plane, smoothness_reward,
                           compactness_reward, estimate_symmetry_plane,
                           compute_initial_huber_delta, _build_face_adjacency)

    mesh = trimesh.load(os.path.join(ROOT, MESHES['baseline']), process=False)
    v = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    f = torch.tensor(np.array(mesh.faces), dtype=torch.long)
    sym_n, sym_d = estimate_symmetry_plane(v.detach())

    weights = torch.tensor([1/3, 1/3, 1/3])
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.005)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)

    with torch.no_grad():
        s0 = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        h0 = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item()
        c0 = compactness_reward(v_opt, f).item()
    ss, hs, cs = max(abs(s0), 1e-6), max(abs(h0), 1e-6), max(abs(c0), 1e-6)

    snap_steps = [0, 10, 25, 50]
    for step in range(51):
        if step in snap_steps:
            snap = trimesh.Trimesh(vertices=v_opt.detach().numpy(),
                                  faces=mesh.faces.copy(), process=False)
            # Save OBJ for potential use in external renderer
            snap.export(os.path.join(OUT, f'vase_step{step:03d}.obj'))

            # Render
            snap_path = os.path.join(OUT, f'vase_step{step:03d}.obj')
            render_mesh(snap_path, os.path.join(OUT, f'vase_step{step:03d}.png'), 30)
            render_with_heatmap(snap_path, os.path.join(OUT, f'vase_step{step:03d}_heat.png'), 30)

            with torch.no_grad():
                sv = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
                hv = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item()
            print(f'  Step {step}: sym={sv:+.5f} hnc={hv:+.4f}')

        if step < 50:
            opt.zero_grad()
            sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
            smo = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj)
            com = compactness_reward(v_opt, f)
            reward = weights[0]*sym/ss + weights[1]*smo/hs + weights[2]*com/cs
            (-reward).backward()
            torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
            opt.step()

    print('\nDone! All assets in:', OUT)


if __name__ == '__main__':
    main()
