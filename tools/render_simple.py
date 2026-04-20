"""Render baseline vs refined vase — clean, same angle, same color, white background."""
import os
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

BASELINE = os.path.join(ROOT, 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj')
REFINED  = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted/symmetry/a_symmetric_vase_seed42.obj')


def render(mesh_path, out_path, angle_deg=30):
    mesh = trimesh.load(mesh_path, process=True)

    # Center + normalize to unit box
    mesh.vertices -= mesh.centroid
    mesh.vertices /= mesh.bounding_box.extents.max()

    # Uniform light gray color — no heatmap, no vertex colors
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.tile([180, 190, 200, 255], (len(mesh.faces), 1))
    )

    scene = trimesh.Scene(mesh)

    # Camera: 3/4 front view
    angle = np.radians(angle_deg)
    dist = 2.8
    pos = np.array([dist * np.sin(angle), dist * 0.25, dist * np.cos(angle)])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    tf = np.eye(4)
    tf[:3, 0] = right
    tf[:3, 1] = up
    tf[:3, 2] = -fwd
    tf[:3, 3] = pos
    scene.camera_transform = tf

    png = scene.save_image(resolution=[1024, 1024])
    with open(out_path, 'wb') as f:
        f.write(png)
    print(f'Saved {out_path} ({len(png):,} bytes)')


if __name__ == '__main__':
    render(BASELINE, os.path.join(OUT, 'coarse_mesh.png'), angle_deg=30)
    render(REFINED,  os.path.join(OUT, 'refined_mesh.png'), angle_deg=30)
    print('Done')
