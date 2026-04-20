"""Render baseline vs refined with pyrender — much better than trimesh's built-in."""
import os

import numpy as np
import trimesh
import pyrender
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

MESHES = {
    'coarse_mesh': 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj',
    'refined_mesh': 'results/mesh_validity_objs/handcrafted/symmetry/a_symmetric_vase_seed42.obj',
}


def render(mesh_path, out_path, elev=20, azim=35, dist=2.2):
    """Render mesh with pyrender: proper lighting, smooth shading, white bg."""
    # Load and prepare mesh
    tm = trimesh.load(mesh_path, process=True)
    tm.vertices -= tm.centroid
    tm.vertices /= tm.bounding_box.extents.max()

    # Create pyrender mesh with smooth shading
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.73, 0.78, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.6,
    )

    # Enable smooth normals
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)

    # Scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=[0.15, 0.15, 0.15])
    scene.add(mesh)

    # Camera
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 6)
    elev_r, azim_r = np.radians(elev), np.radians(azim)
    cx = dist * np.cos(elev_r) * np.sin(azim_r)
    cy = dist * np.sin(elev_r)
    cz = dist * np.cos(elev_r) * np.cos(azim_r)

    # Look-at matrix
    eye = np.array([cx, cy, cz])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = true_up
    cam_pose[:3, 2] = -fwd
    cam_pose[:3, 3] = eye
    scene.add(cam, pose=cam_pose)

    # Lights — 3 point
    # Key light
    key = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=3.0)
    key_pose = np.eye(4)
    key_dir = np.array([1, -1, -0.5])
    key_dir /= np.linalg.norm(key_dir)
    key_right = np.cross(key_dir, [0,1,0])
    key_right /= np.linalg.norm(key_right)
    key_up = np.cross(key_right, key_dir)
    key_pose[:3, 0] = key_right
    key_pose[:3, 1] = key_up
    key_pose[:3, 2] = -key_dir
    scene.add(key, pose=key_pose)

    # Fill light
    fill = pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=1.5)
    fill_pose = np.eye(4)
    fill_dir = np.array([-1, -0.5, -1])
    fill_dir /= np.linalg.norm(fill_dir)
    fill_right = np.cross(fill_dir, [0,1,0])
    fill_right /= np.linalg.norm(fill_right)
    fill_up = np.cross(fill_right, fill_dir)
    fill_pose[:3, 0] = fill_right
    fill_pose[:3, 1] = fill_up
    fill_pose[:3, 2] = -fill_dir
    scene.add(fill, pose=fill_pose)

    # Rim light
    rim = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    rim_pose = np.eye(4)
    rim_dir = np.array([0, -0.3, 1])
    rim_dir /= np.linalg.norm(rim_dir)
    rim_right = np.cross(rim_dir, [0,1,0])
    rim_right /= np.linalg.norm(rim_right)
    rim_up = np.cross(rim_right, rim_dir)
    rim_pose[:3, 0] = rim_right
    rim_pose[:3, 1] = rim_up
    rim_pose[:3, 2] = -rim_dir
    scene.add(rim, pose=rim_pose)

    # Render
    r = pyrender.OffscreenRenderer(1024, 1024)
    color, _ = r.render(scene)
    r.delete()

    img = Image.fromarray(color)
    img.save(out_path)
    print(f'Saved: {out_path} ({os.path.getsize(out_path):,} bytes)')


if __name__ == '__main__':
    for name, rel in MESHES.items():
        path = os.path.join(ROOT, rel)
        render(path, os.path.join(OUT, f'{name}_pyrender.png'))
    print('Done!')
