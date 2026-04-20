"""Render baseline + refined vase from 4 angles, no color, same gray material."""
import os
import numpy as np
import trimesh
import pyrender
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

BASELINE = os.path.join(ROOT, 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj')
REFINED  = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted/symmetry/a_symmetric_vase_seed42.obj')

ANGLES = [
    (20, 0,   'front'),
    (20, 90,  'right'),
    (20, 180, 'back'),
    (20, 270, 'left'),
]


def render(mesh_path, out_path, elev=20, azim=0, dist=2.2):
    tm = trimesh.load(mesh_path, process=True)
    tm.vertices -= tm.centroid
    tm.vertices /= tm.bounding_box.extents.max()

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.73, 0.78, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.6,
    )
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=[0.15, 0.15, 0.15])
    scene.add(mesh)

    # Camera
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 6)
    er, ar = np.radians(elev), np.radians(azim)
    eye = np.array([dist*np.cos(er)*np.sin(ar), dist*np.sin(er), dist*np.cos(er)*np.cos(ar)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0,1,0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3,0], pose[:3,1], pose[:3,2], pose[:3,3] = right, up, -fwd, eye
    scene.add(cam, pose=pose)

    # Lights
    for d, c, i in [([1,-1,-0.5], [1,0.98,0.95], 3.0),
                     ([-1,-0.5,-1], [0.9,0.95,1.0], 1.5),
                     ([0,-0.3,1], [1,1,1], 2.0)]:
        light = pyrender.DirectionalLight(color=c, intensity=i)
        ld = np.array(d, dtype=float); ld /= np.linalg.norm(ld)
        lr = np.cross(ld, [0,1,0]); lr /= np.linalg.norm(lr)
        lu = np.cross(lr, ld)
        lp = np.eye(4); lp[:3,0], lp[:3,1], lp[:3,2] = lr, lu, -ld
        scene.add(light, pose=lp)

    r = pyrender.OffscreenRenderer(1024, 1024)
    color, _ = r.render(scene)
    r.delete()
    Image.fromarray(color).save(out_path)
    print(f'  {os.path.basename(out_path)}')


print('Rendering 4 views x 2 meshes...')
for elev, azim, name in ANGLES:
    render(BASELINE, os.path.join(OUT, f'baseline_{name}.png'), elev, azim)
    render(REFINED,  os.path.join(OUT, f'refined_{name}.png'),  elev, azim)

print('\nDone! Files in:', OUT)
