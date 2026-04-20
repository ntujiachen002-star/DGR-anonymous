"""Render baseline vs refined — clean, same angle, same gray, white bg.
Renders multiple candidates so user can pick the best one for pipeline figure."""
import os, numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

CANDIDATES = [
    ('vase_s42',      'symmetry/a_symmetric_vase_seed42.obj'),
    ('vase_s123',     'symmetry/a_symmetric_vase_seed123.obj'),
    ('trophy_s42',    'symmetry/a_symmetric_trophy_seed42.obj'),
    ('temple_s42',    'symmetry/a_symmetrical_temple_seed42.obj'),
    ('butterfly_s42', 'symmetry/a_symmetric_butterfly_sculpture_seed42.obj'),
    ('chess_s42',     'symmetry/a_balanced_chess_piece_a_king_seed42.obj'),
]


def render(mesh_path, out_path, angle_deg=35):
    mesh = trimesh.load(mesh_path, process=True)
    mesh.vertices -= mesh.centroid
    mesh.vertices /= mesh.bounding_box.extents.max()

    # Uniform light gray
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile([180, 190, 200, 255], (len(mesh.faces), 1)))

    scene = trimesh.Scene(mesh)
    a = np.radians(angle_deg)
    d = 2.8
    pos = np.array([d*np.sin(a), d*0.3, d*np.cos(a)])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0,1,0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tf = np.eye(4)
    tf[:3,0], tf[:3,1], tf[:3,2], tf[:3,3] = right, up, -fwd, pos
    scene.camera_transform = tf

    png = scene.save_image(resolution=[1024, 1024])
    with open(out_path, 'wb') as f:
        f.write(png)
    print(f'  {out_path.split(os.sep)[-1]:40s} {len(png):>8,} bytes')


for name, rel in CANDIDATES:
    base = os.path.join(ROOT, 'results/mesh_validity_objs/baseline', rel)
    ref  = os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted', rel)
    if not os.path.exists(base) or not os.path.exists(ref):
        print(f'SKIP {name} (missing)')
        continue
    print(f'{name}:')
    render(base, os.path.join(OUT, f'{name}_baseline.png'))
    render(ref,  os.path.join(OUT, f'{name}_refined.png'))

print('\nDone! Pick the best pair for the pipeline figure.')
