"""Refine vase with LATEST protocol (multi-start plane) on CPU, then render 4 views."""
import os, sys, numpy as np, trimesh, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
from geo_reward import (symmetry_reward_plane, smoothness_reward, compactness_reward,
                        estimate_symmetry_plane, compute_initial_huber_delta,
                        _build_face_adjacency)
import pyrender
from PIL import Image

OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

BASELINE = os.path.join(ROOT, 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj')

ANGLES = [(20, 0, 'front'), (20, 90, 'right'), (20, 180, 'back'), (20, 270, 'left')]


def refine_mesh(mesh_path, steps=50, lr=0.005):
    """Run refinement with latest multi-start plane protocol."""
    mesh = trimesh.load(mesh_path, process=False)
    v = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    f = torch.tensor(np.array(mesh.faces), dtype=torch.long)

    # NEW PROTOCOL: multi-start plane estimation
    print('  Estimating symmetry plane (multi-start)...')
    sym_n, sym_d = estimate_symmetry_plane(v.detach())
    print(f'  Plane: n=[{sym_n[0]:+.3f},{sym_n[1]:+.3f},{sym_n[2]:+.3f}], d={sym_d.item():+.4f}')

    weights = torch.tensor([1/3, 1/3, 1/3])
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)

    with torch.no_grad():
        s0 = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        h0 = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item()
        c0 = compactness_reward(v_opt, f).item()
    ss, hs, cs = max(abs(s0), 1e-6), max(abs(h0), 1e-6), max(abs(c0), 1e-6)

    print(f'  Before: sym={s0:+.5f} hnc={h0:+.4f} com={c0:+.1f}')

    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
        smo = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj)
        com = compactness_reward(v_opt, f)
        reward = weights[0]*sym/ss + weights[1]*smo/hs + weights[2]*com/cs
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()

    with torch.no_grad():
        sf = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        hf = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item()
        cf = compactness_reward(v_opt, f).item()
    elapsed = time.time() - t0
    print(f'  After:  sym={sf:+.5f} hnc={hf:+.4f} com={cf:+.1f}  ({elapsed:.1f}s)')
    print(f'  Improvement: sym {(sf-s0)/abs(s0)*100:+.1f}%, hnc {(hf-h0)/abs(h0)*100:+.1f}%, com {(cf-c0)/abs(c0)*100:+.1f}%')

    refined = trimesh.Trimesh(vertices=v_opt.detach().numpy(), faces=mesh.faces.copy(), process=False)
    return refined


def render(tm, out_path, elev=20, azim=0, dist=2.2):
    tm_copy = tm.copy()
    tm_copy.vertices -= tm_copy.centroid
    tm_copy.vertices /= tm_copy.bounding_box.extents.max()

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.73, 0.78, 1.0],
        metallicFactor=0.1, roughnessFactor=0.6)
    mesh = pyrender.Mesh.from_trimesh(tm_copy, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=[1,1,1,1], ambient_light=[0.15,0.15,0.15])
    scene.add(mesh)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 6)
    er, ar = np.radians(elev), np.radians(azim)
    eye = np.array([dist*np.cos(er)*np.sin(ar), dist*np.sin(er), dist*np.cos(er)*np.cos(ar)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0,1,0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3,0], pose[:3,1], pose[:3,2], pose[:3,3] = right, up, -fwd, eye
    scene.add(cam, pose=pose)

    for d, c, i in [([1,-1,-0.5],[1,0.98,0.95],3.0),
                     ([-1,-0.5,-1],[0.9,0.95,1.0],1.5),
                     ([0,-0.3,1],[1,1,1],2.0)]:
        light = pyrender.DirectionalLight(color=c, intensity=i)
        ld = np.array(d, dtype=float); ld /= np.linalg.norm(ld)
        lr = np.cross(ld,[0,1,0]); lr /= np.linalg.norm(lr)
        lu = np.cross(lr, ld)
        lp = np.eye(4); lp[:3,0],lp[:3,1],lp[:3,2] = lr,lu,-ld
        scene.add(light, pose=lp)

    r = pyrender.OffscreenRenderer(1024, 1024)
    color, _ = r.render(scene)
    r.delete()
    Image.fromarray(color).save(out_path)


if __name__ == '__main__':
    print('=== Loading baseline ===')
    baseline = trimesh.load(BASELINE, process=False)
    print(f'  {baseline.vertices.shape[0]}v, {baseline.faces.shape[0]}f')

    print('\n=== Refining with NEW protocol ===')
    refined = refine_mesh(BASELINE)

    # Save refined OBJ
    refined_obj = os.path.join(OUT, 'vase_refined_newproto.obj')
    refined.export(refined_obj)
    print(f'\n  Saved refined OBJ: {refined_obj}')

    print('\n=== Rendering 4 views ===')
    for elev, azim, name in ANGLES:
        render(baseline, os.path.join(OUT, f'baseline_{name}.png'), elev, azim)
        render(refined,  os.path.join(OUT, f'refined_{name}.png'),  elev, azim)
        print(f'  {name}: done')

    print('\nAll done! Files in:', OUT)
