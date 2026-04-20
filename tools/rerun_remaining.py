"""Rerun remaining local experiments with new Huber NC metric."""
import sys, os, glob, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from geo_reward import (symmetry_reward, smoothness_reward, smoothness_reward_legacy,
                        compactness_reward, compute_initial_huber_delta, _build_face_adjacency)


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


mesh_dir = 'results/mesh_validity_objs/baseline'

# Load test meshes
meshes = []
for cat in ['symmetry', 'smoothness', 'compactness']:
    for mp in sorted(glob.glob(f'{mesh_dir}/{cat}/*_seed42.obj'))[:10]:
        v, f = load_obj(mp)
        if v.shape[0] >= 20 and f.shape[0] >= 20:
            meshes.append((mp, cat, v, f))
print(f"Loaded {len(meshes)} meshes")

# ============================================================
# 1. Metric Validity
# ============================================================
print("\n=== Metric Validity ===")
all_smo, all_nc = [], []
for cat in ['symmetry', 'smoothness', 'compactness']:
    for seed in [42, 123, 456]:
        for mp in sorted(glob.glob(f'{mesh_dir}/{cat}/*_seed{seed}.obj')):
            v, f = load_obj(mp)
            if v.shape[0] < 20 or f.dim() != 2 or f.shape[1] != 3:
                continue
            try:
                adj = _build_face_adjacency(f)
            except Exception:
                continue
            smo = smoothness_reward(v, f, _adj=adj).item()
            fn = torch.nn.functional.normalize(
                torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1),
                dim=1, eps=1e-8)
            idx_i, idx_j = adj
            if idx_i.shape[0] > 0:
                nc = (fn[idx_i] * fn[idx_j]).sum(dim=1).mean().item()
                all_smo.append(smo)
                all_nc.append(nc)

r_new, p_new = stats.pearsonr(all_smo, all_nc)
print(f"  Huber NC vs Normal Consistency: r={r_new:.4f}, p={p_new:.2e}, n={len(all_smo)}")

# Old metric comparison
all_old, all_nc2 = [], []
for mp in sorted(glob.glob(f'{mesh_dir}/*/*_seed42.obj'))[:60]:
    v, f = load_obj(mp)
    if v.shape[0] < 20 or f.dim() != 2 or f.shape[1] != 3:
        continue
    smo_old = smoothness_reward_legacy(v, f).item()
    try:
        adj = _build_face_adjacency(f)
    except Exception:
        continue
    fn = torch.nn.functional.normalize(
        torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1),
        dim=1, eps=1e-8)
    idx_i, idx_j = adj
    if idx_i.shape[0] > 0:
        nc = (fn[idx_i] * fn[idx_j]).sum(dim=1).mean().item()
        all_old.append(smo_old)
        all_nc2.append(nc)

r_old, p_old = stats.pearsonr(all_old, all_nc2)
print(f"  Old Var(k) vs NC: r={r_old:.4f}, p={p_old:.2e}")
print(f"  Correlation improvement: {r_old:.4f} -> {r_new:.4f}")

# ============================================================
# 2. Scale-Controlled Ablation
# ============================================================
print("\n=== Scale-Controlled Ablation ===")
for config, grad_norm in [('standard', False), ('gradnorm', True)]:
    for rname, w in [('sym_only', [1, 0, 0]), ('smo_only', [0, 1, 0]), ('com_only', [0, 0, 1])]:
        pcts = {'sym': [], 'smo': [], 'com': []}
        for mp, cat, v, f in meshes:
            adj = _build_face_adjacency(f)
            delta = compute_initial_huber_delta(v, f)
            mb = {'sym': symmetry_reward(v).item(),
                  'smo': smoothness_reward(v, f, delta=delta, _adj=adj).item(),
                  'com': compactness_reward(v, f).item()}
            v_opt = v.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([v_opt], lr=0.005)
            r0 = [mb['sym'], mb['smo'], mb['com']]
            eps = 1e-8
            for _ in range(50):
                opt.zero_grad()
                reward = torch.tensor(0.0)
                if w[0] > 0:
                    reward = reward + w[0] * symmetry_reward(v_opt) / (abs(r0[0]) + eps)
                if w[1] > 0:
                    reward = reward + w[1] * smoothness_reward(v_opt, f, delta=delta, _adj=adj) / (abs(r0[1]) + eps)
                if w[2] > 0:
                    reward = reward + w[2] * compactness_reward(v_opt, f) / (abs(r0[2]) + eps)
                (-reward).backward()
                if grad_norm and v_opt.grad.norm() > 1e-8:
                    v_opt.grad.data = v_opt.grad.data * (0.01 / v_opt.grad.norm())
                torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
                opt.step()
            mr = {'sym': symmetry_reward(v_opt.detach()).item(),
                  'smo': smoothness_reward(v_opt.detach(), f, delta=delta, _adj=adj).item(),
                  'com': compactness_reward(v_opt.detach(), f).item()}
            for m in ['sym', 'smo', 'com']:
                pcts[m].append((mr[m] - mb[m]) / (abs(mb[m]) + 1e-10) * 100)
        print(f"  {config:10s} {rname:10s}: sym={np.mean(pcts['sym']):+.1f}% smo={np.mean(pcts['smo']):+.1f}% com={np.mean(pcts['com']):+.1f}%")

# ============================================================
# 3. PCGrad Variants
# ============================================================
print("\n=== PCGrad Variants ===")
for variant in ['standard', 'dn']:
    pcts = {'sym': [], 'smo': [], 'com': []}
    for mp, cat, v, f in meshes[:15]:
        adj = _build_face_adjacency(f)
        delta = compute_initial_huber_delta(v, f)
        mb = {'sym': symmetry_reward(v).item(),
              'smo': smoothness_reward(v, f, delta=delta, _adj=adj).item(),
              'com': compactness_reward(v, f).item()}
        v_opt = v.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([v_opt], lr=0.005)
        r0 = [mb['sym'], mb['smo'], mb['com']]
        eps = 1e-8
        w = [1 / 3, 1 / 3, 1 / 3]
        for _ in range(50):
            opt.zero_grad()
            gs = torch.autograd.grad(symmetry_reward(v_opt) / (abs(r0[0]) + eps), v_opt, retain_graph=True)[0]
            gm = torch.autograd.grad(smoothness_reward(v_opt, f, delta=delta, _adj=adj) / (abs(r0[1]) + eps), v_opt, retain_graph=True)[0]
            gc = torch.autograd.grad(compactness_reward(v_opt, f) / (abs(r0[2]) + eps), v_opt, retain_graph=False)[0]
            grads = [gs, gm, gc]
            mod = [g.clone() for g in grads]
            for i in range(3):
                gi = mod[i].reshape(-1)
                for j in range(3):
                    if i == j:
                        continue
                    gj = grads[j].reshape(-1)
                    dot = gi @ gj
                    if dot < 0:
                        if variant == 'standard':
                            gi = gi - dot / (gj @ gj + 1e-12) * gj
                        elif variant == 'dn':
                            gj_hat = gj / (gj.norm() + 1e-12)
                            gi = gi - (gi @ gj_hat) * gj_hat
                mod[i] = gi.reshape(grads[i].shape)
            v_opt.grad = -sum(w[i] * mod[i] for i in range(3))
            torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
            opt.step()
        mr = {'sym': symmetry_reward(v_opt.detach()).item(),
              'smo': smoothness_reward(v_opt.detach(), f, delta=delta, _adj=adj).item(),
              'com': compactness_reward(v_opt.detach(), f).item()}
        for m in ['sym', 'smo', 'com']:
            pcts[m].append((mr[m] - mb[m]) / (abs(mb[m]) + 1e-10) * 100)
    print(f"  {variant:10s}: sym={np.mean(pcts['sym']):+.1f}%  smo={np.mean(pcts['smo']):+.1f}%  com={np.mean(pcts['com']):+.1f}%")

print("\n=== ALL DONE ===")
