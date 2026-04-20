"""
Rerun core experiments locally on existing OBJ files.
No GPU needed — only uses CPU for mesh refinement.

Runs: Three-reward, Two-reward, PCGrad, Taubin/Laplacian baselines
on the existing 330 baseline OBJ meshes.
"""

import sys, os, glob, json, time
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from geo_reward import (symmetry_reward, smoothness_reward, smoothness_reward_legacy,
                        compactness_reward, compute_initial_huber_delta,
                        _build_face_adjacency)


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def metrics(v, f, adj=None):
    return {
        'sym': symmetry_reward(v).item(),
        'smo_new': smoothness_reward(v, f, _adj=adj).item(),
        'smo_old': smoothness_reward_legacy(v, f).item(),
        'com': compactness_reward(v, f).item(),
    }


def refine(v, f, w, steps=50, lr=0.005):
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)
    r0 = [symmetry_reward(v_opt).item(),
          smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item(),
          compactness_reward(v_opt, f).item()]
    eps = 1e-8
    for _ in range(steps):
        opt.zero_grad()
        reward = torch.tensor(0.0)
        if w[0] > 0:
            reward = reward + w[0] * symmetry_reward(v_opt) / (abs(r0[0]) + eps)
        if w[1] > 0:
            reward = reward + w[1] * smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj) / (abs(r0[1]) + eps)
        if w[2] > 0:
            reward = reward + w[2] * compactness_reward(v_opt, f) / (abs(r0[2]) + eps)
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


def pcgrad_refine(v, f, steps=50, lr=0.005):
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)
    r0 = [symmetry_reward(v_opt).item(),
          smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item(),
          compactness_reward(v_opt, f).item()]
    eps = 1e-8
    w = [1/3, 1/3, 1/3]
    for _ in range(steps):
        opt.zero_grad()
        rs = symmetry_reward(v_opt)
        rm = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj)
        rc = compactness_reward(v_opt, f)
        gs = torch.autograd.grad(rs / (abs(r0[0]) + eps), v_opt, retain_graph=True)[0]
        gm = torch.autograd.grad(rm / (abs(r0[1]) + eps), v_opt, retain_graph=True)[0]
        gc = torch.autograd.grad(rc / (abs(r0[2]) + eps), v_opt, retain_graph=False)[0]
        grads = [gs, gm, gc]
        mod = [g.clone() for g in grads]
        for i in range(3):
            gi = mod[i].reshape(-1)
            for j in range(3):
                if i == j: continue
                gj = grads[j].reshape(-1)
                dot = gi @ gj
                if dot < 0:
                    gi = gi - dot / (gj @ gj + 1e-12) * gj
            mod[i] = gi.reshape(grads[i].shape)
        v_opt.grad = -sum(w[i] * mod[i] for i in range(3))
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


def main():
    mesh_dir = 'results/mesh_validity_objs/baseline'
    all_meshes = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        for seed in [42, 123, 456]:
            for m in sorted(glob.glob(f'{mesh_dir}/{cat}/*_seed{seed}.obj')):
                all_meshes.append((m, cat, seed))

    print(f"Total meshes: {len(all_meshes)}")
    print(f"Methods: baseline, three-reward, two-reward, pcgrad")
    print(f"Metric: NEW Huber Normal Consistency")
    print("=" * 60)

    t0 = time.time()
    results = []
    done = 0

    for mp, cat, seed in all_meshes:
        v, f = load_obj(mp)
        if v.shape[0] < 4 or f.shape[0] < 4:
            continue

        adj = _build_face_adjacency(f)

        rec = {
            'mesh': os.path.basename(mp),
            'cat': cat,
            'seed': seed,
            'base': metrics(v, f, adj),
        }

        # Three-reward (1/3, 1/3, 1/3)
        v3 = refine(v, f, [1/3, 1/3, 1/3])
        rec['three'] = metrics(v3, f, adj)

        # Two-reward (0.5, 0.5, 0)
        v2 = refine(v, f, [0.5, 0.5, 0.0])
        rec['two'] = metrics(v2, f, adj)

        # PCGrad
        vp = pcgrad_refine(v, f)
        rec['pcgrad'] = metrics(vp, f, adj)

        results.append(rec)
        done += 1
        if done % 30 == 0:
            eta = (time.time() - t0) / done * (len(all_meshes) - done) / 60
            print(f"  {done}/{len(all_meshes)}, ETA {eta:.1f}m")

    elapsed = (time.time() - t0) / 60
    print(f"\nDone: {len(results)} meshes in {elapsed:.1f}m")

    # === Prompt-level statistics ===
    print("\n" + "=" * 70)
    print("PROMPT-LEVEL RESULTS (new Huber NC metric)")
    print("=" * 70)

    prompt_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        p = r['mesh'].rsplit('_seed', 1)[0]
        for method in ['base', 'three', 'two', 'pcgrad']:
            prompt_data[p][method].append(r[method])

    for method_name in ['three', 'two', 'pcgrad']:
        print(f"\n--- {method_name} vs baseline ---")
        for metric in ['sym', 'smo_new', 'smo_old', 'com']:
            bl = np.array([np.mean([s[metric] for s in prompt_data[p]['base']])
                          for p in prompt_data if method_name in prompt_data[p]])
            mt = np.array([np.mean([s[metric] for s in prompt_data[p][method_name]])
                          for p in prompt_data if method_name in prompt_data[p]])
            if len(bl) < 3:
                continue
            pct = ((mt - bl) / (np.abs(bl) + 1e-10) * 100).mean()
            t_stat, p_val = stats.ttest_rel(mt, bl)
            d = (mt - bl).mean() / ((mt - bl).std() + 1e-12)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            print(f"  {metric:10s}: {pct:+.1f}%, p={p_val:.2e} {sig}, d={d:.3f}")

    # Save
    out_dir = 'analysis_results/huber_nc_rerun'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/all_results.json")


if __name__ == "__main__":
    main()
