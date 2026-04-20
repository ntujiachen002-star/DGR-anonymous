"""
Rerun core experiments on existing baseline OBJ files using the NEW
multi-start symmetry plane protocol.

This is the new-protocol counterpart of tools/rerun_local.py. CPU only.

For each (prompt, seed):
  - Look up the cached symmetry plane (estimated once per baseline mesh in
    Phase A, lives in analysis_results_newproto/plane_cache/production_plane_cache.json).
  - Run three-reward, two-reward, and PCGrad refinements with that plane
    fixed throughout the 50 optimization steps.
  - Score every metric (init + refined) under the SAME plane.

Reads from results/mesh_validity_objs_baseline_snapshot_2026-04-15/ to avoid
race conditions with the GPU exp_k that may be overwriting the live baseline
dir.

Output: analysis_results_newproto/pcgrad_newplane/all_results.json
"""

import sys, os, glob, json, time
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from geo_reward import (symmetry_reward_plane, smoothness_reward,
                        smoothness_reward_legacy, compactness_reward,
                        compute_initial_huber_delta, _build_face_adjacency)


PLANE_CACHE_PATH = 'analysis_results_newproto/plane_cache/production_plane_cache.json'
SNAPSHOT_DIR = 'results/mesh_validity_objs_baseline_snapshot_2026-04-15'
OUT_DIR = 'analysis_results_newproto/pcgrad_newplane'


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def metrics(v, f, sym_n, sym_d, adj=None):
    return {
        'sym': symmetry_reward_plane(v, sym_n, sym_d).item(),
        'smo_new': smoothness_reward(v, f, _adj=adj).item(),
        'smo_old': smoothness_reward_legacy(v, f).item(),
        'com': compactness_reward(v, f).item(),
    }


def refine(v, f, w, sym_n, sym_d, steps=50, lr=0.005):
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)
    r0 = [symmetry_reward_plane(v_opt, sym_n, sym_d).item(),
          smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item(),
          compactness_reward(v_opt, f).item()]
    eps = 1e-8
    for _ in range(steps):
        opt.zero_grad()
        # Always compute all three rewards (zero-weight terms contribute zero
        # gradient; matches src/shape_gen.py post-fix behavior).
        rs = symmetry_reward_plane(v_opt, sym_n, sym_d)
        rm = smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj)
        rc = compactness_reward(v_opt, f)
        reward = (w[0] * rs / (abs(r0[0]) + eps)
                  + w[1] * rm / (abs(r0[1]) + eps)
                  + w[2] * rc / (abs(r0[2]) + eps))
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


def pcgrad_refine(v, f, sym_n, sym_d, steps=50, lr=0.005):
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)
    r0 = [symmetry_reward_plane(v_opt, sym_n, sym_d).item(),
          smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj).item(),
          compactness_reward(v_opt, f).item()]
    eps = 1e-8
    w = [1/3, 1/3, 1/3]
    for _ in range(steps):
        opt.zero_grad()
        rs = symmetry_reward_plane(v_opt, sym_n, sym_d)
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
                if i == j:
                    continue
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
    # Load the production plane cache (330 entries, one per baseline mesh)
    with open(PLANE_CACHE_PATH) as fp:
        plane_cache = json.load(fp)
    print(f"Loaded plane cache: {len(plane_cache)} entries from {PLANE_CACHE_PATH}")

    # Iterate over the stable snapshot dir to avoid races with the GPU exp_k
    all_meshes = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        for seed in [42, 123, 456]:
            for m in sorted(glob.glob(f'{SNAPSHOT_DIR}/{cat}/*_seed{seed}.obj')):
                all_meshes.append((m, cat, seed))

    print(f"Total meshes: {len(all_meshes)}")
    print(f"Source dir: {SNAPSHOT_DIR}")
    print(f"Methods: three-reward, two-reward, pcgrad")
    print(f"Plane: cached per (prompt, seed) — multi-start estimator")
    print("=" * 60)

    # Resume support: if checkpoint exists, skip already-done meshes
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = f'{OUT_DIR}/checkpoint.json'
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fp:
            results = json.load(fp)
        done_keys = {r['mesh'] for r in results}
        print(f"Resuming from checkpoint: {len(results)} meshes already done")
    else:
        results = []
        done_keys = set()

    t0 = time.time()
    n_skipped_no_plane = 0
    n_processed = 0

    for mp, cat, seed in all_meshes:
        mesh_name = os.path.basename(mp)
        if mesh_name in done_keys:
            continue

        # Look up plane: cache key is "<cat>/<filename>"
        plane_key = f"{cat}/{mesh_name}"
        if plane_key not in plane_cache:
            n_skipped_no_plane += 1
            continue
        cached = plane_cache[plane_key]
        sym_n = torch.tensor(cached['normal'], dtype=torch.float32)
        sym_d = torch.tensor(cached['offset'], dtype=torch.float32)

        v, f = load_obj(mp)
        if v.shape[0] < 4 or f.shape[0] < 4:
            continue

        adj = _build_face_adjacency(f)

        rec = {
            'mesh': mesh_name,
            'cat': cat,
            'seed': seed,
            'plane_normal': cached['normal'],
            'plane_offset': cached['offset'],
            'base': metrics(v, f, sym_n, sym_d, adj),
        }

        try:
            v3 = refine(v, f, [1/3, 1/3, 1/3], sym_n, sym_d)
            rec['three'] = metrics(v3, f, sym_n, sym_d, adj)

            v2 = refine(v, f, [0.5, 0.5, 0.0], sym_n, sym_d)
            rec['two'] = metrics(v2, f, sym_n, sym_d, adj)

            vp = pcgrad_refine(v, f, sym_n, sym_d)
            rec['pcgrad'] = metrics(vp, f, sym_n, sym_d, adj)
        except Exception as e:
            rec['error'] = str(e)

        results.append(rec)
        n_processed += 1

        if n_processed % 20 == 0:
            with open(ckpt_path, 'w') as fp:
                json.dump(results, fp)
            elapsed_min = (time.time() - t0) / 60
            rate = n_processed / max(elapsed_min, 0.01)
            remaining = (len(all_meshes) - len(results)) / max(rate, 0.01)
            print(f"  {len(results)}/{len(all_meshes)}, rate={rate:.1f}/min, "
                  f"ETA {remaining:.1f}m")

    elapsed = (time.time() - t0) / 60
    print(f"\nDone: processed {n_processed} new meshes in {elapsed:.1f}m "
          f"(total results: {len(results)}, no-plane skipped: {n_skipped_no_plane})")

    # === Prompt-level statistics ===
    print("\n" + "=" * 70)
    print("PROMPT-LEVEL RESULTS (NEW symmetry plane protocol)")
    print("=" * 70)

    prompt_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if 'error' in r:
            continue
        p = r['mesh'].rsplit('_seed', 1)[0]
        for method in ['base', 'three', 'two', 'pcgrad']:
            if method in r:
                prompt_data[p][method].append(r[method])

    for method_name in ['three', 'two', 'pcgrad']:
        print(f"\n--- {method_name} vs baseline ---")
        for metric in ['sym', 'smo_new', 'com']:
            bl = np.array([np.mean([s[metric] for s in prompt_data[p]['base']])
                           for p in prompt_data
                           if method_name in prompt_data[p]
                           and 'base' in prompt_data[p]])
            mt = np.array([np.mean([s[metric] for s in prompt_data[p][method_name]])
                           for p in prompt_data
                           if method_name in prompt_data[p]
                           and 'base' in prompt_data[p]])
            if len(bl) < 3:
                continue
            pct = ((mt - bl) / (np.abs(bl) + 1e-10) * 100).mean()
            t_stat, p_val = stats.ttest_rel(mt, bl)
            d = (mt - bl).mean() / ((mt - bl).std() + 1e-12)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            print(f"  {metric:10s}: {pct:+.1f}%, p={p_val:.2e} {sig}, d={d:.3f}")

    # Save final results
    final_path = f'{OUT_DIR}/all_results.json'
    with open(final_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved to {final_path}")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
