"""Evaluate a retrained Lang2Comp checkpoint on the same 236-record test set
used for the original Lang2Comp rerun, reusing cached baseline OBJs and
production planes. CPU-only, ~7-10 minutes.

Loads lang2comp_v2_<tag>.pt, predicts weights for every (prompt, seed), refines
under the paired protocol (same plane as in the Phase 2 lang2comp_blend_sweep),
and saves all_results.json + a summary_vs_equal.json that mirrors the blend
sweep output format.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from lang2comp import Lang2Comp
from geo_reward import (symmetry_reward_plane, smoothness_reward,
                        compactness_reward, compute_initial_huber_delta,
                        _build_face_adjacency)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG2COMP_JSON = f'{ROOT}/analysis_results/lang2comp_rerun/all_results.json'
MESH_ROOT = f'{ROOT}/results/mesh_validity_objs/baseline'
PLANE_CACHE = f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json'
EQUAL_WT = np.array([1/3, 1/3, 1/3], dtype=np.float32)


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return (torch.tensor(verts, dtype=torch.float32),
            torch.tensor(faces, dtype=torch.long))


def find_mesh(prompt, seed):
    name = f'{slug(prompt)}_seed{seed}.obj'
    for cat in ['symmetry', 'smoothness', 'compactness']:
        p = f'{MESH_ROOT}/{cat}/{name}'
        if os.path.exists(p):
            return p, cat
    return None, None


def refine(verts, faces, weights, sym_n, sym_d, steps=50, lr=0.005):
    v_opt = verts.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(faces)
    huber_delta = compute_initial_huber_delta(v_opt, faces)

    with torch.no_grad():
        sym_init = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        smo_init = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=adj).item()
        com_init = compactness_reward(v_opt, faces).item()
    sym_scale = max(abs(sym_init), 1e-6)
    smo_scale = max(abs(smo_init), 1e-6)
    com_scale = max(abs(com_init), 1e-6)

    for _ in range(steps):
        opt.zero_grad()
        sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
        smo = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=adj)
        com = compactness_reward(v_opt, faces)
        reward = (weights[0] * sym / sym_scale
                  + weights[1] * smo / smo_scale
                  + weights[2] * com / com_scale)
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


def compute_metrics(v, f, sym_n, sym_d):
    with torch.no_grad():
        return {
            'symmetry': symmetry_reward_plane(v, sym_n, sym_d).item(),
            'smoothness': smoothness_reward(v, f).item(),
            'compactness': compactness_reward(v, f).item(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--tag', type=str, required=True,
                    help='output subdir under analysis_results_newproto/lang2comp_v2_<tag>')
    args = ap.parse_args()

    out_dir = f'{ROOT}/analysis_results_newproto/lang2comp_v2_{args.tag}'
    os.makedirs(out_dir, exist_ok=True)

    print(f'[eval] checkpoint: {args.checkpoint}')
    print(f'[eval] output dir: {out_dir}')

    print('[eval] Loading Lang2Comp v2...')
    model = Lang2Comp()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    print('[eval] Loading original lang2comp_rerun records (for (prompt, seed) list)...')
    with open(LANG2COMP_JSON) as f:
        l2c = json.load(f)
    keys = sorted({(r['prompt'], r['seed']) for r in l2c
                   if 'prompt' in r and 'seed' in r and 'weights' in r})
    print(f'  {len(keys)} (prompt, seed) pairs')

    print('[eval] Loading plane cache...')
    with open(PLANE_CACHE) as f:
        plane_cache = json.load(f)
    print(f'  {len(plane_cache)} entries')

    # Predict all weights upfront (one batch through encoder).
    prompts = sorted({p for p, _ in keys})
    print(f'[eval] Predicting weights for {len(prompts)} unique prompts...')
    with torch.no_grad():
        pred_w, _ = model([p for p in prompts])
    pred_w = pred_w.cpu().numpy()
    weights_by_prompt = {p: pred_w[i] for i, p in enumerate(prompts)}

    print(f'[eval] Weight prediction stats:')
    max_w = pred_w.max(axis=1)
    print(f'     max weight: mean={max_w.mean():.3f}  median={float(np.median(max_w)):.3f}')
    print(f'                 min={max_w.min():.3f}  max={max_w.max():.3f}')
    print(f'     frac max>=0.6: {(max_w >= 0.6).mean():.2%}')
    print(f'     frac max>=0.8: {(max_w >= 0.8).mean():.2%}')

    results = []
    t0 = time.time()
    n_proc = 0
    n_skip = 0

    for prompt, seed in keys:
        mesh_path, cat = find_mesh(prompt, seed)
        if mesh_path is None:
            n_skip += 1
            continue
        plane_key = f'{cat}/{slug(prompt)}_seed{seed}.obj'
        if plane_key not in plane_cache:
            n_skip += 1
            continue
        verts, faces = load_obj(mesh_path)
        if verts.numel() == 0 or faces.shape[0] == 0:
            n_skip += 1
            continue

        cached = plane_cache[plane_key]
        sym_n = torch.tensor(cached['normal'], dtype=torch.float32)
        sym_d = torch.tensor(cached['offset'], dtype=torch.float32)

        base_m = compute_metrics(verts, faces, sym_n, sym_d)
        w = torch.tensor(weights_by_prompt[prompt], dtype=torch.float32)

        try:
            refined = refine(verts, faces, w, sym_n, sym_d)
            m = compute_metrics(refined, faces, sym_n, sym_d)
            results.append({
                'prompt': prompt, 'seed': seed, 'category': cat,
                'weights': {'symmetry': float(w[0]),
                            'smoothness': float(w[1]),
                            'compactness': float(w[2])},
                **m,
                'base_symmetry': base_m['symmetry'],
                'base_smoothness': base_m['smoothness'],
                'base_compactness': base_m['compactness'],
            })
        except Exception as e:
            results.append({'prompt': prompt, 'seed': seed, 'error': str(e)})

        n_proc += 1
        if n_proc % 40 == 0:
            elapsed = time.time() - t0
            rate = n_proc / max(elapsed / 60, 0.01)
            eta = (len(keys) - n_proc) / max(rate, 0.01)
            print(f'  {n_proc}/{len(keys)}  rate={rate:.1f}/min  ETA {eta:.1f}m')

    elapsed = (time.time() - t0) / 60
    print(f'\n[eval] Done: {n_proc} processed, {n_skip} skipped, {elapsed:.1f} min')

    out_json = f'{out_dir}/all_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  saved {len(results)} records to {out_json}')

    # Aggregate seed-averaged per-prompt stats vs baseline.
    from collections import defaultdict
    prompt_data = defaultdict(list)
    for r in results:
        if 'error' in r:
            continue
        prompt_data[r['prompt']].append(r)
    bl_s, bl_h, bl_c = [], [], []
    mt_s, mt_h, mt_c = [], [], []
    for p, recs in prompt_data.items():
        bl_s.append(np.mean([r['base_symmetry'] for r in recs]))
        bl_h.append(np.mean([r['base_smoothness'] for r in recs]))
        bl_c.append(np.mean([r['base_compactness'] for r in recs]))
        mt_s.append(np.mean([r['symmetry'] for r in recs]))
        mt_h.append(np.mean([r['smoothness'] for r in recs]))
        mt_c.append(np.mean([r['compactness'] for r in recs]))
    bl_s, mt_s = np.array(bl_s), np.array(mt_s)
    bl_h, mt_h = np.array(bl_h), np.array(mt_h)
    bl_c, mt_c = np.array(bl_c), np.array(mt_c)

    def pct(mt, bl):
        return (mt.mean() - bl.mean()) / (abs(bl.mean()) + 1e-10) * 100

    summary = {
        'n_prompts': int(len(bl_s)),
        'sym_mean_pct': float(pct(mt_s, bl_s)),
        'sym_med_pct':  float((np.median(mt_s) - np.median(bl_s)) / (abs(np.median(bl_s)) + 1e-10) * 100),
        'sym_win_pct':  float((mt_s > bl_s).mean() * 100),
        'hnc_mean_pct': float(pct(mt_h, bl_h)),
        'com_mean_pct': float(pct(mt_c, bl_c)),
        'd_sym': float((mt_s - bl_s).mean() / ((mt_s - bl_s).std(ddof=1) + 1e-12)),
    }
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n===== SUMMARY vs baseline (paired, seed-averaged) =====')
    print(f"  n_prompts:    {summary['n_prompts']}")
    print(f"  sym mean%:    {summary['sym_mean_pct']:+.1f}%")
    print(f"  sym median%:  {summary['sym_med_pct']:+.1f}%")
    print(f"  sym win%:     {summary['sym_win_pct']:.1f}%")
    print(f"  HNC mean%:    {summary['hnc_mean_pct']:+.1f}%")
    print(f"  com mean%:    {summary['com_mean_pct']:+.1f}%")
    print(f"  d_sym (d_z):  {summary['d_sym']:+.3f}")

    print('\n===== COMPARE WITH baseline Lang2Comp (alpha=0 in blend sweep) =====')
    print("  Baseline (alpha=0):  sym +77.7%  median +15.0%  win 100%  HNC +131.6%  com +2.2%")
    print("  Equal Wt (alpha=1):  sym +91.9%  median +20.0%  win 100%  HNC +131.7%  com +1.9%")


if __name__ == '__main__':
    main()
