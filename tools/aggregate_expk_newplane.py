"""Aggregate exp_k mesh_validity output under the new plane protocol.

Reads:
  - results/mesh_validity_objs/<method>/<cat>/<prompt>_seed<N>.obj
  - analysis_results/mesh_validity_full/plane_cache.json (per-mesh plane)

Computes:
  - For each (method, prompt, seed) tuple, sym (under cached plane), HNC, com
  - Aggregates per-prompt (seed-averaged), then paired stats vs baseline
  - Prints table of all 5 refinement methods vs baseline
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import torch
from geo_reward import (symmetry_reward_plane, smoothness_reward,
                        compactness_reward)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ_DIR = f'{ROOT}/results/mesh_validity_objs'
PLANE_CACHE = f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json'
CHECKPOINT = f'{ROOT}/analysis_results/mesh_validity_full/generation_checkpoint.json'

METHODS = ['baseline', 'handcrafted', 'sym_only', 'HNC_only', 'compact_only', 'diffgeoreward']


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    if not verts:
        return None, None
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def main():
    plane_cache = json.load(open(PLANE_CACHE))
    print(f'plane cache: {len(plane_cache)} entries')

    done = set(json.load(open(CHECKPOINT)))
    print(f'checkpoint: {len(done)} keys')

    # checkpoint key format: "<method>/<prompt_slug>_seed<N>"
    # plane cache key format: "<cat>/<prompt_slug>_seed<N>.obj"

    results = defaultdict(list)
    n_loaded = 0
    n_skipped = 0
    n_no_obj = 0
    n_no_plane = 0
    n_load_fail = 0

    for i, key in enumerate(sorted(done)):
        # Parse: "baseline/a_balanced_bookshelf_seed42"
        method, name = key.split('/', 1)
        if method not in METHODS:
            continue

        # Find the obj file (could be in symmetry/, smoothness/, or compactness/)
        obj_path = None
        cat_found = None
        for cat in ['symmetry', 'smoothness', 'compactness']:
            candidate = f'{OBJ_DIR}/{method}/{cat}/{name}.obj'
            if os.path.exists(candidate):
                obj_path = candidate
                cat_found = cat
                break
        if i < 3:
            print(f'  [debug {i}] key={key}, method={method}, name={name}, found={obj_path}')
        if not obj_path:
            n_no_obj += 1
            continue

        plane_key = f'{cat_found}/{name}.obj'
        if plane_key not in plane_cache:
            n_no_plane += 1
            continue

        v, f = load_obj(obj_path)
        if v is None or f.shape[0] == 0 or v.shape[0] == 0:
            n_load_fail += 1
            continue

        cached = plane_cache[plane_key]
        sym_n = torch.tensor(cached['normal'], dtype=torch.float32)
        sym_d = torch.tensor(cached['offset'], dtype=torch.float32)

        try:
            with torch.no_grad():
                sym = symmetry_reward_plane(v, sym_n, sym_d).item()
                smo = smoothness_reward(v, f).item()
                com = compactness_reward(v, f).item()
        except Exception as e:
            n_load_fail += 1
            continue

        # prompt id = name without _seedN
        prompt_id = name.rsplit('_seed', 1)[0]
        seed = int(name.rsplit('_seed', 1)[1])
        results[(method, prompt_id, seed)] = (sym, smo, com, cat_found)
        n_loaded += 1

    print(f'loaded: {n_loaded}, no_obj: {n_no_obj}, no_plane: {n_no_plane}, load_fail: {n_load_fail}')
    print()

    # Build prompt-level aggregation: for each (method, prompt_id),
    # average across seeds. Then compute paired t vs baseline.
    by_prompt = defaultdict(lambda: defaultdict(list))
    for (method, prompt, seed), (sym, smo, com, cat) in results.items():
        by_prompt[(method, prompt)]['sym'].append(sym)
        by_prompt[(method, prompt)]['smo'].append(smo)
        by_prompt[(method, prompt)]['com'].append(com)
        by_prompt[(method, prompt)]['cat'] = cat

    # Find prompts present in BOTH baseline and method (paired set)
    baseline_prompts = {k[1] for k in by_prompt if k[0] == 'baseline'}

    print('Method comparison (n=paired-prompts, vs baseline):')
    print('=' * 100)
    header = '{:<14s} {:<6s} {:>4s} {:>10s} {:>10s} {:>9s} {:>10s} {:>8s} {:>8s}'
    print(header.format('Method', 'Metric', 'n', 'baseline', 'method', '%change', 'p', 'd', 'WinRate'))
    print('-' * 100)

    summary = {}
    for method in METHODS:
        if method == 'baseline':
            continue
        for metric in ['sym', 'smo', 'com']:
            bl_vals, mt_vals = [], []
            for prompt in baseline_prompts:
                bl_key = ('baseline', prompt)
                mt_key = (method, prompt)
                if mt_key not in by_prompt:
                    continue
                bl_avg = np.mean(by_prompt[bl_key][metric])
                mt_avg = np.mean(by_prompt[mt_key][metric])
                bl_vals.append(bl_avg)
                mt_vals.append(mt_avg)
            bl_arr = np.array(bl_vals); mt_arr = np.array(mt_vals)
            n = len(bl_arr)
            if n < 3:
                continue
            # Ratio-of-means (paper convention), NOT mean-of-ratios.
            # Avoids per-prompt small-baseline outliers blowing up the average.
            bl_mean = bl_arr.mean()
            mt_mean = mt_arr.mean()
            pct = (mt_mean - bl_mean) / (abs(bl_mean) + 1e-10) * 100
            diff = mt_arr - bl_arr
            t, pv = stats.ttest_rel(mt_arr, bl_arr)
            d_eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
            win = (mt_arr > bl_arr).mean() * 100
            row = '{:<14s} {:<6s} {:>4d} {:>+10.5f} {:>+10.5f} {:>+8.1f}% {:>10.2e} {:>+8.3f} {:>7.1f}%'.format(
                method, metric, n, bl_mean, mt_mean, pct, pv, d_eff, win)
            print(row)
            summary[(method, metric)] = (n, bl_arr.mean(), mt_arr.mean(), pct, pv, d_eff, win)
        print()

    # Save
    out_path = f'{ROOT}/analysis_results_newproto/expk_newplane_aggregated.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({f'{k[0]}/{k[1]}': list(v) for k, v in summary.items()}, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
