"""Lang2Comp blend sweep — measures Lang2Comp + Equal Wt blend at inference time.

Loads existing Lang2Comp predictions (per-prompt weights) from the Phase 2
lang2comp_rerun output, blends them with Equal Wt at alpha in {0, 0.25, 0.5,
0.7, 1.0}, refines each baseline mesh under each blend, and saves the
results for direct comparison.

All CPU. Reuses existing baseline OBJs + cached plane. No Shap-E regen.
Runtime budget: ~20 min on CPU for 236 records × 5 alphas = 1180 refinements.
"""
import os
import sys
import re
import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from geo_reward import (symmetry_reward_plane, smoothness_reward,
                        compactness_reward, compute_initial_huber_delta,
                        _build_face_adjacency)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG2COMP_JSON = f'{ROOT}/analysis_results/lang2comp_rerun/all_results.json'
MESH_ROOT = f'{ROOT}/results/mesh_validity_objs/baseline'
PLANE_CACHE = f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json'
OUT_DIR = f'{ROOT}/analysis_results_newproto/lang2comp_blend_sweep'

ALPHAS = [0.0, 0.25, 0.5, 0.7, 1.0]
EQUAL_WT = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)


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
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def find_mesh(prompt, seed):
    """Locate baseline OBJ for (prompt, seed)."""
    name = f'{slug(prompt)}_seed{seed}.obj'
    for cat in ['symmetry', 'smoothness', 'compactness']:
        p = f'{MESH_ROOT}/{cat}/{name}'
        if os.path.exists(p):
            return p, cat
    return None, None


def refine(verts, faces, weights, sym_n, sym_d, steps=50, lr=0.005):
    """Standard refine_with_geo_reward with fixed plane."""
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
            'symmetry':   symmetry_reward_plane(v, sym_n, sym_d).item(),
            'smoothness': smoothness_reward(v, f).item(),
            'compactness': compactness_reward(v, f).item(),
        }


def main():
    print(f'Loading Lang2Comp predictions from {LANG2COMP_JSON}...')
    l2c = json.load(open(LANG2COMP_JSON))
    print(f'  {len(l2c)} records')

    print(f'Loading plane cache...')
    plane_cache = json.load(open(PLANE_CACHE))
    print(f'  {len(plane_cache)} plane entries')

    os.makedirs(OUT_DIR, exist_ok=True)

    # Organize Lang2Comp records by (prompt, seed) -> weights
    l2c_by_key = {}
    for r in l2c:
        if 'weights' not in r or 'prompt' not in r or 'seed' not in r:
            continue
        key = (r['prompt'], r['seed'])
        l2c_by_key[key] = (r['weights']['symmetry'],
                           r['weights']['smoothness'],
                           r['weights']['compactness'])
    print(f'  Lang2Comp keys: {len(l2c_by_key)}')

    # Results: list of records per alpha
    results_by_alpha = {a: [] for a in ALPHAS}
    baselines = {}  # (prompt, seed) -> {sym, smo, com}

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    for (prompt, seed), l2c_w in l2c_by_key.items():
        mesh_path, cat = find_mesh(prompt, seed)
        if mesh_path is None:
            n_skipped += 1
            continue
        plane_key = f'{cat}/{slug(prompt)}_seed{seed}.obj'
        if plane_key not in plane_cache:
            n_skipped += 1
            continue

        verts, faces = load_obj(mesh_path)
        if verts.numel() == 0 or faces.shape[0] == 0:
            n_skipped += 1
            continue

        cached = plane_cache[plane_key]
        sym_n = torch.tensor(cached['normal'], dtype=torch.float32)
        sym_d = torch.tensor(cached['offset'], dtype=torch.float32)

        # Baseline metrics
        base_m = compute_metrics(verts, faces, sym_n, sym_d)
        baselines[(prompt, seed)] = base_m

        w_l2c = torch.tensor(l2c_w, dtype=torch.float32)

        for alpha in ALPHAS:
            w_blend = (1 - alpha) * w_l2c + alpha * EQUAL_WT
            try:
                refined = refine(verts, faces, w_blend, sym_n, sym_d)
                m = compute_metrics(refined, faces, sym_n, sym_d)
                rec = {
                    'prompt': prompt, 'seed': seed, 'category': cat,
                    'alpha': alpha,
                    'weights': w_blend.tolist(),
                    'weights_l2c': list(l2c_w),
                    **m,
                    'base_symmetry': base_m['symmetry'],
                    'base_smoothness': base_m['smoothness'],
                    'base_compactness': base_m['compactness'],
                }
                results_by_alpha[alpha].append(rec)
            except Exception as e:
                rec = {'prompt': prompt, 'seed': seed, 'alpha': alpha, 'error': str(e)}
                results_by_alpha[alpha].append(rec)

        n_processed += 1
        if n_processed % 30 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed/60, 0.01)
            eta = (len(l2c_by_key) - n_processed) / max(rate, 0.01)
            print(f'  {n_processed}/{len(l2c_by_key)} records, rate={rate:.1f}/min, ETA {eta:.1f}m')

    elapsed = (time.time() - t0) / 60
    print(f'\nDone: processed {n_processed} records, skipped {n_skipped}, in {elapsed:.1f} min')

    # Save per-alpha JSONs
    for alpha, records in results_by_alpha.items():
        path = f'{OUT_DIR}/alpha_{alpha:.2f}.json'
        with open(path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f'  saved {len(records)} records to {path}')

    # Also compute aggregate stats per alpha (seed-averaged per prompt)
    from scipy import stats
    print('\n=== AGGREGATE per alpha (seed-averaged, n paired prompts) ===')
    print('{:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5}'.format(
        'alpha','sym_mean%','sym_med%','sym_win','hnc_mean%','com_mean%','d_sym','n'))
    summary = {}
    for alpha in ALPHAS:
        prompt_data = defaultdict(list)
        for r in results_by_alpha[alpha]:
            if 'error' in r: continue
            prompt_data[r['prompt']].append(r)
        bl_sym, bl_hnc, bl_com = [], [], []
        mt_sym, mt_hnc, mt_com = [], [], []
        for p, recs in prompt_data.items():
            bl_sym.append(np.mean([r['base_symmetry'] for r in recs]))
            bl_hnc.append(np.mean([r['base_smoothness'] for r in recs]))
            bl_com.append(np.mean([r['base_compactness'] for r in recs]))
            mt_sym.append(np.mean([r['symmetry'] for r in recs]))
            mt_hnc.append(np.mean([r['smoothness'] for r in recs]))
            mt_com.append(np.mean([r['compactness'] for r in recs]))
        bl_s = np.array(bl_sym); mt_s = np.array(mt_sym)
        bl_h = np.array(bl_hnc); mt_h = np.array(mt_hnc)
        bl_c = np.array(bl_com); mt_c = np.array(mt_com)
        sym_mean = (mt_s.mean() - bl_s.mean()) / (abs(bl_s.mean())+1e-10)*100
        sym_med  = (np.median(mt_s) - np.median(bl_s)) / (abs(np.median(bl_s))+1e-10)*100
        sym_win  = (mt_s > bl_s).mean() * 100
        hnc_mean = (mt_h.mean() - bl_h.mean()) / (abs(bl_h.mean())+1e-10)*100
        com_mean = (mt_c.mean() - bl_c.mean()) / (abs(bl_c.mean())+1e-10)*100
        diff = mt_s - bl_s
        d_sym = diff.mean() / (diff.std(ddof=1)+1e-12)
        n = len(bl_s)
        summary[alpha] = {'sym_mean': sym_mean, 'sym_med': sym_med, 'sym_win': sym_win,
                          'hnc_mean': hnc_mean, 'com_mean': com_mean, 'd_sym': d_sym, 'n': n}
        print('{:<8.2f} {:>+9.1f}% {:>+9.1f}% {:>9.1f}% {:>+9.1f}% {:>+9.1f}% {:>+10.3f} {:>5d}'.format(
            alpha, sym_mean, sym_med, sym_win, hnc_mean, com_mean, d_sym, n))

    with open(f'{OUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {OUT_DIR}/summary.json')


if __name__ == '__main__':
    main()
