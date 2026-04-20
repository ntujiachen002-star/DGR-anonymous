import os
"""Audit anticollapse — is monotonic degradation a bug or real?"""
import json
import numpy as np
from collections import defaultdict
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANTI = f'{ROOT}/analysis_results/anticollapse/anticollapse_results.json'
ANTI_PLANE  = f'{ROOT}/analysis_results/anticollapse/plane_cache.json'
SCALE_PLANE = f'{ROOT}/analysis_results/scale_controlled_ablation/plane_cache.json'
PROD_PLANE  = f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json'


def load(p):
    with open(p) as f: return json.load(f)


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def main():
    data = load(ANTI)
    print(f'data type: {type(data).__name__}')

    if isinstance(data, dict):
        print(f'keys: {list(data.keys())[:10]}')
        # nested?
        first_k = list(data.keys())[0]
        if isinstance(data[first_k], list):
            recs = []
            for m, rs in data.items():
                for r in rs:
                    r['_method'] = m
                    recs.append(r)
        else:
            recs = list(data.values())
    else:
        recs = data

    print(f'records: {len(recs)}')
    if recs:
        print(f'sample keys: {list(recs[0].keys())}')

    # Group
    by_method = defaultdict(list)
    for r in recs:
        m = r.get('method') or r.get('_method') or r.get('config', '?')
        by_method[m].append(r)
    print(f'\nmethods: {dict((k, len(v)) for k, v in by_method.items())}')

    # === PAIRING CHECK ===
    print('\n' + '=' * 78)
    print('Pairing check: is each (prompt, seed) in all 4 variants?')
    print('=' * 78)

    sets = {}
    for m, rs in by_method.items():
        sets[m] = {(r['prompt'], r['seed']) for r in rs}
    methods = list(sets.keys())
    if len(methods) >= 2:
        common = set.intersection(*sets.values())
        print(f'\n  common (prompt, seed): {len(common)}')
        for m in methods:
            missing = sets[m] - common
            print(f'  {m}: {len(sets[m])} records, {len(missing)} not paired')

    # === RATIO-OF-MEANS FOR EACH METHOD ===
    print('\n' + '=' * 78)
    print('Both aggregations: mean-of-ratios vs ratio-of-means per method')
    print('=' * 78)
    print(f'\n{"method":<28}{"sym MoR":>12}{"sym RoM":>12}{"HNC MoR":>12}{"HNC RoM":>12}{"com MoR":>12}{"com RoM":>12}')
    print('-' * 112)

    standard_pairs = None  # to identify which records dominate standard run
    for m in ['dgr_standard', 'dgr_anticollapse_soft', 'dgr_anticollapse_medium', 'dgr_anticollapse_strong']:
        if m not in by_method: continue
        rs = by_method[m]
        bl_sym = np.array([r['base_symmetry']  for r in rs])
        mt_sym = np.array([r['symmetry']       for r in rs])
        bl_hnc = np.array([r['base_smoothness'] for r in rs])
        mt_hnc = np.array([r['smoothness']      for r in rs])
        bl_com = np.array([r['base_compactness'] for r in rs])
        mt_com = np.array([r['compactness']      for r in rs])

        def rom(mt, bl):
            return (mt.mean() - bl.mean()) / (abs(bl.mean()) + 1e-12) * 100

        def mor(mt, bl):
            pct = (mt - bl) / (np.abs(bl) + 1e-10) * 100
            return pct.mean()

        print(f'  {m:<26}{mor(mt_sym, bl_sym):>+11.1f}%{rom(mt_sym, bl_sym):>+11.1f}%'
              f'{mor(mt_hnc, bl_hnc):>+11.1f}%{rom(mt_hnc, bl_hnc):>+11.1f}%'
              f'{mor(mt_com, bl_com):>+11.1f}%{rom(mt_com, bl_com):>+11.1f}%')

    # === PAIRED ABSOLUTE DELTA ===
    print('\n' + '=' * 78)
    print('Paired absolute-delta test: standard vs anticollapse_strong on same (prompt, seed)')
    print('=' * 78)

    std = {(r['prompt'], r['seed']): r for r in by_method.get('dgr_standard', [])}
    strong = {(r['prompt'], r['seed']): r for r in by_method.get('dgr_anticollapse_strong', [])}
    keys = sorted(set(std) & set(strong))
    print(f'\n  n paired: {len(keys)}')

    for metric, base in [('symmetry', 'base_symmetry'),
                         ('smoothness', 'base_smoothness'),
                         ('compactness', 'base_compactness')]:
        std_gain = np.array([std[k][metric] - std[k][base] for k in keys])
        str_gain = np.array([strong[k][metric] - strong[k][base] for k in keys])
        diff = str_gain - std_gain
        print(f'  {metric}: std_delta={std_gain.mean():+.5f}  strong_delta={str_gain.mean():+.5f}')
        print(f'    per-pair (strong - std): mean={diff.mean():+.5f}  median={np.median(diff):+.5f}')
        better_strong = (diff > 0).sum()  # positive = strong closer to 0 = better
        print(f'    strong beats std on {better_strong}/{len(keys)} prompts')

    # === PLANE CACHE CONSISTENCY (anticollapse vs scale_ablation vs production) ===
    print('\n' + '=' * 78)
    print('Plane cache: anticollapse uses what keys?')
    print('=' * 78)
    anti_pc = load(ANTI_PLANE)
    scale_pc = load(SCALE_PLANE)
    prod_pc  = load(PROD_PLANE)
    print(f'\n  anti:  {len(anti_pc)} entries, sample key: {list(anti_pc)[0]}')
    print(f'  scale: {len(scale_pc)} entries, sample key: {list(scale_pc)[0]}')
    print(f'  prod:  {len(prod_pc)} entries, sample key: {list(prod_pc)[0]}')

    # Check: for first few prompts, are the planes identical modulo format?
    # anti key format: "a symmetric vase|seed=42"
    # prod key format: "symmetry/a_symmetric_vase_seed42.obj"
    def anti_to_prod_key(anti_key):
        try:
            prompt, seed_part = anti_key.rsplit('|seed=', 1)
            seed = int(seed_part)
        except ValueError:
            return None
        prod_name = f'{slug(prompt)}_seed{seed}.obj'
        # look for it in any category
        for cat in ('symmetry', 'smoothness', 'compactness'):
            candidate = f'{cat}/{prod_name}'
            if candidate in prod_pc:
                return candidate
        return None

    print('\nFirst 20 anti keys:')
    matched = 0
    mismatched_normal = 0
    for i, k in enumerate(list(anti_pc)[:20]):
        prod_k = anti_to_prod_key(k)
        if prod_k is None:
            print(f'  {k}  ->  NO MATCH')
            continue
        matched += 1
        anti_n = np.array(anti_pc[k]['normal'])
        prod_n = np.array(prod_pc[prod_k]['normal'])
        diff = min(np.linalg.norm(anti_n - prod_n), np.linalg.norm(anti_n + prod_n))
        if diff > 1e-4:
            mismatched_normal += 1
            print(f'  {k[:40]}  MISMATCH diff={diff:.6f}')

    # Full check
    all_matched = 0
    all_mismatched = 0
    for k in anti_pc:
        prod_k = anti_to_prod_key(k)
        if prod_k is None:
            continue
        all_matched += 1
        anti_n = np.array(anti_pc[k]['normal'])
        prod_n = np.array(prod_pc[prod_k]['normal'])
        diff = min(np.linalg.norm(anti_n - prod_n), np.linalg.norm(anti_n + prod_n))
        if diff > 1e-4:
            all_mismatched += 1

    print(f'\n  full consistency: {all_matched}/{len(anti_pc)} keys mapped to prod, {all_mismatched} normal mismatches')


if __name__ == '__main__':
    main()
