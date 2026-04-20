import os
"""Compute per-category absolute metric improvements for the main-body
`tab:lang2comp` under Option A (targeted-specialization framing).

For each (method, category), reports:
  - target metric mean improvement %  (e.g. symmetry category -> sym %)
  - non-target metric mean improvement % (what you sacrifice)
  - seed-averaged, paired protocol (same baseline plane across methods)
"""
import json
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    'old_Lang2Comp (alpha=0)':   f'{ROOT}/analysis_results_newproto/lang2comp_blend_sweep/alpha_0.00.json',
    'alpha=0.5 blend':           f'{ROOT}/analysis_results_newproto/lang2comp_blend_sweep/alpha_0.50.json',
    'Equal_Wt (alpha=1.0)':      f'{ROOT}/analysis_results_newproto/lang2comp_blend_sweep/alpha_1.00.json',
    'Lang2Comp_v2 (lam=0.5)':    f'{ROOT}/analysis_results_newproto/lang2comp_v2_lam050/all_results.json',
}

CATEGORY_TARGET = {
    'symmetry': 'symmetry',
    'smoothness': 'smoothness',
    'compactness': 'compactness',
}


def load_records(path):
    with open(path) as f:
        recs = json.load(f)
    return [r for r in recs if 'error' not in r]


def seed_avg(records):
    by = defaultdict(list)
    for r in records:
        by[r['prompt']].append(r)
    out = {}
    for p, recs in by.items():
        out[p] = {
            'category': recs[0].get('category', '?'),
            'sym': np.mean([r['symmetry']   for r in recs]),
            'hnc': np.mean([r['smoothness'] for r in recs]),
            'com': np.mean([r['compactness'] for r in recs]),
            'base_sym': np.mean([r['base_symmetry']   for r in recs]),
            'base_hnc': np.mean([r['base_smoothness'] for r in recs]),
            'base_com': np.mean([r['base_compactness'] for r in recs]),
        }
    return out


def category_stats(seed_avged):
    """Return {category: {metric: {mean_pct, median_pct}}}."""
    by_cat = defaultdict(list)
    for p, e in seed_avged.items():
        by_cat[e['category']].append(e)

    out = {}
    for cat, prompts in by_cat.items():
        if not prompts:
            continue
        def _pct(met, base):
            bl = np.array([p[f'base_{base}'] for p in prompts])
            mt = np.array([p[met] for p in prompts])
            return {
                'mean_pct': float((mt.mean() - bl.mean()) / (abs(bl.mean()) + 1e-10) * 100),
                'n': len(prompts),
            }
        out[cat] = {
            'sym': _pct('sym', 'sym'),
            'hnc': _pct('hnc', 'hnc'),
            'com': _pct('com', 'com'),
        }
    return out


def pair_h2h(method_a, method_b):
    """Per-prompt head-to-head wins on each metric."""
    common = sorted(set(method_a) & set(method_b))
    wins_a = {'sym': 0, 'hnc': 0, 'com': 0}
    wins_b = {'sym': 0, 'hnc': 0, 'com': 0}
    for p in common:
        for met in ('sym', 'hnc', 'com'):
            if method_a[p][met] > method_b[p][met] + 1e-6:
                wins_a[met] += 1
            elif method_a[p][met] < method_b[p][met] - 1e-6:
                wins_b[met] += 1
    return wins_a, wins_b, len(common)


def main():
    print('Loading sources...')
    raw = {name: load_records(path) for name, path in SOURCES.items()}
    for name, recs in raw.items():
        print(f'  {name:30s}  {len(recs)} records')

    avg = {name: seed_avg(recs) for name, recs in raw.items()}

    # === PER-CATEGORY ABSOLUTE IMPROVEMENT (for main-body table) ===
    print('\n' + '=' * 78)
    print('PER-CATEGORY target-metric improvement % (seed-averaged, n per category)')
    print('=' * 78)

    stats = {name: category_stats(a) for name, a in avg.items()}

    print(f'\n{"Method":<28}{"sym-cat sym%":>15}{"smo-cat HNC%":>15}{"com-cat com%":>15}')
    print('-' * 73)
    for name, s in stats.items():
        sym_sym = s.get('symmetry',    {}).get('sym', {}).get('mean_pct', float('nan'))
        smo_hnc = s.get('smoothness',  {}).get('hnc', {}).get('mean_pct', float('nan'))
        com_com = s.get('compactness', {}).get('com', {}).get('mean_pct', float('nan'))
        print(f'{name:<28}{sym_sym:>+14.1f}%{smo_hnc:>+14.1f}%{com_com:>+14.1f}%')

    # === FULL CROSS-CATEGORY TABLE ===
    print('\n' + '=' * 78)
    print('Full per-category cross-metric table (showing the tradeoff explicitly)')
    print('=' * 78)

    for cat in ['symmetry', 'smoothness', 'compactness']:
        n = stats[list(stats)[0]].get(cat, {}).get('sym', {}).get('n', 0)
        print(f'\n[category: {cat}]  n={n}')
        print(f'  {"Method":<28}{"sym%":>10}{"HNC%":>10}{"com%":>10}')
        for name, s in stats.items():
            cs = s.get(cat, {})
            sv = cs.get('sym', {}).get('mean_pct', float('nan'))
            hv = cs.get('hnc', {}).get('mean_pct', float('nan'))
            cv = cs.get('com', {}).get('mean_pct', float('nan'))
            print(f'  {name:<28}{sv:>+9.1f}%{hv:>+9.1f}%{cv:>+9.1f}%')

    # === HEAD-TO-HEAD: v2 vs Equal Wt (per-category, for win-rate table) ===
    v2 = avg['Lang2Comp_v2 (lam=0.5)']
    eq = avg['Equal_Wt (alpha=1.0)']

    print('\n' + '=' * 78)
    print('Head-to-head: v2 vs Equal Wt BY CATEGORY')
    print('=' * 78)

    by_cat_v2 = defaultdict(dict)
    by_cat_eq = defaultdict(dict)
    for p, e in v2.items():
        by_cat_v2[e['category']][p] = e
    for p, e in eq.items():
        by_cat_eq[e['category']][p] = e

    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_v2 = by_cat_v2[cat]
        cat_eq = by_cat_eq[cat]
        wins_v2, wins_eq, n = pair_h2h(cat_v2, cat_eq)
        print(f'\n[{cat}]  n={n}')
        target_metric = {'symmetry': 'sym', 'smoothness': 'hnc', 'compactness': 'com'}[cat]
        for met in ('sym', 'hnc', 'com'):
            marker = '  <-- TARGET' if met == target_metric else ''
            print(f'  {met}: v2 wins {wins_v2[met]:3d}/{n}  equal wins {wins_eq[met]:3d}/{n}{marker}')


if __name__ == '__main__':
    main()
