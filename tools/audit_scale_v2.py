import os
"""Deep audit scale_ablation — is -281% sym degradation a bug or real?"""
import json
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = f'{ROOT}/analysis_results/scale_controlled_ablation/scale_controlled_results.json'
SCALE_PLANE = f'{ROOT}/analysis_results/scale_controlled_ablation/plane_cache.json'
PROD_PLANE  = f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json'
ANTI_PLANE  = f'{ROOT}/analysis_results/anticollapse/plane_cache.json'


def load(p):
    with open(p) as f: return json.load(f)


def main():
    recs = load(SCALE)
    by_method = defaultdict(list)
    for r in recs:
        by_method[r['config']].append(r)

    # === AGGREGATION CHECK ===
    print('=' * 78)
    print('Aggregation check: mean-of-ratios vs ratio-of-means')
    print('=' * 78)
    print(f'\n{"method":<26}{"sym m-of-r":>15}{"sym r-of-m":>15}{"HNC m-of-r":>15}{"com m-of-r":>15}')
    for method, rs in by_method.items():
        sym_pct = np.array([r['sym_change_pct'] for r in rs])
        smo_pct = np.array([r['smo_change_pct'] for r in rs])
        com_pct = np.array([r['com_change_pct'] for r in rs])
        # ratio-of-means (more robust to small denominators)
        bl_sym = np.array([r['base_symmetry']   for r in rs])
        mt_sym = np.array([r['symmetry']        for r in rs])
        rom_sym = (mt_sym.mean() - bl_sym.mean()) / (abs(bl_sym.mean()) + 1e-12) * 100
        print(f'  {method:<24}{sym_pct.mean():>+14.1f}%{rom_sym:>+14.1f}%{smo_pct.mean():>+14.1f}%{com_pct.mean():>+14.1f}%')

    # === COMPACT_ONLY DEEP DIVE ===
    print('\n' + '=' * 78)
    print('compact_only STANDARD deep dive: is -281% driven by small denominators?')
    print('=' * 78)

    std_recs = by_method['compact_only_standard']
    bl_sym = np.array([r['base_symmetry'] for r in std_recs])
    mt_sym = np.array([r['symmetry']      for r in std_recs])
    delta_abs = mt_sym - bl_sym
    pct = np.array([r['sym_change_pct'] for r in std_recs])

    print(f'\n  n={len(std_recs)}')
    print(f'  baseline sym: mean={bl_sym.mean():+.5f}  median={np.median(bl_sym):+.5f}  min={bl_sym.min():+.5f}  max={bl_sym.max():+.5f}')
    print(f'  refined sym:  mean={mt_sym.mean():+.5f}  median={np.median(mt_sym):+.5f}')
    print(f'  abs delta:    mean={delta_abs.mean():+.5f}  median={np.median(delta_abs):+.5f}')
    print(f'  reported %:   mean={pct.mean():+.1f}%  median={np.median(pct):+.1f}%')
    print(f'  r-of-m %:     {(mt_sym.mean() - bl_sym.mean()) / (abs(bl_sym.mean()) + 1e-12) * 100:+.1f}%')

    # Distribution of base_sym |value|
    abs_bl = np.abs(bl_sym)
    print(f'\n  |baseline sym| distribution:')
    print(f'    p10: {np.percentile(abs_bl, 10):.6f}')
    print(f'    p25: {np.percentile(abs_bl, 25):.6f}')
    print(f'    p50: {np.percentile(abs_bl, 50):.6f}')
    print(f'    p75: {np.percentile(abs_bl, 75):.6f}')
    print(f'    p90: {np.percentile(abs_bl, 90):.6f}')

    # How many records have |base_sym| < 0.01 (small denominator)?
    small_den = (abs_bl < 0.01).sum()
    print(f'\n  {small_den}/{len(std_recs)} records have |base_sym| < 0.01 (tiny denominator)')
    small_den_tight = (abs_bl < 0.001).sum()
    print(f'  {small_den_tight}/{len(std_recs)} records have |base_sym| < 0.001')

    # Check: excluding these, what's the mean pct?
    mask = abs_bl >= 0.01
    if mask.sum() > 0:
        pct_filtered = pct[mask]
        print(f'\n  Excluding |base_sym|<0.01: n={mask.sum()}  mean pct={pct_filtered.mean():+.1f}%')
    mask2 = abs_bl >= 0.005
    if mask2.sum() > 0:
        print(f'  Excluding |base_sym|<0.005: n={mask2.sum()}  mean pct={pct[mask2].mean():+.1f}%')

    # Top 5 biggest |pct|
    top = np.argsort(np.abs(pct))[-5:][::-1]
    print(f'\n  Top 5 largest |sym_change_pct| (these dominate mean-of-ratios):')
    for i in top:
        r = std_recs[i]
        print(f'    {r["sym_change_pct"]:+12.1f}%  base_sym={r["base_symmetry"]:+.6f}  refined={r["symmetry"]:+.6f}  abs_delta={r["symmetry"]-r["base_symmetry"]:+.6f}  "{r["prompt"][:35]}"')

    # === PLANE CACHE CONSISTENCY CHECK ===
    print('\n' + '=' * 78)
    print('Plane cache consistency: scale_ablation vs production')
    print('=' * 78)
    scale_pc = load(SCALE_PLANE)
    prod_pc  = load(PROD_PLANE)
    print(f'\n  scale_ablation cache: {len(scale_pc)} entries')
    print(f'  production cache:     {len(prod_pc)} entries')
    print(f'  scale sample key: {list(scale_pc.keys())[0]}')
    print(f'  prod  sample key: {list(prod_pc.keys())[0]}')

    # Check overlap + consistency
    scale_keys = set(scale_pc.keys())
    prod_keys  = set(prod_pc.keys())
    overlap = scale_keys & prod_keys
    scale_only = scale_keys - prod_keys
    prod_only  = prod_keys - scale_keys
    print(f'\n  overlap: {len(overlap)}')
    print(f'  scale-only: {len(scale_only)}')
    print(f'  prod-only:  {len(prod_only)}')

    # Compare numerical values for overlapping keys
    mismatches = 0
    max_diff = 0
    for k in list(overlap)[:50]:
        s = scale_pc[k]
        p = prod_pc[k]
        s_n = np.array(s['normal'])
        p_n = np.array(p['normal'])
        diff = float(np.linalg.norm(s_n - p_n))
        if diff > 1e-4:
            mismatches += 1
            max_diff = max(max_diff, diff)
    print(f'  normal mismatches (>1e-4) in first 50 common: {mismatches}  max diff: {max_diff:.6f}')

    # === COMPARE DGR_STANDARD TO ANTICOLLAPSE DGR_STANDARD ===
    print('\n' + '=' * 78)
    print('Cross-check: scale_ablation vs anticollapse dgr_standard (if present)')
    print('=' * 78)
    anti_pc = load(ANTI_PLANE)
    print(f'  anticollapse cache: {len(anti_pc)} entries')
    anti_keys = set(anti_pc.keys())
    overlap_all = scale_keys & prod_keys & anti_keys
    print(f'  3-way plane cache overlap: {len(overlap_all)}')

    # Verify normals match across all 3
    mismatches3 = 0
    for k in list(overlap_all)[:30]:
        if (np.linalg.norm(np.array(scale_pc[k]['normal']) - np.array(prod_pc[k]['normal'])) > 1e-4 or
            np.linalg.norm(np.array(anti_pc[k]['normal']) - np.array(prod_pc[k]['normal'])) > 1e-4):
            mismatches3 += 1
    print(f'  3-way normal mismatches in first 30: {mismatches3}')


if __name__ == '__main__':
    main()
