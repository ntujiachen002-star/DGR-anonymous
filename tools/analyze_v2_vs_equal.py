import os
"""Rigorous per-prompt comparison: Lang2Comp v2 (lam=0.5) vs Equal Wt vs
old Lang2Comp (alpha=0 row in blend sweep), using the SAME 97 prompts.

Checks:
  1. Does v2 win head-to-head against Equal Wt per prompt?
  2. Is the sym mean vs median gap real (outliers)?
  3. Win rate on each metric
  4. Sanity: do blend sweep and v2 eval use the same (prompt, seed) set?
  5. Per-category breakdown
"""
import json
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V2 = f'{ROOT}/analysis_results_newproto/lang2comp_v2_lam050/all_results.json'
BLEND_ALPHA0 = f'{ROOT}/analysis_results_newproto/lang2comp_blend_sweep/alpha_0.00.json'
BLEND_ALPHA1 = f'{ROOT}/analysis_results_newproto/lang2comp_blend_sweep/alpha_1.00.json'


def load(path):
    with open(path) as f:
        return [r for r in json.load(f) if 'error' not in r]


def seed_avg(records):
    """Group by prompt, average metrics across seeds. Returns {prompt: {...}}."""
    by = defaultdict(list)
    for r in records:
        by[r['prompt']].append(r)
    out = {}
    for p, recs in by.items():
        out[p] = {
            'category': recs[0].get('category', '?'),
            'n_seeds': len(recs),
            'sym':       np.mean([r['symmetry']   for r in recs]),
            'hnc':       np.mean([r['smoothness'] for r in recs]),
            'com':       np.mean([r['compactness'] for r in recs]),
            'base_sym':  np.mean([r['base_symmetry']   for r in recs]),
            'base_hnc':  np.mean([r['base_smoothness'] for r in recs]),
            'base_com':  np.mean([r['base_compactness'] for r in recs]),
            'w_raw': recs[0].get('weights'),
        }
    return out


def main():
    print('=' * 72)
    print('Problem hunt: Lang2Comp v2 vs Equal Wt vs old Lang2Comp')
    print('=' * 72)

    v2_raw = load(V2)
    a0_raw = load(BLEND_ALPHA0)  # pure old Lang2Comp
    a1_raw = load(BLEND_ALPHA1)  # pure Equal Wt

    v2 = seed_avg(v2_raw)
    a0 = seed_avg(a0_raw)
    a1 = seed_avg(a1_raw)

    common = sorted(set(v2) & set(a0) & set(a1))
    print(f'\nRecord counts:')
    print(f'  v2 raw: {len(v2_raw)}   seed-avg prompts: {len(v2)}')
    print(f'  a0 raw: {len(a0_raw)}   seed-avg prompts: {len(a0)}')
    print(f'  a1 raw: {len(a1_raw)}   seed-avg prompts: {len(a1)}')
    print(f'  common prompts: {len(common)}')

    # Seed-count mismatch check
    v2_seeds = sum(v.get("n_seeds", 0) for v in v2.values())
    a0_seeds = sum(v.get("n_seeds", 0) for v in a0.values())
    a1_seeds = sum(v.get("n_seeds", 0) for v in a1.values())
    print(f'\nTotal seeds per eval:')
    print(f'  v2: {v2_seeds}  a0: {a0_seeds}  a1: {a1_seeds}')

    # === Per-prompt comparison ===
    print('\n' + '=' * 72)
    print('Per-prompt head-to-head: v2 vs Equal Wt (alpha=1.0)')
    print('=' * 72)

    v2_win_sym = v2_win_hnc = v2_win_com = 0
    a1_win_sym = a1_win_hnc = a1_win_com = 0
    tied_sym = tied_hnc = tied_com = 0

    sym_diffs = []
    hnc_diffs = []
    com_diffs = []

    worst_sym = []
    for p in common:
        ds = v2[p]['sym'] - a1[p]['sym']
        dh = v2[p]['hnc'] - a1[p]['hnc']
        dc = v2[p]['com'] - a1[p]['com']
        sym_diffs.append(ds)
        hnc_diffs.append(dh)
        com_diffs.append(dc)

        # Tiny tolerance for float noise
        eps = 1e-6
        if ds > eps: v2_win_sym += 1
        elif ds < -eps: a1_win_sym += 1
        else: tied_sym += 1
        if dh > eps: v2_win_hnc += 1
        elif dh < -eps: a1_win_hnc += 1
        else: tied_hnc += 1
        if dc > eps: v2_win_com += 1
        elif dc < -eps: a1_win_com += 1
        else: tied_com += 1

        if ds < -0.0005:
            worst_sym.append((p, ds, v2[p]['sym'], a1[p]['sym'], v2[p]['category']))

    n = len(common)
    print(f'\n  (v2 WIN, Equal WIN, tie) out of {n}:')
    print(f'    sym:  v2={v2_win_sym:3d}  equal={a1_win_sym:3d}  tie={tied_sym:3d}')
    print(f'    HNC:  v2={v2_win_hnc:3d}  equal={a1_win_hnc:3d}  tie={tied_hnc:3d}')
    print(f'    com:  v2={v2_win_com:3d}  equal={a1_win_com:3d}  tie={tied_com:3d}')

    sd = np.array(sym_diffs)
    hd = np.array(hnc_diffs)
    cd = np.array(com_diffs)
    print(f'\n  Per-prompt diff stats (v2 - equal):')
    print(f'    sym: mean={sd.mean():+.5f}  median={np.median(sd):+.5f}  min={sd.min():+.5f}  max={sd.max():+.5f}')
    print(f'    hnc: mean={hd.mean():+.5f}  median={np.median(hd):+.5f}  min={hd.min():+.5f}  max={hd.max():+.5f}')
    print(f'    com: mean={cd.mean():+.4f}  median={np.median(cd):+.4f}  min={cd.min():+.4f}  max={cd.max():+.4f}')

    # === Outlier check: mean vs median gap ===
    print('\n' + '=' * 72)
    print('Outlier hunt: why is sym_mean=89.7% but sym_median=93.5%?')
    print('=' * 72)

    bl_sym = np.array([v2[p]['base_sym'] for p in common])
    mt_sym = np.array([v2[p]['sym']       for p in common])
    pct_gain = (mt_sym - bl_sym) / (np.abs(bl_sym) + 1e-10) * 100

    print(f'\n  Distribution of per-prompt sym gain%:')
    print(f'    mean:   {pct_gain.mean():+.1f}%')
    print(f'    median: {np.median(pct_gain):+.1f}%')
    print(f'    p25:    {np.percentile(pct_gain, 25):+.1f}%')
    print(f'    p75:    {np.percentile(pct_gain, 75):+.1f}%')
    print(f'    min:    {pct_gain.min():+.1f}%')
    print(f'    max:    {pct_gain.max():+.1f}%')

    # Outliers pulling mean down
    bottom5 = np.argsort(pct_gain)[:5]
    print(f'\n  5 worst per-prompt sym gains (pulling mean down):')
    for i in bottom5:
        p = common[i]
        print(f'    {pct_gain[i]:+7.1f}%  base_sym={bl_sym[i]:+.4f}  refined={mt_sym[i]:+.4f}  [{v2[p]["category"]}]  {p[:40]}')

    # === v2 vs a0 (did retraining help at all?) ===
    print('\n' + '=' * 72)
    print('Retraining check: v2 vs old Lang2Comp (a0)')
    print('=' * 72)

    v2_beats_a0_sym = 0
    a0_beats_v2_sym = 0
    for p in common:
        if v2[p]['sym'] > a0[p]['sym'] + 1e-6:
            v2_beats_a0_sym += 1
        elif v2[p]['sym'] < a0[p]['sym'] - 1e-6:
            a0_beats_v2_sym += 1

    print(f'  sym: v2 beats old on {v2_beats_a0_sym}/{n},  old beats v2 on {a0_beats_v2_sym}/{n}')

    # === Category breakdown ===
    print('\n' + '=' * 72)
    print('Per-category (v2 vs Equal Wt) — is v2 worse anywhere?')
    print('=' * 72)

    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_prompts = [p for p in common if v2[p]['category'] == cat]
        if not cat_prompts:
            continue
        sd_cat = np.array([v2[p]['sym'] - a1[p]['sym'] for p in cat_prompts])
        hd_cat = np.array([v2[p]['hnc'] - a1[p]['hnc'] for p in cat_prompts])
        cd_cat = np.array([v2[p]['com'] - a1[p]['com'] for p in cat_prompts])
        print(f'\n  [{cat:12s}]  n={len(cat_prompts)}')
        print(f'    sym diff mean: {sd_cat.mean():+.5f}  wins v2={int((sd_cat>0).sum())}  ties={int((sd_cat==0).sum())}  wins eq={int((sd_cat<0).sum())}')
        print(f'    hnc diff mean: {hd_cat.mean():+.5f}  wins v2={int((hd_cat>0).sum())}  wins eq={int((hd_cat<0).sum())}')
        print(f'    com diff mean: {cd_cat.mean():+.4f}  wins v2={int((cd_cat>0).sum())}  wins eq={int((cd_cat<0).sum())}')

    # === Weight prediction distribution ===
    print('\n' + '=' * 72)
    print('v2 weight predictions — check shape of cloud')
    print('=' * 72)
    def _w3(e):
        w = e['w_raw']
        if isinstance(w, dict):
            return w['symmetry'], w['smoothness'], w['compactness']
        return w[0], w[1], w[2]
    w_sym = np.array([_w3(v2[p])[0] for p in common])
    w_hnc = np.array([_w3(v2[p])[1] for p in common])
    w_com = np.array([_w3(v2[p])[2] for p in common])
    max_w = np.maximum.reduce([w_sym, w_hnc, w_com])
    print(f'  max weight: mean={max_w.mean():.3f}  median={np.median(max_w):.3f}  min={max_w.min():.3f}  max={max_w.max():.3f}')
    print(f'  frac max >= 0.5: {(max_w>=0.5).mean():.0%}')
    print(f'  frac max >= 0.6: {(max_w>=0.6).mean():.0%}')
    print(f'  frac max >= 0.7: {(max_w>=0.7).mean():.0%}')

    # Per-category weight
    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_prompts = [p for p in common if v2[p]['category'] == cat]
        if not cat_prompts:
            continue
        ws = np.mean([_w3(v2[p])[0] for p in cat_prompts])
        wh = np.mean([_w3(v2[p])[1] for p in cat_prompts])
        wc = np.mean([_w3(v2[p])[2] for p in cat_prompts])
        print(f'  [{cat:12s}]  mean w: sym={ws:.3f}  hnc={wh:.3f}  com={wc:.3f}')

    # === Final verdict on aggregate ===
    print('\n' + '=' * 72)
    print('VERDICT')
    print('=' * 72)
    v2_sym_agg = (mt_sym.mean() - bl_sym.mean()) / abs(bl_sym.mean()) * 100
    eq_sym_agg = ((np.array([a1[p]['sym'] for p in common]).mean() -
                   np.array([a1[p]['base_sym'] for p in common]).mean()) /
                  abs(np.array([a1[p]['base_sym'] for p in common]).mean()) * 100)
    a0_sym_agg = ((np.array([a0[p]['sym'] for p in common]).mean() -
                   np.array([a0[p]['base_sym'] for p in common]).mean()) /
                  abs(np.array([a0[p]['base_sym'] for p in common]).mean()) * 100)
    print(f'  sym mean%:  old L2C {a0_sym_agg:+.1f}  ->  v2 {v2_sym_agg:+.1f}  ->  Equal {eq_sym_agg:+.1f}')
    print(f'  v2 improvement over old: {v2_sym_agg - a0_sym_agg:+.1f} pp')
    print(f'  Equal Wt still beats v2: {eq_sym_agg - v2_sym_agg:+.1f} pp')

    print()
    print(f'  Per-prompt H2H sym wins: v2 {v2_win_sym} / Equal {a1_win_sym}')
    print(f'  Per-prompt H2H hnc wins: v2 {v2_win_hnc} / Equal {a1_win_hnc}')
    print(f'  Per-prompt H2H com wins: v2 {v2_win_com} / Equal {a1_win_com}')


if __name__ == '__main__':
    main()
