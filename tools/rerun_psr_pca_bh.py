"""Recompute Wilcoxon p-values with Benjamini-Hochberg correction on the
per-category breakdowns of the PSR self-consistency and PCA axis-stability
downstream tests.

This reads the raw results JSONs produced by exp_psr_selfconsistency.py and
exp_pca_stability.py, applies BH correction across the three per-category
tests (plus aggregate where applicable), and prints a corrected summary.
"""
import os
import json
import numpy as np
from scipy import stats


def multipletests_bh(pvals):
    """Simple Benjamini-Hochberg FDR correction without statsmodels."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.argsort(order) + 1
    adj = p * n / ranks
    sorted_adj = adj[order]
    sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    adj_out = np.empty_like(adj)
    adj_out[order] = np.clip(sorted_adj, 0.0, 1.0)
    return adj_out.tolist()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSR_RESULTS = os.path.join(ROOT, 'analysis_results/psr_selfconsistency/psr_results.json')
PCA_RESULTS = os.path.join(ROOT, 'analysis_results/pca_stability/pca_results.json')


def bh_adjust(pvals, method='fdr_bh'):
    return multipletests_bh(pvals)


def psr_bh():
    with open(PSR_RESULTS) as f:
        results = json.load(f)
    categories = ['symmetry', 'smoothness', 'compactness']
    rows = []
    raw_ps = []
    for cat in categories:
        cat_res = [r for r in results if r['category'] == cat]
        if len(cat_res) < 3:
            continue
        bl = np.array([r['bl_self_cd'] for r in cat_res])
        rf = np.array([r['rf_self_cd'] for r in cat_res])
        try:
            _, p = stats.wilcoxon(bl, rf)
        except Exception:
            p = 1.0
        imp = (1 - rf.mean() / bl.mean()) * 100
        wr = (rf < bl).mean() * 100
        rows.append({'cat': cat, 'n': len(cat_res),
                     'bl': float(bl.mean()), 'rf': float(rf.mean()),
                     'imp_pct': float(imp), 'wr': float(wr),
                     'p_raw': float(p)})
        raw_ps.append(p)
    p_adj = bh_adjust(raw_ps)
    for row, p_a in zip(rows, p_adj):
        row['p_bh'] = float(p_a)
    print('=== PSR Self-Chamfer per-category with BH correction ===')
    for row in rows:
        print(f"  [{row['cat']:12s}] n={row['n']:3d}  imp={row['imp_pct']:+6.2f}%  wr={row['wr']:5.1f}%  "
              f"p_raw={row['p_raw']:.2e}  p_bh={row['p_bh']:.2e}")
    return rows


def pca_bh():
    with open(PCA_RESULTS) as f:
        results = json.load(f)
    categories = ['symmetry', 'smoothness', 'compactness']
    rows = []
    raw_ps = []
    for cat in categories:
        cat_res = [r for r in results if r['category'] == cat]
        if len(cat_res) < 3:
            continue
        bl = np.array([r['bl_mean_dev'] for r in cat_res])
        rf = np.array([r['rf_mean_dev'] for r in cat_res])
        try:
            _, p = stats.wilcoxon(bl, rf)
        except Exception:
            p = 1.0
        imp = (1 - rf.mean() / bl.mean()) * 100
        wr = (rf < bl).mean() * 100
        rows.append({'cat': cat, 'n': len(cat_res),
                     'bl': float(bl.mean()), 'rf': float(rf.mean()),
                     'imp_pct': float(imp), 'wr': float(wr),
                     'p_raw': float(p)})
        raw_ps.append(p)
    p_adj = bh_adjust(raw_ps)
    for row, p_a in zip(rows, p_adj):
        row['p_bh'] = float(p_a)
    print('\n=== PCA Axis-Stability per-category with BH correction ===')
    for row in rows:
        print(f"  [{row['cat']:12s}] n={row['n']:3d}  imp={row['imp_pct']:+6.2f}%  wr={row['wr']:5.1f}%  "
              f"p_raw={row['p_raw']:.2e}  p_bh={row['p_bh']:.2e}")
    return rows


if __name__ == '__main__':
    psr_rows = psr_bh()
    pca_rows = pca_bh()
    out = {'psr_bh': psr_rows, 'pca_bh': pca_rows}
    out_path = os.path.join(ROOT, 'analysis_results/psr_pca_bh.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {out_path}')
