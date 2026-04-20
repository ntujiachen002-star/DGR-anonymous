"""
Final Rigorous Statistical Analysis — runs LOCALLY after downloading results.

Loads all experiment outputs and produces:
  1. Main table (exp_k): all 6 methods × 3 metrics with BH-FDR, Cohen's d, 95% CI
  2. Laplacian comparison table (exp_n): DGR vs 4 Laplacian configs
  3. CLIP semantic preservation table (exp_m): per-method CLIP scores + FDR
  4. TripoSR backbone table (exp_t): backbone generalization stats
  5. Convergence summary (exp_p): % improvement at steps 0→100
  6. Summary JSON for paper writing

Statistical methodology (NeurIPS standard):
  - Paired t-test (scipy.stats.ttest_rel, two-sided)
  - Benjamini-Hochberg FDR correction (q=0.05) across all within-table comparisons
  - Cohen's d: paired (mean diff / SD of diffs)
  - 95% bootstrap CI: 2000 resamples, percentile method

Usage:
  python tools/exp_analysis_final.py
"""

import json, sys, os
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control   # scipy >= 1.11

os.chdir(Path(__file__).parent.parent)

OUT_DIR = Path("analysis_results/final_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS  = ["symmetry", "smoothness", "compactness"]
ALPHA    = 0.05
N_BOOT   = 2000
BOOT_RNG = np.random.default_rng(0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path):
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] {path}")
        return None
    with open(p) as f:
        return json.load(f)


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Full paired stats: t, p, Cohen's d, bootstrap CI."""
    diff = b - a
    t, p = stats.ttest_rel(b, a)
    d    = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 1e-12 else 0.0
    boot = [BOOT_RNG.choice(diff, len(diff), replace=True).mean()
            for _ in range(N_BOOT)]
    return dict(n=len(a),
                a_mean=float(a.mean()), b_mean=float(b.mean()),
                delta_pct=float((b.mean()-a.mean())/abs(a.mean())*100),
                t=float(t), p_raw=float(p), cohens_d=float(d),
                ci95_lo=float(np.percentile(boot, 2.5)),
                ci95_hi=float(np.percentile(boot, 97.5)))


def bh_correct(comparisons: list[dict]) -> list[dict]:
    """
    Apply BH-FDR correction to a list of comparison dicts containing 'p_raw'.
    Adds 'p_adj' and 'significant' fields.
    """
    if not comparisons:
        return comparisons
    p_vals = np.array([c["p_raw"] for c in comparisons])
    try:
        # scipy >= 1.11
        p_adj = false_discovery_control(p_vals, method='bh')
    except Exception:
        # Manual BH fallback
        n = len(p_vals)
        order = np.argsort(p_vals)
        p_adj = np.empty(n)
        for rank, idx in enumerate(order):
            p_adj[idx] = min(p_vals[idx] * n / (rank + 1), 1.0)
        # enforce monotonicity (reverse)
        for i in range(n-2, -1, -1):
            if p_adj[order[i]] > p_adj[order[i+1]]:
                p_adj[order[i]] = p_adj[order[i+1]]

    for c, pa in zip(comparisons, p_adj):
        c["p_adj"] = float(pa)
        c["significant"] = bool(pa < ALPHA)
    return comparisons


def filter_paired(records, method_a, method_b, metric):
    """Extract matched (a, b) arrays for paired comparison."""
    a_d = {(r["prompt"], r["seed"]): r[metric]
           for r in records if r["method"] == method_a and r.get(metric) is not None}
    b_d = {(r["prompt"], r["seed"]): r[metric]
           for r in records if r["method"] == method_b and r.get(metric) is not None}
    keys = set(a_d) & set(b_d)
    a = np.array([a_d[k] for k in keys])
    b = np.array([b_d[k] for k in keys])
    return a, b


def print_table(title, rows, cols, fmt=None):
    fmt = fmt or {}
    print(f"\n{'='*70}")
    print(title)
    print("="*70)
    col_w = {c: max(len(c), 10) for c in cols}
    header = " | ".join(f"{c:>{col_w[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = " | ".join(
            f"{row.get(c, '--'):>{col_w[c]}}" if isinstance(row.get(c, '--'), str)
            else f"{row.get(c, float('nan')):{fmt.get(c, '>10.4f')}}"
            for c in cols
        )
        print(line)


# ── Table 1: Main results (exp_n = all geo metrics) ───────────────────────────

def table_main(geo_data):
    """
    Compare all methods vs baseline on geometric metrics.
    Source: analysis_results/laplacian_vs_dgr/all_results.json
    which (after P1a fix) contains ALL methods:
      exp_k: baseline, diffgeoreward, handcrafted, sym_only, smooth_only, compact_only
      Laplacian: lap_light, lap_medium, lap_strong, lap_extreme
    BH-FDR applied per metric across all comparisons in that metric block.
    """
    if not geo_data:
        print("\n[SKIP] Geometric data not available (run exp_n_from_expk.py first)")
        return []

    print("\n\n" + "="*70)
    print("TABLE 1 — Geometric Metrics: All Methods vs Baseline")
    print("Source: analysis_results/laplacian_vs_dgr/all_results.json")
    print("BH-FDR (q=0.05) per metric block; Cohen's d (paired); 95% bootstrap CI")
    print("="*70)

    methods_ordered = [
        # exp_k weight ablations
        "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only",
        # Laplacian baselines
        "lap_light", "lap_medium", "lap_strong", "lap_extreme",
    ]

    all_comparisons = []

    for metric in METRICS:
        print(f"\n  --- {metric.upper()} ---")

        bl = np.array([r[metric] for r in geo_data
                       if r["method"] == "baseline" and r.get(metric) is not None])
        if len(bl) == 0:
            print("    No baseline data — check that exp_n_from_expk.py has run.")
            continue
        print(f"    Baseline: {bl.mean():.5f} ± {bl.std():.5f}  (n={len(bl)})")

        comparisons = []
        for method in methods_ordered:
            a, b = filter_paired(geo_data, "baseline", method, metric)
            if len(a) < 5:
                continue
            s = paired_stats(a, b)
            exp_tag = "exp_n" if method.startswith("lap_") else "exp_k"
            s.update(method=method, metric=metric, experiment=exp_tag)
            comparisons.append(s)

        # BH-FDR across all comparisons within this metric block
        comparisons = bh_correct(comparisons)
        all_comparisons.extend(comparisons)

        print(f"    {'Method':<18} | {'Δ%':>7} | {'d':>6} | "
              f"{'p_adj':>10} | {'sig':>5} | 95% CI")
        print(f"    {'-'*70}")
        for c in comparisons:
            sig = "✓" if c["significant"] else "✗"
            print(f"    {c['method']:<18} | {c['delta_pct']:>+6.1f}% | "
                  f"{c['cohens_d']:>6.3f} | {c['p_adj']:>10.4e} | "
                  f"{sig:>5} | [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}]")

    return all_comparisons


# ── Table 2: CLIP scores (exp_m) ──────────────────────────────────────────────

def table_clip(clip_data):
    if not clip_data:
        print("\n[SKIP] CLIP data not available")
        return []

    print("\n\n" + "="*70)
    print("TABLE 2 — CLIP Semantic Preservation (ViT-B/32, 4-view)")
    print("n = 330 per method  |  BH-FDR across 5 comparisons vs baseline")
    print("="*70)

    methods = ["diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]
    bl_d = {(r["prompt"], r["seed"]): r["clip_score"]
            for r in clip_data if r["method"] == "baseline"
            and r.get("clip_score") is not None}

    comparisons = []
    for method in methods:
        m_d = {(r["prompt"], r["seed"]): r["clip_score"]
               for r in clip_data if r["method"] == method
               and r.get("clip_score") is not None}
        keys = set(bl_d) & set(m_d)
        if len(keys) < 5:
            continue
        a = np.array([bl_d[k] for k in keys])
        b = np.array([m_d[k]  for k in keys])
        s = paired_stats(a, b)
        s.update(method=method, metric="clip_score")
        comparisons.append(s)

    comparisons = bh_correct(comparisons)

    bl_scores = list(bl_d.values())
    print(f"\n  Baseline CLIP: {np.mean(bl_scores):.4f} ± {np.std(bl_scores):.4f}  (n={len(bl_scores)})")
    print(f"  {'Method':<18} | {'Mean':>8} | {'Δ%':>7} | {'d':>6} | {'p_adj':>10} | {'sig':>5}")
    print(f"  {'-'*65}")
    for c in comparisons:
        sig = "✓" if c["significant"] else "✗"
        print(f"  {c['method']:<18} | {c['b_mean']:>8.4f} | "
              f"{c['delta_pct']:>+6.1f}% | {c['cohens_d']:>6.3f} | "
              f"{c['p_adj']:>10.4e} | {sig:>5}")

    print("\n  NOTE: p > 0.05 (not sig) means NO significant CLIP degradation.")
    return comparisons


# ── Table 3: TripoSR backbone (exp_t) ─────────────────────────────────────────

def table_triposr(tsr_data):
    if not tsr_data:
        print("\n[SKIP] TripoSR data not available")
        return []

    print("\n\n" + "="*70)
    print("TABLE 3 — TripoSR Backbone Generalization")
    print("n = 90 (30 prompts × 3 seeds)  |  DGR equal weights [1/3,1/3,1/3]")
    print("Paired t-test + Cohen's d + 95% bootstrap CI")
    print("="*70)

    comparisons = []
    for metric in METRICS:
        a, b = filter_paired(tsr_data, "triposr_baseline", "triposr_dgr", metric)
        if len(a) < 5:
            continue
        s = paired_stats(a, b)
        s.update(method="triposr_dgr", metric=metric)
        comparisons.append(s)

    # Single comparison per metric → no FDR needed (3 metrics, Bonferroni for completeness)
    # Still apply BH for consistency
    comparisons = bh_correct(comparisons)

    print(f"\n  {'Metric':<12} | {'BL':>9} | {'DGR':>9} | "
          f"{'Δ%':>7} | {'d':>6} | {'p_adj':>10} | 95% CI")
    print(f"  {'-'*70}")
    for c in comparisons:
        sig = "✓" if c["significant"] else "✗"
        print(f"  {c['metric']:<12} | {c['a_mean']:>9.5f} | {c['b_mean']:>9.5f} | "
              f"{c['delta_pct']:>+6.1f}% | {c['cohens_d']:>6.3f} | "
              f"{c['p_adj']:>10.4e} | [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] {sig}")

    return comparisons


# ── Table 4: Convergence (exp_p) ──────────────────────────────────────────────

def table_convergence(conv_data):
    if not conv_data:
        print("\n[SKIP] Convergence data not available")
        return

    print("\n\n" + "="*70)
    print("TABLE 4 — DGR Convergence Curve (lr=0.005, equal weights)")
    print("="*70)

    conv = [r for r in conv_data if r.get("experiment") == "convergence"]
    if not conv:
        print("  No convergence records.")
        return

    steps = sorted(set(r["step"] for r in conv))
    bl_step0 = {m: np.mean([r[m] for r in conv if r["step"] == 0 and r.get(m) is not None])
                for m in METRICS}

    print(f"\n  {'Step':>5} | {'Sym Δ%':>8} | {'Smo Δ%':>8} | {'Com Δ%':>8} | N")
    print(f"  {'-'*45}")
    for step in steps:
        recs = [r for r in conv if r["step"] == step]
        row  = []
        for m in METRICS:
            vals = [r[m] for r in recs if r.get(m) is not None]
            if vals and abs(bl_step0[m]) > 1e-12:
                delta = (np.mean(vals) - bl_step0[m]) / abs(bl_step0[m]) * 100
                row.append(f"{delta:>+7.1f}%")
            else:
                row.append("    --  ")
        n = len([r for r in recs if r.get("symmetry") is not None])
        step_lbl = "0 (BL)" if step == 0 else str(step)
        print(f"  {step_lbl:>6} | {row[0]} | {row[1]} | {row[2]} | {n}")

    lr_data = [r for r in conv_data if r.get("experiment") == "lr_sensitivity"
               and r["lr"] != 0.0]
    if lr_data:
        bl_lr = {m: np.mean([r[m] for r in conv_data
                              if r.get("experiment") == "lr_sensitivity"
                              and r["lr"] == 0.0 and r.get(m) is not None])
                 for m in METRICS}
        print(f"\n  LR Sensitivity (step={max(r['step'] for r in lr_data)}):")
        print(f"  {'LR':>8} | {'Sym Δ%':>8} | {'Smo Δ%':>8} | {'Com Δ%':>8} | N")
        print(f"  {'-'*45}")
        for lr_val in sorted(set(r["lr"] for r in lr_data)):
            recs = [r for r in lr_data if r["lr"] == lr_val]
            row  = []
            for m in METRICS:
                vals = [r[m] for r in recs if r.get(m) is not None]
                if vals and abs(bl_lr.get(m, 0)) > 1e-12:
                    delta = (np.mean(vals) - bl_lr[m]) / abs(bl_lr[m]) * 100
                    row.append(f"{delta:>+7.1f}%")
                else:
                    row.append("    --  ")
            n = len(recs)
            print(f"  {lr_val:>8.4f} | {row[0]} | {row[1]} | {row[2]} | {n}")


# ── Summary JSON ──────────────────────────────────────────────────────────────

def build_summary(main_comps, clip_comps, tsr_comps):
    summary = {
        "statistical_method": {
            "test":             "paired t-test (scipy.stats.ttest_rel, two-sided)",
            "correction":       "Benjamini-Hochberg FDR (q=0.05)",
            "effect_size":      "Cohen's d (paired: mean_diff / SD_diff)",
            "confidence_interval": "95% bootstrap (n_boot=2000, percentile)",
        },
        "main_results":  main_comps,
        "clip_results":  clip_comps,
        "triposr_results": tsr_comps,
    }
    # Key numbers for paper
    dgr_main = {c["metric"]: c for c in main_comps if c.get("method") == "diffgeoreward"}
    summary["key_numbers"] = {
        m: {
            "delta_pct": dgr_main[m]["delta_pct"] if m in dgr_main else None,
            "cohens_d":  dgr_main[m]["cohens_d"]  if m in dgr_main else None,
            "p_adj":     dgr_main[m]["p_adj"]      if m in dgr_main else None,
            "significant": dgr_main[m]["significant"] if m in dgr_main else None,
        }
        for m in METRICS
    }
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading experiment results...")

    # exp_n: ALL geometric metrics — baseline, 5 exp_k methods, 4 Laplacian variants
    # Source: exp_n_from_expk.py (after P1a fix includes all 6 exp_k methods)
    expn = load("analysis_results/laplacian_vs_dgr/all_results.json")

    # exp_m: CLIP semantic preservation scores (all 6 exp_k methods)
    expm = load("analysis_results/clip_allmethod/all_results.json")

    # exp_t: TripoSR backbone generalization (30 prompts × 3 seeds × 2 methods)
    expt = load("analysis_results/triposr_backbone/all_results.json")

    # exp_p: convergence curve + LR sensitivity
    expp = load("analysis_results/convergence/all_results.json")

    def n_or_missing(d):
        return len(d) if d else "MISSING"

    # Report available methods in expn to verify the P1a fix worked
    if expn:
        methods_present = sorted(set(r["method"] for r in expn))
        print(f"  exp_n ({len(expn)} records): methods = {methods_present}")
    else:
        print("  exp_n: MISSING — run exp_n_from_expk.py first")
    print(f"  exp_m (clip):        {n_or_missing(expm)} records")
    print(f"  exp_t (triposr):     {n_or_missing(expt)} records")
    print(f"  exp_p (convergence): {n_or_missing(expp)} records")

    # Tables — expn is the single source for all geo metrics
    main_comps = table_main(expn)
    clip_comps = table_clip(expm)
    tsr_comps  = table_triposr(expt)
    table_convergence(expp)

    # Summary
    summary = build_summary(main_comps, clip_comps, tsr_comps)
    out_path = OUT_DIR / "stats_summary.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*70}")
    print("KEY NUMBERS FOR PAPER (DiffGeoReward vs Baseline)")
    print("="*70)
    for metric, kn in summary["key_numbers"].items():
        if kn["delta_pct"] is not None:
            sig = "✓ significant" if kn["significant"] else "✗ not sig"
            print(f"  {metric:<12}: Δ={kn['delta_pct']:+.1f}%  d={kn['cohens_d']:.3f}  "
                  f"p_adj={kn['p_adj']:.4e}  {sig}")

    print(f"\nFull stats: {out_path}")
    print("Paste into paper methods section:")
    print("  Paired t-test (two-sided), BH-FDR correction (q=0.05),")
    print("  Cohen's d (paired), 95% bootstrap CI (n=2000).")


if __name__ == "__main__":
    main()
