"""
Experiment H: Multiple Comparison Correction + Cohen's d Effect Sizes
Reanalyzes existing results with BH correction and standardized effect sizes.
CPU-only: uses existing JSON data.
"""
import os, sys, json, numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FULL = PROJECT_ROOT / "results" / "full"
OUT_DIR = PROJECT_ROOT / "analysis_results" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["symmetry", "smoothness", "compactness"]


def load_metrics(method):
    """Load all_metrics.json for a given method."""
    path = RESULTS_FULL / f"{method}_all_metrics.json"
    with open(path) as f:
        return json.load(f)


def paired_stats(a_list, b_list, metric):
    """Compute paired t-test and Cohen's d between two method results."""
    # Match by (prompt, seed) pairs
    a_dict = {(r["prompt"], r["seed"]): r[metric] for r in a_list}
    b_dict = {(r["prompt"], r["seed"]): r[metric] for r in b_list}

    common_keys = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    if len(common_keys) < 3:
        return None

    a = np.array([a_dict[k] for k in common_keys])
    b = np.array([b_dict[k] for k in common_keys])

    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)

    # Cohen's d for paired samples
    d_mean = diff.mean()
    d_std = diff.std(ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    return {
        "n_pairs": len(common_keys),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "mean_a": float(a.mean()),
        "std_a": float(a.std()),
        "mean_b": float(b.mean()),
        "std_b": float(b.std()),
        "mean_diff": float(d_mean),
        "std_diff": float(d_std),
    }


def bh_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    # BH adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            adjusted[sorted_idx[i]] = sorted_p[i]
        else:
            adjusted[sorted_idx[i]] = min(
                sorted_p[i] * n / rank,
                adjusted[sorted_idx[i + 1]]
            )
    adjusted = np.minimum(adjusted, 1.0)

    rejected = adjusted < alpha
    return adjusted.tolist(), rejected.tolist()


def effect_size_label(d):
    """Cohen's d interpretation."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def main():
    print("Loading metric files...")
    methods = {}
    for m in ["baseline", "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]:
        try:
            methods[m] = load_metrics(m)
            print(f"  {m}: {len(methods[m])} records")
        except FileNotFoundError:
            print(f"  {m}: NOT FOUND, skipping")

    # Define all comparisons
    comparisons = []

    # Core comparisons: DGR vs baseline, HC vs baseline, DGR vs HC
    for metric in METRICS:
        if "diffgeoreward" in methods and "baseline" in methods:
            comparisons.append(("DGR vs BL", "diffgeoreward", "baseline", metric))
        if "handcrafted" in methods and "baseline" in methods:
            comparisons.append(("HC vs BL", "handcrafted", "baseline", metric))
        if "diffgeoreward" in methods and "handcrafted" in methods:
            comparisons.append(("DGR vs HC", "diffgeoreward", "handcrafted", metric))

    # Single-reward vs baseline
    for single in ["sym_only", "smooth_only", "compact_only"]:
        if single in methods and "baseline" in methods:
            for metric in METRICS:
                label = single.replace("_only", "").capitalize()
                comparisons.append((f"{label}-Only vs BL", single, "baseline", metric))

    print(f"\nRunning {len(comparisons)} comparisons...")

    results = {}
    p_values = []

    for label, method_a, method_b, metric in comparisons:
        name = f"{label}: {metric}"
        res = paired_stats(methods[method_a], methods[method_b], metric)
        if res is None:
            print(f"  [SKIP] {name}: insufficient paired data")
            continue
        results[name] = res
        p_values.append(res["p_value"])
        d_label = effect_size_label(res["cohens_d"])
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
        print(f"  {name:40s}  t={res['t_stat']:>8.3f}  p={res['p_value']:.2e}  d={res['cohens_d']:>7.3f} ({d_label:>10s})  {sig}")

    # Apply BH correction
    if p_values:
        print(f"\nApplying Benjamini-Hochberg correction (n={len(p_values)} tests)...")
        adjusted_p, rejected = bh_correction(p_values)

        comparison_names = [n for n in results.keys()]
        for i, name in enumerate(comparison_names):
            results[name]["p_bh"] = adjusted_p[i]
            results[name]["significant_after_bh"] = rejected[i]
            results[name]["effect_size_label"] = effect_size_label(results[name]["cohens_d"])

        n_sig = sum(rejected)
        print(f"  {n_sig}/{len(rejected)} comparisons significant after BH correction (FDR=0.05)")

    # Save full results
    with open(OUT_DIR / "all_comparisons.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Comparison':<42} {'t':>8} {'p':>12} {'p_BH':>12} {'d':>8} {'Effect':>12} {'Sig?':>5}")
    for name, res in results.items():
        sig = "Y" if res.get("significant_after_bh", False) else "N"
        print(f"{name:<42} {res['t_stat']:>8.3f} {res['p_value']:>12.2e} {res.get('p_bh', 0):>12.2e} {res['cohens_d']:>8.3f} {res.get('effect_size_label', ''):>12} {sig:>5}")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
