"""
Experiment J: Clean Evaluation — Remove 7 leaked prompts, recompute all statistics.
CPU only, uses existing all_metrics.json files.
"""
import json, os, sys
import numpy as np
from scipy import stats
from pathlib import Path

LEAKED_PROMPTS = {
    "a dense rock", "a perfectly balanced chair", "a polished sphere",
    "a solid sphere", "a symmetric arch", "a symmetric lamp", "a symmetric vase"
}

METHODS = ["baseline", "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]
METRICS = ["symmetry", "smoothness", "compactness"]
DATA_DIR = Path("results/full")
OUT_DIR = Path("analysis_results/clean_eval")

def load_all_data():
    """Load all method data, return dict[method] -> list of records."""
    data = {}
    for method in METHODS:
        path = DATA_DIR / f"{method}_all_metrics.json"
        with open(path) as f:
            data[method] = json.load(f)
    return data

def filter_clean(records, exclude=True):
    """Filter records: if exclude=True, remove leaked prompts; else keep only leaked."""
    if exclude:
        return [r for r in records if r["prompt"] not in LEAKED_PROMPTS]
    else:
        return [r for r in records if r["prompt"] in LEAKED_PROMPTS]

def paired_ttest(vals_a, vals_b):
    """Paired t-test with Cohen's d."""
    diffs = np.array(vals_a) - np.array(vals_b)
    n = len(diffs)
    if n < 2 or np.std(diffs) == 0:
        return {"t": 0, "p": 1.0, "d": 0, "n": n}
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    cohens_d = np.mean(diffs) / np.std(diffs, ddof=1)
    return {"t": float(t_stat), "p": float(p_val), "d": float(cohens_d), "n": n}

def benjamini_hochberg(p_values, alpha=0.05):
    """BH correction. Returns list of (adjusted_p, significant)."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj_p = min(prev, p * n / rank)
        adjusted[orig_idx] = adj_p
        prev = adj_p
    return [(adj, adj < alpha) for adj in adjusted]

def compute_comparisons(data, label):
    """Compute all pairwise comparisons for a dataset."""
    comparisons = [
        ("DGR vs BL", "diffgeoreward", "baseline"),
        ("HC vs BL", "handcrafted", "baseline"),
        ("DGR vs HC", "diffgeoreward", "handcrafted"),
        ("SYM vs BL", "sym_only", "baseline"),
        ("SMO vs BL", "smooth_only", "baseline"),
        ("COM vs BL", "compact_only", "baseline"),
    ]

    results = {}
    for metric in METRICS:
        for comp_name, method_a, method_b in comparisons:
            key = f"{comp_name}: {metric}"
            # Match by (prompt, seed) pairs
            a_by_key = {(r["prompt"], r["seed"]): r[metric] for r in data[method_a]}
            b_by_key = {(r["prompt"], r["seed"]): r[metric] for r in data[method_b]}
            common_keys = sorted(set(a_by_key.keys()) & set(b_by_key.keys()))

            vals_a = [a_by_key[k] for k in common_keys]
            vals_b = [b_by_key[k] for k in common_keys]

            result = paired_ttest(vals_a, vals_b)
            result["mean_a"] = float(np.mean(vals_a))
            result["mean_b"] = float(np.mean(vals_b))
            result["mean_diff"] = float(np.mean(np.array(vals_a) - np.array(vals_b)))
            results[key] = result

    # BH correction
    all_keys = list(results.keys())
    all_p = [results[k]["p"] for k in all_keys]
    bh_results = benjamini_hochberg(all_p)
    for i, key in enumerate(all_keys):
        results[key]["p_bh"] = bh_results[i][0]
        results[key]["significant_bh"] = bh_results[i][1]

    return results

def per_category_table(data, label):
    """Compute per-category target metric performance (Table 2 style)."""
    # Categories based on prompt keywords
    categories = {}
    for r in data["baseline"]:
        cat = r.get("category", "unknown")
        prompt = r["prompt"]
        if cat not in categories:
            categories[cat] = set()
        categories[cat].add(prompt)

    table = {}
    for cat, prompts in sorted(categories.items()):
        target_metric = cat  # symmetry -> symmetry, etc.
        if target_metric not in METRICS:
            continue

        row = {"category": cat, "n_prompts": len(prompts)}
        for method in METHODS:
            vals = [r[target_metric] for r in data[method] if r["prompt"] in prompts]
            row[f"{method}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{method}_std"] = float(np.std(vals)) if vals else None
            row[f"{method}_n"] = len(vals)

        # Compute improvements vs baseline
        bl_mean = row["baseline_mean"]
        for method in METHODS:
            if method == "baseline":
                continue
            m_mean = row.get(f"{method}_mean")
            if m_mean is not None and bl_mean is not None and abs(bl_mean) > 1e-10:
                row[f"{method}_imp"] = (m_mean - bl_mean) / abs(bl_mean) * 100

        # Paired t-tests for key comparisons on target metric
        for comp_name, ma, mb in [("DGR_vs_HC", "diffgeoreward", "handcrafted"),
                                   ("DGR_vs_BL", "diffgeoreward", "baseline"),
                                   ("HC_vs_BL", "handcrafted", "baseline")]:
            a_by_key = {(r["prompt"], r["seed"]): r[target_metric]
                       for r in data[ma] if r["prompt"] in prompts}
            b_by_key = {(r["prompt"], r["seed"]): r[target_metric]
                       for r in data[mb] if r["prompt"] in prompts}
            common = sorted(set(a_by_key.keys()) & set(b_by_key.keys()))
            if len(common) >= 2:
                va = [a_by_key[k] for k in common]
                vb = [b_by_key[k] for k in common]
                result = paired_ttest(va, vb)
                row[f"{comp_name}_p"] = result["p"]
                row[f"{comp_name}_d"] = result["d"]
                row[f"{comp_name}_n"] = result["n"]

        table[cat] = row

    return table

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    all_data = load_all_data()

    # Full set
    print(f"\n{'='*60}")
    print("FULL SET (110 prompts × 3 seeds = 330)")
    print(f"{'='*60}")
    full_data = all_data
    full_comparisons = compute_comparisons(full_data, "full")
    full_table = per_category_table(full_data, "full")

    # Clean set (exclude leaked)
    print(f"\n{'='*60}")
    print("CLEAN SET (103 prompts × 3 seeds)")
    print(f"{'='*60}")
    clean_data = {m: filter_clean(records, exclude=True) for m, records in all_data.items()}
    for m in METHODS:
        n_clean = len(clean_data[m])
        n_full = len(all_data[m])
        print(f"  {m}: {n_full} -> {n_clean} records ({n_full - n_clean} removed)")

    clean_comparisons = compute_comparisons(clean_data, "clean")
    clean_table = per_category_table(clean_data, "clean")

    # Leaked-only set
    print(f"\n{'='*60}")
    print("LEAKED-ONLY SET (7 prompts × 3 seeds)")
    print(f"{'='*60}")
    leaked_data = {m: filter_clean(records, exclude=False) for m, records in all_data.items()}
    for m in METHODS:
        print(f"  {m}: {len(leaked_data[m])} records")
    leaked_comparisons = compute_comparisons(leaked_data, "leaked")

    # === Print comparison: Full vs Clean ===
    print(f"\n{'='*60}")
    print("COMPARISON: Full vs Clean (key comparisons)")
    print(f"{'='*60}")
    print(f"{'Comparison':<30s} | {'Full p':>10s} {'Full d':>8s} {'Sig':>4s} | {'Clean p':>10s} {'Clean d':>8s} {'Sig':>4s} | {'Stable?':>7s}")
    print("-" * 100)

    stability_results = {}
    for key in sorted(full_comparisons.keys()):
        fc = full_comparisons[key]
        cc = clean_comparisons[key]
        stable = fc["significant_bh"] == cc["significant_bh"]
        stability_results[key] = stable
        print(f"{key:<30s} | {fc['p']:>10.2e} {fc['d']:>8.3f} {'Y' if fc['significant_bh'] else 'N':>4s} | {cc['p']:>10.2e} {cc['d']:>8.3f} {'Y' if cc['significant_bh'] else 'N':>4s} | {'OK' if stable else 'CHANGED':>7s}")

    n_stable = sum(1 for v in stability_results.values() if v)
    n_total = len(stability_results)
    print(f"\nStability: {n_stable}/{n_total} comparisons unchanged after removing leaked prompts")

    # === Print per-category table comparison ===
    print(f"\n{'='*60}")
    print("PER-CATEGORY TABLE: Full vs Clean")
    print(f"{'='*60}")
    for cat in ["symmetry", "smoothness", "compactness"]:
        if cat not in full_table or cat not in clean_table:
            continue
        ft = full_table[cat]
        ct = clean_table[cat]
        print(f"\n  {cat.upper()} (full n={ft.get('diffgeoreward_n', '?')}, clean n={ct.get('diffgeoreward_n', '?')})")
        for method in ["baseline", "diffgeoreward", "handcrafted"]:
            fm = ft.get(f"{method}_mean")
            cm = ct.get(f"{method}_mean")
            fi = ft.get(f"{method}_imp", 0)
            ci = ct.get(f"{method}_imp", 0)
            print(f"    {method:<15s}: full={fm:.6f} ({fi:+.1f}%)  clean={cm:.6f} ({ci:+.1f}%)")

        # P-values
        for comp in ["DGR_vs_HC", "DGR_vs_BL"]:
            fp = ft.get(f"{comp}_p")
            cp = ct.get(f"{comp}_p")
            fd = ft.get(f"{comp}_d")
            cd = ct.get(f"{comp}_d")
            if fp is not None and cp is not None:
                print(f"    {comp}: full p={fp:.4e} d={fd:.3f}, clean p={cp:.4e} d={cd:.3f}")

    # === Leaked-only: Are leaked prompts biased? ===
    print(f"\n{'='*60}")
    print("LEAKED-ONLY: Performance on 7 leaked prompts")
    print(f"{'='*60}")
    for metric in METRICS:
        print(f"\n  {metric}:")
        for method in ["baseline", "diffgeoreward", "handcrafted"]:
            leaked_vals = [r[metric] for r in leaked_data[method]]
            clean_vals = [r[metric] for r in clean_data[method]]
            if leaked_vals and clean_vals:
                print(f"    {method:<15s}: leaked={np.mean(leaked_vals):.6f}, clean={np.mean(clean_vals):.6f}, diff={np.mean(leaked_vals)-np.mean(clean_vals):.6f}")

    # Save results
    report = {
        "leaked_prompts": sorted(LEAKED_PROMPTS),
        "n_full": sum(len(v) for v in all_data.values()),
        "n_clean": sum(len(v) for v in clean_data.values()),
        "n_leaked": sum(len(v) for v in leaked_data.values()),
        "full_comparisons": full_comparisons,
        "clean_comparisons": clean_comparisons,
        "leaked_comparisons": leaked_comparisons,
        "full_table": full_table,
        "clean_table": clean_table,
        "stability": {
            "n_stable": n_stable,
            "n_total": n_total,
            "pct_stable": n_stable / n_total * 100,
            "details": stability_results,
        }
    }

    out_path = OUT_DIR / "clean_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
