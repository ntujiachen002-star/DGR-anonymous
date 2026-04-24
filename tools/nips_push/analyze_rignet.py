"""Paired statistical analysis + LaTeX table for RigNet downstream experiment.

Reads:  analysis_results/nips_push_rignet/all_results.json
Writes: analysis_results/nips_push_rignet/summary.json
        analysis_results/nips_push_rignet/per_metric_table.tex
        stdout summary

Computes:
    - Coverage rate (baseline vs DGR), chi-square test
    - For pairs where BOTH succeed: paired t-test and Wilcoxon on M1-M3
    - Effect size (Cohen's d, paired)
    - Per-category breakdown (if enough pairs)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


OUT_DIR = Path("analysis_results/nips_push_rignet")
RESULTS_PATH = OUT_DIR / "all_results.json"


def load_results():
    if not RESULTS_PATH.exists():
        sys.exit(f"Missing {RESULTS_PATH}; run exp_rignet_downstream.py first.")
    with open(RESULTS_PATH) as f:
        return json.load(f)


def paired_stats(bl_vals, dgr_vals, lower_better=True):
    """Return dict of paired stats. lower_better: True for JSE/offset/angular_dev."""
    bl = np.asarray(bl_vals, dtype=float)
    dg = np.asarray(dgr_vals, dtype=float)
    mask = np.isfinite(bl) & np.isfinite(dg)
    bl, dg = bl[mask], dg[mask]
    if len(bl) < 3:
        return {"n": int(len(bl)), "too_few": True}
    diff = dg - bl
    # For lower-is-better metrics, improvement = reduction, so pct is negative of relative change
    pct = (dg.mean() - bl.mean()) / (abs(bl.mean()) + 1e-12) * 100
    t, p_t = stats.ttest_rel(dg, bl)
    try:
        w, p_w = stats.wilcoxon(dg, bl)
    except Exception:
        w, p_w = float("nan"), float("nan")
    d = diff.mean() / (diff.std(ddof=1) + 1e-12) if len(diff) > 1 else 0.0
    # Win rate: how many meshes DGR improved over baseline
    if lower_better:
        wins = int((dg < bl).sum())
    else:
        wins = int((dg > bl).sum())
    return {
        "n": int(len(bl)),
        "bl_mean": float(bl.mean()), "bl_std": float(bl.std(ddof=1)),
        "dgr_mean": float(dg.mean()), "dgr_std": float(dg.std(ddof=1)),
        "diff_mean": float(diff.mean()),
        "pct": float(pct),
        "t": float(t), "p_t": float(p_t),
        "wilcoxon_w": float(w), "p_w": float(p_w),
        "cohens_d_paired": float(d),
        "win_rate_pct": 100.0 * wins / len(bl),
        "wins": wins, "total": int(len(bl)),
    }


def main():
    results = load_results()
    print(f"[info] Loaded {len(results)} pairs from {RESULTS_PATH}\n")

    # --- Coverage rate (M4) ---
    bl_success = [1 if r["baseline"]["success"] else 0 for r in results]
    dgr_success = [1 if r["dgr"]["success"] else 0 for r in results]
    n = len(results)
    bl_cov = sum(bl_success); dg_cov = sum(dgr_success)
    # McNemar's test for paired binary outcomes
    b_only = sum(b and not d for b, d in zip(bl_success, dgr_success))
    d_only = sum(not b and d for b, d in zip(bl_success, dgr_success))
    if b_only + d_only >= 1:
        # exact McNemar (binomial)
        from scipy.stats import binomtest
        bt = binomtest(min(b_only, d_only), n=b_only + d_only, p=0.5)
        p_mc = bt.pvalue
    else:
        p_mc = float("nan")
    print(f"=== M4 RigNet coverage rate ===")
    print(f"  baseline: {bl_cov}/{n} ({100.0*bl_cov/n:.1f}%)")
    print(f"  DGR:      {dg_cov}/{n} ({100.0*dg_cov/n:.1f}%)")
    print(f"  McNemar paired p={p_mc:.3e} (baseline-only={b_only}, DGR-only={d_only})")
    print()

    # --- Pairs where both succeeded ---
    both = [r for r in results if r["baseline"]["success"] and r["dgr"]["success"]]
    print(f"=== Paired metrics on {len(both)}/{n} mutually-successful pairs ===")

    metric_specs = [
        ("JSE", "JSE (bbox-normalized)", True),
        ("angular_dev_deg", "Bone-plane angular deviation (deg)", True),
        ("root_offset", "Root-joint plane offset (bbox-normalized)", True),
    ]
    summary = {"n_pairs_total": n, "coverage": {
        "baseline_success": bl_cov, "dgr_success": dg_cov,
        "mcnemar_p": p_mc, "baseline_only": b_only, "dgr_only": d_only,
    }, "n_both_succeeded": len(both), "metrics": {}}

    for key, label, lower_better in metric_specs:
        bl_vals = [r["baseline"][key] for r in both]
        dg_vals = [r["dgr"][key] for r in both]
        st = paired_stats(bl_vals, dg_vals, lower_better=lower_better)
        summary["metrics"][key] = st
        if st.get("too_few"):
            print(f"  {label}: only {st['n']} pairs, skipped")
            continue
        arrow = "↓" if lower_better else "↑"
        sig = " *"
        if st["p_t"] < 1e-4:
            sig = " ***"
        elif st["p_t"] < 1e-2:
            sig = " **"
        elif st["p_t"] >= 0.05:
            sig = ""
        print(f"  {label} {arrow}: n={st['n']}  "
              f"bl={st['bl_mean']:.4f}  dgr={st['dgr_mean']:.4f}  "
              f"Δ={st['diff_mean']:+.4f} ({st['pct']:+.1f}%)  "
              f"p_t={st['p_t']:.2e} d={st['cohens_d_paired']:+.2f} "
              f"win={st['win_rate_pct']:.1f}%{sig}")

    # --- LaTeX snippet ---
    latex_rows = []
    for key, label, lower_better in metric_specs:
        st = summary["metrics"].get(key, {})
        if st.get("too_few"):
            continue
        row = (
            f"{label} $\\downarrow$ & "
            f"{st['n']} & "
            f"{st['bl_mean']:.4f} & "
            f"{st['dgr_mean']:.4f} & "
            f"{st['pct']:+.1f}\\% & "
            f"{st['p_t']:.2e} & "
            f"{st['cohens_d_paired']:+.2f} & "
            f"{st['win_rate_pct']:.0f}\\% \\\\"
        )
        latex_rows.append(row)
    latex = [
        "% Auto-generated by tools/nips_push/analyze_rignet.py",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{RigNet downstream validation: three non-circular skeleton metrics on $n$ mesh pairs where RigNet succeeded on both baseline and \\method{{}}-refined outputs. All three metrics lower $=$ better (more bilaterally consistent skeleton). Coverage: baseline {bl_cov}/{n}, \\method{{}} {dg_cov}/{n} (McNemar $p={p_mc:.2e}$).}}",
        "\\label{tab:rignet_downstream}",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Metric & $n$ & Baseline & DGR & $\\Delta$ & $p$ & $d$ & Win \\\\",
        "\\midrule",
        *latex_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    tex_path = OUT_DIR / "per_metric_table.tex"
    tex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(f"\n[tex] {tex_path}")

    # --- Per-category breakdown ---
    # Categorize prompts (cheap heuristic)
    def prompt_cat(p):
        p = p.lower()
        if any(k in p for k in ["butterfly", "spider", "bird", "eagle", "rabbit",
                                 "dog", "cat", "fish", "penguin", "elephant",
                                 "horse", "owl", "lion", "creature", "insect"]):
            return "animal"
        if any(k in p for k in ["chess", "king", "queen", "knight", "bishop",
                                 "person", "figure", "face", "human", "angel",
                                 "statue", "body"]):
            return "humanoid"
        if any(k in p for k in ["chandelier", "candelabra", "chair", "coat hanger",
                                 "picture frame", "window", "door"]):
            return "structure"
        return "object"

    summary["per_category"] = {}
    if both:
        cats = {}
        for r in both:
            cats.setdefault(prompt_cat(r["prompt"]), []).append(r)
        print(f"\n=== Per-category breakdown ({len(cats)} categories) ===")
        for cat, rs in sorted(cats.items(), key=lambda x: -len(x[1])):
            row = {"n": len(rs)}
            for key, _, _ in metric_specs:
                bl_v = [r["baseline"][key] for r in rs]
                dg_v = [r["dgr"][key] for r in rs]
                s = paired_stats(bl_v, dg_v, lower_better=True)
                row[key] = s
            summary["per_category"][cat] = row
            print(f"  {cat:<10s} n={row['n']:3d}  JSE {row['JSE'].get('pct', 0):+.1f}%  "
                  f"Ang {row['angular_dev_deg'].get('pct', 0):+.1f}%  "
                  f"Root {row['root_offset'].get('pct', 0):+.1f}%")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[json] {OUT_DIR / 'summary.json'}")

    # --- KEY CLAIM VERDICT ---
    print("\n" + "=" * 60)
    print("KEY CLAIMS FOR PAPER:")
    jse = summary["metrics"].get("JSE", {})
    if jse and not jse.get("too_few"):
        verdict = "SUPPORTED" if jse["pct"] < -5 and jse["p_t"] < 0.05 else \
                  ("NULL" if jse["p_t"] >= 0.05 else "WEAK")
        print(f"  JSE reduced by {-jse['pct']:.1f}% (p={jse['p_t']:.2e}, "
              f"d={jse['cohens_d_paired']:+.2f}) [{verdict}]")
    cov_lift = 100.0 * (dg_cov - bl_cov) / n
    print(f"  Coverage rate: +{cov_lift:.1f} pp (McNemar p={p_mc:.2e})")
    print("=" * 60)


if __name__ == "__main__":
    main()
