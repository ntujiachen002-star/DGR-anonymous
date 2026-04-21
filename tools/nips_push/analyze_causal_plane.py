"""[P0-1] Analysis + figure generation for the causal plane experiment.

Reads: analysis_results/nips_push_causal_plane/all_results.json
Writes:
    analysis_results/nips_push_causal_plane/summary.json
    analysis_results/nips_push_causal_plane/fig_plane_vs_pcgrad.pdf (and .png)
    analysis_results/nips_push_causal_plane/fig_plane_vs_cosine.pdf (and .png)
    analysis_results/nips_push_causal_plane/per_method_table.tex
    stdout: summary table + correlations

Produces the causal evidence:
    Spearman(plane_error, PCGrad_benefit_sym) > 0 with p < 0.05 would support
    the central claim that plane misestimation *causes* the apparent conflict.

CPU only, < 1 minute.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path("analysis_results/nips_push_causal_plane")
RESULTS_PATH = OUT_DIR / "all_results.json"

PLANE_ORDER = ["fixed_xz", "best_of_3", "pca_single", "multi_start"]
PLANE_LABELS = {
    "fixed_xz": "Fixed xz",
    "best_of_3": "Best-of-3 axes",
    "pca_single": "PCA",
    "multi_start": "Multi-start (ref)",
}


def load_records():
    if not RESULTS_PATH.exists():
        sys.exit(f"Missing {RESULTS_PATH}; run exp_causal_plane.py first.")
    with open(RESULTS_PATH) as f:
        records = json.load(f)
    records = [r for r in records if not r.get("skipped")]
    print(f"[info] Loaded {len(records)} non-skipped meshes")
    return records


def per_method_summary(records):
    """Mean statistics across meshes for each plane method."""
    rows = []
    for m in PLANE_ORDER:
        pe_ang = np.array([r["plane_methods"][m]["plane_angular_err_deg"]
                           for r in records if m in r["plane_methods"]])
        pe_gap = np.array([r["plane_methods"][m]["plane_sym_gap"]
                           for r in records if m in r["plane_methods"]])
        os_i = np.array([r["plane_methods"][m]["own_sym_init"]
                         for r in records if m in r["plane_methods"]])
        pc_ben = np.array([r["plane_methods"][m]["pcgrad_benefit_sym_own"]
                           for r in records if m in r["plane_methods"]])
        pc_ben_ref = np.array([r["plane_methods"][m]["pcgrad_benefit_sym_ref"]
                               for r in records if m in r["plane_methods"]])
        # Grad cosine (if recorded)
        has_cos = "mean_cos_sym_smo" in records[0]["plane_methods"][m]
        if has_cos:
            c_ss = np.array([r["plane_methods"][m]["mean_cos_sym_smo"]
                             for r in records if m in r["plane_methods"]])
            c_sc = np.array([r["plane_methods"][m]["mean_cos_sym_com"]
                             for r in records if m in r["plane_methods"]])
            pct_neg_ss = np.array([r["plane_methods"][m]["pct_neg_sym_smo"]
                                   for r in records if m in r["plane_methods"]])
            pct_neg_sc = np.array([r["plane_methods"][m]["pct_neg_sym_com"]
                                   for r in records if m in r["plane_methods"]])
        else:
            c_ss = c_sc = pct_neg_ss = pct_neg_sc = None

        rows.append({
            "plane": m,
            "n": len(pe_ang),
            "plane_angular_err_deg_mean": float(pe_ang.mean()),
            "plane_sym_gap_mean": float(pe_gap.mean()),
            "own_sym_init_mean": float(os_i.mean()),
            "pcgrad_benefit_sym_own_mean": float(pc_ben.mean()),
            "pcgrad_benefit_sym_ref_mean": float(pc_ben_ref.mean()),
            "mean_cos_sym_smo": float(c_ss.mean()) if has_cos else None,
            "mean_cos_sym_com": float(c_sc.mean()) if has_cos else None,
            "pct_neg_sym_smo_mean": float(pct_neg_ss.mean()) if has_cos else None,
            "pct_neg_sym_com_mean": float(pct_neg_sc.mean()) if has_cos else None,
        })
    return rows


def causal_correlations(records):
    """Pool over (mesh, plane) pairs and compute correlations."""
    pool = {"ang_err": [], "sym_gap": [], "pc_benefit_sym_own": [], "pc_benefit_sym_ref": [],
            "mean_cos_sym_smo": [], "mean_cos_sym_com": [],
            "pct_neg_sym_smo": [], "pct_neg_sym_com": []}
    for r in records:
        for m, d in r["plane_methods"].items():
            pool["ang_err"].append(d["plane_angular_err_deg"])
            pool["sym_gap"].append(d["plane_sym_gap"])
            pool["pc_benefit_sym_own"].append(d["pcgrad_benefit_sym_own"])
            pool["pc_benefit_sym_ref"].append(d["pcgrad_benefit_sym_ref"])
            pool["mean_cos_sym_smo"].append(d.get("mean_cos_sym_smo", float("nan")))
            pool["mean_cos_sym_com"].append(d.get("mean_cos_sym_com", float("nan")))
            pool["pct_neg_sym_smo"].append(d.get("pct_neg_sym_smo", float("nan")))
            pool["pct_neg_sym_com"].append(d.get("pct_neg_sym_com", float("nan")))
    pool = {k: np.array(v) for k, v in pool.items()}

    out = {}
    for err_key in ["ang_err", "sym_gap"]:
        for tgt in ["pc_benefit_sym_own", "pc_benefit_sym_ref",
                    "mean_cos_sym_smo", "mean_cos_sym_com",
                    "pct_neg_sym_smo", "pct_neg_sym_com"]:
            x, y = pool[err_key], pool[tgt]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            out[f"{err_key}_vs_{tgt}"] = {
                "rho": float(rho), "p": float(p), "n": int(mask.sum()),
            }
    return out, pool


def make_figures(records, pool):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping figures")
        return

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 150})

    # Figure 1: plane angular error vs PCGrad benefit
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.0))
    colors = {"fixed_xz": "#d62728", "best_of_3": "#ff7f0e",
              "pca_single": "#2ca02c", "multi_start": "#1f77b4"}
    markers = {"fixed_xz": "x", "best_of_3": "s", "pca_single": "^", "multi_start": "o"}
    for m in PLANE_ORDER:
        xs, ys = [], []
        for r in records:
            if m in r["plane_methods"]:
                xs.append(r["plane_methods"][m]["plane_angular_err_deg"])
                ys.append(r["plane_methods"][m]["pcgrad_benefit_sym_own"])
        ax.scatter(xs, ys, c=colors[m], marker=markers[m], s=18, alpha=0.6,
                   label=PLANE_LABELS[m])
    # Overall trend line
    mask = np.isfinite(pool["ang_err"]) & np.isfinite(pool["pc_benefit_sym_own"])
    if mask.sum() > 5:
        xs_all = pool["ang_err"][mask]
        ys_all = pool["pc_benefit_sym_own"][mask]
        rho, p = stats.spearmanr(xs_all, ys_all)
        # Linear fit for visual
        slope, intercept = np.polyfit(xs_all, ys_all, 1)
        xg = np.linspace(xs_all.min(), xs_all.max(), 50)
        ax.plot(xg, slope * xg + intercept, "k--", alpha=0.5,
                label=rf"linear fit (Spearman $\rho={rho:.2f}$, $p={p:.1e}$)")
    ax.set_xlabel("Plane angular error vs multi-start (deg)")
    ax.set_ylabel(r"PCGrad benefit on $R_\mathrm{sym}$ (own plane)")
    ax.legend(fontsize=8.5, loc="best")
    ax.set_title("Worse plane estimation $\\to$ more PCGrad benefit (causal evidence)")
    plt.tight_layout()
    f1 = OUT_DIR / "fig_plane_vs_pcgrad"
    plt.savefig(str(f1) + ".pdf"); plt.savefig(str(f1) + ".png"); plt.close()
    print(f"[fig] {f1}.pdf")

    # Figure 2: plane error vs mean negative cosine magnitude
    if np.isfinite(pool["pct_neg_sym_smo"]).any():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        for ax, tgt, ylabel in zip(
            axes,
            ["pct_neg_sym_smo", "pct_neg_sym_com"],
            ["% steps with cos(sym,smo)<0", "% steps with cos(sym,com)<0"],
        ):
            for m in PLANE_ORDER:
                xs, ys = [], []
                for r in records:
                    if m in r["plane_methods"]:
                        xs.append(r["plane_methods"][m]["plane_angular_err_deg"])
                        ys.append(r["plane_methods"][m].get(tgt, np.nan))
                ax.scatter(xs, ys, c=colors[m], marker=markers[m], s=18, alpha=0.6,
                           label=PLANE_LABELS[m])
            x, y = pool["ang_err"], pool[tgt]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 5:
                rho, p = stats.spearmanr(x[mask], y[mask])
                ax.set_title(rf"Spearman $\rho={rho:.2f}$, $p={p:.1e}$")
            ax.set_xlabel("Plane angular error (deg)")
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=8.5)
        plt.tight_layout()
        f2 = OUT_DIR / "fig_plane_vs_cosine"
        plt.savefig(str(f2) + ".pdf"); plt.savefig(str(f2) + ".png"); plt.close()
        print(f"[fig] {f2}.pdf")


def write_latex_table(rows):
    latex = [
        "% Auto-generated by tools/nips_push/analyze_causal_plane.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Plane estimation quality predicts PCGrad benefit. Per-plane means across the causal-experiment meshes (n=" + str(rows[0]['n']) + "). PCGrad benefit $= R_\\mathrm{sym}^\\mathrm{PCGrad} - R_\\mathrm{sym}^\\mathrm{equal}$ (higher = more PCGrad advantage over equal weights); plane angular error is deviation from the multi-start normal.}",
        "\\label{tab:causal_plane}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Plane & Angular err (deg) $\\downarrow$ & $R_\\mathrm{sym}$ init $\\uparrow$ & PCGrad benefit ($R_\\mathrm{sym}$) & \\% neg cos(sym,smo) \\\\",
        "\\midrule",
    ]
    for r in rows:
        pct_neg = f"{r['pct_neg_sym_smo_mean']:.1f}" if r['pct_neg_sym_smo_mean'] is not None else "---"
        latex.append(
            f"{PLANE_LABELS[r['plane']]:<20s} & "
            f"{r['plane_angular_err_deg_mean']:>6.1f} & "
            f"{r['own_sym_init_mean']:>+7.4f} & "
            f"{r['pcgrad_benefit_sym_own_mean']:>+7.4f} & "
            f"{pct_neg} \\\\"
        )
    latex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = OUT_DIR / "per_method_table.tex"
    out.write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(f"[tex] {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not records:
        sys.exit("No valid records.")

    rows = per_method_summary(records)

    print("\n=== Per-plane summary (means across meshes) ===")
    hdr = f"{'Plane':<14}{'n':<6}{'ang_err':<12}{'sym_init':<12}{'PCGrad_ben':<14}{'mean_cos_ss':<14}{'%neg_ss':<10}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ss = f"{r['mean_cos_sym_smo']:+.4f}" if r['mean_cos_sym_smo'] is not None else "  ---  "
        pn = f"{r['pct_neg_sym_smo_mean']:.1f}" if r['pct_neg_sym_smo_mean'] is not None else "---"
        print(f"{r['plane']:<14}{r['n']:<6}{r['plane_angular_err_deg_mean']:<+12.2f}"
              f"{r['own_sym_init_mean']:<+12.4f}{r['pcgrad_benefit_sym_own_mean']:<+14.4f}"
              f"{ss:<14}{pn:<10}")

    corr, pool = causal_correlations(records)
    print("\n=== Causal correlations (pooled, Spearman) ===")
    for k, v in corr.items():
        sig = "**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "")
        print(f"  {k:<55s} rho={v['rho']:+.3f}  p={v['p']:.3e}  n={v['n']}  {sig}")

    # Write summary.json
    summary = {"n_meshes": len(records), "per_method": rows, "correlations": corr}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[json] {OUT_DIR / 'summary.json'}")

    make_figures(records, pool)
    write_latex_table(rows)

    print("\n" + "=" * 60)
    print("KEY CLAIM TO CHECK (for paper):")
    print("  Spearman(plane_ang_err, PCGrad_benefit_sym_own) > 0 at p < 0.05")
    key = corr.get("ang_err_vs_pc_benefit_sym_own")
    if key:
        verdict = "SUPPORTED" if key["rho"] > 0 and key["p"] < 0.05 else \
                  ("NULL" if key["p"] >= 0.05 else "CONTRADICTED")
        print(f"  -> rho={key['rho']:+.3f}, p={key['p']:.3e}  [{verdict}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
