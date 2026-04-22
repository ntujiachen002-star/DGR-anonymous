"""[P0 / Round-2 fix #2] Analysis + figures for the controlled plane-perturbation
experiment.

Reads : analysis_results/nips_push_causal_plane_perturbation/all_results.json
Writes:
    summary.json
    fig_angle_vs_pcgrad_benefit.{pdf,png}   — per-angle means + scatter
    fig_angle_vs_cosine.{pdf,png}           — per-angle % negative cosine
    per_angle_table.tex                     — LaTeX table of per-angle stats
    mixed_effects_report.txt                — MixedLM regression results
    stdout                                  — key statistics

Causal analysis performed:
    1. Pooled Spearman (angle, PCGrad benefit) — matches the original experiment.
    2. Per-mesh Kendall tau: for each mesh, correlate angle across its
       perturbations. Summary = mean tau, fraction with tau > 0.
       This is the paired-within-mesh estimand; more robust to between-mesh
       variation than pooled Spearman.
    3. Mixed-effects regression:
           PCGradBenefit ~ log1p(angle_deg) + (1 | mesh_id)
       Coefficient on log1p(angle) with p-value isolates the fixed effect of
       angle after accounting for per-mesh random intercepts.

The strongest causal support comes from (2) and (3). (1) is reported for
comparability with exp_causal_plane.py.

CPU only, < 1 minute.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path("analysis_results/nips_push_causal_plane_perturbation")
RESULTS_PATH = OUT_DIR / "all_results.json"


def load_records():
    if not RESULTS_PATH.exists():
        sys.exit(f"Missing {RESULTS_PATH}; run exp_causal_plane_perturbation.py first.")
    with open(RESULTS_PATH) as f:
        records = json.load(f)
    records = [r for r in records if not r.get("skipped")]
    print(f"[info] Loaded {len(records)} non-skipped meshes")
    return records


def flatten_pool(records):
    """Flatten to a long-format array of perturbation entries."""
    rows = []
    for i, r in enumerate(records):
        mid = r["mesh"]
        for e in r.get("perturbations", []):
            rows.append({
                "mesh_id": i,
                "mesh": mid,
                "category": r.get("category", "?"),
                "angle_nom": e["angle_nominal_deg"],
                "angle_act": e["angle_actual_deg"],
                "rep": e["rep"],
                "own_sym_init": e["own_sym_init"],
                "eq_sym_own": e["equal_sym_own"],
                "pc_sym_own": e["pcgrad_sym_own"],
                "eq_sym_ref": e["equal_sym_ref"],
                "pc_sym_ref": e["pcgrad_sym_ref"],
                "pcgrad_benefit_sym_own": e["pcgrad_benefit_sym_own"],
                "pcgrad_benefit_sym_ref": e["pcgrad_benefit_sym_ref"],
                "mean_cos_sym_smo": e.get("mean_cos_sym_smo", float("nan")),
                "mean_cos_sym_com": e.get("mean_cos_sym_com", float("nan")),
                "pct_neg_sym_smo": e.get("pct_neg_sym_smo", float("nan")),
                "pct_neg_sym_com": e.get("pct_neg_sym_com", float("nan")),
            })
    return rows


def per_angle_summary(rows):
    """Mean + SEM of key metrics stratified by nominal angle."""
    angles = sorted(set(r["angle_nom"] for r in rows))
    summary = []
    for a in angles:
        sub = [r for r in rows if r["angle_nom"] == a]
        if not sub:
            continue

        def stat(key):
            vals = np.array([r[key] for r in sub if np.isfinite(r[key])])
            if len(vals) == 0:
                return None, None
            return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))

        pcb_mean, pcb_sem = stat("pcgrad_benefit_sym_own")
        pcb_ref_mean, pcb_ref_sem = stat("pcgrad_benefit_sym_ref")
        eq_sym_mean, _ = stat("eq_sym_own")
        pc_sym_mean, _ = stat("pc_sym_own")
        cs_ss_mean, _ = stat("mean_cos_sym_smo")
        pn_ss_mean, pn_ss_sem = stat("pct_neg_sym_smo")
        pn_sc_mean, pn_sc_sem = stat("pct_neg_sym_com")
        summary.append({
            "angle": a,
            "n": len(sub),
            "pcgrad_benefit_sym_own_mean": pcb_mean,
            "pcgrad_benefit_sym_own_sem": pcb_sem,
            "pcgrad_benefit_sym_ref_mean": pcb_ref_mean,
            "pcgrad_benefit_sym_ref_sem": pcb_ref_sem,
            "equal_sym_own_mean": eq_sym_mean,
            "pcgrad_sym_own_mean": pc_sym_mean,
            "mean_cos_sym_smo": cs_ss_mean,
            "pct_neg_sym_smo_mean": pn_ss_mean,
            "pct_neg_sym_smo_sem": pn_ss_sem,
            "pct_neg_sym_com_mean": pn_sc_mean,
            "pct_neg_sym_com_sem": pn_sc_sem,
        })
    return summary


def pooled_correlations(rows):
    """Spearman correlations over all (mesh, angle, rep) triples."""
    ang = np.array([r["angle_act"] for r in rows])
    out = {}
    for tgt in ["pcgrad_benefit_sym_own", "pcgrad_benefit_sym_ref",
                "mean_cos_sym_smo", "mean_cos_sym_com",
                "pct_neg_sym_smo", "pct_neg_sym_com"]:
        y = np.array([r[tgt] for r in rows])
        mask = np.isfinite(ang) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        rho, p = stats.spearmanr(ang[mask], y[mask])
        out[f"ang_vs_{tgt}"] = {"rho": float(rho), "p": float(p),
                                "n": int(mask.sum())}
    return out


def per_mesh_kendall(rows):
    """Per-mesh Kendall tau between angle and PCGrad benefit.

    Reports distribution of tau across meshes, fraction > 0, and
    one-sample Wilcoxon signed-rank test against zero.
    """
    by_mesh = {}
    for r in rows:
        by_mesh.setdefault(r["mesh_id"], []).append(r)
    taus, taus_ref = [], []
    for mid, sub in by_mesh.items():
        if len(sub) < 3:
            continue
        ang = np.array([r["angle_act"] for r in sub])
        y = np.array([r["pcgrad_benefit_sym_own"] for r in sub])
        y_ref = np.array([r["pcgrad_benefit_sym_ref"] for r in sub])
        mask = np.isfinite(ang) & np.isfinite(y)
        if mask.sum() >= 3 and np.std(ang[mask]) > 0 and np.std(y[mask]) > 0:
            tau, _ = stats.kendalltau(ang[mask], y[mask])
            if np.isfinite(tau):
                taus.append(tau)
        mask_ref = np.isfinite(ang) & np.isfinite(y_ref)
        if mask_ref.sum() >= 3 and np.std(ang[mask_ref]) > 0 and np.std(y_ref[mask_ref]) > 0:
            tau_ref, _ = stats.kendalltau(ang[mask_ref], y_ref[mask_ref])
            if np.isfinite(tau_ref):
                taus_ref.append(tau_ref)

    def summary_set(tau_list, label):
        if not tau_list:
            return {"n": 0, "label": label}
        tau_arr = np.array(tau_list)
        # One-sample Wilcoxon: test H0: median tau = 0 vs H1: median > 0.
        try:
            stat, p_two = stats.wilcoxon(tau_arr, alternative="greater")
        except Exception:
            stat, p_two = float("nan"), float("nan")
        return {
            "label": label,
            "n": len(tau_arr),
            "tau_mean": float(tau_arr.mean()),
            "tau_median": float(np.median(tau_arr)),
            "tau_std": float(tau_arr.std(ddof=1)) if len(tau_arr) > 1 else 0.0,
            "frac_positive": float((tau_arr > 0).mean()),
            "frac_strictly_positive_gt_0.3": float((tau_arr > 0.3).mean()),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p_onesided_greater": float(p_two),
        }

    return {
        "pcgrad_benefit_sym_own": summary_set(taus, "own plane"),
        "pcgrad_benefit_sym_ref": summary_set(taus_ref, "ref plane"),
    }


def mixed_effects(rows):
    """Mixed-effects linear regression:
        PCGradBenefit ~ log1p(angle_deg) + (1 | mesh_id)
    Returns fixed-effect coefficient and p-value. Falls back to OLS if
    statsmodels is unavailable.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        print("[warn] statsmodels/pandas not installed; skipping mixed-effects")
        return {"error": "statsmodels/pandas unavailable"}

    import pandas as pd
    df = pd.DataFrame([
        {
            "mesh_id": r["mesh_id"],
            "angle": r["angle_act"],
            "log1p_angle": math.log1p(r["angle_act"]),
            "benefit_own": r["pcgrad_benefit_sym_own"],
            "benefit_ref": r["pcgrad_benefit_sym_ref"],
        }
        for r in rows if np.isfinite(r["angle_act"])
    ])
    reports = {}
    for target in ["benefit_own", "benefit_ref"]:
        sub = df[np.isfinite(df[target])]
        if len(sub) < 20 or sub["mesh_id"].nunique() < 5:
            reports[target] = {"error": "not enough data"}
            continue
        try:
            model = smf.mixedlm(
                f"{target} ~ log1p_angle", data=sub, groups=sub["mesh_id"])
            fit = model.fit(method="lbfgs", reml=True)
            reports[target] = {
                "n_obs": int(len(sub)),
                "n_groups": int(sub["mesh_id"].nunique()),
                "coef_log1p_angle": float(fit.params["log1p_angle"]),
                "se_log1p_angle": float(fit.bse["log1p_angle"]),
                "z_log1p_angle": float(fit.tvalues["log1p_angle"]),
                "p_log1p_angle": float(fit.pvalues["log1p_angle"]),
                "intercept": float(fit.params["Intercept"]),
                "group_var": float(fit.cov_re.iloc[0, 0]) if fit.cov_re is not None else None,
                "log_likelihood": float(fit.llf),
                "converged": bool(fit.converged),
            }
        except Exception as e:
            reports[target] = {"error": str(e)}
    return reports


def make_figures(rows, per_angle):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping figures")
        return

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 150})

    # Figure 1: angle vs PCGrad benefit. Per-angle mean +- SEM, with faint scatter.
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.0))
    ang_pool = np.array([r["angle_act"] for r in rows])
    ben_pool = np.array([r["pcgrad_benefit_sym_own"] for r in rows])
    ax.scatter(ang_pool, ben_pool, c="#888", s=8, alpha=0.25, label="per-mesh perturbation")

    a = [p["angle"] for p in per_angle]
    m = [p["pcgrad_benefit_sym_own_mean"] for p in per_angle]
    se = [p["pcgrad_benefit_sym_own_sem"] for p in per_angle]
    ax.errorbar(a, m, yerr=se, fmt="o-", color="#1f77b4", capsize=3,
                linewidth=2, markersize=7, label="per-angle mean ± SEM")

    mask = np.isfinite(ang_pool) & np.isfinite(ben_pool)
    if mask.sum() > 5:
        rho, p = stats.spearmanr(ang_pool[mask], ben_pool[mask])
        ax.set_title(rf"Plane perturbation $\to$ PCGrad benefit "
                     rf"(pooled Spearman $\rho={rho:.2f}$, $p={p:.1e}$)")
    ax.set_xlabel("Plane angular perturbation (deg)")
    ax.set_ylabel(r"PCGrad benefit on $R_\mathrm{sym}$ (own plane)")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    f1 = OUT_DIR / "fig_angle_vs_pcgrad_benefit"
    plt.savefig(str(f1) + ".pdf"); plt.savefig(str(f1) + ".png"); plt.close()
    print(f"[fig] {f1}.pdf")

    # Figure 2: angle vs % negative cosines (sym-smo, sym-com)
    if any(p["pct_neg_sym_smo_mean"] is not None for p in per_angle):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        for ax, key_mean, key_sem, ylabel in [
            (axes[0], "pct_neg_sym_smo_mean", "pct_neg_sym_smo_sem",
             "% steps with cos(sym,smo)<0"),
            (axes[1], "pct_neg_sym_com_mean", "pct_neg_sym_com_sem",
             "% steps with cos(sym,com)<0"),
        ]:
            a = [p["angle"] for p in per_angle]
            m = [p[key_mean] for p in per_angle]
            se = [p[key_sem] for p in per_angle]
            ax.errorbar(a, m, yerr=se, fmt="o-", color="#d62728",
                        capsize=3, linewidth=2, markersize=7)
            ax.set_xlabel("Plane angular perturbation (deg)")
            ax.set_ylabel(ylabel)
        plt.tight_layout()
        f2 = OUT_DIR / "fig_angle_vs_cosine"
        plt.savefig(str(f2) + ".pdf"); plt.savefig(str(f2) + ".png"); plt.close()
        print(f"[fig] {f2}.pdf")


def write_latex_table(per_angle):
    latex = [
        "% Auto-generated by tools/nips_push/analyze_causal_plane_perturbation.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Controlled plane-perturbation experiment: PCGrad benefit vs. angular "
        "perturbation of the multi-start bilateral plane. Values are mean $\\pm$ SEM "
        "across meshes. Same-mesh perturbations isolate the causal effect of plane "
        "error from between-mesh variation.}",
        "\\label{tab:causal_plane_perturbation}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Angle (deg) & $n$ & PCGrad benefit (own) & PCGrad benefit (ref) & "
        "\\% neg cos(sym,smo) & \\% neg cos(sym,com) \\\\",
        "\\midrule",
    ]
    for p in per_angle:
        def fmt(m, s, prec=4):
            if m is None:
                return "---"
            if s is None:
                return f"{m:+.{prec}f}"
            return f"{m:+.{prec}f} $\\pm$ {s:.{prec}f}"
        latex.append(
            f"{p['angle']:>5.0f} & {p['n']} & "
            f"{fmt(p['pcgrad_benefit_sym_own_mean'], p['pcgrad_benefit_sym_own_sem'])} & "
            f"{fmt(p['pcgrad_benefit_sym_ref_mean'], p['pcgrad_benefit_sym_ref_sem'])} & "
            f"{fmt(p['pct_neg_sym_smo_mean'], p['pct_neg_sym_smo_sem'], prec=1)} & "
            f"{fmt(p['pct_neg_sym_com_mean'], p['pct_neg_sym_com_sem'], prec=1)} \\\\"
        )
    latex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = OUT_DIR / "per_angle_table.tex"
    out.write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(f"[tex] {out}")


def write_mixed_report(me, km, pool_corr):
    lines = ["# Causal plane-perturbation: statistical report",
             "", "## Mixed-effects regression",
             "   PCGradBenefit ~ log1p(angle_deg) + (1 | mesh_id)"]
    for tgt, rep in me.items():
        lines.append(f"\n### target: {tgt}")
        if "error" in rep:
            lines.append(f"  ERROR: {rep['error']}")
            continue
        lines.append(f"  n_obs          = {rep['n_obs']}")
        lines.append(f"  n_groups       = {rep['n_groups']}")
        lines.append(f"  coef(log1p_a)  = {rep['coef_log1p_angle']:+.6f}")
        lines.append(f"  SE             = {rep['se_log1p_angle']:.6f}")
        lines.append(f"  z              = {rep['z_log1p_angle']:+.3f}")
        lines.append(f"  p (two-sided)  = {rep['p_log1p_angle']:.3e}")
        lines.append(f"  intercept      = {rep['intercept']:+.6f}")
        lines.append(f"  group var      = {rep.get('group_var')}")
        lines.append(f"  converged      = {rep['converged']}")

    lines += ["", "## Per-mesh Kendall tau (paired within-mesh)"]
    for tgt, rep in km.items():
        lines.append(f"\n### target: {tgt}   ({rep.get('label','?')})")
        if rep.get("n", 0) == 0:
            lines.append("  (no meshes with enough perturbations)")
            continue
        lines.append(f"  n meshes       = {rep['n']}")
        lines.append(f"  mean tau       = {rep['tau_mean']:+.3f}")
        lines.append(f"  median tau     = {rep['tau_median']:+.3f}")
        lines.append(f"  frac tau > 0   = {rep['frac_positive']:.3f}")
        lines.append(f"  frac tau > 0.3 = {rep['frac_strictly_positive_gt_0.3']:.3f}")
        lines.append(f"  Wilcoxon p (greater) = {rep['wilcoxon_p_onesided_greater']:.3e}")

    lines += ["", "## Pooled Spearman (comparability with exp_causal_plane.py)"]
    for k, v in pool_corr.items():
        sig = "**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "")
        lines.append(f"  {k:<40s} rho={v['rho']:+.3f}  p={v['p']:.3e}  "
                     f"n={v['n']}  {sig}")

    out = OUT_DIR / "mixed_effects_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[txt] {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not records:
        sys.exit("No valid records.")
    rows = flatten_pool(records)
    if not rows:
        sys.exit("No perturbations found in records.")
    print(f"[info] {len(rows)} total (mesh, angle, rep) rows")

    per_angle = per_angle_summary(rows)
    print("\n=== Per-angle summary ===")
    hdr = f"{'angle':<8}{'n':<6}{'PCGrad_ben_own':<20}{'%neg_ss':<14}{'%neg_sc':<14}"
    print(hdr); print("-" * len(hdr))
    for p in per_angle:
        pcb = f"{p['pcgrad_benefit_sym_own_mean']:+.4f}"
        if p["pcgrad_benefit_sym_own_sem"] is not None:
            pcb += f" ± {p['pcgrad_benefit_sym_own_sem']:.4f}"
        pn1 = f"{p['pct_neg_sym_smo_mean']:.1f}" if p['pct_neg_sym_smo_mean'] is not None else "---"
        pn2 = f"{p['pct_neg_sym_com_mean']:.1f}" if p['pct_neg_sym_com_mean'] is not None else "---"
        print(f"{p['angle']:<8.0f}{p['n']:<6}{pcb:<20}{pn1:<14}{pn2:<14}")

    pool_corr = pooled_correlations(rows)
    print("\n=== Pooled Spearman (angle vs target) ===")
    for k, v in pool_corr.items():
        sig = "**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "")
        print(f"  {k:<50s} rho={v['rho']:+.3f}  p={v['p']:.3e}  n={v['n']}  {sig}")

    km = per_mesh_kendall(rows)
    print("\n=== Per-mesh Kendall tau (paired within-mesh) ===")
    for tgt, rep in km.items():
        if rep.get("n", 0) == 0:
            continue
        print(f"  {tgt:<30s}: n={rep['n']}  mean_tau={rep['tau_mean']:+.3f}  "
              f"frac>0={rep['frac_positive']:.2f}  "
              f"Wilcox_p={rep['wilcoxon_p_onesided_greater']:.3e}")

    me = mixed_effects(rows)
    print("\n=== Mixed-effects regression: PCGradBenefit ~ log1p(angle) + (1|mesh) ===")
    for tgt, rep in me.items():
        if "error" in rep:
            print(f"  {tgt}: {rep['error']}")
            continue
        print(f"  {tgt}: coef={rep['coef_log1p_angle']:+.5f}  "
              f"SE={rep['se_log1p_angle']:.5f}  "
              f"z={rep['z_log1p_angle']:+.2f}  "
              f"p={rep['p_log1p_angle']:.3e}  "
              f"n_obs={rep['n_obs']}  n_groups={rep['n_groups']}")

    summary = {
        "n_meshes": len(records),
        "n_rows": len(rows),
        "per_angle": per_angle,
        "pooled_spearman": pool_corr,
        "per_mesh_kendall": km,
        "mixed_effects": me,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[json] {OUT_DIR / 'summary.json'}")

    make_figures(rows, per_angle)
    write_latex_table(per_angle)
    write_mixed_report(me, km, pool_corr)

    # Key verdict
    print("\n" + "=" * 70)
    print("KEY CAUSAL CHECKS (for paper):")
    print("  [1] Pooled Spearman(angle, PCGrad_benefit_own) > 0, p<0.05")
    v1 = pool_corr.get("ang_vs_pcgrad_benefit_sym_own")
    if v1:
        verdict = "SUPPORTED" if v1["rho"] > 0 and v1["p"] < 0.05 else \
                  ("NULL" if v1["p"] >= 0.05 else "CONTRADICTED")
        print(f"      rho={v1['rho']:+.3f}, p={v1['p']:.3e}  [{verdict}]")
    print("  [2] Per-mesh Kendall tau: median > 0, Wilcoxon p<0.05")
    k2 = km.get("pcgrad_benefit_sym_own", {})
    if k2.get("n", 0) > 0:
        verdict = "SUPPORTED" if (k2["tau_median"] > 0 and
                                  k2["wilcoxon_p_onesided_greater"] < 0.05) else "NULL"
        print(f"      median_tau={k2['tau_median']:+.3f}, "
              f"Wilcoxon p={k2['wilcoxon_p_onesided_greater']:.3e}  [{verdict}]")
    print("  [3] Mixed-effects coef(log1p_angle) > 0, p<0.05")
    m3 = me.get("benefit_own", {})
    if "coef_log1p_angle" in m3:
        verdict = "SUPPORTED" if (m3["coef_log1p_angle"] > 0 and
                                  m3["p_log1p_angle"] < 0.05) else "NULL"
        print(f"      coef={m3['coef_log1p_angle']:+.5f}, "
              f"p={m3['p_log1p_angle']:.3e}  [{verdict}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
