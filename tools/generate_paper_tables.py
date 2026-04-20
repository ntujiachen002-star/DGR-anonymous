"""
从实验结果自动生成论文 LaTeX 表格。

读取:
  - analysis_results/full_mgda_classical/stats.json
  - analysis_results/mvdream_backbone/stats.json

输出:
  - paper/tables/classical_baselines.tex   (Table: 经典方法对比)
  - paper/tables/mgda_comparison.tex       (Table: MGDA vs weighted sum)
  - paper/tables/independent_metrics.tex   (Table: 独立评估指标)
  - paper/tables/mvdream_backbone.tex      (Table: 第三 backbone)

用法:
  python tools/generate_paper_tables.py
"""

import json
import os
from pathlib import Path

OUT_DIR = Path("paper/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(val, bold=False):
    """格式化百分比改善值。"""
    s = f"{val:+.1f}\\%"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def fmt_p(val):
    """格式化 p 值。"""
    if val < 1e-10:
        return f"$<10^{{-10}}$"
    elif val < 0.001:
        exp = int(f"{val:.0e}".split('e')[1])
        return f"$10^{{{exp}}}$"
    else:
        return f"${val:.3f}$"


def generate_classical_table(stats):
    """Table: DiffGeoReward vs 经典几何方法。"""
    methods = ['handcrafted', 'laplacian', 'normal_consist', 'classical_combined']
    labels = {
        'handcrafted': 'DiffGeoReward (ours)',
        'laplacian': 'Laplacian Smoothing',
        'normal_consist': 'Normal Consistency',
        'classical_combined': 'Classical Combined',
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with classical geometric baselines on 110 prompts ($\times$3 seeds). "
        r"DiffGeoReward is the only method that improves bilateral symmetry. "
        r"Classical methods operate on local surface properties and cannot capture global structure.}",
        r"\label{tab:classical_baselines}",
        r"\vspace{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Symmetry $\uparrow$ & Smoothness $\uparrow$ & Compactness $\uparrow$ \\",
        r"\midrule",
        r"Baseline (no refinement) & --- & --- & --- \\",
    ]

    for method in methods:
        if method not in stats:
            lines.append(f"{labels[method]} & --- & --- & --- \\\\")
            continue
        s = stats[method]
        row_parts = [labels[method]]
        for metric in ['symmetry', 'smoothness', 'compactness']:
            if metric in s:
                m = s[metric]
                val = fmt_pct(m['delta_pct'], bold=(method == 'handcrafted' and metric == 'symmetry'))
                sig = "" if m['significant'] else "$^{\\text{n.s.}}$"
                row_parts.append(f"{val}{sig}")
            else:
                row_parts.append("---")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_mgda_table(stats):
    """Table: MGDA vs Weighted Sum。"""
    methods = ['handcrafted', 'mgda_0.05']
    labels = {
        'handcrafted': 'Weighted Sum (equal)',
        'mgda_0.05': 'Constrained MGDA ($w_{\\min}{=}0.05$)',
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Weighted sum vs.\ constrained MGDA optimization. "
        r"MGDA improves symmetry and reduces compactness degradation by finding Pareto-optimal gradient directions, "
        r"at the cost of smoothness.}",
        r"\label{tab:mgda_comparison}",
        r"\vspace{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Symmetry $\uparrow$ & Smoothness $\uparrow$ & Compactness $\uparrow$ & Cohen's $d$ (sym) \\",
        r"\midrule",
    ]

    for method in methods:
        if method not in stats:
            continue
        s = stats[method]
        row_parts = [labels[method]]
        for metric in ['symmetry', 'smoothness', 'compactness']:
            if metric in s:
                m = s[metric]
                val = fmt_pct(m['delta_pct'])
                row_parts.append(val)
            else:
                row_parts.append("---")
        # Cohen's d for symmetry
        if 'symmetry' in s:
            row_parts.append(f"${s['symmetry']['cohens_d']:+.2f}$")
        else:
            row_parts.append("---")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_independent_metrics_table(stats):
    """Table: 独立评估指标（非优化目标）。"""
    methods = ['handcrafted', 'mgda_0.05', 'laplacian', 'classical_combined']
    labels = {
        'handcrafted': 'DiffGeoReward',
        'mgda_0.05': 'MGDA ($w_{\\min}{=}0.05$)',
        'laplacian': 'Laplacian',
        'classical_combined': 'Classical Combined',
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Independent evaluation metrics (not optimized by any method). "
        r"Edge regularity: coefficient of variation of edge lengths (lower = more regular). "
        r"Normal consistency: mean angular deviation between adjacent face normals in degrees (lower = smoother).}",
        r"\label{tab:independent_metrics}",
        r"\vspace{4pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Edge Regularity $\downarrow$ & Normal Consistency $\downarrow$ \\",
        r"\midrule",
    ]

    for method in methods:
        if method not in stats:
            continue
        s = stats[method]
        row_parts = [labels[method]]
        for metric in ['edge_regularity', 'normal_consistency_deg']:
            if metric in s:
                m = s[metric]
                val = fmt_pct(m['delta_pct'])
                sig = "" if m['significant'] else "$^{\\text{n.s.}}$"
                row_parts.append(f"{val}{sig}")
            else:
                row_parts.append("---")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    print("Generating paper tables from experiment results...\n")

    # P0: MGDA + Classical
    mgda_stats = load_json("analysis_results/full_mgda_classical/stats.json")
    if mgda_stats:
        # Classical baselines table
        tex = generate_classical_table(mgda_stats)
        path = OUT_DIR / "classical_baselines.tex"
        with open(path, 'w') as f:
            f.write(tex)
        print(f"  Written: {path}")

        # MGDA comparison table
        tex = generate_mgda_table(mgda_stats)
        path = OUT_DIR / "mgda_comparison.tex"
        with open(path, 'w') as f:
            f.write(tex)
        print(f"  Written: {path}")

        # Independent metrics table
        tex = generate_independent_metrics_table(mgda_stats)
        path = OUT_DIR / "independent_metrics.tex"
        with open(path, 'w') as f:
            f.write(tex)
        print(f"  Written: {path}")
    else:
        print("  [SKIP] MGDA+Classical results not found. Run exp_full_mgda_classical.py first.")

    # P1: MVDream
    mvdream_stats = load_json("analysis_results/mvdream_backbone/stats.json")
    if mvdream_stats:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-backbone generalization: MVDream $\rightarrow$ TripoSR pipeline. "
            r"DiffGeoReward improves symmetry and smoothness on a third architecturally distinct backbone.}",
            r"\label{tab:mvdream_backbone}",
            r"\vspace{4pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Metric & Baseline & DiffGeoReward & $\Delta\%$ \\",
            r"\midrule",
        ]
        for metric in ['symmetry', 'smoothness', 'compactness']:
            if metric in mvdream_stats:
                m = mvdream_stats[metric]
                sig = f"$p={m.get('p_adj', m.get('p_raw', 0)):.1e}$" if m.get('significant', False) else "n.s."
                lines.append(
                    f"{metric.capitalize()} & ${m['bl_mean']:.5f}$ & ${m['dgr_mean']:.5f}$ & "
                    f"{fmt_pct(m['delta_pct'])} ({sig}) \\\\"
                )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        path = OUT_DIR / "mvdream_backbone.tex"
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        print(f"  Written: {path}")
    else:
        print("  [SKIP] MVDream results not found. Run exp_q_mvdream_backbone.py first.")

    print("\nDone. Include tables in paper with \\input{tables/xxx.tex}")


if __name__ == '__main__':
    main()
