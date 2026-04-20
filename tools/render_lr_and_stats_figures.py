"""
Generate LR sensitivity figure + method comparison bar chart for the paper.
Output: paper/figures/lr_sensitivity.pdf, paper/figures/method_comparison.pdf
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LR Sensitivity Figure ──────────────────────────────────────────────

def plot_lr_sensitivity():
    """Plot LR sensitivity from exp_p Part B data."""
    data_path = Path("analysis_results/convergence/all_results.json")
    if not data_path.exists():
        print("No convergence data for LR sensitivity")
        return

    data = json.load(open(data_path))
    lr_data = [r for r in data if r.get("experiment") == "lr_sensitivity"]
    if not lr_data:
        print("No lr_sensitivity records found")
        return

    # Group by lr
    by_lr = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in lr_data:
        lr = r["lr"]
        by_lr[lr]["sym"].append(r["symmetry"])
        by_lr[lr]["smo"].append(r["smoothness"])
        by_lr[lr]["com"].append(r["compactness"])

    lrs = sorted(by_lr.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    colors = {"sym": "#E63946", "smo": "#457B9D", "com": "#2A9D8F"}
    labels = {"sym": "Symmetry", "smo": "Smoothness", "com": "Compactness"}
    panel_labels = ["(a)", "(b)", "(c)"]

    for i, (m, label) in enumerate(labels.items()):
        means = [np.mean(by_lr[lr][m]) for lr in lrs]
        stds  = [np.std(by_lr[lr][m])  for lr in lrs]
        ax = axes[i]
        ax.errorbar([str(lr) for lr in lrs], means, yerr=stds,
                    fmt='-o', color=colors[m], capsize=3, markersize=6,
                    linewidth=1.5, capthick=1)
        ax.axvline(x=lrs.index(0.005) if 0.005 in lrs else 2,
                   color='orange', linestyle='--', alpha=0.6, linewidth=1)
        ax.set_title(f'{panel_labels[i]} {label}', fontsize=13, fontweight='bold')
        ax.set_xlabel("Learning Rate", fontsize=11)
        if i == 0:
            ax.set_ylabel("Reward", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "lr_sensitivity.pdf"
    fig.patch.set_facecolor('white')
    fig.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(OUT_DIR / "lr_sensitivity.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. Method Comparison Bar Chart ─────────────────────────────────────────

def plot_method_comparison():
    """Bar chart comparing all methods from exp_k full results."""
    # Try multiple possible result paths
    for p in ["analysis_results/clip_allmethod/all_results.json",
              "analysis_results/mesh_validity_full/all_reports.json",
              "analysis_results/pilot_geobench_v2/all_results.json"]:
        if Path(p).exists():
            data = json.load(open(p))
            break
    else:
        print("No method comparison data found")
        return

    # Group by method, compute mean metrics
    by_method = defaultdict(lambda: {"symmetry": [], "smoothness": [], "compactness": []})
    for r in data:
        method = r.get("method", "unknown")
        for m in ["symmetry", "smoothness", "compactness"]:
            if m in r and r[m] is not None:
                by_method[method][m].append(r[m])

    if not by_method:
        print("No method data to plot")
        return

    # Select methods to show (only ones with data)
    show_methods = []
    display_names = {
        "baseline": "Baseline",
        "diffgeoreward": "DGR (Ours)",
        "handcrafted": "HC (1/3,1/3,1/3)",
        "sym_only": "Sym-Only",
        "smooth_only": "Smo-Only",
        "compact_only": "Com-Only",
        "constant": "Constant",
        "lang2comp": "Lang2Comp",
        "uniform": "Uniform",
        "random": "Random",
    }
    for m in ["baseline", "diffgeoreward", "handcrafted", "sym_only",
              "smooth_only", "compact_only", "lang2comp"]:
        if m in by_method and len(by_method[m]["symmetry"]) > 5:
            show_methods.append(m)

    if len(show_methods) < 2:
        print(f"Only {len(show_methods)} methods with data, skipping bar chart")
        return

    metrics = ["symmetry", "smoothness", "compactness"]
    colors_m = {"symmetry": "#E63946", "smoothness": "#457B9D", "compactness": "#2A9D8F"}

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(show_methods))
    width = 0.25

    for i, metric in enumerate(metrics):
        means = [np.mean(by_method[m][metric]) for m in show_methods]
        stds  = [np.std(by_method[m][metric])  for m in show_methods]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=metric.capitalize(), color=colors_m[metric],
               alpha=0.85, capsize=2, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([display_names.get(m, m) for m in show_methods],
                       fontsize=8, rotation=15)
    ax.set_ylabel("Mean Reward", fontsize=10)
    ax.set_title("Geometric Reward Comparison Across Methods", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / "method_comparison.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(OUT_DIR / "method_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Catastrophic Degradation Heatmap ────────────────────────────────────

def plot_degradation_heatmap():
    """Show per-category degradation when using single-reward optimization."""
    for p in ["analysis_results/clip_allmethod/all_results.json",
              "analysis_results/pilot_geobench_v2/all_results.json"]:
        if Path(p).exists():
            data = json.load(open(p))
            break
    else:
        print("No data for degradation heatmap")
        return

    # Compute improvement ratios: (method - baseline) / |baseline|
    by_method_cat = {}
    metrics = ["symmetry", "smoothness", "compactness"]

    # Get baseline means per category
    baseline_by_cat = defaultdict(lambda: {m: [] for m in metrics})
    for r in data:
        if r.get("method") == "baseline":
            cat = r.get("category", "unknown")
            for m in metrics:
                if m in r and r[m] is not None:
                    baseline_by_cat[cat][m].append(r[m])

    # Single-reward methods
    single_methods = ["sym_only", "smooth_only", "compact_only"]
    method_labels = {"sym_only": "Sym-Only", "smooth_only": "Smo-Only",
                     "compact_only": "Com-Only", "diffgeoreward": "DGR (Ours)"}

    for method in single_methods + ["diffgeoreward"]:
        by_method_cat[method] = {}
        for cat in baseline_by_cat:
            by_method_cat[method][cat] = {}
            for m in metrics:
                method_vals = [r[m] for r in data
                               if r.get("method") == method
                               and r.get("category") == cat
                               and m in r and r[m] is not None]
                baseline_mean = np.mean(baseline_by_cat[cat][m]) if baseline_by_cat[cat][m] else 0
                method_mean = np.mean(method_vals) if method_vals else 0
                if abs(baseline_mean) > 1e-8:
                    improvement = (method_mean - baseline_mean) / abs(baseline_mean) * 100
                else:
                    improvement = 0
                by_method_cat[method][cat][m] = improvement

    cats = sorted(baseline_by_cat.keys())
    if not cats:
        print("No categories found for heatmap")
        return

    show_methods = [m for m in single_methods + ["diffgeoreward"]
                    if m in by_method_cat and by_method_cat[m]]

    # Build matrix
    matrix = np.zeros((len(show_methods), len(cats) * len(metrics)))
    col_labels = []
    for cat in cats:
        for m in metrics:
            col_labels.append(f"{cat[:3]}_{m[:3]}")

    for i, method in enumerate(show_methods):
        for j, cat in enumerate(cats):
            for k, m in enumerate(metrics):
                val = by_method_cat[method].get(cat, {}).get(m, 0)
                matrix[i, j * len(metrics) + k] = val

    fig, ax = plt.subplots(figsize=(9, 2.5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(show_methods)))
    ax.set_yticklabels([method_labels.get(m, m) for m in show_methods], fontsize=9)

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if abs(val) > 30 else 'black'
            ax.text(j, i, f"{val:.0f}%", ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Improvement vs Baseline (%)", shrink=0.8)
    ax.set_title("Per-Category Metric Changes: Single-Reward vs. Multi-Reward (DGR)",
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    out = OUT_DIR / "degradation_heatmap.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(OUT_DIR / "degradation_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_lr_sensitivity()
    plot_method_comparison()
    plot_degradation_heatmap()
    print("\nAll figures generated!")
