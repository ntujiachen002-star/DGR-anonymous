"""
Final charts using SciencePlots 'science' style.
This is the de-facto standard used by hundreds of NeurIPS/ICML/CVPR papers.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # registers 'science' style
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

plt.style.use(['science', 'no-latex'])

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# NeurIPS textwidth
TW = 5.5

# Colors from SciencePlots default cycle
C_BLUE  = '#0C5DA5'
C_GREEN = '#00B945'
C_RED   = '#FF2C00'
C_ORANGE= '#FF9500'
C_PURPLE= '#845B97'
C_GRAY  = '#474747'


def fig_convergence():
    data = json.load(open("analysis_results/convergence/all_results.json"))
    conv = [r for r in data if r.get("experiment") == "convergence"]
    if not conv: return

    by_step = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in conv:
        by_step[r["step"]]["sym"].append(r["symmetry"])
        by_step[r["step"]]["smo"].append(r["smoothness"])
        by_step[r["step"]]["com"].append(r["compactness"])
    steps = sorted(by_step.keys())

    fig, ax = plt.subplots(figsize=(TW * 0.45, TW * 0.30))

    for m, label, color, marker in [
        ("sym", "Symmetry",    C_RED,   'o'),
        ("smo", "Smoothness",  C_BLUE,  's'),
        ("com", "Compactness", C_GREEN, '^'),
    ]:
        means = np.array([np.mean(by_step[s][m]) for s in steps])
        base = means[0] if abs(means[0]) > 1e-8 else 1e-8
        nm = means / base
        ax.plot(steps, nm, marker=marker, color=color, label=label,
                markersize=3.5, markeredgewidth=0.3, markeredgecolor='white')

    ax.axvline(x=50, color='#AAAAAA', linestyle=':', linewidth=0.5)
    ax.annotate('default', xy=(52, 0.05), fontsize=6, color='#999999')

    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Normalized reward")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(-2, 102)
    ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC')

    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT / "convergence_curve.pdf"), dpi=300)
    fig.savefig(str(OUT / "convergence_curve.png"), dpi=200)
    plt.close(fig)
    print("  [OK] convergence_curve")


def fig_lr_sensitivity():
    data = json.load(open("analysis_results/convergence/all_results.json"))
    lr_data = [r for r in data if r.get("experiment") == "lr_sensitivity"]
    if not lr_data: return

    by_lr = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in lr_data:
        by_lr[r["lr"]]["sym"].append(r["symmetry"])
        by_lr[r["lr"]]["smo"].append(r["smoothness"])
        by_lr[r["lr"]]["com"].append(r["compactness"])
    lrs = sorted([lr for lr in by_lr.keys() if lr > 0])

    fig, axes = plt.subplots(1, 3, figsize=(TW, TW * 0.22))

    for i, (m, title, color, marker) in enumerate([
        ("sym", "Symmetry", C_RED, 'o'),
        ("smo", "Smoothness", C_BLUE, 's'),
        ("com", "Compactness", C_GREEN, '^'),
    ]):
        ax = axes[i]
        means = [np.mean(by_lr[lr][m]) for lr in lrs]
        ax.plot(range(len(lrs)), means, marker=marker, color=color,
                markersize=3.5, markeredgewidth=0.3, markeredgecolor='white')

        # Circle the default lr
        if 0.005 in lrs:
            idx = lrs.index(0.005)
            ax.plot(idx, means[idx], 'o', markersize=8, markerfacecolor='none',
                    markeredgecolor='#333333', markeredgewidth=1.0, zorder=5)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr}' for lr in lrs], fontsize=5.5, rotation=30, ha='right')
        ax.set_title(title, pad=2)
        if i == 0:
            ax.set_ylabel("Reward")

    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT / "lr_sensitivity.pdf"), dpi=300)
    fig.savefig(str(OUT / "lr_sensitivity.png"), dpi=200)
    plt.close(fig)
    print("  [OK] lr_sensitivity")


def fig_degradation():
    data = json.load(open("analysis_results/laplacian_vs_dgr/checkpoint.json"))
    metrics = ["symmetry", "smoothness", "compactness"]
    metric_labels = ["Sym.", "Smo.", "Com."]
    methods = ["sym_only", "smooth_only", "compact_only", "diffgeoreward", "handcrafted"]
    method_labels = ["Sym-Only", "Smo-Only", "Com-Only", "DGR (Ours)", "Equal Wt."]

    base_means = {}
    for m in metrics:
        vals = [r[m] for r in data if r["method"] == "baseline" and r.get(m) is not None]
        base_means[m] = np.mean(vals) if vals else 0

    matrix = np.zeros((len(methods), len(metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            vals = [r[metric] for r in data if r["method"] == method and r.get(metric) is not None]
            if vals and abs(base_means[metric]) > 1e-8:
                matrix[i, j] = (np.mean(vals) - base_means[metric]) / abs(base_means[metric]) * 100

    fig, ax = plt.subplots(figsize=(TW * 0.42, TW * 0.28))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-120, vmax=120)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(method_labels, fontsize=7)
    ax.tick_params(length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if abs(val) > 60 else '#111111'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.0f}%', ha='center', va='center',
                    fontsize=6.5, color=color, fontweight='bold')

    ax.axhline(y=2.5, color='#333333', linewidth=0.8)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.05, aspect=18)
    cb.set_label("vs. baseline (%)", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=5.5)

    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT / "degradation_heatmap.pdf"), dpi=300)
    fig.savefig(str(OUT / "degradation_heatmap.png"), dpi=200)
    plt.close(fig)
    print("  [OK] degradation_heatmap")


def fig_clip_ir():
    data = json.load(open("analysis_results/clip_allmethod/all_results.json"))
    by_method = defaultdict(lambda: {"clip": [], "ir": []})
    for r in data:
        m = r.get("method", "?")
        if r.get("clip_score") is not None:
            by_method[m]["clip"].append(r["clip_score"])
        if r.get("image_reward") is not None:
            by_method[m]["ir"].append(r["image_reward"])

    order = ["baseline", "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]
    avail = [m for m in order if m in by_method and len(by_method[m]["clip"]) > 10]
    labels = {"baseline": "Base.", "diffgeoreward": "DGR", "handcrafted": "Equal",
              "sym_only": "Sym", "smooth_only": "Smo", "compact_only": "Com"}
    colors = [C_GRAY, C_BLUE, C_GREEN, C_RED, C_ORANGE, C_PURPLE]
    x = np.arange(len(avail))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TW, TW * 0.26))

    # (a) CLIP — use SEM
    clip_m = [np.mean(by_method[m]["clip"]) for m in avail]
    clip_sem = [np.std(by_method[m]["clip"]) / np.sqrt(len(by_method[m]["clip"])) for m in avail]

    ax1.bar(x, clip_m, 0.6, yerr=clip_sem, capsize=2,
            color=colors[:len(avail)], edgecolor='none',
            error_kw={'linewidth': 0.5, 'capthick': 0.4})
    ax1.axhline(y=clip_m[0], color='#CCCCCC', linestyle='--', linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels.get(m, m) for m in avail], fontsize=6.5)
    ax1.set_ylabel("CLIP score")
    ax1.set_title("(a) Semantic alignment")
    ax1.set_ylim(0.225, 0.245)

    # (b) ImageReward — use SEM
    ir_avail = [m for m in avail if len(by_method[m]["ir"]) > 10]
    if ir_avail:
        ir_m = [np.mean(by_method[m]["ir"]) for m in ir_avail]
        ir_sem = [np.std(by_method[m]["ir"]) / np.sqrt(len(by_method[m]["ir"])) for m in ir_avail]
        x2 = np.arange(len(ir_avail))
        c2 = colors[:len(ir_avail)]
        ax2.bar(x2, ir_m, 0.6, yerr=ir_sem, capsize=2,
                color=c2, edgecolor='none',
                error_kw={'linewidth': 0.5, 'capthick': 0.4})
        ax2.axhline(y=ir_m[0], color='#CCCCCC', linestyle='--', linewidth=0.4)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([labels.get(m, m) for m in ir_avail], fontsize=6.5)
        ax2.set_ylabel("ImageReward")
        ax2.set_title("(b) Perceptual quality")

    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT / "clip_comparison.pdf"), dpi=300)
    fig.savefig(str(OUT / "clip_comparison.png"), dpi=200)
    plt.close(fig)
    print("  [OK] clip_comparison")


if __name__ == "__main__":
    print("Generating with SciencePlots 'science' style...")
    fig_convergence()
    fig_lr_sensitivity()
    fig_degradation()
    fig_clip_ir()
    print("Done.")
