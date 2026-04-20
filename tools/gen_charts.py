"""
Publication-quality CHARTS only (no mesh rendering).
Follows NeurIPS/ICML style: serif fonts, clean axes, proper sizing.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tools')
from pub_style import setup_style, figsize, METRIC_COLORS, METHOD_COLORS, METHOD_LABELS, TEXTWIDTH_PT
setup_style()

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

W = TEXTWIDTH_PT / 72.27  # full textwidth in inches


# ═══════════════════════════════════════════════════════════════════════
# 1. Convergence Curve
# ═══════════════════════════════════════════════════════════════════════

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

    fig, ax = plt.subplots(figsize=(W * 0.46, W * 0.32))

    for m, label in [("sym", "Symmetry"), ("smo", "Smoothness"), ("com", "Compactness")]:
        means = np.array([np.mean(by_step[s][m]) for s in steps])
        stds  = np.array([np.std(by_step[s][m])  for s in steps])
        base = means[0] if abs(means[0]) > 1e-8 else 1e-8
        nm = means / base
        ns = stds / abs(base)

        ckey = {'sym': 'symmetry', 'smo': 'smoothness', 'com': 'compactness'}[m]
        color = METRIC_COLORS[ckey]
        ax.plot(steps, nm, '-o', color=color, label=label,
                markersize=2.5, markeredgewidth=0, zorder=3)
        ax.fill_between(steps, nm - 0.5*ns, nm + 0.5*ns,
                        alpha=0.10, color=color, linewidth=0, zorder=1)

    ax.axvline(x=50, color='#BBBBBB', linestyle=':', linewidth=0.7, zorder=0)
    ax.annotate('default', xy=(50, 0.05), fontsize=6, color='#999',
                ha='center', va='bottom')

    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Normalized Reward")
    ax.set_ylim(-0.2, 1.15)
    ax.set_xlim(-2, 105)
    ax.legend(loc='upper right', borderpad=0.3, handlelength=1.2,
              fontsize=7, ncol=1)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)

    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT / "convergence_curve.pdf"))
    fig.savefig(str(OUT / "convergence_curve.png"))
    plt.close(fig)
    print("  [OK] convergence_curve")


# ═══════════════════════════════════════════════════════════════════════
# 2. LR Sensitivity
# ═══════════════════════════════════════════════════════════════════════

def fig_lr_sensitivity():
    data = json.load(open("analysis_results/convergence/all_results.json"))
    lr_data = [r for r in data if r.get("experiment") == "lr_sensitivity"]
    if not lr_data: return

    by_lr = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in lr_data:
        by_lr[r["lr"]]["sym"].append(r["symmetry"])
        by_lr[r["lr"]]["smo"].append(r["smoothness"])
        by_lr[r["lr"]]["com"].append(r["compactness"])
    lrs = sorted([lr for lr in by_lr.keys() if lr > 0])  # exclude 0

    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.26))
    metric_map = [("sym", "Symmetry"), ("smo", "Smoothness"), ("com", "Compactness")]

    for i, (m, title) in enumerate(metric_map):
        ax = axes[i]
        means = [np.mean(by_lr[lr][m]) for lr in lrs]
        stds  = [np.std(by_lr[lr][m])  for lr in lrs]
        color = list(METRIC_COLORS.values())[i]

        ax.errorbar(range(len(lrs)), means, yerr=stds,
                    fmt='-o', color=color, capsize=1.5, markersize=3,
                    linewidth=1.0, capthick=0.6, markeredgewidth=0)

        if 0.005 in lrs:
            idx = lrs.index(0.005)
            ax.axvspan(idx-0.4, idx+0.4, color=color, alpha=0.07, zorder=0)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr:.3f}' for lr in lrs], fontsize=5.5, rotation=40, ha='right')
        ax.set_title(title, fontsize=8, pad=3)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Reward", fontsize=8)

    plt.tight_layout(pad=0.4)
    fig.savefig(str(OUT / "lr_sensitivity.pdf"))
    fig.savefig(str(OUT / "lr_sensitivity.png"))
    plt.close(fig)
    print("  [OK] lr_sensitivity")


# ═══════════════════════════════════════════════════════════════════════
# 3. CLIP + ImageReward
# ═══════════════════════════════════════════════════════════════════════

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
    labels = [METHOD_LABELS.get(m, m) for m in avail]
    colors = [METHOD_COLORS.get(m, '#999') for m in avail]
    x = np.arange(len(avail))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, W * 0.33))

    # (a) CLIP
    clip_m = [np.mean(by_method[m]["clip"]) for m in avail]
    clip_s = [np.std(by_method[m]["clip"])  for m in avail]
    ax1.bar(x, clip_m, 0.55, yerr=clip_s, capsize=1.5,
            color=colors, edgecolor='white', linewidth=0.3,
            error_kw={'linewidth': 0.6, 'capthick': 0.5})
    # Baseline reference line
    ax1.axhline(y=clip_m[0], color='#CCCCCC', linestyle='--', linewidth=0.5, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6, rotation=25, ha='right')
    ax1.set_ylabel("CLIP Score", fontsize=8)
    ax1.set_title("(a) Semantic Alignment", fontsize=8, pad=4)
    ax1.set_ylim(0.20, 0.26)
    ax1.grid(True, axis='y', linewidth=0.3, alpha=0.3)

    # (b) ImageReward
    ir_avail = [m for m in avail if len(by_method[m]["ir"]) > 10]
    if ir_avail:
        ir_m = [np.mean(by_method[m]["ir"]) for m in ir_avail]
        ir_s = [np.std(by_method[m]["ir"])  for m in ir_avail]
        x2 = np.arange(len(ir_avail))
        c2 = [METHOD_COLORS.get(m, '#999') for m in ir_avail]
        lb2 = [METHOD_LABELS.get(m, m) for m in ir_avail]

        ax2.bar(x2, ir_m, 0.55, yerr=ir_s, capsize=1.5,
                color=c2, edgecolor='white', linewidth=0.3,
                error_kw={'linewidth': 0.6, 'capthick': 0.5})
        ax2.axhline(y=ir_m[0], color='#CCCCCC', linestyle='--', linewidth=0.5, zorder=0)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(lb2, fontsize=6, rotation=25, ha='right')
        ax2.set_ylabel("ImageReward", fontsize=8)
        ax2.set_title("(b) Perceptual Quality", fontsize=8, pad=4)
        ax2.grid(True, axis='y', linewidth=0.3, alpha=0.3)

    plt.tight_layout(pad=0.4)
    fig.savefig(str(OUT / "clip_comparison.pdf"))
    fig.savefig(str(OUT / "clip_comparison.png"))
    plt.close(fig)
    print("  [OK] clip_comparison")


# ═══════════════════════════════════════════════════════════════════════
# 4. Catastrophic Degradation Heatmap
# ═══════════════════════════════════════════════════════════════════════

def fig_degradation():
    data = json.load(open("analysis_results/laplacian_vs_dgr/checkpoint.json"))
    metrics = ["symmetry", "smoothness", "compactness"]
    metric_labels = ["Sym.", "Smooth.", "Compact."]
    methods_show = ["sym_only", "smooth_only", "compact_only", "diffgeoreward", "handcrafted"]
    method_names = ["Sym-Only", "Smo-Only", "Com-Only", "DGR (Ours)", "Equal Wt."]

    base_means = {}
    for m in metrics:
        vals = [r[m] for r in data if r["method"] == "baseline" and r.get(m) is not None]
        base_means[m] = np.mean(vals) if vals else 0

    matrix = np.zeros((len(methods_show), len(metrics)))
    for i, method in enumerate(methods_show):
        for j, metric in enumerate(metrics):
            vals = [r[metric] for r in data if r["method"] == method and r.get(metric) is not None]
            if vals and abs(base_means[metric]) > 1e-8:
                matrix[i, j] = (np.mean(vals) - base_means[metric]) / abs(base_means[metric]) * 100

    fig, ax = plt.subplots(figsize=(W * 0.42, W * 0.34))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=100)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=7.5)
    ax.set_yticks(range(len(methods_show)))
    ax.set_yticklabels(method_names, fontsize=7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if abs(val) > 55 else '#222222'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.0f}%', ha='center', va='center',
                    fontsize=6.5, color=color, fontweight='bold')

    # Divider between single-reward and multi-reward
    ax.axhline(y=2.5, color='#333333', linewidth=0.8)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.06, aspect=15)
    cb.set_label("vs. Baseline (%)", fontsize=6.5, labelpad=3)
    cb.ax.tick_params(labelsize=5.5)

    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT / "degradation_heatmap.pdf"))
    fig.savefig(str(OUT / "degradation_heatmap.png"))
    plt.close(fig)
    print("  [OK] degradation_heatmap")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating charts...")
    fig_convergence()
    fig_lr_sensitivity()
    fig_clip_ir()
    fig_degradation()
    print("Done.")
