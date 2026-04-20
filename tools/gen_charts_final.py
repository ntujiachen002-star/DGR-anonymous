"""
Final publication charts — only convergence + degradation heatmap.
CLIP/ImageReward results → Table in LaTeX (not bar chart).
Style: STIX fonts (Computer Modern lookalike), minimal, NeurIPS column width.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# ── NeurIPS-matching style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#111111',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'axes.grid': False,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# NeurIPS textwidth = 5.5 inches
TW = 5.5
GOLDEN = 0.618

# Colors — from SciencePlots, colorblind-friendly
C_RED   = '#D62728'
C_BLUE  = '#1F77B4'
C_GREEN = '#2CA02C'


# ═══════════════════════════════════════════════════════════════════════
# Figure: Convergence Curve (half-width, goes in main text)
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

    w = TW * 0.48
    fig, ax = plt.subplots(figsize=(w, w * GOLDEN))

    for m, label, color in [
        ("sym", "Symmetry",    C_RED),
        ("smo", "Smoothness",  C_BLUE),
        ("com", "Compactness", C_GREEN),
    ]:
        means = np.array([np.mean(by_step[s][m]) for s in steps])
        stds  = np.array([np.std(by_step[s][m])  for s in steps])
        base = means[0] if abs(means[0]) > 1e-8 else 1e-8
        nm = means / base
        ax.plot(steps, nm, '-o', color=color, label=label,
                markersize=3, markeredgewidth=0, zorder=3)

    ax.axvline(x=50, color='#999999', linestyle=':', linewidth=0.6, zorder=0)
    ax.text(52, 0.08, 'default', fontsize=7, color='#888888', va='bottom')

    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Normalized reward")
    ax.set_ylim(-0.1, 1.12)
    ax.set_xlim(-2, 102)
    ax.legend(loc='upper right', borderpad=0.3)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.3)

    plt.tight_layout(pad=0.4)
    fig.savefig(str(OUT / "convergence_curve.pdf"))
    fig.savefig(str(OUT / "convergence_curve.png"))
    plt.close(fig)
    print("  [OK] convergence_curve")


# ═══════════════════════════════════════════════════════════════════════
# Figure: Catastrophic Degradation Heatmap (half-width, main text)
# ═══════════════════════════════════════════════════════════════════════

def fig_degradation():
    data = json.load(open("analysis_results/laplacian_vs_dgr/checkpoint.json"))
    metrics = ["symmetry", "smoothness", "compactness"]
    metric_labels = ["Sym.", "Smo.", "Com."]
    methods = ["sym_only", "smooth_only", "compact_only", "diffgeoreward", "handcrafted"]
    method_labels = ["Sym-Only", "Smo-Only", "Com-Only", "DGR (Ours)", "Equal Weights"]

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

    w = TW * 0.48
    fig, ax = plt.subplots(figsize=(w, w * 0.72))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-120, vmax=120)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(method_labels, fontsize=7.5)
    ax.tick_params(length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if abs(val) > 60 else '#111111'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.0f}%', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    ax.axhline(y=2.5, color='#333333', linewidth=0.8)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.05, aspect=18)
    cb.set_label("vs. baseline (%)", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(OUT / "degradation_heatmap.pdf"))
    fig.savefig(str(OUT / "degradation_heatmap.png"))
    plt.close(fig)
    print("  [OK] degradation_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# Figure: LR Sensitivity (full-width, appendix)
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
    lrs = sorted([lr for lr in by_lr.keys() if lr > 0])

    fig, axes = plt.subplots(1, 3, figsize=(TW, TW * 0.25))

    for i, (m, title, color) in enumerate([
        ("sym", "Symmetry", C_RED),
        ("smo", "Smoothness", C_BLUE),
        ("com", "Compactness", C_GREEN),
    ]):
        ax = axes[i]
        means = [np.mean(by_lr[lr][m]) for lr in lrs]

        # No error bars — just mean line (standard in A-tier papers for hyperparam sensitivity)
        ax.plot(range(len(lrs)), means, '-o', color=color,
                markersize=4, linewidth=1.3, markeredgewidth=0, zorder=3)

        # Highlight default lr=0.005 with a subtle marker
        if 0.005 in lrs:
            idx = lrs.index(0.005)
            ax.plot(idx, means[idx], 'o', color=color, markersize=7,
                    markeredgecolor='#333333', markeredgewidth=1.2,
                    markerfacecolor='none', zorder=4)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr}' for lr in lrs], fontsize=6.5, rotation=30, ha='right')
        ax.set_title(title, fontsize=9, pad=3)
        ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)
        if i == 0:
            ax.set_ylabel("Reward", fontsize=9)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(OUT / "lr_sensitivity.pdf"))
    fig.savefig(str(OUT / "lr_sensitivity.png"))
    plt.close(fig)
    print("  [OK] lr_sensitivity")


# ═══════════════════════════════════════════════════════════════════════
# Figure: CLIP + ImageReward (full-width, compact style)
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
    labels_map = {"baseline": "Base.", "diffgeoreward": "DGR", "handcrafted": "Equal",
                  "sym_only": "Sym", "smooth_only": "Smo", "compact_only": "Com"}
    avail = [m for m in order if m in by_method and len(by_method[m]["clip"]) > 10]
    x = np.arange(len(avail))

    # Muted, consistent palette
    bar_colors = {
        'baseline': '#AAAAAA', 'diffgeoreward': '#1F77B4',
        'handcrafted': '#2CA02C', 'sym_only': '#D62728',
        'smooth_only': '#FF7F0E', 'compact_only': '#9467BD'
    }

    # Use SEM (standard error of mean) instead of std — much smaller, appropriate for comparing means
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    clip_m = [np.mean(by_method[m]["clip"]) for m in avail]
    clip_sem = [np.std(by_method[m]["clip"]) / np.sqrt(len(by_method[m]["clip"])) for m in avail]
    colors = [bar_colors.get(m, '#AAA') for m in avail]

    ax1.bar(x, clip_m, 0.6, yerr=clip_sem, capsize=3,
            color=colors, edgecolor='none',
            error_kw={'linewidth': 1, 'capthick': 0.8})
    ax1.axhline(y=clip_m[0], color='#CCCCCC', linestyle='--', linewidth=0.5, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels_map.get(m, m) for m in avail], fontsize=11)
    ax1.set_ylabel("CLIP score", fontsize=12)
    ax1.set_title("(a) Semantic alignment", fontsize=13, fontweight='bold')
    ax1.set_ylim(0.225, 0.245)
    ax1.tick_params(labelsize=10)
    ax1.yaxis.grid(True, linewidth=0.3, alpha=0.25)

    # (b) ImageReward
    ir_avail = [m for m in avail if len(by_method[m]["ir"]) > 10]
    if ir_avail:
        ir_m = [np.mean(by_method[m]["ir"]) for m in ir_avail]
        ir_sem = [np.std(by_method[m]["ir"]) / np.sqrt(len(by_method[m]["ir"])) for m in ir_avail]
        x2 = np.arange(len(ir_avail))
        c2 = [bar_colors.get(m, '#AAA') for m in ir_avail]

        ax2.bar(x2, ir_m, 0.6, yerr=ir_sem, capsize=3,
                color=c2, edgecolor='none',
                error_kw={'linewidth': 1, 'capthick': 0.8})
        ax2.axhline(y=ir_m[0], color='#CCCCCC', linestyle='--', linewidth=0.5, zorder=0)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([labels_map.get(m, m) for m in ir_avail], fontsize=11)
        ax2.set_ylabel("ImageReward", fontsize=12)
        ax2.set_title("(b) Perceptual quality", fontsize=13, fontweight='bold')
        ax2.tick_params(labelsize=10)
        ax2.yaxis.grid(True, linewidth=0.3, alpha=0.25)

    plt.tight_layout(pad=0.8)
    fig.patch.set_facecolor('white')
    fig.savefig(str(OUT / "clip_comparison.pdf"), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(OUT / "clip_comparison.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  [OK] clip_comparison")


if __name__ == "__main__":
    print("Generating final charts...")
    fig_convergence()
    fig_degradation()
    fig_lr_sensitivity()
    fig_clip_ir()
    print("Done.")
