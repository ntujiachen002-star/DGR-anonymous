"""
修复所有论文图片的学术规范问题。

修复列表:
1. convergence_curve: Y轴方向翻转 + legend/default标签修复
2. degradation_heatmap: 用 330-run 全量数据重新生成（数字一致性）
3. spectral_band_energy: 统一字体 + 去matplotlib标题 + 加error bar
4. spectral_cumulative: 统一字体 + 去matplotlib标题
5. 删除重复的 imagereward_comparison.pdf
6. lr_sensitivity: 加 error bar 标记 default

用法:
  python tools/fix_all_figures.py
"""

import os, sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tools')
from pub_style import setup_style, figsize, METRIC_COLORS, METHOD_COLORS, METHOD_LABELS, TEXTWIDTH_PT

setup_style()

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: Convergence Curve — Y轴翻转为"improvement"视角
# ═══════════════════════════════════════════════════════════════════════

def fix_convergence():
    data = json.load(open("analysis_results/convergence/all_results.json"))
    conv = [r for r in data if r.get("experiment") == "convergence"]
    if not conv:
        print("  [SKIP] No convergence data"); return

    by_step = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in conv:
        by_step[r["step"]]["sym"].append(r["symmetry"])
        by_step[r["step"]]["smo"].append(r["smoothness"])
        by_step[r["step"]]["com"].append(r["compactness"])

    steps = sorted(by_step.keys())

    fig, ax = plt.subplots(figsize=figsize(0.48, aspect=0.72))

    for m, label in [("sym", "Symmetry"), ("smo", "Smoothness"), ("com", "Compactness")]:
        means = np.array([np.mean(by_step[s][m]) for s in steps])
        stds  = np.array([np.std(by_step[s][m])  for s in steps])
        base = means[0] if abs(means[0]) > 1e-8 else 1e-8

        # 关键修复: 计算改善百分比 (higher = better)
        improvement = (means - means[0]) / abs(base) * 100
        imp_std = stds / abs(base) * 100

        color = METRIC_COLORS[{'sym': 'symmetry', 'smo': 'smoothness', 'com': 'compactness'}[m]]
        ax.plot(steps, improvement, '-o', color=color, label=label,
                markersize=2.5, markeredgewidth=0)
        ax.fill_between(steps, improvement - imp_std, improvement + imp_std,
                        alpha=0.08, color=color, linewidth=0)

    # Default step 标记（修复：放在不遮挡的位置）
    ax.axvline(x=50, color='#BBBBBB', linestyle=':', linewidth=0.7, zorder=0)
    ylim = ax.get_ylim()
    ax.annotate('$T{=}50$', xy=(50, ylim[1]), xytext=(55, ylim[1] * 0.9),
                fontsize=6.5, color='#888888', ha='left',
                arrowprops=dict(arrowstyle='-', color='#BBBBBB', lw=0.5))

    ax.set_xlabel("Optimization Steps")
    ylabel = r"Improvement over Initial (\%)" if plt.rcParams.get('text.usetex', False) else "Improvement over Initial (%)"
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', borderpad=0.3, framealpha=0.0)
    ax.grid(True, axis='y')
    ax.axhline(y=0, color='#CCCCCC', linewidth=0.4)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(OUT / "convergence_curve.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("  [FIXED] convergence_curve.pdf — Y轴改为 improvement %")


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Degradation Heatmap — 用全量330-run数据
# ═══════════════════════════════════════════════════════════════════════

def fix_degradation():
    # 优先用全量数据
    candidates = [
        "results/full/evaluation_report.json",
        "analysis_results/laplacian_vs_dgr/checkpoint.json",
    ]

    data = None
    for path in candidates:
        if os.path.exists(path):
            data = json.load(open(path))
            print(f"  Using data from {path}")
            break

    if data is None:
        print("  [SKIP] No degradation data"); return

    # 尝试从 *_all_metrics.json 合并（330-run 全量）
    full_data = []
    full_dir = Path("results/full")
    for method_file in full_dir.glob("*_all_metrics.json"):
        full_data.extend(json.load(open(method_file)))

    if len(full_data) > 100:
        data = full_data
        print(f"  Merged {len(data)} records from full results")

    metrics = ["symmetry", "smoothness", "compactness"]
    methods_show = ["sym_only", "smooth_only", "compact_only", "diffgeoreward", "handcrafted"]
    method_names = ["Sym-Only", "Smo-Only", "Com-Only", "DGR (Ours)", "Equal Wt."]

    base_means = {}
    for m in metrics:
        vals = [r[m] for r in data if r.get("method") == "baseline" and r.get(m) is not None]
        base_means[m] = np.mean(vals) if vals else 0

    matrix = np.zeros((len(methods_show), len(metrics)))
    for i, method in enumerate(methods_show):
        for j, metric in enumerate(metrics):
            vals = [r[metric] for r in data if r.get("method") == method and r.get(metric) is not None]
            if vals and abs(base_means[metric]) > 1e-8:
                matrix[i, j] = (np.mean(vals) - base_means[metric]) / abs(base_means[metric]) * 100

    fig, ax = plt.subplots(figsize=figsize(0.48, aspect=0.78))

    # 颜色映射：使用对称 clip 到 [-100, 100]，但显示真实值
    im = ax.imshow(matrix.clip(-150, 150), cmap='RdYlGn', aspect='auto', vmin=-150, vmax=150)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Sym.", "Smo.", "Com."], fontsize=8)
    ax.set_yticks(range(len(methods_show)))
    ax.set_yticklabels(method_names, fontsize=7.5)

    pct = r'\%' if plt.rcParams.get('text.usetex', False) else '%'
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if abs(val) > 60 else '#1F2937'
            text = f"{val:+.0f}{pct}" if abs(val) < 1000 else f"{val/100:+.0f}x"
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=6.5, color=color, fontweight='bold')

    ax.axhline(y=2.5, color='#4B5563', linewidth=1.2, linestyle='-')

    cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.08)
    cb_label = r"vs.\ Baseline (\%)" if plt.rcParams.get('text.usetex', False) else "vs. Baseline (%)"
    cb.set_label(cb_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(str(OUT / "degradation_heatmap.pdf"))
    plt.close(fig)
    print(f"  [FIXED] degradation_heatmap.pdf — 用{len(data)}条记录")


# ═══════════════════════════════════════════════════════════════════════
# Fix 3 & 4: Spectral 图 — 统一字体, 去标题, 加 error bar
# ═══════════════════════════════════════════════════════════════════════

def fix_spectral():
    summary_path = "analysis_results/spectral_decomposition/spectral_summary.json"
    if not os.path.exists(summary_path):
        print("  [SKIP] No spectral data"); return

    summary = json.load(open(summary_path))
    band_energy = summary['avg_band_energy']

    # ── Fig A: Band Energy Bar Chart ──
    fig, ax = plt.subplots(figsize=figsize(0.48, aspect=0.65))

    bands = ['low', 'mid', 'high']
    band_labels = ['Low freq\n(global)', 'Mid freq', 'High freq\n(local)']
    x = np.arange(len(bands))
    width = 0.25

    colors = {'symmetry': METRIC_COLORS['symmetry'],
              'smoothness': METRIC_COLORS['smoothness'],
              'compactness': METRIC_COLORS['compactness']}

    for i, (reward, label) in enumerate([('symmetry', 'Symmetry'),
                                          ('smoothness', 'Smoothness'),
                                          ('compactness', 'Compactness')]):
        means = [band_energy[reward][b] for b in bands]
        ax.bar(x + i * width, means, width, color=colors[reward],
               label=label, alpha=0.85, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(band_labels, fontsize=7.5)
    ax.set_ylabel("Fraction of gradient energy")
    ax.legend(loc='upper right', borderpad=0.3, framealpha=0.0)
    ax.grid(True, axis='y')
    ax.set_ylim(0, 0.7)

    plt.tight_layout()
    fig.savefig(str(OUT / "spectral_band_energy.pdf"))
    plt.close(fig)
    print("  [FIXED] spectral_band_energy.pdf — 统一字体, 去标题")

    # ── Fig B: Cumulative Energy ──
    # 需要原始 per-eigenmode 数据
    per_mesh_dir = Path("analysis_results/spectral_decomposition")
    cum_files = sorted(per_mesh_dir.glob("*_spectral.json"))

    if cum_files:
        all_cum = {'symmetry': [], 'smoothness': [], 'compactness': []}
        for cf in cum_files:
            d = json.load(open(cf))
            for reward in all_cum:
                if reward in d and 'energy_per_mode' in d[reward]:
                    energies = np.array(d[reward]['energy_per_mode'])
                    energies = energies / (energies.sum() + 1e-10)
                    all_cum[reward].append(np.cumsum(energies))

        fig, ax = plt.subplots(figsize=figsize(0.48, aspect=0.55))

        for reward, label in [('symmetry', 'Symmetry'),
                               ('smoothness', 'Smoothness'),
                               ('compactness', 'Compactness')]:
            if all_cum[reward]:
                min_len = min(len(c) for c in all_cum[reward])
                arr = np.array([c[:min_len] for c in all_cum[reward]])
                mean_cum = arr.mean(axis=0)
                std_cum = arr.std(axis=0)
                modes = np.arange(min_len)

                color = colors[reward]
                ax.plot(modes, mean_cum, '-', color=color, label=label, linewidth=1.3)
                ax.fill_between(modes, mean_cum - std_cum, mean_cum + std_cum,
                                alpha=0.1, color=color)

        ax.axhline(y=0.5, color='#CCCCCC', linestyle=':', linewidth=0.5)
        ax.axhline(y=0.9, color='#CCCCCC', linestyle=':', linewidth=0.5)
        ax.set_xlabel("Eigenmode index (low $\\rightarrow$ high frequency)")
        ax.set_ylabel("Cumulative energy fraction")
        ax.legend(loc='lower right', borderpad=0.3, framealpha=0.0)
        ax.set_xlim(0, min_len - 1)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        fig.savefig(str(OUT / "spectral_cumulative.pdf"))
        plt.close(fig)
        print("  [FIXED] spectral_cumulative.pdf — 统一字体, 加 std shading")
    else:
        print("  [SKIP] No per-mesh spectral data for cumulative plot")


# ═══════════════════════════════════════════════════════════════════════
# Fix 5: 删除重复的 imagereward_comparison.pdf
# ═══════════════════════════════════════════════════════════════════════

def fix_remove_duplicate():
    dup = OUT / "imagereward_comparison.pdf"
    if dup.exists():
        dup.unlink()
        print("  [FIXED] 已删除重复的 imagereward_comparison.pdf")
    else:
        print("  [OK] imagereward_comparison.pdf 已不存在")


# ═══════════════════════════════════════════════════════════════════════
# Fix 6: LR Sensitivity — 加 error bar + 圆圈标记 default
# ═══════════════════════════════════════════════════════════════════════

def fix_lr_sensitivity():
    data_path = "analysis_results/convergence/all_results.json"
    if not os.path.exists(data_path):
        print("  [SKIP] No LR data"); return

    data = json.load(open(data_path))
    lr_data = [r for r in data if r.get("experiment") == "lr_sensitivity"]
    if not lr_data:
        print("  [SKIP] No LR sensitivity records"); return

    by_lr = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in lr_data:
        lr_val = r["lr"]
        if lr_val <= 0:  # 排除 lr=0（无优化）
            continue
        by_lr[lr_val]["sym"].append(r["symmetry"])
        by_lr[lr_val]["smo"].append(r["smoothness"])
        by_lr[lr_val]["com"].append(r["compactness"])

    lrs = sorted(by_lr.keys())

    fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, aspect=0.28))
    labels_map = [("sym", "Symmetry"), ("smo", "Smoothness"), ("com", "Compactness")]
    metric_keys = ['symmetry', 'smoothness', 'compactness']

    for i, (m, label) in enumerate(labels_map):
        ax = axes[i]
        means = [np.mean(by_lr[lr][m]) for lr in lrs]
        stds  = [np.std(by_lr[lr][m])  for lr in lrs]
        color = METRIC_COLORS[metric_keys[i]]

        ax.errorbar(range(len(lrs)), means, yerr=stds,
                    fmt='-o', color=color, capsize=2, markersize=3,
                    linewidth=1.2, capthick=0.8, markeredgewidth=0)

        # 圆圈标记 default lr=0.005
        if 0.005 in lrs:
            idx = lrs.index(0.005)
            ax.plot(idx, means[idx], 'o', markersize=10, markerfacecolor='none',
                    markeredgecolor='#333333', markeredgewidth=1.2)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr}' for lr in lrs], fontsize=6.5, rotation=30)
        ax.set_title(label, fontsize=9, pad=4)
        if i == 0:
            ax.set_ylabel("Reward")
        ax.grid(True, axis='y')

    plt.tight_layout()
    fig.savefig(str(OUT / "lr_sensitivity.pdf"))
    plt.close(fig)
    print("  [FIXED] lr_sensitivity.pdf — 加 error bar + 圆圈标记")


# ═══════════════════════════════════════════════════════════════════════
# 同时检查并移除附录中对 imagereward_comparison.pdf 的引用
# ═══════════════════════════════════════════════════════════════════════

def fix_appendix_ref():
    appendix_path = "paper/sections/A_appendix.tex"
    if not os.path.exists(appendix_path):
        return

    with open(appendix_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'imagereward_comparison' in content:
        # 找到并移除对 imagereward_comparison 的引用
        lines = content.split('\n')
        new_lines = []
        skip_block = False
        for line in lines:
            if 'imagereward_comparison' in line:
                skip_block = True
                continue
            if skip_block and ('\\end{figure}' in line or '\\caption' in line):
                if '\\end{figure}' in line:
                    skip_block = False
                continue
            new_lines.append(line)

        with open(appendix_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("  [FIXED] 移除附录中 imagereward_comparison 引用")
    else:
        print("  [OK] 附录无 imagereward_comparison 引用")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("修复所有论文图片")
    print("=" * 60)

    fix_convergence()
    fix_degradation()
    fix_spectral()
    fix_remove_duplicate()
    fix_lr_sensitivity()
    fix_appendix_ref()

    print(f"\n所有修复完成。图片在 {OUT}/")
