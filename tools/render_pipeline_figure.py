"""
Generate Pipeline Overview Figure (Figure 1) for the paper.
Shows: Text Prompt -> Backbone -> Coarse Mesh -> DiffGeoReward -> Refined Mesh
Output: paper/figures/pipeline.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
OUT = Path("paper/figures")

def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color scheme
    c_input = '#E8F4FD'
    c_backbone = '#FFF3E0'
    c_reward = '#E8F5E9'
    c_output = '#F3E5F5'
    c_lang = '#FCE4EC'
    border = '#37474F'

    def box(x, y, w, h, text, color, fontsize=8, bold=False):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.12",
                               facecolor=color, edgecolor=border,
                               linewidth=1.2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color='#1a1a1a',
                wrap=True)

    def arrow(x1, y1, x2, y2, style='->', color='#546E7A'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    lw=1.5, connectionstyle='arc3,rad=0'))

    def curved_arrow(x1, y1, x2, y2, rad=0.3, color='#546E7A'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.2, connectionstyle=f'arc3,rad={rad}'))

    # ── Row 1: Main pipeline ──
    # Text Prompt
    box(0, 1.2, 1.8, 1.0, '"a symmetric\nvase"', c_input, fontsize=9, bold=True)

    # Arrow
    arrow(1.8, 1.7, 2.3, 1.7)

    # Backbone
    box(2.3, 1.2, 1.8, 1.0, 'Text-to-3D\nBackbone\n(Shap-E /\nTripoSR)', c_backbone, fontsize=7)

    # Arrow
    arrow(4.1, 1.7, 4.6, 1.7)

    # Coarse Mesh
    box(4.6, 1.2, 1.6, 1.0, 'Coarse\nMesh', c_backbone, fontsize=9)

    # Arrow to DGR
    arrow(6.2, 1.7, 6.7, 1.7)

    # DiffGeoReward block (main)
    rect_dgr = FancyBboxPatch((6.7, 0.4), 3.6, 2.4,
                               boxstyle="round,pad=0.15",
                               facecolor='#E3F2FD', edgecolor='#1565C0',
                               linewidth=2.0, linestyle='-')
    ax.add_patch(rect_dgr)
    ax.text(8.5, 2.55, 'DiffGeoReward', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1565C0')

    # Three reward boxes inside DGR
    rw = 1.0
    rh = 0.55
    ry = 1.55
    box(6.95, ry, rw, rh, 'Symmetry\n$\\mathcal{R}_{sym}$', '#FFCDD2', fontsize=7)
    box(8.0, ry, rw, rh, 'Smoothness\n$\\mathcal{R}_{smo}$', '#C8E6C9', fontsize=7)
    box(9.05, ry, rw, rh, 'Compactness\n$\\mathcal{R}_{com}$', '#BBDEFB', fontsize=7)

    # Gradient ascent box
    box(7.5, 0.6, 2.0, 0.65, 'Adam Gradient Ascent\n(50 steps on vertices)', '#E1F5FE', fontsize=7, bold=True)

    # Arrows from rewards to gradient
    for rx in [7.45, 8.5, 9.55]:
        arrow(rx, ry, 8.5, 1.25, color='#90A4AE')

    # Arrow out of DGR
    arrow(10.3, 1.7, 10.8, 1.7)

    # Refined Mesh
    box(10.8, 1.2, 1.5, 1.0, 'Refined\nMesh', c_output, fontsize=9, bold=True)

    # ── Row 2: Lang2Comp (optional) ──
    # Lang2Comp box below
    box(1.5, -0.7, 2.2, 0.7, 'Lang2Comp\n(Optional)', c_lang, fontsize=8)

    # Arrow from text prompt to Lang2Comp
    curved_arrow(0.9, 1.2, 2.0, 0.0, rad=0.3, color='#E91E63')

    # Arrow from Lang2Comp to weights
    ax.annotate('', xy=(7.2, 0.6), xytext=(3.7, -0.35),
                arrowprops=dict(arrowstyle='->', color='#E91E63',
                                lw=1.2, linestyle='--',
                                connectionstyle='arc3,rad=-0.2'))
    ax.text(5.3, -0.3, '$w_{sym}, w_{smo}, w_{com}$',
            fontsize=7, color='#E91E63', fontstyle='italic')

    # Legend annotations
    ax.text(0.0, 3.0, 'Training-free | Zero overhead | Backbone-agnostic',
            fontsize=8, color='#666666', fontstyle='italic')

    plt.tight_layout()
    out_path = OUT / "pipeline.pdf"
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    fig.savefig(str(OUT / "pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    draw_pipeline()
