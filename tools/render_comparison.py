#!/usr/bin/env python
"""M1: Render mesh comparison figures for qualitative evaluation."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

RESULTS_DIR = 'results/full'


def render_mesh(mesh_path, ax, color='lightblue'):
    mesh = trimesh.load(mesh_path)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale
    if len(faces) > 5000:
        idx = np.random.choice(len(faces), 5000, replace=False)
        faces = faces[idx]
    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=0.7, facecolor=color, edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=25, azim=45)
    ax.axis('off')


def get_category(prompt):
    if prompt in SYMMETRY_PROMPTS:
        return 'symmetry'
    elif prompt in SMOOTHNESS_PROMPTS:
        return 'smoothness'
    elif prompt in COMPACTNESS_PROMPTS:
        return 'compactness'
    return 'symmetry'


def find_mesh(method, prompt, seed=42):
    cat = get_category(prompt)
    path = f'{RESULTS_DIR}/{method}/{cat}/{method}_seed{seed}.obj'
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return path
    return None


def load_all_metrics():
    metrics = {}
    for name in ['baseline', 'diffgeoreward', 'handcrafted']:
        for p in ['results/', 'results/full/']:
            path = f'{p}{name}_all_metrics.json'
            if os.path.exists(path) and os.path.getsize(path) > 200:
                with open(path) as f:
                    data = json.load(f)
                for entry in data:
                    key = (name, entry.get('prompt', ''), entry.get('seed', 42))
                    metrics[key] = entry
                break
    return metrics


def figure1():
    """Figure 1: Representative results 3x4 grid."""
    prompts = [
        "a symmetric vase",
        "a smooth river stone",
        "a compact stone",
        "a balanced chess piece, a king",
    ]
    methods = ['baseline', 'diffgeoreward', 'handcrafted']
    labels = ['Baseline (Shap-E)', 'DiffGeoReward', 'Handcrafted (Equal)']
    colors = ['#AEC6CF', '#77DD77', '#FFB347']
    all_m = load_all_metrics()

    fig, axes = plt.subplots(len(methods), len(prompts), figsize=(16, 12),
                              subplot_kw={'projection': '3d'})

    for j, prompt in enumerate(prompts):
        for i, method in enumerate(methods):
            ax = axes[i][j]
            mesh_path = find_mesh(method, prompt)
            if mesh_path:
                render_mesh(mesh_path, ax, color=colors[i])
                m = all_m.get((method, prompt, 42))
                if m:
                    s = m.get('symmetry', 0)
                    sm = m.get('smoothness', 0)
                    c = m.get('compactness', 0)
                    ax.text2D(0.5, -0.05, f'Sym:{s:.4f} Smo:{sm:.5f} Com:{c:.1f}',
                             transform=ax.transAxes, ha='center', fontsize=6)
            else:
                ax.text(0, 0, 0, 'N/A', ha='center', fontsize=12)
                ax.axis('off')

            if i == 0:
                short = prompt[:30] + '...' if len(prompt) > 30 else prompt
                ax.set_title(f'"{short}"', fontsize=9, pad=10)
            if j == 0:
                ax.text2D(-0.15, 0.5, labels[i], transform=ax.transAxes,
                         ha='center', va='center', fontsize=10, rotation=90)

    fig.suptitle('Qualitative Comparison of Mesh Refinement', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    os.makedirs('paper/figures', exist_ok=True)
    fig.savefig('paper/figures/qualitative_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('paper/figures/qualitative_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")


def figure2():
    """Figure 2: Failure cases."""
    all_m = load_all_metrics()
    bl_metrics = {(m['prompt'], m['seed']): m for k, m in all_m.items() if k[0] == 'baseline'}
    dgr_metrics = {(m['prompt'], m['seed']): m for k, m in all_m.items() if k[0] == 'diffgeoreward'}

    failures = []
    for key, m in dgr_metrics.items():
        bl = bl_metrics.get(key)
        if bl and abs(bl['compactness']) > 0.1:
            change = (m['compactness'] - bl['compactness']) / abs(bl['compactness']) * 100
            if change < -30:
                failures.append({'prompt': key[0], 'seed': key[1], 'change': change})

    failures.sort(key=lambda x: x['change'])
    selected = failures[:3] if failures else []

    if not selected:
        print("No failure cases found, skipping Figure 2.")
        return

    fig, axes = plt.subplots(2, len(selected), figsize=(5 * len(selected), 10),
                              subplot_kw={'projection': '3d'})
    if len(selected) == 1:
        axes = axes.reshape(2, 1)

    for j, fail in enumerate(selected):
        for i, (method, color, label) in enumerate([
            ('baseline', '#AEC6CF', 'Before'),
            ('diffgeoreward', '#FF6B6B', 'After')
        ]):
            ax = axes[i][j]
            mesh_path = find_mesh(method, fail['prompt'], fail['seed'])
            if mesh_path:
                render_mesh(mesh_path, ax, color=color)
            else:
                ax.text(0, 0, 0, 'N/A', ha='center')
                ax.axis('off')
            if i == 0:
                ax.set_title(f'"{fail["prompt"][:35]}"\nCompact: {fail["change"]:.0f}%',
                           fontsize=8, pad=5)
            if j == 0:
                ax.text2D(-0.1, 0.5, label, transform=ax.transAxes,
                         ha='center', va='center', fontsize=10, rotation=90)

    fig.suptitle('Failure Cases: Compactness Degradation', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.savefig('paper/figures/failure_cases.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('paper/figures/failure_cases.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    figure1()
    figure2()
