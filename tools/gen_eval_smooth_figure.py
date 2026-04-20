"""
Generate comparison figure for smoothness human evaluation.
Renders baseline vs DGR (HNC) side-by-side for polished gemstone prompt.
"""
import sys, os, numpy as np
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT_DIR = Path("analysis_results/huber_nc_rerun/human_eval_v12/smooth_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    {
        "prompt": "a polished gemstone",
        "slug": "a_polished_gemstone",
        "base_path": "results/mesh_validity_objs/baseline/smoothness/a_polished_gemstone_seed42.obj",
        "dgr_path": "results/mesh_validity_objs/diffgeoreward_huber/smoothness/a_polished_gemstone_seed42.obj",
        "A_is_dgr": True,   # randomized: DGR on left (A), baseline on right (B)
        "out": "qs01.png"
    }
]


def load_obj_np(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def render_mesh(ax, verts, faces, angle_elev=25, angle_azim=35):
    """Render mesh with Phong-like shading."""
    # Normalize
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale * 0.85

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals = normals / norms

    light_dir = np.array([0.5, 0.7, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    diffuse = 0.25 + 0.75 * np.clip(normals @ light_dir, 0, 1)
    base_c = np.array([0.55, 0.68, 0.82])
    fc = np.column_stack([np.outer(diffuse, base_c), np.ones(len(diffuse))])
    fc = np.clip(fc, 0, 1)

    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=1.0)
    pc.set_facecolor(fc)
    pc.set_edgecolor('none')
    ax.add_collection3d(pc)

    margin = 1.0
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)
    ax.view_init(elev=angle_elev, azim=angle_azim)
    ax.set_axis_off()


def make_comparison(pair):
    verts_b, faces_b = load_obj_np(pair["base_path"])
    verts_d, faces_d = load_obj_np(pair["dgr_path"])

    # 2 views per mesh: front-3/4 and back-3/4
    angles = [(25, 35), (25, 155)]

    fig = plt.figure(figsize=(14, 4.2), facecolor='white')

    if pair["A_is_dgr"]:
        left_v, left_f = verts_d, faces_d
        right_v, right_f = verts_b, faces_b
        left_label = "Method A"
        right_label = "Method B"
    else:
        left_v, left_f = verts_b, faces_b
        right_v, right_f = verts_d, faces_d
        left_label = "Method A"
        right_label = "Method B"

    n_views = len(angles)
    # Left side: n_views columns for A, then gap, then n_views columns for B
    total_cols = n_views * 2 + 1   # +1 for divider
    ax_left = []
    ax_right = []
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, total_cols, i + 1, projection='3d')
        render_mesh(ax, left_v, left_f, angle_elev=elev, angle_azim=azim)
        ax_left.append(ax)
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, total_cols, n_views + 2 + i, projection='3d')
        render_mesh(ax, right_v, right_f, angle_elev=elev, angle_azim=azim)
        ax_right.append(ax)

    # Add labels
    fig.text(0.24, 0.96, left_label, ha='center', va='top', fontsize=15,
             fontweight='bold', color='#1a73e8')
    fig.text(0.74, 0.96, right_label, ha='center', va='top', fontsize=15,
             fontweight='bold', color='#e53935')

    # Divider line
    fig.add_artist(plt.Line2D([0.495, 0.495], [0.0, 1.0],
                               transform=fig.transFigure,
                               color='#cccccc', linewidth=1.5, linestyle='--'))

    # Prompt
    fig.text(0.5, 0.02, f'Prompt: "{pair["prompt"]}"', ha='center', va='bottom',
             fontsize=10, color='#555', style='italic')

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    out_path = OUT_DIR / pair["out"]
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for pair in PAIRS:
        print(f"Generating: {pair['prompt']}")
        make_comparison(pair)
    print("Done.")
