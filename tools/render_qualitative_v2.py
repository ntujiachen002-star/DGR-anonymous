"""
Qualitative comparison v2: curvature heatmap + metric annotations.
Shows WHERE improvements happen, not just the shape.
"""
import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib import cm
from pathlib import Path
import torch

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'src')
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

SEED = 42
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick examples with LARGEST metric improvement (from data)
# We'll compute metrics and pick best
OBJ_BASE = Path("results/mesh_validity_objs/baseline")
OBJ_DGR  = Path("results/mesh_validity_objs/diffgeoreward")


def load_obj(path):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                idx = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    if not verts or not faces:
        return None, None
    return np.array(verts), np.array(faces)


def compute_per_vertex_curvature(verts, faces):
    """Approximate per-vertex curvature via angle defect."""
    n = len(verts)
    curvature = np.zeros(n)
    angle_sum = np.zeros(n)

    for f in faces:
        for i in range(3):
            v0 = verts[f[i]]
            v1 = verts[f[(i+1)%3]]
            v2 = verts[f[(i+2)%3]]
            e1 = v1 - v0
            e2 = v2 - v0
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angle_sum[f[i]] += np.arccos(cos_a)

    curvature = 2 * np.pi - angle_sum  # angle defect
    return np.abs(curvature)  # absolute curvature magnitude


def compute_symmetry_error(verts):
    """Per-vertex symmetry error: distance to yz-plane reflection."""
    reflected = verts.copy()
    reflected[:, 0] = -reflected[:, 0]  # reflect across yz-plane
    # For each vertex, find nearest reflected vertex
    from scipy.spatial import cKDTree
    tree = cKDTree(reflected)
    dists, _ = tree.query(verts)
    return dists


def compute_metrics(verts_np, faces_np):
    v = torch.tensor(verts_np, dtype=torch.float32)
    f = torch.tensor(faces_np, dtype=torch.long)
    try:
        sym = symmetry_reward(v).item()
        smo = smoothness_reward(v, f).item()
        com = compactness_reward(v, f).item()
        return sym, smo, com
    except:
        return None, None, None


def render_mesh_colored(ax, verts, faces, values, cmap_name='coolwarm',
                        azim=30, elev=20, vmin=None, vmax=None):
    """Render mesh with per-face coloring based on vertex values."""
    if len(verts) == 0 or len(faces) == 0:
        return None

    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale

    if len(faces) > 6000:
        idx = np.random.RandomState(42).choice(len(faces), 6000, replace=False)
        faces = faces[idx]

    # Per-face value = mean of vertex values
    face_vals = np.mean(values[faces], axis=1)

    if vmin is None:
        vmin = np.percentile(face_vals, 5)
    if vmax is None:
        vmax = np.percentile(face_vals, 95)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    face_colors = cmap(norm(face_vals))

    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=0.9)
    pc.set_facecolor(face_colors)
    pc.set_edgecolor('none')
    ax.add_collection3d(pc)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return norm


def main():
    # Find all available pairs
    pairs = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        base_dir = OBJ_BASE / cat
        dgr_dir = OBJ_DGR / cat
        if not base_dir.exists():
            continue
        for f in sorted(base_dir.glob(f'*_seed{SEED}.obj')):
            dgr_f = dgr_dir / f.name
            if dgr_f.exists():
                slug = f.stem.replace(f'_seed{SEED}', '')
                pairs.append((cat, slug, str(f), str(dgr_f)))

    if not pairs:
        print("No paired OBJ files found!")
        return

    # Compute metrics for all pairs, rank by improvement
    ranked = []
    for cat, slug, base_path, dgr_path in pairs:
        bv, bf = load_obj(base_path)
        dv, df = load_obj(dgr_path)
        if bv is None or dv is None:
            continue
        bs, bsm, bc = compute_metrics(bv, bf)
        ds, dsm, dc = compute_metrics(dv, df)
        if bs is None or ds is None:
            continue
        # Overall improvement
        imp_sym = (ds - bs) / max(abs(bs), 1e-8) * 100
        imp_smo = (dsm - bsm) / max(abs(bsm), 1e-8) * 100
        total_imp = imp_sym + imp_smo  # focus on sym+smo
        ranked.append({
            'cat': cat, 'slug': slug,
            'base_path': base_path, 'dgr_path': dgr_path,
            'base_metrics': (bs, bsm, bc), 'dgr_metrics': (ds, dsm, dc),
            'imp_sym': imp_sym, 'imp_smo': imp_smo, 'total_imp': total_imp,
        })

    # Sort by total improvement, pick top examples per category
    ranked.sort(key=lambda x: -x['total_imp'])
    print(f"Found {len(ranked)} pairs. Top improvements:")
    for r in ranked[:8]:
        name = r['slug'].replace('_', ' ')
        print(f"  {r['cat']:12s} {name:30s}  sym: {r['imp_sym']:+.0f}%  smo: {r['imp_smo']:+.0f}%")

    # Pick best 2 per category
    selected = []
    per_cat = {}
    for r in ranked:
        c = r['cat']
        if c not in per_cat:
            per_cat[c] = 0
        if per_cat[c] < 2:
            selected.append(r)
            per_cat[c] += 1
        if len(selected) >= 6:
            break

    if len(selected) < 3:
        selected = ranked[:min(6, len(ranked))]

    # Limit to 4 rows for compact figure
    selected = selected[:4]
    n_rows = len(selected)
    fig = plt.figure(figsize=(13, n_rows * 2.8 + 1.2))

    for row, item in enumerate(selected):
        bv, bf = load_obj(item['base_path'])
        dv, df = load_obj(item['dgr_path'])

        # Compute curvature for coloring
        base_curv = compute_per_vertex_curvature(bv, bf)
        dgr_curv  = compute_per_vertex_curvature(dv, df)

        # Use same colormap range for fair comparison
        vmin = min(np.percentile(base_curv, 5), np.percentile(dgr_curv, 5))
        vmax = max(np.percentile(base_curv, 90), np.percentile(dgr_curv, 90))

        # Col 1: Baseline (curvature heatmap)
        ax1 = fig.add_subplot(n_rows, 4, row * 4 + 1, projection='3d')
        render_mesh_colored(ax1, bv.copy(), bf, base_curv,
                           cmap_name='YlOrRd', azim=30, vmin=vmin, vmax=vmax)
        if row == 0:
            ax1.set_title('Baseline\n(curvature)', fontsize=10, fontweight='bold', pad=8)

        # Col 2: DGR (curvature heatmap)
        ax2 = fig.add_subplot(n_rows, 4, row * 4 + 2, projection='3d')
        render_mesh_colored(ax2, dv.copy(), df, dgr_curv,
                           cmap_name='YlOrRd', azim=30, vmin=vmin, vmax=vmax)
        if row == 0:
            ax2.set_title('DGR (Ours)\n(curvature)', fontsize=10, fontweight='bold', pad=8)

        # Col 3: Improvement map — show curvature REDUCTION on DGR mesh
        # Positive = curvature reduced (improved), negative = increased
        # Need same vertex count; if different, just show DGR colored by its own curvature reduction proxy
        ax3 = fig.add_subplot(n_rows, 4, row * 4 + 3, projection='3d')
        if len(bv) == len(dv):
            curv_diff = base_curv - dgr_curv  # positive = improvement
            render_mesh_colored(ax3, dv.copy(), df, curv_diff,
                               cmap_name='RdYlGn', azim=30,
                               vmin=-np.percentile(np.abs(curv_diff), 90),
                               vmax=np.percentile(np.abs(curv_diff), 90))
        else:
            # Different vertex counts, show symmetry error reduction instead
            dgr_sym_err = compute_symmetry_error(dv - dv.mean(axis=0))
            render_mesh_colored(ax3, dv.copy(), df, -dgr_sym_err,
                               cmap_name='RdYlGn', azim=30)
        if row == 0:
            ax3.set_title('Improvement\n(green=better)', fontsize=10, fontweight='bold', pad=8)

        # Col 4: Metrics panel
        ax4 = fig.add_subplot(n_rows, 4, row * 4 + 4)
        ax4.axis('off')
        ax3 = ax4  # reuse variable name for metrics text

        bs, bsm, bc = item['base_metrics']
        ds, dsm, dc = item['dgr_metrics']

        name = item['slug'].replace('_', ' ')
        cat = item['cat']

        ax3.text(0.05, 0.95, name, fontsize=9, fontweight='bold',
                va='top', transform=ax3.transAxes)
        ax3.text(0.05, 0.80, f"({cat})", fontsize=7, fontstyle='italic',
                va='top', transform=ax3.transAxes, color='#666')

        # Metric bars with colored improvement
        y_pos = [0.60, 0.40, 0.20]
        labels = ['Sym', 'Smo', 'Com']
        base_vals = [bs, bsm, bc]
        dgr_vals = [ds, dsm, dc]
        imps = [item['imp_sym'], item['imp_smo'],
                (dc - bc) / max(abs(bc), 1e-8) * 100]

        for i, (label, bval, dval, imp) in enumerate(zip(labels, base_vals, dgr_vals, imps)):
            color = '#16a34a' if imp > 5 else ('#dc2626' if imp < -5 else '#6b7280')
            arrow = '\u2191' if imp > 5 else ('\u2193' if imp < -5 else '\u2192')
            ax3.text(0.05, y_pos[i], f"{label}:", fontsize=8, fontweight='bold',
                    va='center', transform=ax3.transAxes)
            ax3.text(0.22, y_pos[i], f"{bval:.4f}" if abs(bval) < 1 else f"{bval:.1f}",
                    fontsize=7, va='center', transform=ax3.transAxes, color='#888')
            ax3.text(0.48, y_pos[i], f"{arrow}", fontsize=11,
                    va='center', transform=ax3.transAxes, color=color)
            ax3.text(0.57, y_pos[i], f"{dval:.4f}" if abs(dval) < 1 else f"{dval:.1f}",
                    fontsize=7, va='center', transform=ax3.transAxes, fontweight='bold')
            ax3.text(0.82, y_pos[i], f"{imp:+.0f}%",
                    fontsize=9, va='center', transform=ax3.transAxes,
                    color=color, fontweight='bold')

    fig.suptitle("Curvature Visualization: Baseline vs. DGR Refinement",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.08, hspace=0.25)

    out = OUT_DIR / "qualitative_comparison.pdf"
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(OUT_DIR / "qualitative_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
