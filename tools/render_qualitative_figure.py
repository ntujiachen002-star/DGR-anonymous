"""
Generate qualitative comparison figure for the paper.
Renders baseline vs DGR meshes from multiple angles.
Output: paper/figures/qualitative_comparison.pdf
"""
import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# ── Config ───────────────────────────────────────────────────────────────────

SEED = 42
ANGLE = 30  # azimuth for rendering

# Select representative prompts (2 per category, paired baseline + DGR)
SAMPLES = [
    # (category, slug, display_name)
    ("symmetry",   "a_symmetric_vase",       "symmetric vase"),
    ("symmetry",   "a_balanced_bookshelf",    "balanced bookshelf"),
    ("smoothness", "a_polished_apple",        "polished apple"),
    ("smoothness", "a_smooth_river_stone",    "smooth river stone"),
    ("compactness","a_compact_backpack",      "compact backpack"),
    ("compactness","a_compact_cube",          "compact cube"),
]

METHODS = [
    ("baseline",       "Shap-E (Baseline)"),
    ("diffgeoreward",  "DGR (Ours)"),
]

OBJ_DIR = Path("results/mesh_validity_objs")
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_obj(path):
    """Load OBJ file, return vertices and faces as numpy arrays."""
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                idx = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    return np.array(verts), np.array(faces) if faces else np.zeros((0, 3), dtype=int)


def render_mesh(ax, verts, faces, azim=30, elev=20):
    """Render a mesh onto a matplotlib 3D axis."""
    if len(verts) == 0:
        ax.text(0, 0, 0, "N/A", ha='center', fontsize=10)
        return

    # Center and normalize
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale

    if len(faces) > 0:
        # Subsample if too many faces
        if len(faces) > 8000:
            idx = np.random.RandomState(42).choice(len(faces), 8000, replace=False)
            faces = faces[idx]

        polys = verts[faces]
        pc = Poly3DCollection(polys, alpha=0.85,
                              facecolor='#6CB4EE', edgecolor='#4A4A4A',
                              linewidth=0.05)
        ax.add_collection3d(pc)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def compute_metrics(obj_path):
    """Compute symmetry, smoothness, compactness from OBJ file."""
    try:
        sys.path.insert(0, 'src')
        import torch
        from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

        verts_np, faces_np = load_obj(obj_path)
        if len(verts_np) == 0 or len(faces_np) == 0:
            return None
        v = torch.tensor(verts_np, dtype=torch.float32)
        f = torch.tensor(faces_np, dtype=torch.long)
        sym = symmetry_reward(v).item()
        smo = smoothness_reward(v, f).item()
        com = compactness_reward(v, f).item()
        return {"sym": sym, "smo": smo, "com": com}
    except Exception as e:
        return None


def main():
    n_rows = len(SAMPLES)
    n_cols = len(METHODS)

    fig = plt.figure(figsize=(n_cols * 3.2 + 1.5, n_rows * 2.8 + 0.8))

    for row, (cat, slug, display) in enumerate(SAMPLES):
        for col, (method, method_label) in enumerate(METHODS):
            obj_path = OBJ_DIR / method / cat / f"{slug}_seed{SEED}.obj"

            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1,
                                 projection='3d')

            if obj_path.exists():
                verts, faces = load_obj(str(obj_path))
                render_mesh(ax, verts, faces, azim=ANGLE)
            else:
                ax.text(0.5, 0.5, 0.5, "Missing", ha='center', fontsize=8,
                        transform=ax.transAxes)
                ax.set_axis_off()

            # Column headers
            if row == 0:
                ax.set_title(method_label, fontsize=11, fontweight='bold', pad=8)

        # Row labels (prompt name + category)
        # Add text annotation on the left
        y_pos = 1.0 - (row + 0.5) / n_rows
        fig.text(0.02, y_pos, f"{display}\n({cat})",
                 fontsize=8, ha='left', va='center',
                 fontstyle='italic', color='#444444')

    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.05, hspace=0.15)
    fig.suptitle("Qualitative Comparison: Shap-E Baseline vs. DGR Refinement",
                 fontsize=13, fontweight='bold', y=0.98)

    out_path = OUT_DIR / "qualitative_comparison.pdf"
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Also save PNG for quick preview
    out_png = OUT_DIR / "qualitative_comparison.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
    print(f"Preview: {out_png}")
    plt.close(fig)

    # ── Also generate convergence curve figure ──
    print("\n=== Generating convergence curve figure ===")
    conv_path = Path("analysis_results/convergence/all_results.json")
    if conv_path.exists():
        data = json.load(open(conv_path))
        conv = [r for r in data if r.get("experiment") == "convergence"]

        if conv:
            # Group by step, compute mean + std for each metric
            from collections import defaultdict
            by_step = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
            for r in conv:
                s = r["step"]
                by_step[s]["sym"].append(r["symmetry"])
                by_step[s]["smo"].append(r["smoothness"])
                by_step[s]["com"].append(r["compactness"])

            steps = sorted(by_step.keys())
            metrics = {}
            for m in ["sym", "smo", "com"]:
                means = [np.mean(by_step[s][m]) for s in steps]
                stds  = [np.std(by_step[s][m])  for s in steps]
                # Normalize to step 0
                base = means[0] if abs(means[0]) > 1e-8 else 1e-8
                metrics[m] = {
                    "steps": steps,
                    "norm_means": [v / base for v in means],
                    "norm_stds": [s / abs(base) for s in stds],
                }

            fig2, ax2 = plt.subplots(1, 1, figsize=(5.5, 3.5))
            colors = {"sym": "#E63946", "smo": "#457B9D", "com": "#2A9D8F"}
            labels = {"sym": "Symmetry", "smo": "Smoothness", "com": "Compactness"}

            for m in ["sym", "smo", "com"]:
                d = metrics[m]
                ax2.plot(d["steps"], d["norm_means"], '-o', color=colors[m],
                         label=labels[m], markersize=4, linewidth=1.8)
                lower = [v - s for v, s in zip(d["norm_means"], d["norm_stds"])]
                upper = [v + s for v, s in zip(d["norm_means"], d["norm_stds"])]
                ax2.fill_between(d["steps"], lower, upper,
                                 alpha=0.15, color=colors[m])

            ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax2.annotate('Default (50 steps)', xy=(50, ax2.get_ylim()[0]),
                         fontsize=7, color='gray', ha='center')
            ax2.set_xlabel("Optimization Steps", fontsize=10)
            ax2.set_ylabel("Normalized Reward (vs. step 0)", fontsize=10)
            ax2.set_title("DGR Convergence Curve (30 prompts x 3 seeds)", fontsize=11)
            ax2.legend(fontsize=9, loc='best')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()

            conv_out = OUT_DIR / "convergence_curve.pdf"
            fig2.savefig(str(conv_out), dpi=300, bbox_inches='tight')
            print(f"Saved: {conv_out}")
            fig2.savefig(str(OUT_DIR / "convergence_curve.png"), dpi=150,
                         bbox_inches='tight')
            plt.close(fig2)
    else:
        print("No convergence data found, skipping.")

    print("\nDone! Figures saved to paper/figures/")


if __name__ == "__main__":
    main()
