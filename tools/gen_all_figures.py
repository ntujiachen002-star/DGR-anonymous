"""
Generate ALL publication-quality figures for DiffGeoReward NeurIPS paper.
Style: clean serif fonts, muted palette, NeurIPS column widths.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tools')
from pub_style import *

setup_style()

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Convergence Curve
# ═══════════════════════════════════════════════════════════════════════

def fig_convergence():
    data = json.load(open("analysis_results/convergence/all_results.json"))
    conv = [r for r in data if r.get("experiment") == "convergence"]
    if not conv:
        print("  No convergence data"); return

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
        norm_m = means / base
        norm_s = stds / abs(base)

        color = METRIC_COLORS[{'sym': 'symmetry', 'smo': 'smoothness', 'com': 'compactness'}[m]]
        ax.plot(steps, norm_m, '-o', color=color, label=label,
                markersize=2.5, markeredgewidth=0)
        ax.fill_between(steps, norm_m - norm_s, norm_m + norm_s,
                        alpha=0.08, color=color, linewidth=0)

    ax.axvline(x=50, color='#AAAAAA', linestyle='--', linewidth=0.6, zorder=0)
    ax.text(52, ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            'default', fontsize=6, color='#999999', va='bottom')

    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Normalized Reward")
    ax.legend(loc='best', borderpad=0.2)
    ax.grid(True, axis='y')

    plt.tight_layout()
    fig.savefig(str(OUT / "convergence_curve.pdf"))
    fig.savefig(str(OUT / "convergence_curve.png"))
    plt.close(fig)
    print("  convergence_curve.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: LR Sensitivity (for appendix)
# ═══════════════════════════════════════════════════════════════════════

def fig_lr_sensitivity():
    data = json.load(open("analysis_results/convergence/all_results.json"))
    lr_data = [r for r in data if r.get("experiment") == "lr_sensitivity"]
    if not lr_data:
        print("  No LR data"); return

    by_lr = defaultdict(lambda: {"sym": [], "smo": [], "com": []})
    for r in lr_data:
        by_lr[r["lr"]]["sym"].append(r["symmetry"])
        by_lr[r["lr"]]["smo"].append(r["smoothness"])
        by_lr[r["lr"]]["com"].append(r["compactness"])

    lrs = sorted(by_lr.keys())

    fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, aspect=0.28))
    labels_map = {"sym": "Symmetry", "smo": "Smoothness", "com": "Compactness"}

    for i, (m, label) in enumerate(labels_map.items()):
        ax = axes[i]
        means = [np.mean(by_lr[lr][m]) for lr in lrs]
        stds  = [np.std(by_lr[lr][m])  for lr in lrs]
        color = list(METRIC_COLORS.values())[i]

        ax.errorbar(range(len(lrs)), means, yerr=stds,
                    fmt='-o', color=color, capsize=2, markersize=3,
                    linewidth=1.2, capthick=0.8, markeredgewidth=0)

        # Mark default LR
        if 0.005 in lrs:
            idx = lrs.index(0.005)
            ax.axvline(x=idx, color='#D1D5DB', linestyle='--', linewidth=0.7)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr}' for lr in lrs], fontsize=6.5, rotation=30)
        ax.set_title(label, fontsize=9, pad=4)
        if i == 0:
            ax.set_ylabel("Reward")
        ax.grid(True, axis='y')

    fig.supxlabel("Learning Rate", fontsize=9, y=-0.02)
    plt.tight_layout()
    fig.savefig(str(OUT / "lr_sensitivity.pdf"))
    fig.savefig(str(OUT / "lr_sensitivity.png"))
    plt.close(fig)
    print("  lr_sensitivity.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: CLIP + ImageReward Comparison
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
    available = [m for m in order if m in by_method and len(by_method[m]["clip"]) > 10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.38))
    x = np.arange(len(available))
    width = 0.6

    # CLIP
    clip_means = [np.mean(by_method[m]["clip"]) for m in available]
    clip_stds  = [np.std(by_method[m]["clip"])  for m in available]
    colors = [METHOD_COLORS.get(m, '#9CA3AF') for m in available]

    bars1 = ax1.bar(x, clip_means, width, yerr=clip_stds, capsize=2,
                    color=colors, edgecolor='white', linewidth=0.3, alpha=0.85,
                    error_kw={'linewidth': 0.8})
    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_LABELS.get(m, m) for m in available],
                        fontsize=6.5, rotation=20, ha='right')
    ax1.set_ylabel("CLIP Score")
    ax1.set_title("(a) Semantic Alignment", fontsize=9, pad=4)

    # Add tiny value labels
    for bar, val in zip(bars1, clip_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#6B7280')

    # ImageReward
    ir_available = [m for m in available if len(by_method[m]["ir"]) > 10]
    if ir_available:
        ir_means = [np.mean(by_method[m]["ir"]) for m in ir_available]
        ir_stds  = [np.std(by_method[m]["ir"])  for m in ir_available]
        x2 = np.arange(len(ir_available))
        colors2 = [METHOD_COLORS.get(m, '#9CA3AF') for m in ir_available]

        bars2 = ax2.bar(x2, ir_means, width, yerr=ir_stds, capsize=2,
                        color=colors2, edgecolor='white', linewidth=0.3, alpha=0.85,
                        error_kw={'linewidth': 0.8})
        ax2.set_xticks(x2)
        ax2.set_xticklabels([METHOD_LABELS.get(m, m) for m in ir_available],
                            fontsize=6.5, rotation=20, ha='right')
        ax2.set_ylabel("ImageReward")
        ax2.set_title("(b) Perceptual Quality", fontsize=9, pad=4)

        for bar, val in zip(bars2, ir_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                    f'{val:.2f}', ha='center', va='top', fontsize=5.5, color='#6B7280')

    plt.tight_layout()
    fig.savefig(str(OUT / "clip_comparison.pdf"))
    fig.savefig(str(OUT / "clip_comparison.png"))
    plt.close(fig)
    print("  clip_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Catastrophic Degradation (single-reward heatmap)
# ═══════════════════════════════════════════════════════════════════════

def fig_degradation():
    """Compact heatmap showing cross-metric degradation from single-reward optimization."""
    data = json.load(open("analysis_results/laplacian_vs_dgr/checkpoint.json"))

    metrics = ["symmetry", "smoothness", "compactness"]
    methods_show = ["sym_only", "smooth_only", "compact_only", "diffgeoreward", "handcrafted"]
    method_names = ["Sym-Only", "Smo-Only", "Com-Only", "DGR (Ours)", "Equal Wt."]

    # Compute improvement vs baseline
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

    fig, ax = plt.subplots(figsize=figsize(0.48, aspect=0.78))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=100)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=8)
    ax.set_yticks(range(len(methods_show)))
    ax.set_yticklabels(method_names, fontsize=7.5)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if abs(val) > 60 else '#1F2937'
            ax.text(j, i, f"{val:+.0f}%", ha='center', va='center',
                    fontsize=6.5, color=color, fontweight='bold')

    # Divider between single-reward and multi-reward
    ax.axhline(y=2.5, color='#4B5563', linewidth=1.0, linestyle='-')

    cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.08)
    cb.set_label("vs. Baseline (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(str(OUT / "degradation_heatmap.pdf"))
    fig.savefig(str(OUT / "degradation_heatmap.png"))
    plt.close(fig)
    print("  degradation_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Figure: Qualitative Comparison (trimesh scene render)
# ═══════════════════════════════════════════════════════════════════════

def fig_qualitative():
    """Render meshes using trimesh scene with proper lighting."""
    import trimesh
    from PIL import Image
    import io

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
            return None
        return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)

    def render_trimesh(mesh, resolution=(400, 400), angle=30):
        """Render using pyrender with PBR lighting for publication quality."""
        if mesh is None:
            return Image.new('RGB', resolution, (255, 255, 255))

        # Normalize mesh
        mesh.vertices -= mesh.centroid
        scale = mesh.extents.max()
        if scale > 0:
            mesh.vertices /= scale * 1.2  # leave some margin

        try:
            import pyrender
            # Create pyrender mesh with metallic material
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.55, 0.68, 0.82, 1.0],  # medium steel blue
                metallicFactor=0.1,
                roughnessFactor=0.55,
            )
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

            scene = pyrender.Scene(bg_color=[0.97, 0.97, 0.97, 1.0],
                                   ambient_light=[0.3, 0.3, 0.32])
            scene.add(pr_mesh)

            # Camera — closer for larger mesh in frame
            cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            angle_rad = np.radians(angle)
            dist = 1.6
            cam_pos = np.array([dist * np.sin(angle_rad), 0.6, dist * np.cos(angle_rad)])
            # Look-at matrix
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(forward, [0, 1, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            cam_mat = np.eye(4)
            cam_mat[:3, 0] = right
            cam_mat[:3, 1] = up
            cam_mat[:3, 2] = -forward
            cam_mat[:3, 3] = cam_pos
            scene.add(cam, pose=cam_mat)

            # Key light (strong, from upper-right)
            key_light = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=3.5)
            key_pose = np.eye(4)
            key_dir = np.array([0.5, -0.8, -0.3])
            key_dir = key_dir / np.linalg.norm(key_dir)
            key_pose[:3, 2] = key_dir
            scene.add(key_light, pose=key_pose)

            # Fill light (softer, from left)
            fill_light = pyrender.DirectionalLight(color=[0.85, 0.90, 1.0], intensity=1.5)
            fill_pose = np.eye(4)
            fill_dir = np.array([-0.6, -0.4, 0.5])
            fill_dir = fill_dir / np.linalg.norm(fill_dir)
            fill_pose[:3, 2] = fill_dir
            scene.add(fill_light, pose=fill_pose)

            # Render
            r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
            color, _ = r.render(scene)
            r.delete()
            return Image.fromarray(color)

        except Exception as e:
            print(f"    pyrender failed ({e}), falling back to matplotlib")
            # Matplotlib fallback with Phong-like shading
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            verts = mesh.vertices
            faces = mesh.faces
            if len(faces) > 6000:
                idx = np.random.RandomState(42).choice(len(faces), 6000, replace=False)
                faces = faces[idx]
            v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(normals, axis=1, keepdims=True); norms[norms < 1e-10] = 1
            normals = normals / norms
            light_dir = np.array([0.4, 0.6, 0.8]); light_dir /= np.linalg.norm(light_dir)
            diffuse = 0.3 + 0.7 * np.abs(normals @ light_dir)
            base_c = np.array([0.65, 0.75, 0.85])
            fc = np.column_stack([np.outer(diffuse, base_c), np.ones(len(diffuse))])
            polys = verts[faces]
            pc = Poly3DCollection(polys); pc.set_facecolor(np.clip(fc, 0, 1)); pc.set_edgecolor('none')
            ax.add_collection3d(pc)
            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.8, 0.8)
            ax.view_init(elev=25, azim=angle); ax.set_axis_off()
            fig.patch.set_facecolor('white')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
            plt.close(fig); buf.seek(0)
            return Image.open(buf).convert('RGB')

    # Find best pairs
    import torch
    sys.path.insert(0, 'src')
    from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

    pairs = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        base_dir = OBJ_BASE / cat
        dgr_dir = OBJ_DGR / cat
        if not base_dir.exists():
            continue
        for f in sorted(base_dir.glob('*_seed42.obj')):
            dgr_f = dgr_dir / f.name
            if dgr_f.exists():
                slug = f.stem.replace('_seed42', '')
                bm = load_obj(str(f))
                dm = load_obj(str(dgr_f))
                if bm is None or dm is None:
                    continue

                bv = torch.tensor(np.array(bm.vertices), dtype=torch.float32)
                bf = torch.tensor(np.array(bm.faces), dtype=torch.long)
                dv = torch.tensor(np.array(dm.vertices), dtype=torch.float32)
                df_t = torch.tensor(np.array(dm.faces), dtype=torch.long)
                try:
                    bs = symmetry_reward(bv).item()
                    bsm = smoothness_reward(bv, bf).item()
                    ds = symmetry_reward(dv).item()
                    dsm = smoothness_reward(dv, df_t).item()
                    imp = (ds - bs)/max(abs(bs),1e-8)*100 + (dsm - bsm)/max(abs(bsm),1e-8)*100
                    pairs.append((cat, slug, str(f), str(dgr_f), imp,
                                  bs, bsm, ds, dsm))
                except:
                    pass

    pairs.sort(key=lambda x: -x[4])
    # Filter out very flat meshes (aspect ratio < 0.3) — they look bad rendered
    good_pairs = []
    for p in pairs:
        try:
            bm = trimesh.load(p[2], force='mesh')
            ext = bm.extents
            aspect = min(ext) / max(ext) if max(ext) > 0 else 0
            if aspect > 0.5 and len(bm.faces) >= 80:
                good_pairs.append(p)
        except:
            pass

    if len(good_pairs) >= 4:
        pairs = good_pairs

    # Score = improvement * log(nfaces) — prefer high-face meshes for visual quality
    scored_pairs = []
    for p in pairs:
        try:
            bm = trimesh.load(p[2], force='mesh')
            nf = max(len(bm.faces), 1)
            visual_score = p[4] * np.log10(nf + 1)
            scored_pairs.append((visual_score, p))
        except:
            scored_pairs.append((p[4], p))
    scored_pairs.sort(key=lambda x: -x[0])
    pairs = [sp[1] for sp in scored_pairs]

    # Pick top 4 (balanced across categories)
    selected = []
    per_cat = defaultdict(int)
    for p in pairs:
        if per_cat[p[0]] < 2 and len(selected) < 4:
            selected.append(p)
            per_cat[p[0]] += 1

    if len(selected) < 4:
        selected = pairs[:4]

    # Render figure: 4 rows x 3 cols (baseline | DGR | metrics)
    fig = plt.figure(figsize=(TEXTWIDTH_PT / 72.27, len(selected) * 1.7 + 0.6))
    gs = GridSpec(len(selected), 3, width_ratios=[1, 1, 0.8],
                  hspace=0.25, wspace=0.1)

    for row, (cat, slug, bp, dp, imp, bs, bsm, ds, dsm) in enumerate(selected):
        bm = load_obj(bp)
        dm = load_obj(dp)

        # Render both meshes
        img_base = render_trimesh(bm, resolution=(350, 350), angle=35)
        img_dgr  = render_trimesh(dm, resolution=(350, 350), angle=35)

        # Plot baseline
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(img_base)
        ax1.axis('off')
        if row == 0:
            ax1.set_title('Shap-E Baseline', fontsize=9, fontweight='bold', pad=6)

        # Plot DGR
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(img_dgr)
        ax2.axis('off')
        if row == 0:
            ax2.set_title('DGR (Ours)', fontsize=9, fontweight='bold', pad=6)

        # Metrics panel
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis('off')

        name = slug.replace('_', ' ')
        imp_sym = (ds - bs) / max(abs(bs), 1e-8) * 100
        imp_smo = (dsm - bsm) / max(abs(bsm), 1e-8) * 100

        ax3.text(0.0, 0.92, f'"{name}"', fontsize=7.5, fontweight='bold',
                va='top', transform=ax3.transAxes, fontstyle='italic')
        ax3.text(0.0, 0.76, f'({cat})', fontsize=6.5,
                va='top', transform=ax3.transAxes, color='#9CA3AF')

        # Metric improvement bars
        for i, (label, bval, dval, pct) in enumerate([
            ('Sym', bs, ds, imp_sym),
            ('Smo', bsm, dsm, imp_smo),
        ]):
            y = 0.55 - i * 0.28
            color = '#16A34A' if pct > 5 else ('#DC2626' if pct < -5 else '#9CA3AF')
            ax3.text(0.0, y, f'{label}:', fontsize=7, fontweight='bold',
                    va='center', transform=ax3.transAxes)
            ax3.text(0.95, y, f'{pct:+.0f}%', fontsize=8.5, fontweight='bold',
                    va='center', ha='right', transform=ax3.transAxes, color=color)

    fig.savefig(str(OUT / "qualitative_comparison.pdf"))
    fig.savefig(str(OUT / "qualitative_comparison.png"))
    plt.close(fig)
    print("  qualitative_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating publication-quality figures...")
    fig_convergence()
    fig_lr_sensitivity()
    fig_clip_ir()
    fig_degradation()
    fig_qualitative()
    print(f"\nAll figures saved to {OUT}/")
