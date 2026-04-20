"""
Render meshes with PyVista PBR for publication-quality figures.
Produces smooth-shaded, well-lit renders with white background.
"""
import os, sys, json
import numpy as np
import pyvista as pv
from pathlib import Path
from PIL import Image

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Offscreen rendering
pv.OFF_SCREEN = True
pv.global_theme.anti_aliasing = 'ssaa'

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

MESH_COLOR = '#7EB8D4'  # steel blue — standard academic mesh color


def render_obj(obj_path, output_path, azimuth=35, elevation=25, resolution=(800, 800)):
    """Render a single OBJ file with PyVista PBR."""
    mesh = pv.read(str(obj_path))

    # Center and normalize
    center = mesh.center
    mesh.translate(-np.array(center), inplace=True)
    bounds = mesh.bounds
    scale = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    if scale > 0:
        mesh.scale(1.8 / scale, inplace=True)

    # Compute normals for smooth shading
    mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)

    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background('white')

    # Add mesh with PBR material
    pl.add_mesh(
        mesh,
        smooth_shading=True,
        split_sharp_edges=True,
        color=MESH_COLOR,
        pbr=True,
        metallic=0.08,
        roughness=0.45,
        diffuse=1.0,
        specular=0.5,
    )

    # Three-point lighting
    pl.add_light(pv.Light(
        position=(4, 4, 6), focal_point=(0, 0, 0),
        color='white', intensity=0.7, light_type='scenelight'
    ))
    pl.add_light(pv.Light(
        position=(-4, 3, 4), focal_point=(0, 0, 0),
        color=[0.9, 0.92, 1.0], intensity=0.35, light_type='scenelight'
    ))
    pl.add_light(pv.Light(
        position=(0, -4, 2), focal_point=(0, 0, 0),
        color='white', intensity=0.2, light_type='scenelight'
    ))

    # Camera position
    dist = 3.5
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    x = dist * np.cos(el_rad) * np.sin(az_rad)
    y = dist * np.cos(el_rad) * np.cos(az_rad)
    z = dist * np.sin(el_rad)
    pl.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]

    pl.enable_anti_aliasing('ssaa')

    # Render
    pl.screenshot(str(output_path), transparent_background=False)
    pl.close()
    return True


def main():
    import trimesh
    import torch
    sys.path.insert(0, 'src')
    from geo_reward import symmetry_reward, smoothness_reward

    OBJ_BASE = Path("results/mesh_validity_objs/baseline")
    OBJ_DGR  = Path("results/mesh_validity_objs/diffgeoreward")

    # Find pairs and compute improvement
    pairs = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        base_dir = OBJ_BASE / cat
        dgr_dir = OBJ_DGR / cat
        if not base_dir.exists():
            continue
        for f in sorted(base_dir.glob('*_seed42.obj')):
            dgr_f = dgr_dir / f.name
            if not dgr_f.exists():
                continue

            # Filter: good 3D shape
            try:
                tm = trimesh.load(str(f), force='mesh')
                ext = tm.extents
                aspect = min(ext) / max(ext) if max(ext) > 0 else 0
                nfaces = len(tm.faces)
                if aspect < 0.5 or nfaces < 80:
                    continue
            except:
                continue

            slug = f.stem.replace('_seed42', '')

            # Compute metrics
            try:
                bv = torch.tensor(np.array(tm.vertices), dtype=torch.float32)
                bf = torch.tensor(np.array(tm.faces), dtype=torch.long)
                dm = trimesh.load(str(dgr_f), force='mesh')
                dv = torch.tensor(np.array(dm.vertices), dtype=torch.float32)
                df_t = torch.tensor(np.array(dm.faces), dtype=torch.long)

                bs = symmetry_reward(bv).item()
                bsm = smoothness_reward(bv, bf).item()
                ds = symmetry_reward(dv).item()
                dsm = smoothness_reward(dv, df_t).item()
                imp = (ds-bs)/max(abs(bs),1e-8)*100 + (dsm-bsm)/max(abs(bsm),1e-8)*100

                # Bonus for high face count (renders better)
                visual_score = imp * np.log10(nfaces + 1)
                pairs.append((cat, slug, str(f), str(dgr_f), visual_score,
                              bs, bsm, ds, dsm))
            except:
                continue

    pairs.sort(key=lambda x: -x[4])

    # Pick top 4, balanced by category
    from collections import defaultdict
    selected = []
    per_cat = defaultdict(int)
    for p in pairs:
        if per_cat[p[0]] < 2 and len(selected) < 4:
            selected.append(p)
            per_cat[p[0]] += 1
    if len(selected) < 4:
        selected = pairs[:4]

    print(f"Selected {len(selected)} meshes for rendering:")
    for cat, slug, bp, dp, vs, *_ in selected:
        print(f"  {cat:12s} {slug}")

    # Render each pair
    render_dir = OUT / "renders"
    render_dir.mkdir(exist_ok=True)

    for cat, slug, bp, dp, vs, bs, bsm, ds, dsm in selected:
        for label, path in [("baseline", bp), ("dgr", dp)]:
            out_path = render_dir / f"{slug}_{label}.png"
            if out_path.exists():
                print(f"  skip {out_path.name}")
                continue
            print(f"  rendering {out_path.name}...")
            try:
                render_obj(path, out_path)
            except Exception as e:
                print(f"    FAILED: {e}")

    # Compose final figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sys.path.insert(0, 'tools')
    from pub_style import setup_style, figsize, TEXTWIDTH_PT, METRIC_COLORS
    setup_style()

    fig = plt.figure(figsize=(TEXTWIDTH_PT / 72.27, len(selected) * 1.7 + 0.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(len(selected), 3, width_ratios=[1, 1, 0.7],
                  hspace=0.15, wspace=0.08)

    for row, (cat, slug, bp, dp, vs, bs, bsm, ds, dsm) in enumerate(selected):
        for col, label in enumerate(["baseline", "dgr"]):
            ax = fig.add_subplot(gs[row, col])
            img_path = render_dir / f"{slug}_{label}.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.axis('off')
            if row == 0:
                title = "Shap-E Baseline" if label == "baseline" else "DGR (Ours)"
                ax.set_title(title, fontsize=9, fontweight='bold', pad=6)

        # Metrics panel
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis('off')

        name = slug.replace('_', ' ')
        imp_sym = (ds - bs) / max(abs(bs), 1e-8) * 100
        imp_smo = (dsm - bsm) / max(abs(bsm), 1e-8) * 100

        ax3.text(0.0, 0.90, f'"{name}"', fontsize=7, fontweight='bold',
                va='top', transform=ax3.transAxes, fontstyle='italic')
        ax3.text(0.0, 0.75, f'({cat})', fontsize=6, va='top',
                transform=ax3.transAxes, color='#999999')

        for i, (lb, pct) in enumerate([('Sym', imp_sym), ('Smo', imp_smo)]):
            y = 0.55 - i * 0.25
            color = '#16A34A' if pct > 5 else ('#DC2626' if pct < -5 else '#9CA3AF')
            ax3.text(0.0, y, f'{lb}:', fontsize=7.5, fontweight='bold',
                    va='center', transform=ax3.transAxes)
            ax3.text(0.92, y, f'{pct:+.0f}%', fontsize=9, fontweight='bold',
                    va='center', ha='right', transform=ax3.transAxes, color=color)

    fig.savefig(str(OUT / "qualitative_comparison.pdf"), dpi=300)
    fig.savefig(str(OUT / "qualitative_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nFinal figure: {OUT / 'qualitative_comparison.pdf'}")


if __name__ == "__main__":
    main()
