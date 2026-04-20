"""
Render: Baseline vs Single-Reward (catastrophic degradation) vs DGR.
This is the paper's CORE FINDING — much more visually striking than before/after.
Shows that single-reward optimization destroys the mesh while DGR preserves it.
"""
import os, sys, json
import numpy as np
import pyvista as pv
from pathlib import Path
from PIL import Image
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
pv.OFF_SCREEN = True
pv.global_theme.anti_aliasing = 'ssaa'

OUT = Path("paper/figures")
render_dir = OUT / "renders"
render_dir.mkdir(parents=True, exist_ok=True)

MESH_COLOR = '#7EB8D4'


def render_obj(obj_path, output_path, azimuth=35, elevation=25, resolution=(600, 600)):
    """Render OBJ with PyVista PBR."""
    mesh = pv.read(str(obj_path))
    mesh.translate(-np.array(mesh.center), inplace=True)
    scale = max(mesh.bounds[1]-mesh.bounds[0], mesh.bounds[3]-mesh.bounds[2], mesh.bounds[5]-mesh.bounds[4])
    if scale > 0:
        mesh.scale(1.8/scale, inplace=True)
    mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)

    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background('white')
    pl.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                color=MESH_COLOR, pbr=True, metallic=0.08, roughness=0.45)
    pl.add_light(pv.Light(position=(4,4,6), focal_point=(0,0,0),
                          color='white', intensity=0.7, light_type='scenelight'))
    pl.add_light(pv.Light(position=(-4,3,4), focal_point=(0,0,0),
                          color=[0.9,0.92,1.0], intensity=0.35, light_type='scenelight'))

    dist = 3.5
    az_rad, el_rad = np.radians(azimuth), np.radians(elevation)
    x = dist*np.cos(el_rad)*np.sin(az_rad)
    y = dist*np.cos(el_rad)*np.cos(az_rad)
    z = dist*np.sin(el_rad)
    pl.camera_position = [(x,y,z), (0,0,0), (0,0,1)]
    pl.enable_anti_aliasing('ssaa')
    pl.screenshot(str(output_path), transparent_background=False)
    pl.close()


def main():
    # We need: baseline, sym_only, smooth_only, compact_only, diffgeoreward
    # From results/full/ directory (old pilot runs with all methods)
    FULL_DIR = Path("results/full")

    # Check what methods are available
    methods_available = []
    for m in ['baseline', 'sym_only', 'smooth_only', 'compact_only', 'diffgeoreward', 'handcrafted']:
        d = FULL_DIR / m
        if d.exists() and any(d.rglob('*.obj')):
            methods_available.append(m)
            n = len(list(d.rglob('*.obj')))
            print(f"  {m}: {n} OBJ files")

    if len(methods_available) < 3:
        print("Not enough methods in results/full/. Trying mesh_validity_objs/...")
        FULL_DIR = Path("results/mesh_validity_objs")
        methods_available = []
        for m in ['baseline', 'diffgeoreward']:
            d = FULL_DIR / m
            if d.exists():
                methods_available.append(m)
                n = len(list(d.rglob('*.obj')))
                print(f"  {m}: {n} OBJ files")

    # Select prompts — SKIP symmetry (vertex displacement too large, looks like different mesh)
    # Only use smoothness + compactness where baseline-DGR correspondence is verified
    prompts = {}
    for cat in ['smoothness', 'compactness']:
        cat_dir = FULL_DIR / 'baseline' / cat
        if not cat_dir.exists():
            continue
        for base_path in sorted(cat_dir.glob('*seed42.obj')):
            slug = cat
            all_exist = all(
                len(list((FULL_DIR / m / cat).glob('*seed42.obj'))) > 0
                for m in methods_available
            )
            if all_exist and cat not in prompts:
                prompts[cat] = (cat, slug, base_path)
                break

    print(f"\nSelected prompts: {list(prompts.keys())}")

    # Render comparison grid
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    sys.path.insert(0, 'tools')
    from pub_style import setup_style, TEXTWIDTH_PT
    setup_style()

    # Layout: rows = prompts, cols = methods
    show_methods = [m for m in ['baseline', 'sym_only', 'smooth_only', 'compact_only',
                                'diffgeoreward'] if m in methods_available]
    method_titles = {
        'baseline': 'Baseline',
        'sym_only': 'Sym-Only',
        'smooth_only': 'Smo-Only',
        'compact_only': 'Com-Only',
        'diffgeoreward': 'DGR (Ours)',
        'handcrafted': 'Equal Wt.',
    }

    rows = list(prompts.values())
    n_rows = len(rows)
    n_cols = len(show_methods)

    # Render each mesh
    for cat, slug, base_path in rows:
        for method in show_methods:
            method_dir = FULL_DIR / method / cat
            obj_files = list(method_dir.glob('*seed42.obj'))
            if not obj_files:
                continue
            obj_path = obj_files[0]
            out_path = render_dir / f"cata_{slug}_{method}.png"
            if not out_path.exists():
                print(f"  rendering {out_path.name}...")
                try:
                    render_obj(obj_path, out_path)
                except Exception as e:
                    print(f"    FAILED: {e}")

    # Compose figure
    fig = plt.figure(figsize=(TEXTWIDTH_PT/72.27, n_rows * 1.5 + 0.6))
    gs = GridSpec(n_rows, n_cols, hspace=0.08, wspace=0.04)

    for row, (cat, slug, _) in enumerate(rows):
        for col, method in enumerate(show_methods):
            ax = fig.add_subplot(gs[row, col])
            img_path = render_dir / f"cata_{slug}_{method}.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.axis('off')

            if row == 0:
                title = method_titles.get(method, method)
                bold = method == 'diffgeoreward'
                color = '#0C5DA5' if bold else '#333333'
                ax.set_title(title, fontsize=8, fontweight='bold' if bold else 'normal',
                            pad=4, color=color)

            if col == 0:
                name = slug.replace('_', ' ').replace('baseline ', '')
                ax.text(-0.05, 0.5, f'"{name}"\n({cat})',
                        fontsize=6, va='center', ha='right',
                        transform=ax.transAxes, fontstyle='italic', color='#666')

    fig.savefig(str(OUT / "qualitative_comparison.pdf"), dpi=300)
    fig.savefig(str(OUT / "qualitative_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT / 'qualitative_comparison.pdf'}")


if __name__ == "__main__":
    main()
