"""
Render meshes with per-vertex curvature coloring via PyVista.
Shows WHERE geometric quality improves, not just the shape.
This is the standard approach in geometry processing papers (SIGGRAPH/EG).
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
OUT.mkdir(parents=True, exist_ok=True)


def compute_curvature_pyvista(mesh):
    """Compute mean curvature using PyVista."""
    try:
        curv = mesh.curvature(curv_type='mean')
        return np.abs(curv)  # absolute mean curvature
    except:
        return np.zeros(mesh.n_points)


def render_mesh_with_curvature(mesh, output_path, azimuth=35, elevation=25,
                                resolution=(700, 700), clim=None):
    """Render mesh colored by curvature magnitude."""
    # Center and normalize
    center = mesh.center
    mesh.translate(-np.array(center), inplace=True)
    bounds = mesh.bounds
    scale = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    if scale > 0:
        mesh.scale(1.8 / scale, inplace=True)

    # Compute curvature
    curv = compute_curvature_pyvista(mesh)
    mesh.point_data['curvature'] = curv

    if clim is None:
        clim = [np.percentile(curv, 5), np.percentile(curv, 90)]

    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background('white')

    pl.add_mesh(
        mesh,
        scalars='curvature',
        cmap='YlOrRd',        # yellow(low curv) → orange → red(high curv)
        clim=clim,
        smooth_shading=True,
        show_scalar_bar=False,
        opacity=1.0,
    )

    # Lighting
    pl.add_light(pv.Light(position=(4, 4, 6), focal_point=(0,0,0),
                          color='white', intensity=0.6, light_type='scenelight'))
    pl.add_light(pv.Light(position=(-3, 3, 4), focal_point=(0,0,0),
                          color='white', intensity=0.3, light_type='scenelight'))

    dist = 3.5
    az_rad, el_rad = np.radians(azimuth), np.radians(elevation)
    x = dist * np.cos(el_rad) * np.sin(az_rad)
    y = dist * np.cos(el_rad) * np.cos(az_rad)
    z = dist * np.sin(el_rad)
    pl.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]
    pl.enable_anti_aliasing('ssaa')
    pl.screenshot(str(output_path), transparent_background=False)
    pl.close()
    return clim


def main():
    import trimesh, torch
    sys.path.insert(0, 'src')
    from geo_reward import symmetry_reward, smoothness_reward

    OBJ_BASE = Path("results/mesh_validity_objs/baseline")
    OBJ_DGR  = Path("results/mesh_validity_objs/diffgeoreward")

    # Find good pairs
    pairs = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        base_dir = OBJ_BASE / cat
        dgr_dir = OBJ_DGR / cat
        if not base_dir.exists(): continue
        for f in sorted(base_dir.glob('*_seed42.obj')):
            dgr_f = dgr_dir / f.name
            if not dgr_f.exists(): continue
            try:
                tm = trimesh.load(str(f), force='mesh')
                ext = tm.extents
                aspect = min(ext)/max(ext) if max(ext)>0 else 0
                if aspect < 0.5 or len(tm.faces) < 80: continue

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
                vs = imp * np.log10(len(tm.faces)+1)

                pairs.append((cat, f.stem.replace('_seed42',''), str(f), str(dgr_f),
                              vs, bs, bsm, ds, dsm))
            except: continue

    pairs.sort(key=lambda x: -x[4])
    selected = []
    per_cat = defaultdict(int)
    for p in pairs:
        if per_cat[p[0]] < 2 and len(selected) < 4:
            selected.append(p); per_cat[p[0]] += 1
    if len(selected) < 4:
        selected = pairs[:4]

    print(f"Rendering {len(selected)} mesh pairs with curvature visualization...")

    render_dir = OUT / "renders"
    render_dir.mkdir(exist_ok=True)

    # Render each pair with SAME color scale
    for cat, slug, bp, dp, vs, bs, bsm, ds, dsm in selected:
        # Load both to compute shared clim
        mesh_b = pv.read(bp)
        mesh_d = pv.read(dp)

        # Normalize both consistently
        for m in [mesh_b, mesh_d]:
            m.translate(-np.array(m.center), inplace=True)
            s = max(m.bounds[1]-m.bounds[0], m.bounds[3]-m.bounds[2], m.bounds[5]-m.bounds[4])
            if s > 0: m.scale(1.8/s, inplace=True)

        cb = compute_curvature_pyvista(mesh_b)
        cd = compute_curvature_pyvista(mesh_d)
        shared_clim = [
            min(np.percentile(cb, 5), np.percentile(cd, 5)),
            max(np.percentile(cb, 85), np.percentile(cd, 85))
        ]

        # Re-read (since we modified in place)
        for label, path in [("baseline", bp), ("dgr", dp)]:
            out_path = render_dir / f"{slug}_{label}_curv.png"
            print(f"  {out_path.name}")
            mesh = pv.read(path)
            render_mesh_with_curvature(mesh, out_path, clim=shared_clim)

    # ── Compose final figure ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    sys.path.insert(0, 'tools')
    from pub_style import setup_style, TEXTWIDTH_PT
    setup_style()

    n = len(selected)
    fig = plt.figure(figsize=(TEXTWIDTH_PT/72.27, n * 1.65 + 0.8))
    gs = GridSpec(n, 3, width_ratios=[1, 1, 0.65], hspace=0.12, wspace=0.06)

    for row, (cat, slug, bp, dp, vs, bs, bsm, ds, dsm) in enumerate(selected):
        for col, label in enumerate(["baseline", "dgr"]):
            ax = fig.add_subplot(gs[row, col])
            img_path = render_dir / f"{slug}_{label}_curv.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.axis('off')
            if row == 0:
                t = "Baseline (curvature)" if label == "baseline" else "DGR (curvature)"
                ax.set_title(t, fontsize=8, fontweight='bold', pad=4)

        # Metrics
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis('off')
        name = slug.replace('_', ' ')
        imp_sym = (ds-bs)/max(abs(bs),1e-8)*100
        imp_smo = (dsm-bsm)/max(abs(bsm),1e-8)*100

        ax3.text(0.0, 0.92, f'"{name}"', fontsize=7, fontweight='bold',
                va='top', transform=ax3.transAxes, fontstyle='italic')
        ax3.text(0.0, 0.76, f'({cat})', fontsize=6, va='top',
                transform=ax3.transAxes, color='#999')

        for i, (lb, pct) in enumerate([('Sym', imp_sym), ('Smo', imp_smo)]):
            y = 0.55 - i * 0.25
            c = '#16A34A' if pct > 5 else ('#DC2626' if pct < -5 else '#9CA3AF')
            ax3.text(0.0, y, f'{lb}:', fontsize=7.5, fontweight='bold',
                    va='center', transform=ax3.transAxes)
            ax3.text(0.92, y, f'{pct:+.0f}%', fontsize=9, fontweight='bold',
                    va='center', ha='right', transform=ax3.transAxes, color=c)

    # Colorbar at bottom
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cbar_ax = fig.add_axes([0.15, 0.01, 0.5, 0.015])
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Curvature magnitude (yellow=low, red=high)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(str(OUT / "qualitative_comparison.pdf"), dpi=300)
    fig.savefig(str(OUT / "qualitative_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nFinal: {OUT / 'qualitative_comparison.pdf'}")


if __name__ == "__main__":
    main()
