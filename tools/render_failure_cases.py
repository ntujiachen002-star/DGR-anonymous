"""
Render failure case figure for paper.
2×3 grid: top row = success cases, bottom row = failure cases.
Each cell shows baseline (gray) and DiffGeoReward (colored) side by side.

Uses curvature heatmap coloring for visual impact.
Output: paper/figures/failure_cases.pdf / .png
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

OBJ_BASE = Path("results/mesh_validity_objs/baseline")
OBJ_DGR  = Path("results/mesh_validity_objs/diffgeoreward")

# ── Success prompts (known improvements) ──
SUCCESS_PROMPTS = [
    "a symmetric vase",
    "a polished marble sphere",
    "a compact stone",
]
# ── Failure prompts (known failures — irregular shapes) ──
FAILURE_PROMPTS = [
    "a banana",
    "a fluffy cloud",
    "a spoon",
]
# Fallback if exact files not found
SUCCESS_FALLBACKS = [
    "a perfectly balanced chair",
    "a smooth river stone",
    "a solid bowling ball",
]
FAILURE_FALLBACKS = [
    "a twisted branch",
    "a crumpled paper ball",
    "a coral reef",
]


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


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
    """Per-vertex curvature via normal deviation from neighbors."""
    n = len(verts)
    # Build adjacency
    adj = [[] for _ in range(n)]
    for f in faces:
        for i in range(3):
            adj[f[i]].append(f[(i+1)%3])
            adj[f[i]].append(f[(i+2)%3])

    # Compute face normals
    face_normals = np.zeros((len(faces), 3))
    for fi, f in enumerate(faces):
        e1 = verts[f[1]] - verts[f[0]]
        e2 = verts[f[2]] - verts[f[0]]
        n_vec = np.cross(e1, e2)
        norm = np.linalg.norm(n_vec)
        if norm > 1e-10:
            face_normals[fi] = n_vec / norm

    # Vertex normals (area-weighted)
    vert_normals = np.zeros((n, 3))
    for fi, f in enumerate(faces):
        for vi in f:
            vert_normals[vi] += face_normals[fi]
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vert_normals /= norms

    # Curvature = 1 - cos(normal, mean_neighbor_normal)
    curvature = np.zeros(n)
    for i in range(n):
        neighbors = list(set(adj[i]))
        if not neighbors:
            continue
        mean_n = vert_normals[neighbors].mean(axis=0)
        mn = np.linalg.norm(mean_n)
        if mn > 1e-10:
            cos_val = np.dot(vert_normals[i], mean_n / mn)
            curvature[i] = 1.0 - np.clip(cos_val, -1, 1)
    return curvature


def compute_metrics_np(verts, faces):
    """Compute metrics using torch backend."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v = torch.tensor(verts, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device)
    with torch.no_grad():
        sym = symmetry_reward(v).item()
        smo = smoothness_reward(v, f).item()
        com = compactness_reward(v, f).item()
    return dict(symmetry=sym, smoothness=smo, compactness=com)


def render_mesh(ax, verts, faces, curvature, title, cmap='YlOrRd', vmin=0, vmax=0.5):
    """Render mesh with curvature heatmap."""
    # Subsample faces for rendering speed
    max_faces = 6000
    if len(faces) > max_faces:
        idx = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
        faces_sub = faces[idx]
    else:
        faces_sub = faces

    # Get per-face color from mean vertex curvature
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    polys = []
    colors = []
    for f in faces_sub:
        tri = verts[f]
        polys.append(tri)
        mean_curv = curvature[f].mean()
        colors.append(colormap(norm(mean_curv)))

    pc = Poly3DCollection(polys, facecolors=colors, edgecolors='none', linewidths=0.1, alpha=0.9)
    ax.add_collection3d(pc)

    # Set view
    margin = 0.1
    mins = verts.min(axis=0) - margin
    maxs = verts.max(axis=0) + margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.view_init(elev=20, azim=30)
    ax.set_title(title, fontsize=9, pad=2)
    ax.axis('off')


def find_prompt_objs(prompt):
    """Find baseline and DGR obj files for a prompt."""
    ps = slug(prompt)
    # Try multiple naming patterns
    patterns = [
        (OBJ_BASE / f"{ps}_s{SEED}.obj", OBJ_DGR / f"{ps}_s{SEED}.obj"),
        (OBJ_BASE / f"{ps}.obj", OBJ_DGR / f"{ps}.obj"),
    ]
    for bl_path, dgr_path in patterns:
        if bl_path.exists() and dgr_path.exists():
            return bl_path, dgr_path
    return None, None


def main():
    # Resolve which prompts have available meshes
    success_prompts = []
    for p in SUCCESS_PROMPTS + SUCCESS_FALLBACKS:
        bl, dgr = find_prompt_objs(p)
        if bl is not None:
            success_prompts.append(p)
        if len(success_prompts) >= 3:
            break

    failure_prompts = []
    for p in FAILURE_PROMPTS + FAILURE_FALLBACKS:
        bl, dgr = find_prompt_objs(p)
        if bl is not None:
            failure_prompts.append(p)
        if len(failure_prompts) >= 3:
            break

    if len(success_prompts) < 2 or len(failure_prompts) < 2:
        # Fallback: scan all available objs and pick by metric delta
        print("Not enough specific prompts found. Scanning all available meshes...")
        all_deltas = []
        if OBJ_BASE.exists():
            for bl_path in sorted(OBJ_BASE.glob("*.obj")):
                name = bl_path.stem
                dgr_path = OBJ_DGR / bl_path.name
                if not dgr_path.exists():
                    continue
                bl_v, bl_f = load_obj(bl_path)
                dgr_v, dgr_f = load_obj(dgr_path)
                if bl_v is None or dgr_v is None:
                    continue
                bl_m = compute_metrics_np(bl_v, bl_f)
                dgr_m = compute_metrics_np(dgr_v, dgr_f)
                # Total improvement = sum of relative improvements
                total_imp = 0
                for k in ['symmetry', 'smoothness']:
                    if abs(bl_m[k]) > 1e-10:
                        total_imp += (dgr_m[k] - bl_m[k]) / abs(bl_m[k])
                all_deltas.append((name, total_imp, bl_path, dgr_path))

        all_deltas.sort(key=lambda x: x[1])
        # Best = most improved (success), worst = most degraded (failure)
        if len(all_deltas) >= 6:
            success_prompts = [d[0] for d in all_deltas[-3:]]
            failure_prompts = [d[0] for d in all_deltas[:3]]
            success_paths = [(d[2], d[3]) for d in all_deltas[-3:]]
            failure_paths = [(d[2], d[3]) for d in all_deltas[:3]]
        else:
            print(f"Only {len(all_deltas)} meshes found. Need at least 6.")
            return
    else:
        success_paths = [find_prompt_objs(p) for p in success_prompts]
        failure_paths = [find_prompt_objs(p) for p in failure_prompts]

    # Pad to exactly 3
    while len(success_prompts) < 3:
        success_prompts.append(success_prompts[-1])
        success_paths.append(success_paths[-1])
    while len(failure_prompts) < 3:
        failure_prompts.append(failure_prompts[-1])
        failure_paths.append(failure_paths[-1])

    # ── Render figure ──
    fig = plt.figure(figsize=(14, 9))
    rows = 2   # success / failure
    cols = 6   # 3 prompts × (baseline + DGR)

    all_prompts = success_prompts[:3] + failure_prompts[:3]
    all_paths   = success_paths[:3] + failure_paths[:3]
    row_labels  = ['Success cases\n(geometric improvement)', 'Failure cases\n(semantic conflict)']

    for row in range(2):
        for col in range(3):
            idx = row * 3 + col
            prompt = all_prompts[idx]
            bl_path, dgr_path = all_paths[idx]

            bl_v, bl_f = load_obj(bl_path)
            dgr_v, dgr_f = load_obj(dgr_path)

            if bl_v is None or dgr_v is None:
                continue

            bl_curv  = compute_per_vertex_curvature(bl_v, bl_f)
            dgr_curv = compute_per_vertex_curvature(dgr_v, dgr_f)

            # Compute shared color scale
            vmax = max(np.percentile(bl_curv, 95), np.percentile(dgr_curv, 95))
            vmax = max(vmax, 0.1)

            # Compute metrics
            bl_m  = compute_metrics_np(bl_v, bl_f)
            dgr_m = compute_metrics_np(dgr_v, dgr_f)

            # Baseline subplot
            ax_bl = fig.add_subplot(rows, cols, row * cols + col * 2 + 1, projection='3d')
            render_mesh(ax_bl, bl_v, bl_f, bl_curv, 'Baseline', vmax=vmax)

            # DGR subplot
            ax_dgr = fig.add_subplot(rows, cols, row * cols + col * 2 + 2, projection='3d')
            cmap = 'YlGn' if row == 0 else 'OrRd'
            render_mesh(ax_dgr, dgr_v, dgr_f, dgr_curv, 'DGR', cmap=cmap, vmax=vmax)

            # Add prompt label above pair
            mid_x = (ax_bl.get_position().x0 + ax_dgr.get_position().x1) / 2
            y_top = ax_bl.get_position().y1 + 0.02
            # Truncate long prompts
            label = prompt.replace('_', ' ')
            if len(label) > 25:
                label = label[:22] + '...'

            # Add metric delta annotation below
            sym_d = (dgr_m['symmetry'] - bl_m['symmetry']) / abs(bl_m['symmetry']) * 100 if abs(bl_m['symmetry']) > 1e-10 else 0
            smo_d = (dgr_m['smoothness'] - bl_m['smoothness']) / abs(bl_m['smoothness']) * 100 if abs(bl_m['smoothness']) > 1e-10 else 0

    # Row labels
    fig.text(0.02, 0.75, row_labels[0], fontsize=11, fontweight='bold',
             rotation=90, va='center', ha='center', color='#2d7d46')
    fig.text(0.02, 0.30, row_labels[1], fontsize=11, fontweight='bold',
             rotation=90, va='center', ha='center', color='#c0392b')

    fig.suptitle('DiffGeoReward: Success vs Failure Cases (Curvature Heatmap)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.04, 0.02, 1.0, 0.95])

    # Save
    for fmt in ['pdf', 'png']:
        out = OUT_DIR / f"failure_cases.{fmt}"
        fig.savefig(str(out), dpi=300 if fmt == 'pdf' else 150,
                    bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
