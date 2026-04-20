#!/usr/bin/env python3
"""
Generate before/after qualitative comparison figure for DiffGeoReward.

Renders 4 representative prompts showing Baseline vs DiffGeoReward meshes
side-by-side from 2 views (front + 45-degree).

Output: paper/figures/qualitative.pdf and .png

Usage:
    python tools/render_qualitative.py
"""

import json, os, re, sys
from pathlib import Path

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PROMPTS = [
    "a symmetric vase",
    "a polished marble sphere",
    "a compact stone",
    "a perfectly balanced chair",
]
SEED = 42
OUT_DIR = Path("paper/figures")
PAIR_ROOT = Path("results/mesh_validity_objs")

# ── Imports with fallback ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

class SimpleMesh:
    """Minimal mesh container to avoid trimesh dependency."""
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def prompt_from_stem(stem: str) -> str:
    stem = re.sub(r"_(seed|s)\d+$", "", stem)
    return stem.replace("_", " ")


def load_mesh(obj_path):
    """Load OBJ file and return vertices, faces, mesh object."""
    verts, faces = [], []
    with open(obj_path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                idx = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    if not verts or not faces:
        return None, None, None
    v = np.array(verts)
    f = np.array(faces)
    return v, f, SimpleMesh(v, f)


def compute_curvature_colors(mesh):
    """Compute per-vertex curvature proxy via normal deviation from neighbors."""
    verts = mesh.vertices
    faces = mesh.faces
    n = len(verts)

    # Build adjacency
    adj = [set() for _ in range(n)]
    for f in faces:
        for i in range(3):
            adj[f[i]].add(f[(i+1) % 3])
            adj[f[i]].add(f[(i+2) % 3])

    # Face normals
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
    curv = np.zeros(n)
    for i in range(n):
        neighbors = list(adj[i])
        if not neighbors:
            continue
        mean_n = vert_normals[neighbors].mean(axis=0)
        mn = np.linalg.norm(mean_n)
        if mn > 1e-10:
            cos_val = np.dot(vert_normals[i], mean_n / mn)
            curv[i] = 1.0 - np.clip(cos_val, -1, 1)

    # Normalize to [0, 1]
    if curv.max() > curv.min():
        curv = (curv - curv.min()) / (curv.max() - curv.min())
    return curv


def render_mesh_matplotlib(ax, mesh, title="", elev=25, azim=45):
    """Render mesh on a matplotlib 3D axis with curvature coloring."""
    vertices = mesh.vertices
    faces = mesh.faces

    # Center and normalize
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / scale

    # Compute face colors from curvature
    curv = compute_curvature_colors(mesh)
    face_colors = []
    cmap = plt.cm.coolwarm
    for f in faces:
        face_curv = curv[f].mean()
        color = cmap(face_curv)
        face_colors.append(color)

    # Create polygon collection
    polys = [[vertices[f[0]], vertices[f[1]], vertices[f[2]]] for f in faces]
    poly_collection = Poly3DCollection(polys, alpha=0.9)
    poly_collection.set_facecolor(face_colors)
    poly_collection.set_edgecolor("none")
    ax.add_collection3d(poly_collection)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9, pad=2)
    ax.axis("off")


def find_mesh_pair(prompt, seed=42):
    """Find baseline and DiffGeoReward OBJ files for a prompt+seed."""
    ps = slug(prompt)
    base_root = PAIR_ROOT / "baseline"
    dgr_root = PAIR_ROOT / "diffgeoreward"

    patterns = [
        f"{ps}_seed{seed}.obj",
        f"{ps}_s{seed}.obj",
        f"{ps}.obj",
    ]
    for pattern in patterns:
        for base_path in sorted(base_root.rglob(pattern)):
            rel = base_path.relative_to(base_root)
            dgr_path = dgr_root / rel
            if dgr_path.exists():
                return base_path, dgr_path

    return None, None


def find_available_pairs():
    """Scan for any available mesh pairs if specific prompts not found."""
    pairs = []
    base_dir = PAIR_ROOT / "baseline"
    dgr_dir = PAIR_ROOT / "diffgeoreward"

    if not base_dir.exists():
        return pairs

    seen = set()
    candidates = sorted(
        base_dir.rglob("*.obj"),
        key=lambda p: ("_seed42" not in p.stem and "_s42" not in p.stem, str(p)),
    )
    for base_obj in candidates:
        rel = base_obj.relative_to(base_dir)
        dgr_obj = dgr_dir / rel
        if dgr_obj.exists():
            prompt_name = prompt_from_stem(base_obj.stem)
            if prompt_name in seen:
                continue
            seen.add(prompt_name)
            pairs.append((prompt_name, base_obj, dgr_obj))
        if len(pairs) >= 20:
            break

    return pairs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try to find specific prompt pairs
    found_pairs = []
    for prompt in PROMPTS:
        base_path, dgr_path = find_mesh_pair(prompt, SEED)
        if base_path and dgr_path:
            found_pairs.append((prompt, base_path, dgr_path))
            print(f"  Found: {prompt}")
        else:
            print(f"  Miss:  {prompt}")

    # Fallback: scan available pairs
    if len(found_pairs) < 3:
        print("\nNot enough specific prompts found, scanning available meshes...")
        available = find_available_pairs()
        for name, bp, dp in available:
            if len(found_pairs) >= 4:
                break
            if not any(p[0] == name for p in found_pairs):
                found_pairs.append((name, bp, dp))
                print(f"  Auto:  {name}")

    if not found_pairs:
        print("ERROR: No mesh pairs found. Run main experiments first.")
        sys.exit(1)

    n_prompts = min(len(found_pairs), 4)
    found_pairs = found_pairs[:n_prompts]

    # Create figure: n_prompts rows x 2 columns (Baseline | DiffGeoReward)
    fig = plt.figure(figsize=(8, 2.5 * n_prompts))

    for i, (prompt, base_path, dgr_path) in enumerate(found_pairs):
        print(f"  Rendering: {prompt}")

        # Load meshes
        _, _, base_mesh = load_mesh(base_path)
        _, _, dgr_mesh = load_mesh(dgr_path)

        # Baseline
        ax1 = fig.add_subplot(n_prompts, 2, 2*i + 1, projection="3d")
        label = "Baseline" if i == 0 else ""
        render_mesh_matplotlib(ax1, base_mesh,
                               title=f"{label}\n\"{prompt}\"" if i == 0 else f"\"{prompt}\"",
                               elev=25, azim=45)

        # DiffGeoReward
        ax2 = fig.add_subplot(n_prompts, 2, 2*i + 2, projection="3d")
        label = "DiffGeoReward" if i == 0 else ""
        render_mesh_matplotlib(ax2, dgr_mesh,
                               title=label if i == 0 else "",
                               elev=25, azim=45)

    plt.tight_layout(pad=0.5)

    # Save
    pdf_path = OUT_DIR / "qualitative.pdf"
    png_path = OUT_DIR / "qualitative.png"
    pdf_alias = OUT_DIR / "qualitative_comparison.pdf"
    png_alias = OUT_DIR / "qualitative_comparison.png"
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_alias), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_alias), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {pdf_path}, {png_path}, {pdf_alias}, {png_alias}")


if __name__ == "__main__":
    main()
