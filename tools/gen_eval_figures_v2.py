"""
Human evaluation figures v2:
- Large A/B labels
- No edge lines (clean surface render)
- Symmetry axis reference line for symmetry prompts
- Filter out unrecognizable meshes (face mask, etc.)
- Better color contrast between A/B
"""
import trimesh
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import json, os, random

OBJ_ROOT = Path("results/mesh_validity_objs")
OUT_DIR = Path("analysis_results/human_eval_v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
MIN_VERTS = 100  # higher threshold for recognizability
ANGLES = [(25, 45), (25, 135), (25, -45)]  # 3 views

# Curated prompts: only clearly recognizable objects
EVAL_PROMPTS = {
    "symmetry": [
        "a symmetric vase",
        "a perfectly balanced chair",
        "a symmetric trophy",
        "a symmetric cathedral door",
        "a symmetric wine glass",
        "a symmetric crown",
        "a balanced chess piece a king",
        "a symmetric butterfly sculpture",
        "a symmetrical temple",
        "a balanced candelabra",
        "a symmetric bell",
        "a symmetric goblet",
        "a symmetric pagoda",
        # skip face mask - unrecognizable
    ],
    "smoothness": [
        "a smooth melting chocolate drop",
        "a polished gemstone",
    ],
    "compactness": [
        "a compact cube",
        "a compact birdhouse",
        "a solid ice cube",
        "a solid metal die",
        "a tight ball",
        "a compact backpack",
    ],
}

CATEGORY_QUESTIONS = {
    "symmetry": "Which mesh has better bilateral symmetry?",
    "smoothness": "Which mesh has a smoother, more regular surface?",
    "compactness": "Which mesh has a more compact, well-proportioned shape?",
}


def render_mesh(mesh_path, angles, resolution=450, color='#6BAED6', show_sym_axis=False):
    """Render mesh from multiple angles. No edge lines for clean look."""
    mesh = trimesh.load(str(mesh_path), force='mesh')
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    if len(verts) < 4 or len(faces) < 4:
        return None

    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale

    images = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(4.5, 4.5), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Subsample for speed but keep enough for smooth look
        if len(faces) > 10000:
            idx = np.random.RandomState(42).choice(len(faces), 10000, replace=False)
            render_faces = faces[idx]
        else:
            render_faces = faces

        polys = verts[render_faces]

        # Compute simple face normals for shading
        v0 = polys[:, 0]
        v1 = polys[:, 1]
        v2 = polys[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        normals = normals / norms

        # Simple directional lighting
        light_dir = np.array([0.3, 0.5, 0.8])
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = np.abs(np.dot(normals, light_dir))
        intensity = 0.3 + 0.7 * intensity  # ambient + diffuse

        # Convert base color to RGB
        from matplotlib.colors import to_rgb
        base_rgb = np.array(to_rgb(color))
        face_colors = np.outer(intensity, base_rgb)
        face_colors = np.clip(face_colors, 0, 1)

        pc = Poly3DCollection(polys, alpha=0.95,
                              facecolors=face_colors,
                              edgecolors='none')  # NO edge lines
        ax.add_collection3d(pc)

        # Symmetry axis reference line (Y axis = xz plane symmetry)
        if show_sym_axis:
            ax.plot([0, 0], [0, 0], [-1.2, 1.2],
                    color='red', linewidth=1.5, linestyle='--', alpha=0.6)

        lim = 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        images.append(buf[:, :, :3].copy())
        plt.close(fig)

    return images


def create_comparison(a_imgs, b_imgs, prompt, category, question, question_id, out_path):
    """A (top row) vs B (bottom row), 3 views, big labels."""
    n_views = len(a_imgs)
    fig, axes = plt.subplots(2, n_views + 1, figsize=(4.5 * n_views + 2, 9),
                             gridspec_kw={'width_ratios': [0.4] + [1]*n_views})

    # Big A/B labels in first column
    for row, label, color in [(0, 'A', '#2166AC'), (1, 'B', '#B2182B')]:
        axes[row, 0].text(0.5, 0.5, label, fontsize=56, fontweight='bold',
                         color=color, ha='center', va='center',
                         transform=axes[row, 0].transAxes)
        axes[row, 0].set_axis_off()

    imgs = [a_imgs, b_imgs]
    for row in range(2):
        for j in range(n_views):
            axes[row, j+1].imshow(imgs[row][j])
            axes[row, j+1].set_axis_off()
            # View angle labels
            if row == 0:
                view_labels = ['Front-Left', 'Front-Right', 'Back-Left']
                axes[row, j+1].set_title(view_labels[j], fontsize=11, color='gray')

    # Title and question
    fig.suptitle(f'Q{question_id}: "{prompt}"',
                 fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.92, question, fontsize=13, ha='center',
             style='italic', color='#444444')
    fig.text(0.5, 0.01, 'Choose: A is better / B is better / About the same',
             fontsize=11, ha='center', color='#888888')

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def find_mesh_path(method, category, prompt):
    """Find OBJ file matching prompt."""
    slug = prompt.replace(' ', '_').replace(',', '')
    direct = OBJ_ROOT / method / category / f"{slug}_seed{SEED}.obj"
    if direct.exists():
        return direct

    # Fuzzy match
    target = prompt.replace(' ', '_').lower()
    for p in (OBJ_ROOT / method / category).glob(f"*seed{SEED}.obj"):
        name = p.stem.replace(f"_seed{SEED}", "").replace(f"seed{SEED}", "").lower()
        if name == target or target in name or name in target:
            return p
    return None


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    questions = []
    qi = 0
    skipped = []

    for category, prompts in EVAL_PROMPTS.items():
        for prompt in prompts:
            bl_path = find_mesh_path("baseline", category, prompt)
            dgr_path = find_mesh_path("handcrafted", category, prompt)

            if not bl_path or not dgr_path:
                skipped.append(f"{prompt}: mesh not found")
                continue

            # Check vertex count
            try:
                bl_mesh = trimesh.load(str(bl_path), force='mesh')
                nv = len(bl_mesh.vertices)
                if nv < MIN_VERTS:
                    skipped.append(f"{prompt}: only {nv} verts")
                    continue
            except:
                skipped.append(f"{prompt}: load failed")
                continue

            show_axis = (category == "symmetry")

            bl_imgs = render_mesh(bl_path, ANGLES, color='#6BAED6', show_sym_axis=show_axis)
            dgr_imgs = render_mesh(dgr_path, ANGLES, color='#6BAED6', show_sym_axis=show_axis)

            if not bl_imgs or not dgr_imgs:
                skipped.append(f"{prompt}: render failed")
                continue

            qi += 1

            # Randomize A/B
            if random.random() < 0.5:
                a_imgs, b_imgs = bl_imgs, dgr_imgs
                a_method, b_method = "baseline", "DGR"
            else:
                a_imgs, b_imgs = dgr_imgs, bl_imgs
                a_method, b_method = "DGR", "baseline"

            out_path = FIG_DIR / f"q{qi:03d}.png"
            create_comparison(a_imgs, b_imgs, prompt, category,
                            CATEGORY_QUESTIONS[category], qi, out_path)

            questions.append({
                "question_id": qi,
                "prompt": prompt,
                "category": category,
                "question": CATEGORY_QUESTIONS[category],
                "figure": f"q{qi:03d}.png",
                "answer_key": {"A": a_method, "B": b_method},
                "n_verts": nv,
            })

            print(f"  Q{qi}: [{category}] {prompt} ({nv}v) -> A={a_method}")

    # Save
    with open(OUT_DIR / "questionnaire.json", 'w') as f:
        json.dump({
            "instructions": (
                "For each question, compare meshes A (blue label) and B (red label) "
                "rendered from 3 viewing angles.\n"
                "Answer the question shown below the prompt.\n"
                "Choose: A is better / B is better / About the same"
            ),
            "questions": questions,
        }, f, indent=2)

    with open(OUT_DIR / "answer_key.json", 'w') as f:
        json.dump({q["question_id"]: q["answer_key"] for q in questions}, f, indent=2)

    print(f"\nGenerated: {qi} questions")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
