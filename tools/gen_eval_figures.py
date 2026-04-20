"""
Generate human evaluation figures: high-quality side-by-side comparisons.
Only uses recognizable meshes (>=80 vertices).
Evaluator sees baseline (left) vs DGR (right) from 3 angles.
Per-category evaluation questions.
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
OUT_DIR = Path("analysis_results/human_eval_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
MIN_VERTS = 80
ANGLES = [(30, 45), (30, 135), (30, -45)]  # 3 views: elev, azim

# Hand-picked recognizable prompts with category-specific eval questions
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
        "a mirror symmetric face mask",
        "a balanced candelabra",
        "a symmetric bell",
        "a symmetric goblet",
        "a symmetric pagoda",
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


def render_mesh(mesh_path, angles, resolution=400):
    """Render mesh from multiple angles."""
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
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Subsample faces for rendering speed
        if len(faces) > 8000:
            idx = np.random.RandomState(42).choice(len(faces), 8000, replace=False)
            render_faces = faces[idx]
        else:
            render_faces = faces

        polys = verts[render_faces]
        pc = Poly3DCollection(polys, alpha=0.9, facecolor='#87CEEB',
                              edgecolor='#4A4A4A', linewidth=0.1)
        ax.add_collection3d(pc)

        lim = 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        images.append(buf.copy())
        plt.close(fig)

    return images


def create_comparison(baseline_imgs, dgr_imgs, prompt, category, question_id, out_path):
    """Create side-by-side comparison: Baseline (top) vs DGR (bottom), 3 views each."""
    n_views = len(baseline_imgs)
    fig, axes = plt.subplots(2, n_views, figsize=(4*n_views, 8))

    for j in range(n_views):
        axes[0, j].imshow(baseline_imgs[j])
        axes[0, j].set_axis_off()
        if j == 0:
            axes[0, j].set_ylabel('A', fontsize=24, fontweight='bold', rotation=0,
                                   labelpad=30, va='center')

        axes[1, j].imshow(dgr_imgs[j])
        axes[1, j].set_axis_off()
        if j == 0:
            axes[1, j].set_ylabel('B', fontsize=24, fontweight='bold', rotation=0,
                                   labelpad=30, va='center')

    question = CATEGORY_QUESTIONS[category]
    fig.suptitle(f'Q{question_id}: "{prompt}"\n{question}',
                 fontsize=14, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0.03, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    questions = []
    qi = 0
    n_skipped = 0

    for category, prompts in EVAL_PROMPTS.items():
        for prompt in prompts:
            slug = prompt.replace(' ', '_').replace(',', '')

            bl_path = OBJ_ROOT / "baseline" / category / f"{slug}_seed{SEED}.obj"
            dgr_path = OBJ_ROOT / "handcrafted" / category / f"{slug}_seed{SEED}.obj"

            if not bl_path.exists() or not dgr_path.exists():
                # Try alternative naming
                found = False
                for p in (OBJ_ROOT / "baseline" / category).glob("*.obj"):
                    if f"seed{SEED}" in p.stem:
                        name_part = p.stem.replace(f"_seed{SEED}", "").replace("seed{SEED}", "")
                        if name_part.replace("_", " ") == prompt or prompt.replace(" ", "_") in name_part:
                            bl_path = p
                            dgr_path = OBJ_ROOT / "handcrafted" / category / p.name
                            if dgr_path.exists():
                                found = True
                                break
                if not found:
                    print(f"  SKIP {prompt}: mesh not found")
                    n_skipped += 1
                    continue

            # Check vertex count
            try:
                bl_mesh = trimesh.load(str(bl_path), force='mesh')
                if len(bl_mesh.vertices) < MIN_VERTS:
                    print(f"  SKIP {prompt}: only {len(bl_mesh.vertices)} verts")
                    n_skipped += 1
                    continue
            except:
                print(f"  SKIP {prompt}: failed to load")
                n_skipped += 1
                continue

            # Render both
            bl_imgs = render_mesh(bl_path, ANGLES)
            dgr_imgs = render_mesh(dgr_path, ANGLES)

            if bl_imgs is None or dgr_imgs is None:
                print(f"  SKIP {prompt}: render failed")
                n_skipped += 1
                continue

            qi += 1

            # Randomize A/B order
            if random.random() < 0.5:
                a_imgs, b_imgs = bl_imgs, dgr_imgs
                a_method, b_method = "baseline", "handcrafted"
            else:
                a_imgs, b_imgs = dgr_imgs, bl_imgs
                a_method, b_method = "handcrafted", "baseline"

            out_path = FIG_DIR / f"q{qi:03d}.png"
            create_comparison(a_imgs, b_imgs, prompt, category, qi, out_path)

            questions.append({
                "question_id": qi,
                "prompt": prompt,
                "category": category,
                "question": CATEGORY_QUESTIONS[category],
                "figure": f"q{qi:03d}.png",
                "answer_key": {"A": a_method, "B": b_method},
                "n_verts": len(bl_mesh.vertices),
                "n_faces": len(bl_mesh.faces),
            })

            print(f"  Q{qi}: [{category}] {prompt} ({len(bl_mesh.vertices)}v)")

    # Save questionnaire
    questionnaire = {
        "instructions": (
            "For each question, compare meshes A and B rendered from 3 viewing angles.\n"
            "Answer the question shown above each pair.\n"
            "Choose: A is better / B is better / About the same"
        ),
        "questions": questions,
        "answer_key_warning": "DO NOT share with evaluators",
    }

    with open(OUT_DIR / "questionnaire.json", 'w') as f:
        json.dump(questionnaire, f, indent=2)

    # Answer key (separate file)
    key = {q["question_id"]: q["answer_key"] for q in questions}
    with open(OUT_DIR / "answer_key.json", 'w') as f:
        json.dump(key, f, indent=2)

    print(f"\nGenerated: {qi} questions, skipped: {n_skipped}")
    print(f"Figures: {FIG_DIR}")
    print(f"Questionnaire: {OUT_DIR / 'questionnaire.json'}")


if __name__ == "__main__":
    main()
