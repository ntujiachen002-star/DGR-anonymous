"""
Generate a valid blinded pairwise user-study set from prompt-specific meshes.

Uses the prompt-specific OBJ files stored in results/mesh_validity_objs for:
  - baseline
  - diffgeoreward

This avoids the overwritten per-category OBJ paths in results/full/.
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


ROOT = Path(".")
OUT_DIR = ROOT / "analysis_results" / "user_study_pairwise_valid"
FIG_DIR = OUT_DIR / "figures"
METHODS = ["baseline", "diffgeoreward"]
ANGLES = [0, 45, 135, 270]
SEED = 42


def canvas_to_rgb(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[..., :3].copy()


def render_mesh_multiview(mesh_path, angles, resolution=256):
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
    except Exception:
        return None

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if verts.size == 0 or faces.size == 0:
        return None
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale

    if len(faces) > 5000:
        idx = np.random.choice(len(faces), 5000, replace=False)
        faces = faces[idx]

    images = []
    for angle in angles:
        fig = plt.figure(figsize=(resolution / 72, resolution / 72), dpi=72)
        ax = fig.add_subplot(111, projection="3d")
        polys = verts[faces]
        pc = Poly3DCollection(
            polys,
            alpha=0.88,
            facecolor="#87CEEB",
            edgecolor="#666666",
            linewidth=0.1,
        )
        ax.add_collection3d(pc)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=20, azim=angle)
        ax.axis("off")
        images.append(canvas_to_rgb(fig))
        plt.close(fig)

    return images


def parse_prompt_and_seed(filename):
    stem = Path(filename).stem
    prompt_slug, seed_str = stem.rsplit("_seed", 1)
    return prompt_slug, int(seed_str)


def promptify(prompt_slug):
    return prompt_slug.replace("_", " ")


def collect_pairs():
    """Collect prompt-specific mesh pairs shared by baseline and diffgeoreward."""
    base_root = ROOT / "results" / "mesh_validity_objs" / "baseline"
    dgr_root = ROOT / "results" / "mesh_validity_objs" / "diffgeoreward"
    pairs = []
    for category_dir in sorted(base_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        dgr_cat = dgr_root / category
        if not dgr_cat.exists():
            continue
        for base_file in sorted(category_dir.glob("*.obj")):
            dgr_file = dgr_cat / base_file.name
            if not dgr_file.exists():
                continue
            prompt_slug, seed = parse_prompt_and_seed(base_file.name)
            pairs.append(
                {
                    "category": category,
                    "prompt_slug": prompt_slug,
                    "prompt": promptify(prompt_slug),
                    "seed": seed,
                    "baseline": base_file,
                    "diffgeoreward": dgr_file,
                }
            )
    return pairs


def create_pair_figure(images_dict, prompt, question_id, out_path):
    methods = list(images_dict.keys())
    labels = ["A", "B"]

    fig, axes = plt.subplots(len(methods), len(ANGLES), figsize=(12, 6))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)

    for i, method in enumerate(methods):
        for j, img in enumerate(images_dict[method]):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(f"Option {labels[i]}", fontsize=14, fontweight="bold")

    fig.suptitle(f'Q{question_id}: "{prompt}"', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    pairs = collect_pairs()
    random.shuffle(pairs)

    answer_key = []
    questions = []

    for idx, row in enumerate(pairs, start=1):
        method_order = METHODS[:]
        random.shuffle(method_order)

        images_dict = {}
        for method in method_order:
            imgs = render_mesh_multiview(row[method], ANGLES)
            if imgs is None:
                images_dict = {}
                break
            images_dict[method] = imgs

        if len(images_dict) != 2:
            continue

        fig_path = FIG_DIR / f"q{idx:03d}.png"
        create_pair_figure(images_dict, row["prompt"], idx, fig_path)

        labels = {chr(65 + i): method for i, method in enumerate(method_order)}
        answer_key.append(
            {
                "question_id": idx,
                "prompt": row["prompt"],
                "prompt_slug": row["prompt_slug"],
                "category": row["category"],
                "seed": row["seed"],
                "label_to_method": labels,
                "figure": str(fig_path),
            }
        )
        questions.append(
            {
                "question_id": idx,
                "prompt": row["prompt"],
                "category": row["category"],
                "seed": row["seed"],
                "options": "A, B",
                "figure": fig_path.name,
            }
        )

    with open(OUT_DIR / "answer_key.json", "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2, ensure_ascii=False)

    questionnaire = {
        "title": "3D Mesh Quality Evaluation (Pairwise, Blinded)",
        "instructions": (
            "For each question, compare Option A and Option B. "
            "Each option shows the same mesh from 4 viewing angles.\n\n"
            "Please provide:\n"
            "1. Geometric Quality for A and B (1-5)\n"
            "2. Text Alignment for A and B (1-5)\n"
            "3. Overall Preference (A or B)\n"
        ),
        "questions": questions,
    }
    with open(OUT_DIR / "questionnaire.json", "w", encoding="utf-8") as f:
        json.dump(questionnaire, f, indent=2, ensure_ascii=False)

    csv_lines = [
        "question_id,prompt,category,seed,geo_quality_A,geo_quality_B,text_align_A,text_align_B,overall_preference"
    ]
    for q in questions:
        csv_lines.append(
            f'{q["question_id"]},"{q["prompt"]}",{q["category"]},{q["seed"]},,,,,'
        )
    with open(OUT_DIR / "evaluation_form.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    protocol = (
        "# Pairwise User Study Protocol\n\n"
        f"- Questions: {len(questions)}\n"
        "- Methods: Baseline vs DiffGeoReward\n"
        "- Blinding: randomized A/B order per question\n"
        "- Source meshes: prompt-specific OBJ pairs from results/mesh_validity_objs\n"
    )
    with open(OUT_DIR / "study_protocol.md", "w", encoding="utf-8") as f:
        f.write(protocol)

    print("=== VALID PAIRWISE USER STUDY MATERIALS ===")
    print(f"Questions generated: {len(questions)}")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Answer key: {OUT_DIR / 'answer_key.json'} (DO NOT share)")
    print(f"Questionnaire: {OUT_DIR / 'questionnaire.json'}")
    print(f"Evaluation form: {OUT_DIR / 'evaluation_form.csv'}")
    print(f"Protocol: {OUT_DIR / 'study_protocol.md'}")


if __name__ == "__main__":
    main()
