"""
Experiment O: User Study Material Generation.
Generates side-by-side comparison images for human evaluation.
Renders baseline vs DiffGeoReward vs Laplacian from 4 angles.
Outputs randomized A/B pairs + answer key for blinded evaluation.
CPU-compatible (matplotlib rendering). ~30min.
"""
import os, sys, json, random, re
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CATEGORIES = {}
for p in SYMMETRY_PROMPTS:
    PROMPT_CATEGORIES[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "smoothness"
for p in COMPACTNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "compactness"

N_STUDY_PROMPTS = 30   # sample 30 prompts (10 per category)
SEED = 42
EVAL_SEED = 42          # seed for mesh generation
METHODS = ["baseline", "diffgeoreward", "handcrafted"]
ANGLES = [0, 45, 135, 270]  # 4 viewing angles

RESULT_INDEX = {}


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def build_result_index():
    """Index prompt/seed pairs to concrete mesh paths from experiment logs."""
    root = Path("results/full")
    index = {}
    for method in METHODS:
        metrics_path = root / f"{method}_all_metrics.json"
        if not metrics_path.exists():
            index[method] = {}
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        method_index = {}
        for row in data:
            mesh_rel = row.get("mesh_path")
            if not mesh_rel:
                continue
            mesh_path = Path(mesh_rel)
            if not mesh_path.is_absolute():
                mesh_path = Path(".") / mesh_path
            method_index[(row["category"], row["prompt"], row["seed"])] = mesh_path
        index[method] = method_index
    return index


def canvas_to_rgb(fig):
    """Convert a Matplotlib canvas to an RGB numpy array across versions."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[..., :3].copy()


def render_mesh_multiview(mesh_path, angles, resolution=256):
    """Render mesh from multiple angles, return list of numpy arrays."""
    try:
        mesh = trimesh.load(str(mesh_path), force='mesh')
    except Exception:
        return None

    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
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
        ax = fig.add_subplot(111, projection='3d')
        polys = verts[faces]
        pc = Poly3DCollection(polys, alpha=0.85, facecolor='#87CEEB',
                              edgecolor='#666666', linewidth=0.1)
        ax.add_collection3d(pc)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=20, azim=angle)
        ax.axis('off')
        img = canvas_to_rgb(fig)
        images.append(img)
        plt.close(fig)

    return images


def find_mesh_file(method, category, prompt, seed):
    """Try to find mesh .obj file."""
    p = RESULT_INDEX.get(method, {}).get((category, prompt, seed))
    if p is not None and p.exists() and p.stat().st_size > 100:
        return p
    return None


def collect_available_prompts():
    """Find prompts that have meshes for every method in the study."""
    available = {"symmetry": [], "smoothness": [], "compactness": []}
    for prompt, category in PROMPT_CATEGORIES.items():
        ok = all(find_mesh_file(method, category, prompt, EVAL_SEED) is not None
                 for method in METHODS)
        if ok:
            available[category].append(prompt)
    return available


def create_comparison_figure(images_dict, prompt, question_id, out_path):
    """Create a single comparison figure with methods side by side.

    images_dict: {method_name: [img_angle0, img_angle1, ...]}
    Shows methods as rows, angles as columns.
    Method labels are anonymized (A, B, C).
    """
    methods = list(images_dict.keys())
    n_methods = len(methods)
    n_angles = len(ANGLES)

    fig, axes = plt.subplots(n_methods, n_angles,
                             figsize=(n_angles * 3, n_methods * 3))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    if n_angles == 1:
        axes = axes.reshape(-1, 1)

    labels = [chr(65 + i) for i in range(n_methods)]  # A, B, C

    for i, method in enumerate(methods):
        imgs = images_dict[method]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(f"Option {labels[i]}",
                                       fontsize=14, fontweight='bold')

    # Title with prompt (truncated)
    title = f"Q{question_id}: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\""
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    global RESULT_INDEX
    RESULT_INDEX = build_result_index()

    out_dir = Path("analysis_results/user_study")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    # Sample prompts: 10 per category for balanced evaluation, using only
    # prompt/seed pairs that have meshes for every blinded method.
    available = collect_available_prompts()
    study_prompts = []
    for cat, prompt_list in [("symmetry", available["symmetry"]),
                             ("smoothness", available["smoothness"]),
                             ("compactness", available["compactness"])]:
        n = min(10, len(prompt_list))
        selected = random.sample(prompt_list, n)
        study_prompts.extend([(p, cat) for p in selected])

    print(f"Selected {len(study_prompts)} prompts for user study")

    answer_key = []
    questions = []
    n_generated = 0

    for qi, (prompt, cat) in enumerate(study_prompts):
        question_id = qi + 1

        # Find mesh files for each method
        images_dict = {}
        method_order = list(METHODS)
        random.shuffle(method_order)  # randomize order for blinding

        all_found = True
        for method in method_order:
            mesh_path = find_mesh_file(method, cat, prompt, EVAL_SEED)
            if mesh_path is None:
                all_found = False
                break
            imgs = render_mesh_multiview(mesh_path, ANGLES)
            if imgs is None:
                all_found = False
                break
            images_dict[method] = imgs

        if not all_found or len(images_dict) < 2:
            print(f"  Q{question_id}: skipped (meshes not found)")
            continue

        # Create comparison figure
        fig_path = fig_dir / f"q{question_id:03d}.png"
        create_comparison_figure(images_dict, prompt, question_id, fig_path)

        # Record answer key (which label = which method)
        labels = {chr(65 + i): m for i, m in enumerate(method_order)}
        answer_key.append({
            "question_id": question_id,
            "prompt": prompt,
            "category": cat,
            "label_to_method": labels,
            "figure": str(fig_path),
        })

        # Create question entry
        label_str = ", ".join([f"{chr(65+i)}" for i in range(len(method_order))])
        questions.append({
            "question_id": question_id,
            "prompt": prompt,
            "category": cat,
            "options": label_str,
            "figure": f"q{question_id:03d}.png",
        })

        n_generated += 1
        if n_generated % 5 == 0:
            print(f"  Generated {n_generated}/{len(study_prompts)} questions")

    # Save answer key (DO NOT share with evaluators)
    with open(out_dir / "answer_key.json", 'w') as f:
        json.dump(answer_key, f, indent=2)

    # Save questionnaire (share with evaluators)
    questionnaire = {
        "title": "3D Mesh Quality Evaluation",
        "instructions": (
            "For each question, you will see 3D meshes generated from a text prompt, "
            "rendered from 4 viewing angles. Each row (A, B, C) is a different method.\n\n"
            "Please rate each option on three criteria (1-5 scale):\n"
            "1. Geometric Quality: How clean and well-formed is the 3D shape?\n"
            "2. Text Alignment: How well does the shape match the text description?\n"
            "3. Overall Preference: Which option would you prefer overall?\n\n"
            "For 'Overall Preference', just pick the best option (A, B, or C)."
        ),
        "questions": questions,
    }
    with open(out_dir / "questionnaire.json", 'w') as f:
        json.dump(questionnaire, f, indent=2)

    # Generate evaluation form template (CSV)
    csv_lines = ["question_id,prompt,geo_quality_A,geo_quality_B,geo_quality_C,"
                 "text_align_A,text_align_B,text_align_C,overall_preference"]
    for q in questions:
        csv_lines.append(f"{q['question_id']},\"{q['prompt']}\",,,,,,,,")
    with open(out_dir / "evaluation_form.csv", 'w') as f:
        f.write("\n".join(csv_lines))

    # Summary
    print(f"\n=== USER STUDY MATERIALS ===")
    print(f"Questions generated: {n_generated}")
    print(f"Figures saved to: {fig_dir}")
    print(f"Answer key: {out_dir / 'answer_key.json'} (DO NOT share)")
    print(f"Questionnaire: {out_dir / 'questionnaire.json'}")
    print(f"Evaluation form: {out_dir / 'evaluation_form.csv'}")

    # Also generate a study protocol document
    protocol = """# User Study Protocol: 3D Mesh Geometric Quality

## Objective
Evaluate whether DiffGeoReward refinement improves perceived geometric quality
of text-to-3D generated meshes compared to baseline and traditional methods.

## Study Design
- **Type**: Blinded A/B/C comparison
- **Participants**: Target N=20 (mix of 3D graphics experts and non-experts)
- **Stimuli**: {n_q} text-prompt/mesh triplets, rendered from 4 viewing angles
- **Methods compared** (blinded as A/B/C, randomized per question):
  - Baseline (vanilla Shap-E)
  - DiffGeoReward (our method)
  - Handcrafted equal weights (ablation)

## Evaluation Criteria
1. **Geometric Quality** (1-5): Shape regularity, surface smoothness, symmetry
2. **Text Alignment** (1-5): How well the shape matches the text description
3. **Overall Preference** (pick best): Which mesh is best overall

## Analysis Plan
- Mean rating per method (with 95% CI)
- Preference win rate
- Wilcoxon signed-rank test for paired comparisons
- Inter-rater reliability (Krippendorff's alpha)
- Stratified analysis by prompt category (symmetry/smoothness/compactness)

## Ethical Considerations
- No personal data collected beyond ratings
- Participation is voluntary
- Results anonymized
""".format(n_q=n_generated)

    with open(out_dir / "study_protocol.md", 'w') as f:
        f.write(protocol)

    print(f"Protocol: {out_dir / 'study_protocol.md'}")


if __name__ == "__main__":
    main()
