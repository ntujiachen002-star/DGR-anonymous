"""
GPT-4V Perceptual Quality Evaluation for DiffGeoReward.

Renders Baseline vs DiffGeoReward meshes from 4 views, then asks GPT-4V
to judge which has better geometric quality in a pairwise comparison.

This addresses Limitation #3 (no human perceptual evaluation) by using
GPT-4V as a proxy, following the GPTEval3D methodology (Wu et al., CVPR 2024).

Input:  results/mesh_validity_objs/{baseline,diffgeoreward}/<cat>/<slug>.obj
Output: analysis_results/gpt4v_eval/results.json
        analysis_results/gpt4v_eval/summary.json

Requirements:
    pip install openai trimesh pyrender pillow numpy
    export OPENAI_API_KEY=sk-...

Usage:
    PYTHONPATH=src python tools/exp_gpt4v_eval.py
    # Or with custom API key:
    OPENAI_API_KEY=sk-xxx python tools/exp_gpt4v_eval.py
"""

import os, sys, json, time, base64, io, re
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# ── Config ────────────────────────────────────────────────────────────────────
MESH_DIR = Path("results/mesh_validity_objs")
OUT_DIR = Path("analysis_results/gpt4v_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use 50 prompts (balanced across categories) for cost efficiency
N_PER_CAT = 17  # ~50 total
SEED = 42       # Single seed for evaluation (reduce cost)
MODEL = "gpt-4o"  # GPT-4V model name (gpt-4o has vision)

VIEWS = [
    {"elev": 20, "azim": 45,  "name": "front-right"},
    {"elev": 20, "azim": 135, "name": "back-right"},
    {"elev": 20, "azim": 225, "name": "back-left"},
    {"elev": 20, "azim": 315, "name": "front-left"},
]


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_mesh_views(obj_path, views=VIEWS, resolution=(512, 512)):
    """Render mesh from multiple viewpoints. Returns list of PIL images."""
    try:
        import trimesh
        import pyrender
        from PIL import Image
    except ImportError:
        print("ERROR: pip install trimesh pyrender pillow")
        sys.exit(1)

    mesh = trimesh.load(str(obj_path), force='mesh')

    # Center and normalize
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center
    scale = np.max(np.abs(mesh.vertices))
    if scale > 0:
        mesh.vertices /= scale

    # Create pyrender mesh
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)

    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    images = []
    r = pyrender.OffscreenRenderer(*resolution)

    for view in views:
        # Camera pose from elevation and azimuth
        elev = np.radians(view["elev"])
        azim = np.radians(view["azim"])
        dist = 2.5

        x = dist * np.cos(elev) * np.sin(azim)
        y = dist * np.sin(elev)
        z = dist * np.cos(elev) * np.cos(azim)

        camera_pos = np.array([x, y, z])
        up = np.array([0, 1, 0])
        forward = -camera_pos / np.linalg.norm(camera_pos)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = camera_pos

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4)
        cam_node = scene.add(camera, pose=cam_pose)

        color, _ = r.render(scene)
        images.append(Image.fromarray(color))
        scene.remove_node(cam_node)

    r.delete()
    return images


def create_comparison_image(baseline_views, dgr_views):
    """Create a 2×4 comparison grid: top=A, bottom=B (randomized order)."""
    from PIL import Image

    # Randomize which is A and which is B
    if np.random.random() > 0.5:
        top_views, top_label = baseline_views, "baseline"
        bot_views, bot_label = dgr_views, "dgr"
    else:
        top_views, top_label = dgr_views, "dgr"
        bot_views, bot_label = baseline_views, "baseline"

    w, h = top_views[0].size
    grid = Image.new('RGB', (w * 4, h * 2 + 40), (255, 255, 255))

    # Add label
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 5), "Mesh A", fill=(0, 0, 0), font=font)
    draw.text((10, h + 25), "Mesh B", fill=(0, 0, 0), font=font)

    for i, (tv, bv) in enumerate(zip(top_views, bot_views)):
        grid.paste(tv, (i * w, 20))
        grid.paste(bv, (i * w, h + 40))

    return grid, top_label, bot_label


def image_to_base64(img):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ── GPT-4V Evaluation ────────────────────────────────────────────────────────

def ask_gpt4v(image_b64, prompt_text):
    """Query GPT-4V with an image and text prompt."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high"
                }}
            ]
        }],
        max_tokens=300,
        temperature=0.0,
    )

    return response.choices[0].message.content


EVAL_PROMPT = """You are evaluating the geometric quality of two 3D meshes (Mesh A and Mesh B) shown from 4 viewpoints each.

Focus ONLY on geometric quality — NOT texture or color. Evaluate these three properties:
1. **Symmetry**: Does the mesh exhibit bilateral symmetry where expected?
2. **Curvature regularity**: Is the surface curvature uniform and free of irregular bumps/artifacts?
3. **Compactness**: Is the shape well-defined without unnecessary bloating?

The text prompt for this object was: "{prompt}"

Please respond in EXACTLY this JSON format:
{{
  "symmetry_winner": "A" or "B" or "tie",
  "curvature_winner": "A" or "B" or "tie",
  "compactness_winner": "A" or "B" or "tie",
  "overall_winner": "A" or "B" or "tie",
  "reasoning": "brief explanation"
}}"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

    prompts_by_cat = {
        "symmetry": SYMMETRY_PROMPTS[:N_PER_CAT],
        "smoothness": SMOOTHNESS_PROMPTS[:N_PER_CAT],
        "compactness": COMPACTNESS_PROMPTS[:N_PER_CAT - 1],  # 16 to make ~50 total
    }

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Load checkpoint
    checkpoint_path = OUT_DIR / "checkpoint.json"
    if checkpoint_path.exists():
        state = json.load(open(checkpoint_path))
        results = state["results"]
        completed = set(state["completed"])
        print(f"Resuming from checkpoint: {len(completed)} done")
    else:
        results = []
        completed = set()

    all_pairs = []
    for cat, prompts in prompts_by_cat.items():
        for prompt in prompts:
            slug_p = re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')
            bl_path = MESH_DIR / "baseline" / cat / f"{slug_p}_seed{SEED}.obj"
            dgr_path = MESH_DIR / "diffgeoreward" / cat / f"{slug_p}_seed{SEED}.obj"
            if bl_path.exists() and dgr_path.exists():
                all_pairs.append((prompt, cat, bl_path, dgr_path))

    print(f"Total pairs to evaluate: {len(all_pairs)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(all_pairs) - len(completed)}")
    print()

    for i, (prompt, cat, bl_path, dgr_path) in enumerate(all_pairs):
        key = f"{prompt}_{SEED}"
        if key in completed:
            continue

        print(f"[{i+1}/{len(all_pairs)}] {prompt[:50]}... ({cat})")

        try:
            # Render views
            bl_views = render_mesh_views(bl_path)
            dgr_views = render_mesh_views(dgr_path)

            # Create comparison image (randomized A/B)
            grid, top_label, bot_label = create_comparison_image(bl_views, dgr_views)

            # Save comparison image for reference
            img_path = OUT_DIR / f"{re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')}.png"
            grid.save(str(img_path))

            # Query GPT-4V
            img_b64 = image_to_base64(grid)
            eval_prompt = EVAL_PROMPT.format(prompt=prompt)
            response = ask_gpt4v(img_b64, eval_prompt)
            print(f"  GPT-4V: {response[:100]}...")

            # Parse response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = {"overall_winner": "error", "reasoning": response}
            except:
                parsed = {"overall_winner": "error", "reasoning": response}

            # Map A/B back to baseline/dgr
            mapping = {"A": top_label, "B": bot_label}
            record = {
                "prompt": prompt,
                "category": cat,
                "seed": SEED,
                "a_is": top_label,
                "b_is": bot_label,
                "raw_response": parsed,
                "symmetry_winner": mapping.get(parsed.get("symmetry_winner", ""), "tie"),
                "curvature_winner": mapping.get(parsed.get("curvature_winner", ""), "tie"),
                "compactness_winner": mapping.get(parsed.get("compactness_winner", ""), "tie"),
                "overall_winner": mapping.get(parsed.get("overall_winner", ""), "tie"),
                "reasoning": parsed.get("reasoning", ""),
            }

            results.append(record)
            completed.add(key)

            # Save checkpoint
            json.dump({"results": results, "completed": list(completed)},
                      open(checkpoint_path, "w"), indent=2)

            # Rate limit
            time.sleep(1)

        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(2)

    # ── Summary statistics ────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("GPT-4V Perceptual Evaluation Summary")
    print("=" * 60)

    n = len(results)
    if n == 0:
        print("No results to summarize")
        return

    for metric in ["overall", "symmetry", "curvature", "compactness"]:
        key = f"{metric}_winner"
        counts = Counter(r[key] for r in results)
        dgr_wins = counts.get("dgr", 0)
        bl_wins = counts.get("baseline", 0)
        ties = counts.get("tie", 0)
        errors = counts.get("error", 0)
        valid = dgr_wins + bl_wins + ties

        print(f"\n{metric.upper()}:")
        print(f"  DGR wins: {dgr_wins}/{valid} ({dgr_wins/valid*100:.0f}%)" if valid > 0 else "  N/A")
        print(f"  Baseline wins: {bl_wins}/{valid} ({bl_wins/valid*100:.0f}%)" if valid > 0 else "")
        print(f"  Ties: {ties}/{valid} ({ties/valid*100:.0f}%)" if valid > 0 else "")
        if errors:
            print(f"  Errors: {errors}")

    # Save summary
    summary = {
        "n_evaluated": n,
        "model": MODEL,
        "seed": SEED,
    }
    for metric in ["overall", "symmetry", "curvature", "compactness"]:
        key = f"{metric}_winner"
        counts = Counter(r[key] for r in results)
        valid = counts.get("dgr", 0) + counts.get("baseline", 0) + counts.get("tie", 0)
        summary[metric] = {
            "dgr_wins": counts.get("dgr", 0),
            "baseline_wins": counts.get("baseline", 0),
            "ties": counts.get("tie", 0),
            "dgr_win_rate": counts.get("dgr", 0) / valid if valid > 0 else 0,
        }

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    json.dump(results, open(OUT_DIR / "results.json", "w"), indent=2)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
