"""
Experiment M: CLIP + ImageReward Comparison Across All Methods.
Renders each mesh from 4 angles, computes:
  - CLIP similarity with prompt (semantic preservation, external metric)
  - ImageReward score (perceptual quality, comparable to DreamCS's VisionReward)
This is an EXTERNAL metric (not optimized) — breaks circular evaluation.
110 prompts x 3 seeds x 6 methods = 1980 meshes.
GPU recommended. ~1.2h on V100.

ImageReward reference: Xu et al., NeurIPS 2023. Used as perceptual quality metric
in DreamReward (ECCV 2024) and comparable to VisionReward used in DreamCS (2025).
"""
import os, sys, json, torch, numpy as np, time, re, glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import clip
from PIL import Image
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import io

from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CATEGORIES = {}
for p in SYMMETRY_PROMPTS:
    PROMPT_CATEGORIES[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "smoothness"
for p in COMPACTNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "compactness"

METHODS = ["baseline", "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]
SEEDS = [42, 123, 456]


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def render_mesh_to_image(mesh_path, angle_deg=0, resolution=224):
    """Render mesh from one angle, return PIL Image."""
    mesh = trimesh.load(str(mesh_path), force='mesh')
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

    fig = plt.figure(figsize=(resolution / 72, resolution / 72), dpi=72)
    ax = fig.add_subplot(111, projection='3d')
    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=0.8, facecolor='lightblue',
                          edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=angle_deg)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def compute_scores(mesh_path, prompt, clip_model, preprocess, ir_model, device):
    """Render 4 views, compute CLIP + ImageReward for each, return averages."""
    clip_scores, ir_scores = [], []
    text_input = clip.tokenize([prompt], truncate=True).to(device)

    for angle in [0, 90, 180, 270]:
        try:
            image = render_mesh_to_image(mesh_path, angle)

            # CLIP
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(image_input)
                txt_feat = clip_model.encode_text(text_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                clip_scores.append((img_feat @ txt_feat.T).item())

            # ImageReward
            if ir_model is not None:
                try:
                    ir_scores.append(ir_model.score(prompt, image))
                except Exception:
                    pass

        except Exception:
            pass

    return (
        float(np.mean(clip_scores)) if clip_scores else None,
        float(np.mean(ir_scores))   if ir_scores   else None,
    )


def find_mesh_file(method, category, prompt, seed):
    """Try multiple naming conventions to find the mesh .obj file."""
    ps = re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')
    candidates = [
        # Exp K naming: results/mesh_validity_objs/{method}/{category}/{slug}_seed{seed}.obj
        Path(f"results/mesh_validity_objs/{method}/{category}/{ps}_seed{seed}.obj"),
        # Full experiment naming: results/full/{method}/{category}/{method}_seed{seed}.obj
        Path(f"results/full/{method}/{category}/{method}_seed{seed}.obj"),
        # Alternative with prompt in name
        Path(f"results/full/{method}/{category}/{ps}_seed{seed}.obj"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100:
            return p
    return None


def load_image_reward(device):
    """Load ImageReward model. Returns None if not installed."""
    try:
        # Fix compatibility: newer huggingface_hub removed cached_download
        import huggingface_hub
        if not hasattr(huggingface_hub, 'cached_download'):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
        import os
        # Try local cache first (set HF_HOME to a writable drive if needed).
        import ImageReward as RM
        for cache_dir in [os.path.join(ROOT, "models"),
                          os.path.expanduser("~/.cache/imagereward"),
                          os.path.expanduser("~/.cache")]:
            local_pt = os.path.join(cache_dir, 'ImageReward.pt')
            if os.path.exists(local_pt):
                model = RM.load("ImageReward-v1.0", device=device,
                                download_root=cache_dir)
                print(f"  ImageReward-v1.0 ready (from {cache_dir}).")
                return model
        # Fallback: let it download
        model = RM.load("ImageReward-v1.0", device=device)
        print("  ImageReward-v1.0 ready.")
        return model
    except Exception as e:
        print(f"  ImageReward unavailable ({e}); skipping perceptual score.")
        print("  Install with: pip install image-reward")
        return None


def main():
    out_dir = Path("analysis_results/clip_allmethod")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading CLIP (ViT-B/32) on {device}...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    ir_model = load_image_reward(device)

    # Load checkpoint if exists
    checkpoint_path = out_dir / "checkpoint.json"
    results = []
    done_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["seed"], r["method"]) for r in results}
        # Rerun entries that are missing image_reward if ir_model is available
        if ir_model is not None:
            done_keys = {k for k in done_keys
                         if any(r.get("image_reward") is not None
                                for r in results
                                if (r["prompt"], r["seed"], r["method"]) == k)}
            results = [r for r in results if (r["prompt"], r["seed"], r["method"]) in done_keys]
        print(f"Resuming: {len(results)} done")

    t0 = time.time()
    total = len(ALL_PROMPTS) * len(SEEDS) * len(METHODS)
    n_done = len(results)
    n_missing = 0

    for pi, prompt in enumerate(ALL_PROMPTS):
        cat = PROMPT_CATEGORIES[prompt]
        for seed in SEEDS:
            for method in METHODS:
                if (prompt, seed, method) in done_keys:
                    continue

                mesh_path = find_mesh_file(method, cat, prompt, seed)
                if mesh_path is None:
                    n_missing += 1
                    continue

                clip_score, ir_score = compute_scores(
                    str(mesh_path), prompt, clip_model, preprocess, ir_model, device)
                if clip_score is None:
                    continue

                record = {
                    "prompt": prompt,
                    "seed": seed,
                    "method": method,
                    "category": cat,
                    "clip_score": clip_score,
                    "image_reward": ir_score,
                    "mesh_path": str(mesh_path),
                }
                results.append(record)
                done_keys.add((prompt, seed, method))
                n_done += 1

                if n_done % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_done}/{total} ({n_missing} missing, {elapsed:.0f}s)")
                    with open(checkpoint_path, 'w') as f:
                        json.dump(results, f)

    with open(checkpoint_path, 'w') as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} scored, {n_missing} meshes missing, {elapsed:.0f}s")

    # Save full results
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # === ANALYSIS ===
    print("\n=== CLIP SCORE BY METHOD ===")
    print(f"{'Method':<16s} | {'Mean':>8s} | {'Std':>8s} | {'N':>5s}")
    print("-" * 45)
    method_scores = {}
    for method in METHODS:
        scores = [r["clip_score"] for r in results if r["method"] == method]
        if scores:
            method_scores[method] = scores
            print(f"{method:<16s} | {np.mean(scores):>8.4f} | {np.std(scores):>8.4f} | {len(scores):>5d}")

    # Key comparison: baseline vs diffgeoreward
    if "baseline" in method_scores and "diffgeoreward" in method_scores:
        bl = method_scores["baseline"]
        dgr = method_scores["diffgeoreward"]
        diff = np.mean(dgr) - np.mean(bl)
        print(f"\nDiffGeoReward vs Baseline CLIP: {diff:+.4f} ({diff/abs(np.mean(bl))*100:+.1f}%)")

        # Paired comparison (match prompt+seed)
        bl_dict = {(r["prompt"], r["seed"]): r["clip_score"]
                   for r in results if r["method"] == "baseline"}
        dgr_dict = {(r["prompt"], r["seed"]): r["clip_score"]
                    for r in results if r["method"] == "diffgeoreward"}
        common = set(bl_dict.keys()) & set(dgr_dict.keys())
        if len(common) > 10:
            paired_bl = [bl_dict[k] for k in common]
            paired_dgr = [dgr_dict[k] for k in common]
            from scipy import stats
            t, p = stats.ttest_rel(paired_dgr, paired_bl)
            d = np.mean(np.array(paired_dgr) - np.array(paired_bl))
            print(f"  Paired t-test (n={len(common)}): diff={d:.4f}, t={t:.3f}, p={p:.4e}")
            wins = sum(1 for k in common if dgr_dict[k] > bl_dict[k])
            print(f"  Win rate: {wins}/{len(common)} ({100*wins/len(common):.1f}%)")

    # Per-category
    print("\n=== CLIP BY METHOD x CATEGORY ===")
    print(f"{'Method':<16s} | {'Symmetry':>10s} | {'Smoothness':>10s} | {'Compactness':>10s}")
    print("-" * 55)
    for method in METHODS:
        vals = {}
        for cat in ["symmetry", "smoothness", "compactness"]:
            scores = [r["clip_score"] for r in results
                      if r["method"] == method and r["category"] == cat]
            vals[cat] = np.mean(scores) if scores else 0
        print(f"{method:<16s} | {vals['symmetry']:>10.4f} | {vals['smoothness']:>10.4f} | {vals['compactness']:>10.4f}")

    # All methods vs baseline paired (CLIP)
    print("\n=== ALL METHODS vs BASELINE (paired CLIP) ===")
    bl_dict = {(r["prompt"], r["seed"]): r["clip_score"]
               for r in results if r["method"] == "baseline"}
    for method in METHODS:
        if method == "baseline":
            continue
        m_dict = {(r["prompt"], r["seed"]): r["clip_score"]
                  for r in results if r["method"] == method}
        common = set(bl_dict.keys()) & set(m_dict.keys())
        if len(common) > 10:
            from scipy import stats
            paired_bl = [bl_dict[k] for k in common]
            paired_m = [m_dict[k] for k in common]
            t, p = stats.ttest_rel(paired_m, paired_bl)
            d = np.mean(np.array(paired_m) - np.array(paired_bl))
            wins = sum(1 for k in common if m_dict[k] > bl_dict[k])
            print(f"  {method}: diff={d:+.4f}, t={t:.3f}, p={p:.4e}, wins={wins}/{len(common)}")

    # ImageReward analysis (comparable to DreamCS's VisionReward)
    ir_results = [r for r in results if r.get("image_reward") is not None]
    if ir_results:
        print("\n=== ImageReward BY METHOD (comparable to DreamCS VisionReward) ===")
        print(f"{'Method':<16s} | {'Mean':>8s} | {'Std':>8s} | {'N':>5s}")
        print("-" * 45)
        ir_method_scores = {}
        for method in METHODS:
            sc = [r["image_reward"] for r in ir_results if r["method"] == method]
            if sc:
                ir_method_scores[method] = sc
                print(f"{method:<16s} | {np.mean(sc):>8.4f} | {np.std(sc):>8.4f} | {len(sc):>5d}")

        print("\n=== ALL METHODS vs BASELINE (paired ImageReward) ===")
        from scipy import stats as _stats
        bl_ir = {(r["prompt"], r["seed"]): r["image_reward"]
                 for r in ir_results if r["method"] == "baseline"}
        for method in METHODS:
            if method == "baseline":
                continue
            m_ir = {(r["prompt"], r["seed"]): r["image_reward"]
                    for r in ir_results if r["method"] == method}
            common = set(bl_ir.keys()) & set(m_ir.keys())
            if len(common) > 10:
                a = [bl_ir[k] for k in common]
                b = [m_ir[k]  for k in common]
                t, p = _stats.ttest_rel(b, a)
                d = np.mean(np.array(b) - np.array(a))
                wins = sum(1 for k in common if m_ir[k] > bl_ir[k])
                print(f"  {method}: diff={d:+.4f}, t={t:.3f}, p={p:.4e}, wins={wins}/{len(common)}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
