"""[P1-1] Scale adversarial stress test using TripoSR backbone.

Motivation: the existing Shap-E-based adversarial test only yielded 22/60 valid
runs because Shap-E cannot render asymmetric / broken / compositional objects.
TripoSR (via SDXL-Turbo single-view conditioning) handles those concepts much
better and should give 80%+ valid rate.

Protocol:
    - 50 adversarial prompts across 4 semantic categories
    - 3 seeds each -> 150 runs
    - SDXL-Turbo -> rembg -> TripoSR -> DGR (equal weights, multi-start plane)
    - Score: (R_sym, R_HNC, R_compact), CLIP against adversarial prompt,
      shape-CD(baseline <-> refined), per-category diagonal preservation

Output: analysis_results/nips_push_adversarial_triposr/all_results.json
        analysis_results/nips_push_adversarial_triposr/summary.json

Runtime (V100): ~3 hours (~150 images + 150 TripoSR + 150 DGR).

Launch:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/nips_push/exp_adversarial_triposr.py
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from geo_reward import (  # noqa: E402
    compactness_reward,
    estimate_symmetry_plane,
    smoothness_reward,
    symmetry_reward_plane,
)
from shape_gen import refine_with_geo_reward  # noqa: E402


# ---------- Prompt set (50 total) ----------

ADVERSARIAL_PROMPTS = {
    # Asymmetric structural (15)
    "asymmetric_structural": [
        "a broken wooden chair",
        "a leaning clock tower",
        "a bent old tree",
        "a half-eaten red apple",
        "a crooked house",
        "a cracked egg on a table",
        "a tilted ceramic vase",
        "a crashed car on its side",
        "a slanted mushroom",
        "a melting candle",
        "a torn paper bag",
        "a dented metal bucket",
        "a bent coat hanger",
        "a damaged wooden fence",
        "a tilted street lamp",
    ],
    # Pose-asymmetric (10)
    "pose_asymmetric": [
        "a person raising one arm",
        "a runner in motion",
        "a sitting dog with one paw up",
        "a waving human figure",
        "a kneeling statue",
        "a reaching human hand",
        "a dog begging on hind legs",
        "a figure leaning forward",
        "a person throwing a ball",
        "a soldier saluting",
    ],
    # Thin/elongated structures (15)
    "thin_structure": [
        "a fishing rod",
        "a sword on a stand",
        "a spear leaning against a wall",
        "a long wooden staff",
        "an axe with a long handle",
        "a guitar",
        "a violin",
        "a tall floor lamp",
        "an umbrella open",
        "a garden rake",
        "a tall thin tree",
        "a walking cane",
        "a wine glass",
        "a microphone stand",
        "a tall narrow vase",
    ],
    # Compositional with handles / appendages (10)
    "compositional_handle": [
        "a coffee mug with a handle",
        "a teapot with spout and handle",
        "a basket with one handle",
        "a hammer",
        "a pair of scissors",
        "a frying pan with a handle",
        "a watering can",
        "a pitcher with a handle",
        "a suitcase with a handle",
        "a shovel",
    ],
}
assert sum(len(v) for v in ADVERSARIAL_PROMPTS.values()) == 50, "Expected 50 prompts"

SEEDS = [42, 123, 456]
WEIGHTS = [1.0 / 3, 1.0 / 3, 1.0 / 3]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIPOSR_PATH = os.environ.get("TRIPOSR_PATH", os.path.expanduser("~/TripoSR"))

OUT_DIR = Path("analysis_results/nips_push_adversarial_triposr")
IMG_DIR = OUT_DIR / "images"
MESH_DIR = OUT_DIR / "meshes"
for d in [OUT_DIR, IMG_DIR, MESH_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---------- Image generation ----------

def load_sdxl_pipe():
    from diffusers import AutoPipelineForText2Image
    candidates = [
        ("stabilityai/sdxl-turbo", dict(torch_dtype=torch.float16, variant="fp16")),
        ("stabilityai/stable-diffusion-xl-base-1.0",
         dict(torch_dtype=torch.float16, variant="fp16")),
        ("runwayml/stable-diffusion-v1-5", dict(torch_dtype=torch.float16)),
    ]
    for mid, kw in candidates:
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(mid, use_safetensors=True, **kw).to(DEVICE)
            pipe.set_progress_bar_config(disable=True)
            print(f"  Loaded image model: {mid}")
            return pipe, "turbo" in mid
        except Exception as e:
            print(f"  {mid} unavailable: {e}")
    sys.exit("FATAL: no SDXL model available")


def generate_image(pipe, is_turbo, prompt, seed, out_path):
    if out_path.exists():
        from PIL import Image
        return Image.open(out_path).convert("RGB")
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    kw = dict(prompt=prompt, generator=gen)
    if is_turbo:
        kw.update(num_inference_steps=4, guidance_scale=0.0)
    else:
        kw.update(num_inference_steps=20, guidance_scale=7.5)
    img = pipe(**kw).images[0]
    img.save(str(out_path))
    return img


# ---------- TripoSR ----------

def load_triposr():
    sys.path.insert(0, TRIPOSR_PATH)
    from tsr.system import TSR
    print(f"Loading TripoSR from {TRIPOSR_PATH} ...")
    model = TSR.from_pretrained("stabilityai/TripoSR",
                                config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(DEVICE)
    rembg_session = None
    try:
        import rembg
        rembg_session = rembg.new_session()
    except Exception as e:
        print(f"  rembg unavailable: {e}")
    return model, rembg_session


def triposr_infer(tsr, rembg_session, image):
    from tsr.utils import resize_foreground
    from PIL import Image as PILImage
    if rembg_session is not None:
        import rembg
        img = rembg.remove(image, session=rembg_session)
        img = resize_foreground(img, 0.85)
        if img.mode == "RGBA":
            bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")
    else:
        img = image.convert("RGB")
    with torch.no_grad():
        codes = tsr([img], device=DEVICE)
        mesh_out = tsr.extract_mesh(codes, False, resolution=256)[0]
    tmesh = mesh_out if hasattr(mesh_out, "vertices") else \
        trimesh.Trimesh(vertices=mesh_out[0], faces=mesh_out[1])
    if len(tmesh.faces) > 20000:
        try:
            tmesh = tmesh.simplify_quadratic_decimation(10000)
        except Exception:
            pass
    if len(tmesh.faces) < 4:
        return None, None
    V = torch.tensor(np.asarray(tmesh.vertices), dtype=torch.float32, device=DEVICE)
    F = torch.tensor(np.asarray(tmesh.faces), dtype=torch.long, device=DEVICE)
    return V, F


# ---------- Scoring ----------

def score(V, F, pn, pd):
    with torch.no_grad():
        return {
            "sym": float(symmetry_reward_plane(V, pn, pd)),
            "hnc": float(smoothness_reward(V, F)),
            "com": float(compactness_reward(V, F)),
        }


def chamfer_l1(A, B, n=4096, rng_seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(rng_seed)
    if len(A) > n:
        A = A[rng.choice(len(A), n, replace=False)]
    if len(B) > n:
        B = B[rng.choice(len(B), n, replace=False)]
    ta, tb = cKDTree(A), cKDTree(B)
    return float(0.5 * (tb.query(A)[0].mean() + ta.query(B)[0].mean()))


def compute_clip(path, text, clip_model, clip_preprocess):
    """CLIP score of a rendered mesh against a text prompt."""
    from compute_baseline_clip import compute_clip_score
    return compute_clip_score(path, text, clip_model, clip_preprocess, DEVICE)


# ---------- Per-run pipeline ----------

def run_one(prompt, seed, category, pipe, is_turbo, tsr, rembg_session, clip_model, clip_preprocess):
    torch.manual_seed(seed); np.random.seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    ps = slug(prompt)
    img_path = IMG_DIR / f"{ps}_s{seed}.png"

    # 1. Image
    try:
        img = generate_image(pipe, is_turbo, prompt, seed, img_path)
    except Exception as e:
        return None, f"img_fail: {e}"

    # 2. TripoSR mesh
    try:
        V, F = triposr_infer(tsr, rembg_session, img)
        if V is None:
            return None, "triposr_degenerate"
    except Exception as e:
        return None, f"triposr_fail: {e}"
    if V.shape[0] < 50 or F.shape[0] < 100:
        return None, f"mesh_too_small V={V.shape[0]} F={F.shape[0]}"

    # 3. Estimate plane
    try:
        pn, pd = estimate_symmetry_plane(V)
    except Exception as e:
        return None, f"plane_fail: {e}"

    # 4. Score baseline
    bl = score(V, F, pn, pd)

    # 5. DGR refine
    w = torch.tensor(WEIGHTS, dtype=torch.float32, device=DEVICE)
    V1, _ = refine_with_geo_reward(V, F, w, steps=50, lr=0.005,
                                   sym_normal=pn, sym_offset=pd)

    # 6. Score refined
    rf = score(V1, F, pn, pd)

    # 7. Save meshes
    bl_obj = MESH_DIR / f"baseline_{ps}_s{seed}.obj"
    rf_obj = MESH_DIR / f"refined_{ps}_s{seed}.obj"
    trimesh.Trimesh(vertices=V.cpu().numpy(), faces=F.cpu().numpy(),
                    process=False).export(str(bl_obj))
    trimesh.Trimesh(vertices=V1.cpu().numpy(), faces=F.cpu().numpy(),
                    process=False).export(str(rf_obj))

    # 8. Shape drift
    shape_cd = chamfer_l1(V.cpu().numpy(), V1.cpu().numpy())

    # 9. CLIP vs ADVERSARIAL prompt
    try:
        bl_clip = compute_clip(bl_obj, prompt, clip_model, clip_preprocess)
        rf_clip = compute_clip(rf_obj, prompt, clip_model, clip_preprocess)
    except Exception as e:
        bl_clip = rf_clip = None
        print(f"    [warn] clip failed: {e}")

    return {
        "prompt": prompt, "category": category, "seed": seed,
        "n_verts": int(V.shape[0]), "n_faces": int(F.shape[0]),
        "plane_n": pn.detach().cpu().tolist(), "plane_d": float(pd),
        "bl_sym": bl["sym"], "rf_sym": rf["sym"],
        "bl_hnc": bl["hnc"], "rf_hnc": rf["hnc"],
        "bl_com": bl["com"], "rf_com": rf["com"],
        "bl_clip": bl_clip, "rf_clip": rf_clip,
        "shape_cd": shape_cd,
    }, None


def summarize(results):
    by_cat = {}
    for cat in list(ADVERSARIAL_PROMPTS) + ["OVERALL"]:
        rows = results if cat == "OVERALL" else [r for r in results if r["category"] == cat]
        if not rows:
            by_cat[cat] = {"n": 0}
            continue
        d = {"n": len(rows)}
        for key in ["sym", "hnc", "com"]:
            bl = np.array([r[f"bl_{key}"] for r in rows])
            rf = np.array([r[f"rf_{key}"] for r in rows])
            d[f"{key}_bl_mean"] = float(bl.mean())
            d[f"{key}_rf_mean"] = float(rf.mean())
            d[f"{key}_pct"] = float((rf.mean() - bl.mean()) / (abs(bl.mean()) + 1e-8) * 100)
        clips = [(r["bl_clip"], r["rf_clip"]) for r in rows
                 if r["bl_clip"] is not None and r["rf_clip"] is not None]
        if clips:
            bl_c = np.array([x[0] for x in clips])
            rf_c = np.array([x[1] for x in clips])
            d["clip_delta_mean"] = float((rf_c - bl_c).mean())
            d["n_clip"] = len(clips)
        d["shape_cd_mean"] = float(np.mean([r["shape_cd"] for r in rows]))
        by_cat[cat] = d
    return by_cat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=None,
                        help="Comma list of category keys (default: all)")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    keys = args.subset.split(",") if args.subset else list(ADVERSARIAL_PROMPTS)
    prompts = [(k, p) for k in keys for p in ADVERSARIAL_PROMPTS[k]]
    total = len(prompts) * len(SEEDS)
    print(f"=== Adversarial stress test (TripoSR backbone) ===")
    print(f"Device: {DEVICE}, prompts: {len(prompts)}, seeds: {len(SEEDS)}, total runs: {total}")

    # Resume
    results_path = OUT_DIR / "all_results.json"
    errors_path = OUT_DIR / "errors.json"
    results, errors = [], []
    done_keys = set()
    if args.resume and results_path.exists():
        with open(results_path) as f: results = json.load(f)
        done_keys = {(r["prompt"], r["seed"]) for r in results}
        print(f"Resume: {len(results)} runs already done")

    print("Loading models...")
    pipe, is_turbo = load_sdxl_pipe()
    tsr, rembg_session = load_triposr()
    import clip  # openai clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

    t0 = time.time()
    idx = 0
    for cat, prompt in prompts:
        for seed in SEEDS:
            idx += 1
            if (prompt, seed) in done_keys:
                continue
            tm0 = time.time()
            print(f"[{idx}/{total}] {cat[:8]}  {prompt[:50]}  seed={seed}  ", end="", flush=True)
            rec, err = run_one(prompt, seed, cat, pipe, is_turbo, tsr, rembg_session,
                                clip_model, clip_preprocess)
            if rec is None:
                errors.append({"prompt": prompt, "seed": seed, "category": cat, "error": err})
                print(f"[SKIP] {err}  ({time.time()-tm0:.1f}s)")
                continue
            results.append(rec)
            print(f"sym {rec['bl_sym']:+.4f}->{rec['rf_sym']:+.4f}  "
                  f"cd={rec['shape_cd']:.4f}  ({time.time()-tm0:.1f}s)")
            if idx % 10 == 0:
                with open(results_path, "w") as f: json.dump(results, f, indent=2)
                with open(errors_path, "w") as f: json.dump(errors, f, indent=2)

    with open(results_path, "w") as f: json.dump(results, f, indent=2)
    with open(errors_path, "w") as f: json.dump(errors, f, indent=2)

    print(f"\n[done] {len(results)} valid / {total} total  ({time.time()-t0:.0f}s)")
    print(f"       {len(errors)} errors, see {errors_path}")

    by_cat = summarize(results)
    with open(OUT_DIR / "summary.json", "w") as f: json.dump(by_cat, f, indent=2)

    print("\nPer-category results:")
    hdr = f"{'Category':<24}{'n':<5}{'sym%':<10}{'hnc%':<10}{'com%':<10}{'dCLIP':<10}{'shape_cd':<10}"
    print(hdr); print("-" * len(hdr))
    for cat, d in by_cat.items():
        if d["n"] == 0:
            print(f"{cat:<24}{d['n']:<5}--"); continue
        dc = f"{d.get('clip_delta_mean', float('nan')):+.4f}" if d.get("clip_delta_mean") is not None else "---"
        print(f"{cat:<24}{d['n']:<5}"
              f"{d.get('sym_pct',0):<+10.1f}{d.get('hnc_pct',0):<+10.1f}{d.get('com_pct',0):<+10.1f}"
              f"{dc:<10}{d.get('shape_cd_mean',0):<10.4f}")

    # Valid rate comparison
    valid_rate = 100.0 * len(results) / total if total else 0.0
    print(f"\nValid rate: {valid_rate:.1f}% (vs Shap-E version: 36.7%)")


if __name__ == "__main__":
    main()
