"""
Exp: SDXL-Turbo + TripoSR backbone -- third pipeline generalization test.

Pipeline: text -> SDXL-Turbo (text-to-image, 4 steps) -> rembg -> TripoSR -> [DGR]

This is the third distinct generation pipeline tested:
  1. Shap-E: text -> mesh (direct, latent diffusion)
  2. MVDream + TripoSR: text -> multi-view -> mesh (multi-view aware)
  3. SDXL-Turbo + TripoSR: text -> single-view -> mesh (THIS, turbo diffusion)

The T2I backbone (SDXL-Turbo vs MVDream) produces different image distributions,
leading to different mesh characteristics from TripoSR.

Methods:
  sdxl_triposr_baseline -> raw TripoSR mesh from SDXL-Turbo image
  sdxl_triposr_dgr      -> + DiffGeoReward (50 Adam steps, lr=5e-3)

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/exp_sf3d_backbone.py
"""

import os, sys, json, time, gc
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]
STEPS = 50
LR = 0.005
DGR_W = [1/3, 1/3, 1/3]
TRIPOSR_PATH = os.environ.get('TRIPOSR_PATH', os.environ.get('TRIPOSR_PATH', os.path.join(os.path.dirname(ROOT), 'TripoSR')))
N_PER_CAT = 10  # 10 prompts per category = 30 total
OUTPUT_DIR = 'analysis_results/sdxl_triposr_backbone'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sdxl_pipeline():
    """Load a text-to-image pipeline for generating input images.
    Uses locally cached SDXL-Turbo (no network needed).
    """
    from diffusers import AutoPipelineForText2Image
    print("Loading SDXL-Turbo for text-to-image (from cache)...")
    # Try local cache first
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    print("  SDXL-Turbo loaded from cache")
    return pipe


def load_triposr_model():
    """Load TripoSR model."""
    sys.path.insert(0, TRIPOSR_PATH)
    from tsr.system import TSR
    print("Loading TripoSR...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(DEVICE)
    model.eval()
    print("  TripoSR loaded")
    return model


def generate_image(pipe, prompt, seed, size=512):
    """Generate an image from text prompt using SDXL-Turbo."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=4,  # SDXL-Turbo: 1-4 steps
        guidance_scale=0.0,     # SDXL-Turbo: no CFG needed
        height=size, width=size,
        generator=generator,
    ).images[0]
    return image


def image_to_mesh_triposr(triposr_model, image):
    """Convert image to 3D mesh using TripoSR."""
    import rembg
    from PIL import Image

    # Remove background
    rembg_session = rembg.new_session()
    image_nobg = rembg.remove(image, session=rembg_session)

    # Ensure RGBA, paste on white
    if image_nobg.mode == 'RGBA':
        bg = Image.new('RGBA', image_nobg.size, (255, 255, 255, 255))
        bg.paste(image_nobg, mask=image_nobg.split()[3])
        image_input = bg.convert('RGB')
    else:
        image_input = image_nobg.convert('RGB')

    # Run TripoSR
    import trimesh
    with torch.no_grad():
        codes = triposr_model(image_input, device=DEVICE)
        mesh = triposr_model.extract_mesh(codes, False, resolution=256)[0]

    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        t_mesh = geoms[0] if geoms else None
    elif isinstance(mesh, trimesh.Trimesh):
        t_mesh = mesh
    else:
        t_mesh = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1]) if isinstance(mesh, tuple) else None

    if t_mesh is None or len(t_mesh.vertices) < 10:
        return None, None

    # Simplify if too large
    if len(t_mesh.faces) > 10000:
        t_mesh = t_mesh.simplify_quadric_decimation(10000)

    verts = torch.tensor(np.array(t_mesh.vertices), dtype=torch.float32, device=DEVICE)
    faces = torch.tensor(np.array(t_mesh.faces), dtype=torch.long, device=DEVICE)

    return verts, faces


def eval_metrics(verts, faces):
    """Evaluate all three geometric metrics."""
    with torch.no_grad():
        sym = symmetry_reward(verts, axis=1).item()
        smo = smoothness_reward(verts, faces).item()
        com = compactness_reward(verts, faces).item()
    return sym, smo, com


def main():
    print(f"Device: {DEVICE}")
    print(f"TripoSR path: {TRIPOSR_PATH}")

    # Select prompts
    all_prompts = []
    for cat_name, cat_prompts in [
        ('symmetry', SYMMETRY_PROMPTS),
        ('smoothness', SMOOTHNESS_PROMPTS),
        ('compactness', COMPACTNESS_PROMPTS),
    ]:
        for p in cat_prompts[:N_PER_CAT]:
            if isinstance(p, tuple):
                all_prompts.append({'prompt': p[0], 'category': cat_name})
            else:
                all_prompts.append({'prompt': p, 'category': cat_name})

    print(f"Prompts: {len(all_prompts)} ({N_PER_CAT} per category)")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(all_prompts) * len(SEEDS) * 2}")

    # Load models
    t2i_pipe = load_sdxl_pipeline()
    triposr_model = load_triposr_model()

    results = []
    total = len(all_prompts) * len(SEEDS)
    count = 0

    for pinfo in all_prompts:
        prompt = pinfo['prompt']
        category = pinfo['category']

        for seed in SEEDS:
            count += 1
            t0 = time.time()

            try:
                # Step 1: Text -> Image
                image = generate_image(t2i_pipe, prompt, seed)

                # Save image
                img_dir = os.path.join(OUTPUT_DIR, 'images')
                os.makedirs(img_dir, exist_ok=True)
                slug = prompt.lower().replace(' ', '_')[:30]
                image.save(os.path.join(img_dir, f'{slug}_s{seed}.png'))

                # Step 2: Image -> Mesh (TripoSR)
                verts, faces = image_to_mesh_triposr(triposr_model, image)

                if verts is None:
                    print(f"[{count}/{total}] SKIP {prompt[:30]}... seed={seed} (degenerate mesh)")
                    continue

                # Step 3: Evaluate baseline
                bl_sym, bl_smo, bl_com = eval_metrics(verts, faces)

                # Step 4: DiffGeoReward refinement
                weights = torch.tensor(DGR_W, dtype=torch.float32)
                refined_verts, history = refine_with_geo_reward(
                    verts, faces, weights, steps=STEPS, lr=LR
                )

                # Step 5: Evaluate refined
                dgr_sym, dgr_smo, dgr_com = eval_metrics(refined_verts, faces)

                elapsed = time.time() - t0

                result = {
                    'prompt': prompt,
                    'category': category,
                    'seed': seed,
                    'n_vertices': verts.shape[0],
                    'n_faces': faces.shape[0],
                    'baseline': {'symmetry': bl_sym, 'smoothness': bl_smo, 'compactness': bl_com},
                    'dgr': {'symmetry': dgr_sym, 'smoothness': dgr_smo, 'compactness': dgr_com},
                    'sym_pct': (dgr_sym - bl_sym) / max(abs(bl_sym), 1e-8) * 100,
                    'smo_pct': (dgr_smo - bl_smo) / max(abs(bl_smo), 1e-8) * 100,
                    'com_pct': (dgr_com - bl_com) / max(abs(bl_com), 1e-8) * 100,
                    'time': elapsed,
                }
                results.append(result)

                print(f"[{count}/{total}] {prompt[:30]:30s} s={seed} V={verts.shape[0]:5d} "
                      f"sym={result['sym_pct']:+.0f}% smo={result['smo_pct']:+.0f}% "
                      f"com={result['com_pct']:+.0f}% ({elapsed:.1f}s)")

            except Exception as e:
                print(f"[{count}/{total}] ERROR {prompt[:30]}... seed={seed}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Aggregate
    if results:
        valid = [r for r in results if abs(r['sym_pct']) > 0.1 or abs(r['smo_pct']) > 0.1]
        print(f"\n{'='*60}")
        print(f"SDXL-Turbo + TripoSR BACKBONE RESULTS")
        print(f"{'='*60}")
        print(f"Total: {len(results)}, Valid: {len(valid)}")

        sym_pcts = [r['sym_pct'] for r in valid]
        smo_pcts = [r['smo_pct'] for r in valid]
        com_pcts = [r['com_pct'] for r in valid]

        print(f"Symmetry:    {np.mean(sym_pcts):+.1f}% +/- {np.std(sym_pcts):.1f}%")
        print(f"Smoothness:  {np.mean(smo_pcts):+.1f}% +/- {np.std(smo_pcts):.1f}%")
        print(f"Compactness: {np.mean(com_pcts):+.1f}% +/- {np.std(com_pcts):.1f}%")

        # Paired t-test
        from scipy.stats import ttest_rel
        for name, bl_key, dgr_key in [
            ('symmetry', 'symmetry', 'symmetry'),
            ('smoothness', 'smoothness', 'smoothness'),
            ('compactness', 'compactness', 'compactness'),
        ]:
            bl_vals = [r['baseline'][bl_key] for r in valid]
            dgr_vals = [r['dgr'][dgr_key] for r in valid]
            t, p = ttest_rel(dgr_vals, bl_vals)
            print(f"  {name}: t={t:.2f}, p={p:.2e}")

    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
