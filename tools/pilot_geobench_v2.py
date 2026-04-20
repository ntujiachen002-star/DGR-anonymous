"""
GeoBench v2 — Cross-Model Geometric Quality + CLIP Orthogonality.

Extended version: Shap-E (text-to-3D) + TripoSR + SF3D (image-to-3D).
110 prompts × 3 models × 1 seed. GPU required. ~8-10h on V100.

For image-to-3D models, we render Shap-E meshes to front-view images,
providing a fair comparison where all models reconstruct from the same reference.
"""
import os, sys, json, torch, numpy as np, time, gc
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

import trimesh

DEVICE = 'cuda:0'
SEED = 42

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CATEGORIES = {}
for p in SYMMETRY_PROMPTS:
    PROMPT_CATEGORIES[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "smoothness"
for p in COMPACTNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "compactness"


def simplify_mesh(mesh, max_faces=10000):
    """Simplify mesh to max_faces if needed."""
    if len(mesh.faces) > max_faces:
        target_reduction = 1.0 - (max_faces / len(mesh.faces))
        mesh = mesh.simplify_quadric_decimation(target_reduction)
    return mesh


def mesh_to_tensors(mesh, device):
    """Convert trimesh to (verts, faces) tensors."""
    verts = torch.tensor(np.array(mesh.vertices, dtype=np.float32), dtype=torch.float32, device=device)
    faces = torch.tensor(np.array(mesh.faces, dtype=np.int64), dtype=torch.long, device=device)
    return verts, faces


# ============================================================
# Generator: Shap-E (text-to-3D)
# ============================================================
def generate_shap_e(prompts, device, seed=42):
    """Generate meshes from Shap-E."""
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh

    print("Loading Shap-E...")
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    results = []
    for pi, prompt in enumerate(prompts):
        torch.manual_seed(seed + pi)
        np.random.seed(seed + pi)
        try:
            latents = sample_latents(
                batch_size=1, model=model, diffusion=diffusion,
                guidance_scale=15.0,
                model_kwargs=dict(texts=[prompt]),
                progress=False, clip_denoised=True, use_fp16=True,
                use_karras=True, karras_steps=64,
                sigma_min=1e-3, sigma_max=160, s_churn=0,
            )
            t = decode_latent_mesh(xm, latents[0]).tri_mesh()
            mesh = trimesh.Trimesh(vertices=t.verts, faces=t.faces)
            mesh = simplify_mesh(mesh)
            verts, faces = mesh_to_tensors(mesh, device)
            results.append((prompt, verts, faces))
        except Exception as e:
            print(f"  Shap-E failed for '{prompt[:30]}': {e}")
            results.append((prompt, None, None))

        if (pi + 1) % 20 == 0:
            print(f"  Shap-E: {pi+1}/{len(prompts)}")

    del xm, model, diffusion
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ============================================================
# Render Shap-E meshes to images for image-to-3D models
# ============================================================
def render_meshes_to_images(shap_e_results, cache_dir=None):
    """Render Shap-E meshes to front-view images for image-to-3D input.

    This is more fair than text-to-image: all models reconstruct from
    the same reference view, eliminating the text-to-image variable.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    import io

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    images = {}
    n_rendered = 0
    for prompt, verts, faces in shap_e_results:
        if verts is None:
            continue

        cache_path = os.path.join(cache_dir, f"{n_rendered:03d}.png") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            images[prompt] = Image.open(cache_path).convert("RGB")
            n_rendered += 1
            continue

        v_np = verts.cpu().numpy()
        f_np = faces.cpu().numpy()
        center = v_np.mean(axis=0)
        v_np = v_np - center
        scale = np.abs(v_np).max()
        if scale > 0:
            v_np = v_np / scale
        if len(f_np) > 5000:
            idx = np.random.choice(len(f_np), 5000, replace=False)
            f_np = f_np[idx]

        try:
            fig = plt.figure(figsize=(512/72, 512/72), dpi=72)
            ax = fig.add_subplot(111, projection='3d')
            polys = v_np[f_np]
            pc = Poly3DCollection(polys, alpha=0.9, facecolor='lightblue',
                                  edgecolor='gray', linewidth=0.1)
            ax.add_collection3d(pc)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.view_init(elev=20, azim=30)
            ax.set_facecolor('white')
            ax.axis('off')
            fig.patch.set_facecolor('white')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                        dpi=72, facecolor='white')
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

            if cache_path:
                image.save(cache_path)
            images[prompt] = image
            n_rendered += 1
        except Exception as e:
            print(f"  Render failed for '{prompt[:30]}': {e}")

    print(f"  Rendered {len(images)}/{len(shap_e_results)} meshes to images")
    return images


# ============================================================
# Generator: TripoSR (image-to-3D)
# ============================================================
def generate_triposr(prompts, images, device, seed=42):
    """Generate meshes from TripoSR using pre-generated images."""
    if not images:
        print("No images available for TripoSR. Skipping.")
        return [(p, None, None) for p in prompts]

    sys.path.insert(0, os.environ.get('TRIPOSR_PATH', os.path.join(os.path.dirname(ROOT), 'TripoSR')))
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
    except ImportError as e:
        print(f"TripoSR not available: {e}")
        return [(p, None, None) for p in prompts]

    print("Loading TripoSR...")
    tsr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    tsr_model.renderer.set_chunk_size(8192)
    tsr_model.to(device)

    # Pre-init rembg session once
    try:
        import rembg
        rembg_session = rembg.new_session()
        print("  rembg initialized")
    except Exception as e:
        print(f"  rembg not available ({e}), using images directly")
        rembg_session = None

    results = []
    for pi, prompt in enumerate(prompts):
        if prompt not in images:
            results.append((prompt, None, None))
            continue

        try:
            image = images[prompt]
            # Images are rendered with white bg, rembg optional
            if rembg_session is not None:
                try:
                    image_nobg = rembg.remove(image, session=rembg_session)
                    image_nobg = resize_foreground(image_nobg, 0.85)
                    if image_nobg.mode == "RGBA":
                        from PIL import Image as PILImage
                        bg = PILImage.new("RGBA", image_nobg.size, (255, 255, 255, 255))
                        bg.paste(image_nobg, mask=image_nobg.split()[3])
                        image_input = bg.convert("RGB")
                    else:
                        image_input = image_nobg.convert("RGB")
                except Exception:
                    image_input = image.convert("RGB")
            else:
                image_input = image.convert("RGB")

            with torch.no_grad():
                scene_codes = tsr_model([image_input], device=device)
                mesh_out = tsr_model.extract_mesh(scene_codes, False, resolution=256)[0]

            # mesh_out is a trimesh
            if hasattr(mesh_out, 'vertices'):
                mesh = mesh_out
            else:
                mesh = trimesh.Trimesh(vertices=mesh_out[0], faces=mesh_out[1])

            mesh = simplify_mesh(mesh)
            if len(mesh.faces) < 4:
                results.append((prompt, None, None))
                continue

            verts, faces = mesh_to_tensors(mesh, device)
            results.append((prompt, verts, faces))
        except Exception as e:
            print(f"  TripoSR failed for '{prompt[:30]}': {e}")
            results.append((prompt, None, None))

        if (pi + 1) % 20 == 0:
            print(f"  TripoSR: {pi+1}/{len(prompts)}")

    del tsr_model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ============================================================
# Generator: SF3D (image-to-3D)
# ============================================================
def generate_sf3d(prompts, images, device, seed=42):
    """Generate meshes from SF3D (Stable Fast 3D) using pre-generated images."""
    if not images:
        print("No images available for SF3D. Skipping.")
        return [(p, None, None) for p in prompts]

    sys.path.insert(0, '/root/autodl-tmp/stable-fast-3d')
    try:
        from sf3d.system import SF3D
    except ImportError as e:
        print(f"SF3D not available: {e}")
        return [(p, None, None) for p in prompts]

    print("Loading SF3D...")
    try:
        sf3d_model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        sf3d_model.to(device)
        sf3d_model.eval()
    except Exception as e:
        print(f"SF3D model loading failed: {e}")
        print("  SF3D requires HuggingFace access approval. Skipping.")
        return [(p, None, None) for p in prompts]

    results = []
    for pi, prompt in enumerate(prompts):
        if prompt not in images:
            results.append((prompt, None, None))
            continue

        try:
            image = images[prompt]
            # Remove background
            import rembg
            rembg_session = rembg.new_session()
            image_nobg = rembg.remove(image, session=rembg_session)
            # Convert to RGB with white bg
            if image_nobg.mode == "RGBA":
                from PIL import Image as PILImage
                bg = PILImage.new("RGBA", image_nobg.size, (255, 255, 255, 255))
                bg.paste(image_nobg, mask=image_nobg.split()[3])
                image_input = bg.convert("RGB")
            else:
                image_input = image_nobg.convert("RGB")

            with torch.no_grad():
                mesh_out = sf3d_model.run_image(
                    image_input,
                    bake_resolution=512,
                    remesh="none",
                )

            # Extract mesh - SF3D returns a trimesh or similar
            if hasattr(mesh_out, 'vertices'):
                mesh = mesh_out
            elif isinstance(mesh_out, tuple):
                mesh = trimesh.Trimesh(vertices=mesh_out[0], faces=mesh_out[1])
            elif isinstance(mesh_out, list) and len(mesh_out) > 0:
                m = mesh_out[0]
                if hasattr(m, 'vertices'):
                    mesh = m
                else:
                    mesh = trimesh.Trimesh(vertices=m[0], faces=m[1])
            else:
                results.append((prompt, None, None))
                continue

            mesh = simplify_mesh(mesh)
            if len(mesh.faces) < 4:
                results.append((prompt, None, None))
                continue

            verts, faces = mesh_to_tensors(mesh, device)
            results.append((prompt, verts, faces))
        except Exception as e:
            print(f"  SF3D failed for '{prompt[:30]}': {e}")
            results.append((prompt, None, None))

        if (pi + 1) % 20 == 0:
            print(f"  SF3D: {pi+1}/{len(prompts)}")

    del sf3d_model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ============================================================
# Metrics
# ============================================================
def compute_metrics(verts, faces):
    """Compute all geometric metrics."""
    if verts is None or faces is None:
        return None
    try:
        with torch.no_grad():
            sym = symmetry_reward(verts).item()
            smo = smoothness_reward(verts, faces).item()
            com = compactness_reward(verts, faces).item()
        if np.isnan(sym) or np.isnan(smo) or np.isnan(com):
            return None
        return {"symmetry": sym, "smoothness": smo, "compactness": com}
    except Exception:
        return None


def compute_clip_scores(results_list, model_name, device):
    """Compute CLIP scores for generated meshes."""
    try:
        import clip
        from PIL import Image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import io
    except ImportError:
        print("CLIP not available. Skipping CLIP scores.")
        return {}

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    scores = {}

    for prompt, verts, faces in results_list:
        if verts is None:
            continue
        v_np = verts.cpu().numpy()
        f_np = faces.cpu().numpy()
        center = v_np.mean(axis=0)
        v_np = v_np - center
        scale = np.abs(v_np).max()
        if scale > 0:
            v_np = v_np / scale
        if len(f_np) > 5000:
            idx = np.random.choice(len(f_np), 5000, replace=False)
            f_np = f_np[idx]

        view_scores = []
        for angle in [0, 90, 180, 270]:
            try:
                fig = plt.figure(figsize=(224/72, 224/72), dpi=72)
                ax = fig.add_subplot(111, projection='3d')
                polys = v_np[f_np]
                pc = Poly3DCollection(polys, alpha=0.8, facecolor='lightblue',
                                      edgecolor='gray', linewidth=0.1)
                ax.add_collection3d(pc)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
                ax.view_init(elev=20, azim=angle)
                ax.axis('off')
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=72)
                plt.close(fig)
                buf.seek(0)
                image = Image.open(buf).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)
                text_input = clip.tokenize([prompt], truncate=True).to(device)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(image_input)
                    txt_feat = clip_model.encode_text(text_input)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                    sim = (img_feat @ txt_feat.T).item()
                    view_scores.append(sim)
            except Exception:
                pass

        if view_scores:
            scores[prompt] = float(np.mean(view_scores))

    del clip_model
    torch.cuda.empty_cache()
    return scores


def main():
    out_dir = Path("analysis_results/pilot_geobench_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mesh_cache = str(out_dir / "cached_meshes")
    os.makedirs(mesh_cache, exist_ok=True)

    # === Step 1: Generate from Shap-E (text-to-3D) — with caching ===
    print("\n" + "="*60)
    print("STEP 1: Shap-E (text-to-3D)")
    print("="*60)
    # Check cache
    cache_file = os.path.join(mesh_cache, "shap_e_results.json")
    if os.path.exists(cache_file):
        print("  Loading cached Shap-E results...")
        with open(cache_file) as f:
            cached = json.load(f)
        shap_e_results = []
        for item in cached:
            if item["verts"] is not None:
                v = torch.tensor(item["verts"], dtype=torch.float32, device=DEVICE)
                fa = torch.tensor(item["faces"], dtype=torch.long, device=DEVICE)
                shap_e_results.append((item["prompt"], v, fa))
            else:
                shap_e_results.append((item["prompt"], None, None))
        print(f"  Loaded {len(shap_e_results)} cached results")
    else:
        shap_e_results = generate_shap_e(ALL_PROMPTS, DEVICE, SEED)
        # Save cache
        cache_data = []
        for prompt, verts, faces in shap_e_results:
            if verts is not None:
                cache_data.append({"prompt": prompt,
                    "verts": verts.cpu().tolist(), "faces": faces.cpu().tolist()})
            else:
                cache_data.append({"prompt": prompt, "verts": None, "faces": None})
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"  Cached {len(shap_e_results)} results")

    # === Step 2: Render Shap-E meshes to images for image-to-3D models ===
    print("\n" + "="*60)
    print("STEP 2: Rendering Shap-E meshes to images")
    print("="*60)
    img_cache = str(out_dir / "cached_images")
    images = render_meshes_to_images(shap_e_results, img_cache)

    # === Step 3: Generate from TripoSR (image-to-3D) ===
    print("\n" + "="*60)
    print("STEP 3: TripoSR (image-to-3D)")
    print("="*60)
    triposr_results = generate_triposr(ALL_PROMPTS, images, DEVICE, SEED)

    # === Step 3b: Generate from SF3D (image-to-3D) ===
    print("\n" + "="*60)
    print("STEP 3b: SF3D (image-to-3D)")
    print("="*60)
    sf3d_results = generate_sf3d(ALL_PROMPTS, images, DEVICE, SEED)

    # === Step 4: Compute geometric metrics ===
    print("\n" + "="*60)
    print("STEP 4: Computing geometric metrics")
    print("="*60)
    records = []
    model_results = [
        ("shap_e", shap_e_results),
        ("triposr", triposr_results),
        ("sf3d", sf3d_results),
    ]
    for model_name, gen_results in model_results:
        n_ok = 0
        for prompt, verts, faces in gen_results:
            metrics = compute_metrics(verts, faces)
            cat = PROMPT_CATEGORIES.get(prompt, "unknown")
            record = {
                "model": model_name, "prompt": prompt, "category": cat,
                "metrics_valid": metrics is not None,
            }
            if metrics:
                record.update(metrics)
                n_ok += 1
            records.append(record)
        print(f"  {model_name}: {n_ok}/{len(gen_results)} valid metrics")

    # === Step 5: Compute CLIP scores ===
    print("\n" + "="*60)
    print("STEP 5: Computing CLIP scores")
    print("="*60)
    for model_name, gen_results in model_results:
        clip_scores = compute_clip_scores(gen_results, model_name, DEVICE)
        for record in records:
            if record["model"] == model_name and record["prompt"] in clip_scores:
                record["clip_score"] = clip_scores[record["prompt"]]
        print(f"  {model_name}: {len(clip_scores)} CLIP scores")

    # Save all data
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(records, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nData collection: {elapsed:.0f}s")

    # === ANALYSIS ===
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Model comparison
    print("\n=== MODEL COMPARISON ===")
    all_models = ["shap_e", "triposr", "sf3d"]
    for model_name in all_models:
        valid = [r for r in records if r["model"] == model_name and r["metrics_valid"]]
        if not valid:
            print(f"  {model_name}: no valid results")
            continue
        sym = np.mean([r["symmetry"] for r in valid])
        smo = np.mean([r["smoothness"] for r in valid])
        com = np.mean([r["compactness"] for r in valid])
        clips = [r["clip_score"] for r in valid if "clip_score" in r]
        clip_mean = np.mean(clips) if clips else float('nan')
        print(f"  {model_name} (n={len(valid)}): "
              f"sym={sym:.4f}, smo={smo:.8f}, com={com:.2f}, clip={clip_mean:.4f}")

    # T-tests between models
    print("\n=== STATISTICAL COMPARISON ===")
    from scipy import stats
    for metric in ["symmetry", "smoothness", "compactness"]:
        vals_by_model = {}
        for model_name in all_models:
            vals_by_model[model_name] = [
                r[metric] for r in records
                if r["model"] == model_name and r["metrics_valid"]
            ]

        for m1, m2 in [("shap_e", "triposr"), ("shap_e", "sf3d"), ("triposr", "sf3d")]:
            if len(vals_by_model.get(m1, [])) > 5 and len(vals_by_model.get(m2, [])) > 5:
                t, p = stats.ttest_ind(vals_by_model[m1], vals_by_model[m2])
                d = np.mean(vals_by_model[m1]) - np.mean(vals_by_model[m2])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {metric}: {m1} vs {m2} diff={d:+.4f}, p={p:.4e} {sig}")

    # CLIP vs Geometric correlation
    print("\n=== CLIP vs GEOMETRIC CORRELATION ===")
    valid_clip = [r for r in records if "clip_score" in r and r["metrics_valid"]]
    if len(valid_clip) > 10:
        clips = [r["clip_score"] for r in valid_clip]
        all_r = []
        for metric in ["symmetry", "smoothness", "compactness"]:
            vals = [r[metric] for r in valid_clip]
            r_val, p = stats.spearmanr(clips, vals)
            print(f"  CLIP vs {metric}: r={r_val:.3f}, p={p:.4e}")
            all_r.append(abs(r_val))

        mean_abs_r = np.mean(all_r)
        print(f"\n  Mean |correlation|: {mean_abs_r:.3f}")
        if mean_abs_r < 0.3:
            print(f"  POSITIVE: CLIP and geometric metrics are orthogonal (|r|<0.3)")
        elif mean_abs_r < 0.5:
            print(f"  WEAK POSITIVE: partial orthogonality (0.3<|r|<0.5)")
        else:
            print(f"  NEGATIVE: CLIP and geometry are correlated (|r|>0.5)")

    # Per-category analysis
    print("\n=== PER-CATEGORY ANALYSIS ===")
    for cat in ["symmetry", "smoothness", "compactness"]:
        print(f"\n  Category: {cat}")
        for model_name in all_models:
            valid = [r for r in records
                     if r["model"] == model_name and r["category"] == cat and r["metrics_valid"]]
            if not valid:
                print(f"    {model_name}: no valid")
                continue
            vals = {m: np.mean([r[m] for r in valid]) for m in ["symmetry", "smoothness", "compactness"]}
            print(f"    {model_name} (n={len(valid)}): sym={vals['symmetry']:.4f}, "
                  f"smo={vals['smoothness']:.8f}, com={vals['compactness']:.2f}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
