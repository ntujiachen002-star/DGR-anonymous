"""
Pilot Test 3: GeoBench — Cross-Model Geometric Quality + CLIP Orthogonality.

Generates shapes from Shap-E and Point-E, computes geometric metrics + CLIP,
tests whether CLIP and geometric quality are orthogonal.

110 prompts x 2 models x 1 seed. GPU required. ~4-5h on V100.
"""
import os, sys, json, torch, numpy as np, time
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
            if len(mesh.faces) > 10000:
                target_reduction = 1.0 - (10000 / len(mesh.faces))
                mesh = mesh.simplify_quadric_decimation(target_reduction)

            verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
            faces = torch.tensor(np.array(mesh.faces), dtype=torch.long, device=device)
            results.append((prompt, verts, faces))
        except Exception as e:
            print(f"  Shap-E failed for '{prompt[:30]}': {e}")
            results.append((prompt, None, None))

        if (pi + 1) % 20 == 0:
            print(f"  Shap-E: {pi+1}/{len(prompts)}")

    # Free memory
    del xm, model, diffusion
    torch.cuda.empty_cache()
    return results


def generate_point_e(prompts, device, seed=42):
    """Generate meshes from Point-E (if available)."""
    try:
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config as pe_diffusion
        from point_e.diffusion.sampler import PointCloudSampler
        from point_e.models.download import load_checkpoint
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        from point_e.util.pc_to_mesh import marching_cubes_mesh
    except ImportError:
        print("Point-E not installed. Skipping.")
        return [(p, None, None) for p in prompts]

    print("Loading Point-E...")
    # Load base model
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = pe_diffusion(DIFFUSION_CONFIGS[base_name])

    # Load upsampler
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = pe_diffusion(DIFFUSION_CONFIGS['upsample'])

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),
    )

    results = []
    for pi, prompt in enumerate(prompts):
        torch.manual_seed(seed + pi)
        np.random.seed(seed + pi)
        try:
            samples = None
            for x in sampler.sample_batch_progressive(
                batch_size=1, model_kwargs=dict(texts=[prompt])
            ):
                samples = x

            pc = sampler.output_to_point_clouds(samples)[0]
            # Convert point cloud to mesh via marching cubes
            mesh = marching_cubes_mesh(
                pc=pc, grid_size=128,
                progress=False
            )
            if len(mesh.verts) > 10000:
                tm = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
                tm = tm.simplify_quadric_decimation(1.0 - 10000/len(mesh.faces))
                verts = torch.tensor(np.array(tm.vertices), dtype=torch.float32, device=device)
                faces = torch.tensor(np.array(tm.faces), dtype=torch.long, device=device)
            else:
                verts = torch.tensor(np.array(mesh.verts), dtype=torch.float32, device=device)
                faces = torch.tensor(np.array(mesh.faces), dtype=torch.long, device=device)

            results.append((prompt, verts, faces))
        except Exception as e:
            print(f"  Point-E failed for '{prompt[:30]}': {e}")
            results.append((prompt, None, None))

        if (pi + 1) % 20 == 0:
            print(f"  Point-E: {pi+1}/{len(prompts)}")

    return results


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
    out_dir = Path("analysis_results/pilot_geobench")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Generate from Shap-E
    shap_e_results = generate_shap_e(ALL_PROMPTS, DEVICE, SEED)

    # Generate from Point-E
    point_e_results = generate_point_e(ALL_PROMPTS, DEVICE, SEED)

    # Compute geometric metrics
    print("\nComputing geometric metrics...")
    records = []
    for model_name, gen_results in [("shap_e", shap_e_results),
                                     ("point_e", point_e_results)]:
        for prompt, verts, faces in gen_results:
            metrics = compute_metrics(verts, faces)
            cat = PROMPT_CATEGORIES.get(prompt, "unknown")
            record = {
                "model": model_name, "prompt": prompt, "category": cat,
                "metrics_valid": metrics is not None,
            }
            if metrics:
                record.update(metrics)
            records.append(record)

    # Compute CLIP scores
    print("\nComputing CLIP scores...")
    for model_name, gen_results in [("shap_e", shap_e_results),
                                     ("point_e", point_e_results)]:
        clip_scores = compute_clip_scores(gen_results, model_name, DEVICE)
        for record in records:
            if record["model"] == model_name and record["prompt"] in clip_scores:
                record["clip_score"] = clip_scores[record["prompt"]]

    # Save all data
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(records, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nData collection: {elapsed:.0f}s")

    # === ANALYSIS ===
    print("\n=== MODEL COMPARISON ===")
    for model_name in ["shap_e", "point_e"]:
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
        shap_vals = [r[metric] for r in records
                     if r["model"] == "shap_e" and r["metrics_valid"]]
        point_vals = [r[metric] for r in records
                      if r["model"] == "point_e" and r["metrics_valid"]]
        if len(shap_vals) > 5 and len(point_vals) > 5:
            t, p = stats.ttest_ind(shap_vals, point_vals)
            d = np.mean(shap_vals) - np.mean(point_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {metric}: shap_e vs point_e diff={d:+.4f}, p={p:.4e} {sig}")

    # CLIP vs Geometric metrics correlation
    print("\n=== CLIP vs GEOMETRIC CORRELATION ===")
    valid_clip = [r for r in records if "clip_score" in r and r["metrics_valid"]]
    if len(valid_clip) > 10:
        clips = [r["clip_score"] for r in valid_clip]
        for metric in ["symmetry", "smoothness", "compactness"]:
            vals = [r[metric] for r in valid_clip]
            r, p = stats.spearmanr(clips, vals)
            print(f"  CLIP vs {metric}: r={r:.3f}, p={p:.4e}")

        # Key finding: is CLIP orthogonal to geometric quality?
        all_r = []
        for metric in ["symmetry", "smoothness", "compactness"]:
            vals = [r[metric] for r in valid_clip]
            r_val, _ = stats.spearmanr(clips, vals)
            all_r.append(abs(r_val))

        mean_abs_r = np.mean(all_r)
        print(f"\n  Mean |correlation|: {mean_abs_r:.3f}")
        if mean_abs_r < 0.3:
            print(f"  POSITIVE: CLIP and geometric metrics are orthogonal (|r|<0.3)")
            print(f"  This validates GeoBench as measuring a distinct quality axis.")
        elif mean_abs_r < 0.5:
            print(f"  WEAK POSITIVE: partial orthogonality (0.3<|r|<0.5)")
        else:
            print(f"  NEGATIVE: CLIP and geometry are correlated (|r|>0.5)")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
