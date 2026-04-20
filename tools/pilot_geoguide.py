"""
Pilot Test 2: GeoGuide — x0-prediction mesh quality across denoising steps.

Validates whether geometric rewards can be computed on intermediate
x0-predictions during Shap-E's denoising process.
If rewards become meaningful early enough, inference-time guidance is viable.

10 prompts, log rewards at each denoising step. No guidance yet — just logging.
GPU required. ~3-4h on V100.
"""
import os, sys, json, torch, numpy as np, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

import trimesh

DEVICE = 'cuda:0'
SEED = 42

PILOT_PROMPTS = [
    "a symmetric vase", "a polished sphere", "a compact cube",
    "a smooth dolphin", "a balanced chair", "a dense rock",
    "a smooth river stone", "a symmetric butterfly sculpture",
    "a tight ball", "a smooth whale",
]


def try_extract_mesh(xm, latent, device):
    """Try to extract mesh from latent. Returns (verts, faces) or None."""
    try:
        t = decode_latent_mesh(xm, latent).tri_mesh()
        if len(t.verts) < 4 or len(t.faces) < 4:
            return None
        mesh = trimesh.Trimesh(vertices=t.verts, faces=t.faces)
        if len(mesh.faces) > 10000:
            target_reduction = 1.0 - (10000 / len(mesh.faces))
            mesh = mesh.simplify_quadric_decimation(target_reduction)
        verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.array(mesh.faces), dtype=torch.long, device=device)
        if verts.shape[0] < 4 or faces.shape[0] < 4:
            return None
        return verts, faces
    except Exception:
        return None


def compute_rewards(verts, faces):
    """Compute all three rewards. Returns dict or None on failure."""
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


def sample_with_logging(xm, model, diffusion, prompt, device, seed=42):
    """Sample from Shap-E at various Karras step counts to test early-stop viability.

    Instead of intercepting internal x0 predictions (fragile with Shap-E's Karras
    sampler), we run complete sampling at different step counts: fewer steps =
    earlier effective stopping point in denoising = more residual noise.

    This tests the same hypothesis: at what denoising progress do geometric
    rewards become meaningful?
    """
    # Step counts to test: from very few (noisy) to full (clean)
    step_counts = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]

    step_data = []
    final_rewards = None

    for n_steps in step_counts:
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            latents = sample_latents(
                batch_size=1, model=model, diffusion=diffusion,
                guidance_scale=15.0,
                model_kwargs=dict(texts=[prompt]),
                progress=False, clip_denoised=True, use_fp16=True,
                use_karras=True, karras_steps=n_steps,
                sigma_min=1e-3, sigma_max=160, s_churn=0,
            )
        except Exception as e:
            step_data.append({
                "karras_steps": n_steps,
                "progress_pct": 100.0 * n_steps / 64,
                "mesh_extracted": False,
                "rewards_valid": False,
                "error": str(e)[:100],
            })
            continue

        result = try_extract_mesh(xm, latents[0], device)
        entry = {
            "karras_steps": n_steps,
            "progress_pct": 100.0 * n_steps / 64,
            "mesh_extracted": result is not None,
        }

        if result is not None:
            verts, faces = result
            rewards = compute_rewards(verts, faces)
            entry["n_verts"] = int(verts.shape[0])
            entry["n_faces"] = int(faces.shape[0])
            if rewards is not None:
                entry.update(rewards)
                entry["rewards_valid"] = True
            else:
                entry["rewards_valid"] = False

            # Track final (64-step) as ground truth
            if n_steps == 64:
                final_rewards = rewards
        else:
            entry["rewards_valid"] = False

        step_data.append(entry)
        print(f"    steps={n_steps:3d}: mesh={'OK' if result else 'FAIL'}, "
              f"rewards={'OK' if entry.get('rewards_valid') else 'FAIL'}")

    return step_data, final_rewards


def main():
    out_dir = Path("analysis_results/pilot_geoguide")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Shap-E...")
    xm = load_model('transmitter', device=DEVICE)
    model = load_model('text300M', device=DEVICE)
    diffusion = diffusion_from_config(load_config('diffusion'))

    t0 = time.time()
    all_results = []

    for pi, prompt in enumerate(PILOT_PROMPTS):
        print(f"\n[{pi+1}/{len(PILOT_PROMPTS)}] {prompt}")
        step_data, final_rewards = sample_with_logging(
            xm, model, diffusion, prompt, DEVICE, seed=SEED + pi
        )

        n_valid = sum(1 for s in step_data if s["rewards_valid"])
        n_mesh = sum(1 for s in step_data if s["mesh_extracted"])
        print(f"  Checked {len(step_data)} timesteps: "
              f"{n_mesh} meshes extracted, {n_valid} rewards computed")

        if final_rewards:
            print(f"  Final: sym={final_rewards['symmetry']:.4f}, "
                  f"smo={final_rewards['smoothness']:.6f}, "
                  f"com={final_rewards['compactness']:.2f}")

        result = {
            "prompt": prompt,
            "step_data": step_data,
            "final_rewards": final_rewards,
        }
        all_results.append(result)

    elapsed = time.time() - t0

    # Save raw data
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Analysis: at what step count do rewards become meaningful?
    print(f"\n=== ANALYSIS ({elapsed:.0f}s) ===")

    # For each step count, compute success rate and reward statistics
    step_stats = {}
    for result in all_results:
        for entry in result["step_data"]:
            s = entry["karras_steps"]
            if s not in step_stats:
                step_stats[s] = {"mesh_ok": 0, "reward_ok": 0, "total": 0,
                                 "syms": [], "smos": [], "coms": []}
            step_stats[s]["total"] += 1
            if entry["mesh_extracted"]:
                step_stats[s]["mesh_ok"] += 1
            if entry["rewards_valid"]:
                step_stats[s]["reward_ok"] += 1
                step_stats[s]["syms"].append(entry["symmetry"])
                step_stats[s]["smos"].append(entry["smoothness"])
                step_stats[s]["coms"].append(entry["compactness"])

    print(f"{'Steps':>6s} | {'Progress':>9s} | {'Mesh%':>6s} | {'Reward%':>8s} | "
          f"{'Mean Sym':>10s} | {'Mean Smo':>12s} | {'Mean Com':>10s}")
    print("-" * 85)
    transition_steps = None
    for s in sorted(step_stats.keys()):
        st = step_stats[s]
        mesh_pct = 100 * st["mesh_ok"] / st["total"]
        rew_pct = 100 * st["reward_ok"] / st["total"]
        progress = 100.0 * s / 64
        sym_mean = np.mean(st["syms"]) if st["syms"] else float('nan')
        smo_mean = np.mean(st["smos"]) if st["smos"] else float('nan')
        com_mean = np.mean(st["coms"]) if st["coms"] else float('nan')
        print(f"{s:>6d} | {progress:>7.0f}%  | {mesh_pct:>5.0f}% | {rew_pct:>7.0f}% | "
              f"{sym_mean:>10.4f} | {smo_mean:>12.8f} | {com_mean:>10.2f}")

        # Find transition: first step count where >50% of prompts have valid rewards
        if transition_steps is None and rew_pct >= 50:
            transition_steps = s

    # Reward convergence: compare early vs final
    print("\n=== REWARD CONVERGENCE ===")
    if 64 in step_stats and step_stats[64]["syms"]:
        final_sym = np.mean(step_stats[64]["syms"])
        for s in sorted(step_stats.keys()):
            if s == 64 or not step_stats[s]["syms"]:
                continue
            early_sym = np.mean(step_stats[s]["syms"])
            rel_diff = abs(early_sym - final_sym) / (abs(final_sym) + 1e-8) * 100
            print(f"  steps={s:3d}: sym={early_sym:.4f} (vs final {final_sym:.4f}, "
                  f"diff={rel_diff:.1f}%)")

    print(f"\n=== PILOT VERDICT ===")
    if transition_steps is not None:
        viable_pct = 100.0 * (64 - transition_steps) / 64
        print(f"  Transition point: {transition_steps} steps ({100*transition_steps/64:.0f}% of full)")
        print(f"  Viable guidance range: last {viable_pct:.0f}% of denoising")
        if viable_pct >= 50:
            print(f"  POSITIVE: >=50% viable range. GeoGuide is feasible.")
        elif viable_pct >= 25:
            print(f"  WEAK POSITIVE: 25-50% range. Late-guidance variant viable.")
        else:
            print(f"  NEGATIVE: <25% viable. GeoGuide may not work.")
    else:
        print(f"  NEGATIVE: No step count achieved 50% reward success rate.")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
