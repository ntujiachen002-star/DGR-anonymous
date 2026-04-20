"""
Pilot Test 1: GeoReward-FT — Reward-Weighted Fine-Tuning of Shap-E.

Strategy: REINFORCE-style reward-weighted regression on Shap-E latent diffusion.
1. Generate N shapes per prompt, cache latents + geometric rewards
2. Compute advantage = reward - baseline (mean reward per prompt)
3. Fine-tune denoiser: L = -advantage * log_prob(latent|text)
4. Evaluate: compare geometric scores before/after fine-tuning

Pilot scale: 20 prompts × 50 samples = 1000 shapes
GPU required. ~8h on V100.
"""
import os, sys, json, torch, numpy as np, time, copy
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

import trimesh

DEVICE = 'cuda:0'
SEED = 42

# Pilot scale
N_PROMPTS = 20
N_SAMPLES_PER_PROMPT = 50
FT_STEPS = 500
FT_LR = 1e-5
FT_BATCH_SIZE = 4
EVAL_SAMPLES = 20  # samples per prompt for evaluation

# Prompts: balanced 7 sym + 7 smo + 6 com
PILOT_PROMPTS = [
    # Symmetry
    "a symmetric vase", "a perfectly balanced chair", "an hourglass shape",
    "a symmetric wine glass", "a symmetric trophy", "a balanced chess piece, a king",
    "a symmetric butterfly sculpture",
    # Smoothness
    "a smooth organic blob", "a polished sphere", "a smooth river stone",
    "a smooth dolphin", "a polished marble egg", "a smooth soap bar",
    "a smooth whale",
    # Compactness
    "a compact cube", "a tight ball", "a dense solid shape",
    "a solid brick", "a compact robot", "a dense rock",
]


def extract_mesh(xm, latent, device):
    """Decode latent to mesh vertices and faces."""
    t = decode_latent_mesh(xm, latent).tri_mesh()
    mesh = trimesh.Trimesh(vertices=t.verts, faces=t.faces)
    if len(mesh.faces) > 10000:
        target_reduction = 1.0 - (10000 / len(mesh.faces))
        mesh = mesh.simplify_quadric_decimation(target_reduction)
    verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.array(mesh.faces), dtype=torch.long, device=device)
    return verts, faces


def compute_geo_reward(verts, faces, weights=None):
    """Compute weighted geometric reward."""
    if weights is None:
        weights = [0.33, 0.33, 0.34]
    with torch.no_grad():
        sym = symmetry_reward(verts).item()
        smo = smoothness_reward(verts, faces).item()
        com = compactness_reward(verts, faces).item()
    # Normalize to similar scale (approximate)
    sym_n = max(sym / 0.1, -10)  # typical range [-0.5, 0]
    smo_n = max(smo / 0.01, -10)  # typical range [-0.05, 0]
    com_n = max(com / 10, -10)  # typical range [-50, -5]
    reward = weights[0] * sym_n + weights[1] * smo_n + weights[2] * com_n
    return reward, sym, smo, com


def phase1_collect_data(xm, model, diffusion, out_dir):
    """Phase 1: Generate shapes and compute rewards."""
    data_path = out_dir / "collected_data.json"
    if data_path.exists():
        print("Phase 1: Loading cached data...")
        with open(data_path) as f:
            return json.load(f)

    print(f"Phase 1: Generating {N_PROMPTS} x {N_SAMPLES_PER_PROMPT} = {N_PROMPTS * N_SAMPLES_PER_PROMPT} shapes...")
    all_data = []
    latent_dir = out_dir / "latents"
    latent_dir.mkdir(exist_ok=True)

    t0 = time.time()
    for pi, prompt in enumerate(PILOT_PROMPTS[:N_PROMPTS]):
        prompt_data = []
        for si in range(0, N_SAMPLES_PER_PROMPT, 4):
            bs = min(4, N_SAMPLES_PER_PROMPT - si)
            torch.manual_seed(SEED + pi * 1000 + si)
            np.random.seed(SEED + pi * 1000 + si)

            latents = sample_latents(
                batch_size=bs, model=model, diffusion=diffusion,
                guidance_scale=15.0,
                model_kwargs=dict(texts=[prompt] * bs),
                progress=False, clip_denoised=True, use_fp16=True,
                use_karras=True, karras_steps=64,
                sigma_min=1e-3, sigma_max=160, s_churn=0,
            )

            for j, latent in enumerate(latents):
                idx = si + j
                try:
                    verts, faces = extract_mesh(xm, latent, DEVICE)
                    reward, sym, smo, com = compute_geo_reward(verts, faces)

                    # Save latent for fine-tuning
                    lat_path = str(latent_dir / f"p{pi:02d}_s{idx:03d}.pt")
                    torch.save(latent.detach().cpu(), lat_path)

                    prompt_data.append({
                        "prompt_idx": pi, "sample_idx": idx,
                        "prompt": prompt,
                        "reward": reward, "symmetry": sym,
                        "smoothness": smo, "compactness": com,
                        "latent_path": lat_path,
                        "n_verts": int(verts.shape[0]),
                        "n_faces": int(faces.shape[0]),
                    })
                except Exception as e:
                    print(f"  Error p{pi} s{idx}: {e}")

        all_data.extend(prompt_data)
        rewards = [d["reward"] for d in prompt_data]
        elapsed = time.time() - t0
        print(f"  [{pi+1}/{N_PROMPTS}] {prompt[:30]}... "
              f"n={len(prompt_data)}, r={np.mean(rewards):.3f}+-{np.std(rewards):.3f} "
              f"({elapsed:.0f}s)")

    with open(data_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Phase 1 done: {len(all_data)} samples collected")
    return all_data


class LatentRewardDataset(Dataset):
    """Dataset of (latent, prompt, advantage) for fine-tuning."""
    def __init__(self, data_records, device):
        self.records = data_records
        self.device = device

        # Compute per-prompt baseline (mean reward)
        prompt_rewards = {}
        for r in data_records:
            pi = r["prompt_idx"]
            if pi not in prompt_rewards:
                prompt_rewards[pi] = []
            prompt_rewards[pi].append(r["reward"])

        self.baselines = {pi: np.mean(rews) for pi, rews in prompt_rewards.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        latent = torch.load(r["latent_path"], map_location='cpu')
        advantage = r["reward"] - self.baselines[r["prompt_idx"]]
        return {
            "latent": latent,
            "prompt": r["prompt"],
            "advantage": torch.tensor(advantage, dtype=torch.float32),
        }


def phase2_finetune(model, diffusion, data_records, out_dir):
    """Phase 2: REINFORCE-style fine-tuning of the denoiser."""
    print(f"\nPhase 2: Fine-tuning for {FT_STEPS} steps...")

    # Save original weights for comparison
    original_state = copy.deepcopy(model.state_dict())

    dataset = LatentRewardDataset(data_records, DEVICE)
    loader = DataLoader(dataset, batch_size=FT_BATCH_SIZE, shuffle=True,
                        collate_fn=lambda batch: batch)

    # Only fine-tune the last few layers to avoid catastrophic forgetting
    # Freeze most of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze output layers (last transformer block + output projection)
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['output_proj', 'final_ln', 'out.',
                                     'backbone.resblocks.23',
                                     'backbone.resblocks.22']):
            param.requires_grad = True
            trainable_params.append(param)

    if not trainable_params:
        # Fallback: unfreeze all parameters with 'out' in name
        for name, param in model.named_parameters():
            if 'out' in name.lower():
                param.requires_grad = True
                trainable_params.append(param)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params "
          f"({100*n_trainable/n_total:.1f}%)")

    if n_trainable == 0:
        print("  WARNING: No trainable params found. Unfreezing all...")
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = list(model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=FT_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FT_STEPS)

    history = []
    step = 0
    t0 = time.time()

    while step < FT_STEPS:
        for batch in loader:
            if step >= FT_STEPS:
                break

            # REINFORCE: approximate log_prob via denoising loss weighted by advantage
            # For each sample: compute MSE denoising loss, weight by advantage
            total_loss = 0
            n_valid = 0

            for sample in batch:
                latent = sample["latent"].to(DEVICE)
                advantage = sample["advantage"].to(DEVICE)

                # Sample random timestep
                t = torch.randint(0, diffusion.num_timesteps, (1,),
                                  device=DEVICE)

                # Add noise
                noise = torch.randn_like(latent)
                x_t = diffusion.q_sample(latent.unsqueeze(0), t, noise=noise.unsqueeze(0))

                # Predict noise (simplified — actual Shap-E uses text conditioning)
                # We compute denoising loss and weight by advantage
                model_output = model(x_t, t,
                                     texts=[sample["prompt"]])

                # MSE loss (standard denoising objective)
                # Shap-E outputs [eps, v] concatenated (2x channels).
                # Take first half as epsilon prediction.
                noise_target = noise.unsqueeze(0)
                if hasattr(model_output, 'predicted_noise'):
                    pred = model_output.predicted_noise
                elif isinstance(model_output, tuple):
                    pred = model_output[0]
                else:
                    pred = model_output
                C = noise_target.shape[-1]
                if pred.shape[-1] > C:
                    pred = pred[..., :C]  # take eps, discard v
                mse = ((pred - noise_target) ** 2).mean()

                # REINFORCE: weight denoising loss by advantage
                # Positive advantage → reduce loss (encourage this latent)
                # Negative advantage → increase loss (discourage this latent)
                weighted_loss = mse * (-advantage)  # negative because we minimize

                total_loss = total_loss + weighted_loss
                n_valid += 1

            if n_valid > 0:
                loss = total_loss / n_valid
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()

                history.append({
                    "step": step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })

                if step % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  Step {step}/{FT_STEPS}: loss={loss.item():.4f}, "
                          f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s)")

            step += 1

    # Save fine-tuned model
    ft_path = out_dir / "shape_ft_pilot.pt"
    torch.save(model.state_dict(), ft_path)
    print(f"  Fine-tuned model saved to {ft_path}")

    # Save training history
    with open(out_dir / "ft_history.json", 'w') as f:
        json.dump(history, f)

    return model, original_state


def phase3_evaluate(xm, model, diffusion, original_state, out_dir):
    """Phase 3: Compare original vs fine-tuned generation quality."""
    print(f"\nPhase 3: Evaluating ({EVAL_SAMPLES} samples per prompt)...")

    results = {"original": [], "finetuned": []}

    for phase_name, state_dict in [("finetuned", None), ("original", original_state)]:
        if state_dict is not None:
            model.load_state_dict(state_dict)

        for pi, prompt in enumerate(PILOT_PROMPTS[:N_PROMPTS]):
            prompt_metrics = []
            for si in range(0, EVAL_SAMPLES, 4):
                bs = min(4, EVAL_SAMPLES - si)
                torch.manual_seed(SEED + 99999 + pi * 1000 + si)  # different seed from training
                np.random.seed(SEED + 99999 + pi * 1000 + si)

                latents = sample_latents(
                    batch_size=bs, model=model, diffusion=diffusion,
                    guidance_scale=15.0,
                    model_kwargs=dict(texts=[prompt] * bs),
                    progress=False, clip_denoised=True, use_fp16=True,
                    use_karras=True, karras_steps=64,
                    sigma_min=1e-3, sigma_max=160, s_churn=0,
                )

                for latent in latents:
                    try:
                        verts, faces = extract_mesh(xm, latent, DEVICE)
                        reward, sym, smo, com = compute_geo_reward(verts, faces)
                        prompt_metrics.append({
                            "reward": reward, "symmetry": sym,
                            "smoothness": smo, "compactness": com,
                        })
                    except Exception:
                        pass

            if prompt_metrics:
                results[phase_name].append({
                    "prompt": prompt, "prompt_idx": pi,
                    "n_samples": len(prompt_metrics),
                    "mean_reward": np.mean([m["reward"] for m in prompt_metrics]),
                    "mean_symmetry": np.mean([m["symmetry"] for m in prompt_metrics]),
                    "mean_smoothness": np.mean([m["smoothness"] for m in prompt_metrics]),
                    "mean_compactness": np.mean([m["compactness"] for m in prompt_metrics]),
                })

        print(f"  {phase_name}: {sum(r['n_samples'] for r in results[phase_name])} samples")

    # Save results
    with open(out_dir / "eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Analysis
    print(f"\n=== PILOT RESULTS ===")
    for metric in ["mean_reward", "mean_symmetry", "mean_smoothness", "mean_compactness"]:
        orig_vals = [r[metric] for r in results["original"]]
        ft_vals = [r[metric] for r in results["finetuned"]]
        orig_mean = np.mean(orig_vals)
        ft_mean = np.mean(ft_vals)
        change = (ft_mean - orig_mean) / abs(orig_mean) * 100 if abs(orig_mean) > 1e-10 else 0
        print(f"  {metric}: orig={orig_mean:.4f}, ft={ft_mean:.4f} ({change:+.1f}%)")

    # Per-prompt comparison
    print(f"\n=== PER-PROMPT ===")
    wins = 0
    for i in range(len(results["original"])):
        o = results["original"][i]
        f_ = results["finetuned"][i]
        better = f_["mean_reward"] > o["mean_reward"]
        if better:
            wins += 1
        marker = "+" if better else "-"
        print(f"  [{marker}] {o['prompt'][:35]}: {o['mean_reward']:.3f} -> {f_['mean_reward']:.3f}")
    print(f"\n  Win rate: {wins}/{len(results['original'])} ({100*wins/len(results['original']):.0f}%)")

    # Success criteria
    orig_reward = np.mean([r["mean_reward"] for r in results["original"]])
    ft_reward = np.mean([r["mean_reward"] for r in results["finetuned"]])
    improvement = (ft_reward - orig_reward) / abs(orig_reward) * 100 if abs(orig_reward) > 1e-10 else 0

    print(f"\n=== PILOT VERDICT ===")
    print(f"  Overall reward: {orig_reward:.4f} -> {ft_reward:.4f} ({improvement:+.1f}%)")
    if improvement > 10:
        print(f"  POSITIVE: >10% improvement. Proceed to full experiment.")
    elif improvement > 0:
        print(f"  WEAK POSITIVE: some improvement. May need tuning.")
    else:
        print(f"  NEGATIVE: no improvement. Reconsider approach.")

    return results


def main():
    out_dir = Path("analysis_results/pilot_georeward_ft")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Shap-E...")
    xm = load_model('transmitter', device=DEVICE)
    model = load_model('text300M', device=DEVICE)
    diffusion = diffusion_from_config(load_config('diffusion'))

    t0 = time.time()

    # Phase 1: Collect training data
    data = phase1_collect_data(xm, model, diffusion, out_dir)

    # Phase 2: Fine-tune
    model, original_state = phase2_finetune(model, diffusion, data, out_dir)

    # Phase 3: Evaluate
    results = phase3_evaluate(xm, model, diffusion, original_state, out_dir)

    total_time = time.time() - t0
    print(f"\nTotal pilot time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
