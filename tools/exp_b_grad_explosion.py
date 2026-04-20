"""
Experiment B: Gradient Explosion Root Cause Analysis
Decomposes gradient norms by reward component and tests clip threshold ablation.
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

FAILURE_PROMPTS = [
    "a symmetric vase",
    "a perfectly balanced chair",
    "an hourglass shape",
    "a symmetric wine glass",
    "a balanced chess piece, a king",
]

SUCCESS_PROMPTS = [
    "a smooth sphere",
    "a compact stone",
    "a polished marble",
    "a round pebble",
    "a smooth egg",
]

SEED = 42
STEPS = 50
LR = 0.005
DEVICE = 'cuda:0'
WEIGHTS = torch.tensor([0.33, 0.33, 0.34])

def decompose_gradients(vertices, faces, weights, steps=50, lr=0.005, clip_norm=1.0):
    """Run refinement with per-step gradient decomposition by reward component."""
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    with torch.no_grad():
        sym_init = symmetry_reward(v_opt, axis=1).item()
        smo_init = smoothness_reward(v_opt, faces).item()
        com_init = compactness_reward(v_opt, faces).item()
    sym_scale = max(abs(sym_init), 1e-6)
    smo_scale = max(abs(smo_init), 1e-6)
    com_scale = max(abs(com_init), 1e-6)

    records = []
    for step in range(steps):
        # Compute rewards
        sym = symmetry_reward(v_opt, axis=1)
        smo = smoothness_reward(v_opt, faces)
        com = compactness_reward(v_opt, faces)

        # Decompose: grad norm per reward component
        grad_norms = {}
        for name, reward, scale in [("sym", sym, sym_scale), ("smo", smo, smo_scale), ("com", com, com_scale)]:
            optimizer.zero_grad()
            (-reward / scale).backward(retain_graph=True)
            grad_norms[f"grad_norm_{name}"] = v_opt.grad.norm().item() if v_opt.grad is not None else 0.0

        # Combined gradient
        optimizer.zero_grad()
        combined = weights[0] * sym / sym_scale + weights[1] * smo / smo_scale + weights[2] * com / com_scale
        (-combined).backward()
        grad_norm_combined = v_opt.grad.norm().item() if v_opt.grad is not None else 0.0
        clip_triggered = grad_norm_combined > clip_norm

        torch.nn.utils.clip_grad_norm_([v_opt], clip_norm)
        optimizer.step()

        records.append({
            "step": step,
            **grad_norms,
            "grad_norm_combined": grad_norm_combined,
            "clip_triggered": clip_triggered,
            "symmetry": sym.item(),
            "smoothness": smo.item(),
            "compactness": com.item(),
        })

    return records

def clip_ablation(vertices, faces, weights, clip_norms=[1.0, 0.1, 0.01]):
    """Test different clip thresholds on the same mesh."""
    results = {}
    for clip_val in clip_norms:
        v_opt = vertices.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([v_opt], lr=LR)

        with torch.no_grad():
            sym_init = symmetry_reward(v_opt, axis=1).item()
            smo_init = smoothness_reward(v_opt, faces).item()
            com_init = compactness_reward(v_opt, faces).item()
        sym_scale = max(abs(sym_init), 1e-6)
        smo_scale = max(abs(smo_init), 1e-6)
        com_scale = max(abs(com_init), 1e-6)

        clip_count = 0
        for step in range(STEPS):
            optimizer.zero_grad()
            sym = symmetry_reward(v_opt, axis=1)
            smo = smoothness_reward(v_opt, faces)
            com = compactness_reward(v_opt, faces)
            combined = weights[0] * sym / sym_scale + weights[1] * smo / smo_scale + weights[2] * com / com_scale
            (-combined).backward()
            gn = v_opt.grad.norm().item() if v_opt.grad is not None else 0.0
            if gn > clip_val:
                clip_count += 1
            torch.nn.utils.clip_grad_norm_([v_opt], clip_val)
            optimizer.step()

        with torch.no_grad():
            final_sym = symmetry_reward(v_opt, axis=1).item()
            final_smo = smoothness_reward(v_opt, faces).item()
            final_com = compactness_reward(v_opt, faces).item()

        results[f"clip_{clip_val}"] = {
            "clip_norm": clip_val,
            "symmetry": final_sym,
            "smoothness": final_smo,
            "compactness": final_com,
            "clip_triggered_count": clip_count,
            "clip_triggered_pct": clip_count / STEPS * 100,
        }

    return results

def main():
    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)
    weights = WEIGHTS.to(DEVICE)

    out_dir = "analysis_results/grad_explosion"
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Gradient decomposition
    print("\n=== Step 1: Gradient Decomposition ===")
    grad_data = {"failure": {}, "success": {}}

    for group_name, prompts in [("failure", FAILURE_PROMPTS), ("success", SUCCESS_PROMPTS)]:
        for prompt in prompts:
            print(f"\n  [{group_name}] {prompt}")
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            results = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
            verts, faces, _ = results[0]
        if faces.shape[0] == 0:
            print(f"  [SKIP] {prompt}: 0 faces")
            continue
            print(f"    Mesh: {verts.shape[0]}v, {faces.shape[0]}f")

            records = decompose_gradients(verts, faces, weights, steps=STEPS)
            grad_data[group_name][prompt] = records

            # Summary
            avg_gn = np.mean([r["grad_norm_combined"] for r in records])
            max_gn = max(r["grad_norm_combined"] for r in records)
            clips = sum(1 for r in records if r["clip_triggered"])
            print(f"    avg_grad={avg_gn:.2f}, max_grad={max_gn:.2f}, clips={clips}/{STEPS}")

    with open(os.path.join(out_dir, "per_step_grad.json"), 'w') as f:
        json.dump(grad_data, f, indent=2)

    # Step 2: Clip threshold ablation (failure prompts only)
    print("\n=== Step 2: Clip Threshold Ablation ===")
    clip_data = {}

    for prompt in FAILURE_PROMPTS:
        print(f"\n  {prompt}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        results = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        verts, faces, _ = results[0]
        if faces.shape[0] == 0:
            print(f"  [SKIP] {prompt}: 0 faces")
            continue

        clip_results = clip_ablation(verts, faces, weights)
        clip_data[prompt] = clip_results

        for k, v in clip_results.items():
            print(f"    {k}: sym={v['symmetry']:.6f}, smo={v['smoothness']:.6f}, com={v['compactness']:.2f}, clips={v['clip_triggered_pct']:.0f}%")

    with open(os.path.join(out_dir, "clip_ablation.json"), 'w') as f:
        json.dump(clip_data, f, indent=2)

    # Summary table
    print("\n=== CLIP ABLATION SUMMARY ===")
    print(f"{'Clip':>8} {'Sym':>12} {'Smo':>12} {'Com':>10} {'Clip%':>8}")
    for clip_val in [1.0, 0.1, 0.01]:
        syms, smos, coms, clips = [], [], [], []
        for prompt in FAILURE_PROMPTS:
            d = clip_data.get(prompt, None)
            if d is None: continue
            d = d[f"clip_{clip_val}"]
            syms.append(d["symmetry"])
            smos.append(d["smoothness"])
            coms.append(d["compactness"])
            clips.append(d["clip_triggered_pct"])
        print(f"{clip_val:>8.2f} {np.mean(syms):>12.6f} {np.mean(smos):>12.6f} {np.mean(coms):>10.2f} {np.mean(clips):>7.1f}%")

    print(f"\nResults saved to {out_dir}/")

if __name__ == "__main__":
    main()
