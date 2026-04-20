#!/usr/bin/env python
"""M7: Scale-Controlled Ablation.
Verify catastrophic degradation is intrinsic, not a scale artifact.
Runs on 20-prompt subset with 3 conditions: raw, normalized, gradient-matched.
"""
import os, sys, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from shape_gen import load_shap_e, generate_mesh, save_mesh
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS


def refine_raw_single(vertices, faces, target_idx, steps=50, lr=0.005):
    """Single-reward optimization with RAW (unnormalized) reward."""
    v = vertices.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        rewards = [symmetry_reward(v), smoothness_reward(v, faces), compactness_reward(v, faces)]
        loss = -rewards[target_idx]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        opt.step()
    return v.detach()


def refine_normalized_single(vertices, faces, target_idx, steps=50, lr=0.005):
    """Single-reward optimization with initial-value normalization (Eq.9)."""
    v = vertices.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    with torch.no_grad():
        init_vals = [
            abs(symmetry_reward(v).item()),
            abs(smoothness_reward(v, faces).item()),
            abs(compactness_reward(v, faces).item()),
        ]
    scale = max(init_vals[target_idx], 1e-6)
    for _ in range(steps):
        opt.zero_grad()
        rewards = [symmetry_reward(v), smoothness_reward(v, faces), compactness_reward(v, faces)]
        loss = -(rewards[target_idx] / scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        opt.step()
    return v.detach()


def refine_gradmatched_single(vertices, faces, target_idx, steps=50, lr=0.005):
    """Single-reward optimization with gradient-norm matching."""
    v = vertices.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)

    # First compute reference gradient norm from combined reward
    v_ref = vertices.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        rewards = [symmetry_reward(v_ref), smoothness_reward(v_ref, faces), compactness_reward(v_ref, faces)]
        combined = sum(r / max(abs(r.item()), 1e-6) for r in rewards) / 3.0
        combined.backward()
        ref_norm = v_ref.grad.norm().item()

    for _ in range(steps):
        opt.zero_grad()
        rewards = [symmetry_reward(v), smoothness_reward(v, faces), compactness_reward(v, faces)]
        loss = -rewards[target_idx]
        loss.backward()
        # Rescale gradient to match reference norm
        grad_norm = v.grad.norm().item()
        if grad_norm > 0:
            v.grad.data *= (ref_norm / grad_norm)
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        opt.step()
    return v.detach()


def evaluate_metrics(vertices, faces):
    with torch.no_grad():
        return {
            'symmetry': symmetry_reward(vertices).item(),
            'smoothness': smoothness_reward(vertices, faces).item(),
            'compactness': compactness_reward(vertices, faces).item(),
        }


def main():
    device = 'cuda:0'
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device)

    # 20-prompt subset: ~7 from each category
    subset = (SYMMETRY_PROMPTS[:7] + SMOOTHNESS_PROMPTS[:7] + COMPACTNESS_PROMPTS[:6])

    targets = {'sym_only': 0, 'smooth_only': 1, 'compact_only': 2}
    conditions = ['raw', 'normalized', 'gradmatched']
    refine_fns = {
        'raw': refine_raw_single,
        'normalized': refine_normalized_single,
        'gradmatched': refine_gradmatched_single,
    }

    results = []
    total = len(subset) * len(targets) * len(conditions)
    idx = 0

    for prompt in subset:
        print(f"\n=== Generating: {prompt[:50]}...")
        meshes = generate_mesh(xm, model, diffusion, prompt, device=device)
        verts, faces, _ = meshes[0]

        baseline_m = evaluate_metrics(verts, faces)
        results.append({
            'prompt': prompt, 'target': 'baseline', 'condition': 'none',
            'seed': seed, **baseline_m,
        })

        for target_name, target_idx in targets.items():
            for cond in conditions:
                idx += 1
                print(f"  [{idx}/{total}] {target_name} / {cond}...")
                refined = refine_fns[cond](verts, faces, target_idx)
                m = evaluate_metrics(refined, faces)
                results.append({
                    'prompt': prompt, 'target': target_name, 'condition': cond,
                    'seed': seed, **m,
                })

    # Save results
    os.makedirs('results/ablations', exist_ok=True)
    with open('results/ablations/scale_controlled.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("Scale-Controlled Ablation Summary")
    print("=" * 80)
    for target_name in targets:
        print(f"\n--- {target_name} ---")
        for cond in conditions:
            entries = [r for r in results if r['target'] == target_name and r['condition'] == cond]
            baselines = [r for r in results if r['target'] == 'baseline']
            if not entries:
                continue
            sym_mean = np.mean([e['symmetry'] for e in entries])
            smo_mean = np.mean([e['smoothness'] for e in entries])
            com_mean = np.mean([e['compactness'] for e in entries])
            bl_sym = np.mean([b['symmetry'] for b in baselines])
            bl_smo = np.mean([b['smoothness'] for b in baselines])
            bl_com = np.mean([b['compactness'] for b in baselines])
            print(f"  {cond:12s}: sym={sym_mean:.6f} ({(sym_mean-bl_sym)/abs(bl_sym)*100:+.0f}%), "
                  f"smo={smo_mean:.6f} ({(smo_mean-bl_smo)/abs(bl_smo)*100:+.0f}%), "
                  f"com={com_mean:.2f} ({(com_mean-bl_com)/abs(bl_com)*100:+.0f}%)")

    print(f"\nResults saved to results/ablations/scale_controlled.json")


if __name__ == '__main__':
    main()
