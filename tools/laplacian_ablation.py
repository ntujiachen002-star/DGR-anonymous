#!/usr/bin/env python
"""M8: Anti-Collapse Regularization (Laplacian Smoothing + Displacement Penalty).
Tests on 20-prompt subset.
"""
import os, sys, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from shape_gen import load_shap_e, generate_mesh
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS


def compute_laplacian_loss(vertices, faces):
    """Uniform Laplacian smoothing loss."""
    V = vertices.shape[0]
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]
    ], dim=0)
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = torch.zeros(V, 1, device=vertices.device)
    neighbor_sum.scatter_add_(0, edges[:, 1:2].expand(-1, 3), vertices[edges[:, 0]])
    neighbor_count.scatter_add_(0, edges[:, 1:2], torch.ones(edges.shape[0], 1, device=vertices.device))
    neighbor_count = neighbor_count.clamp(min=1)
    laplacian = vertices - neighbor_sum / neighbor_count
    return (laplacian ** 2).sum(dim=1).mean()


def refine_with_reg(vertices, faces, weights, reg_type='none', reg_lambda=0.01,
                     steps=50, lr=0.005):
    """Refine with optional regularization."""
    v_init = vertices.detach().clone()
    v = vertices.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)

    with torch.no_grad():
        scales = [
            max(abs(symmetry_reward(v).item()), 1e-6),
            max(abs(smoothness_reward(v, faces).item()), 1e-6),
            max(abs(compactness_reward(v, faces).item()), 1e-6),
        ]

    for _ in range(steps):
        opt.zero_grad()
        sym = symmetry_reward(v)
        smo = smoothness_reward(v, faces)
        com = compactness_reward(v, faces)
        reward = (weights[0] * sym / scales[0] +
                  weights[1] * smo / scales[1] +
                  weights[2] * com / scales[2])

        reg = torch.tensor(0.0, device=v.device)
        if reg_type == 'laplacian':
            reg = compute_laplacian_loss(v, faces)
        elif reg_type == 'displacement':
            reg = (v - v_init).norm(dim=-1).mean()

        loss = -reward + reg_lambda * reg
        loss.backward()
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

    subset = SYMMETRY_PROMPTS[:7] + SMOOTHNESS_PROMPTS[:7] + COMPACTNESS_PROMPTS[:6]
    weights = torch.tensor([0.33, 0.33, 0.34])

    configs = [
        ('no_reg', 'none', 0.0),
        ('laplacian_0.001', 'laplacian', 0.001),
        ('laplacian_0.01', 'laplacian', 0.01),
        ('laplacian_0.1', 'laplacian', 0.1),
        ('disp_0.001', 'displacement', 0.001),
        ('disp_0.01', 'displacement', 0.01),
        ('disp_0.1', 'displacement', 0.1),
    ]

    results = []
    for prompt in subset:
        print(f"\n=== {prompt[:50]}...")
        meshes = generate_mesh(xm, model, diffusion, prompt, device=device)
        verts, faces, _ = meshes[0]

        bl = evaluate_metrics(verts, faces)
        results.append({'prompt': prompt, 'config': 'baseline', 'seed': seed, **bl})

        for config_name, reg_type, reg_lambda in configs:
            print(f"  {config_name}...")
            refined = refine_with_reg(verts, faces, weights, reg_type, reg_lambda)
            m = evaluate_metrics(refined, faces)
            results.append({'prompt': prompt, 'config': config_name, 'seed': seed, **m})

    os.makedirs('results/ablations', exist_ok=True)
    with open('results/ablations/laplacian.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("Regularization Ablation Summary")
    print("=" * 80)
    for config_name, _, _ in configs:
        entries = [r for r in results if r['config'] == config_name]
        baselines = [r for r in results if r['config'] == 'baseline']
        if not entries:
            continue
        sym = np.mean([e['symmetry'] for e in entries])
        smo = np.mean([e['smoothness'] for e in entries])
        com = np.mean([e['compactness'] for e in entries])
        bl_com = np.mean([b['compactness'] for b in baselines])

        # Count compact failures
        n_fail = sum(1 for e, b in zip(
            sorted(entries, key=lambda x: x['prompt']),
            sorted(baselines, key=lambda x: x['prompt']))
            if e['compactness'] < b['compactness'])

        print(f"  {config_name:20s}: sym={sym:.6f} smo={smo:.6f} com={com:.2f} "
              f"(compact fail: {n_fail}/{len(entries)})")

    print(f"\nSaved to results/ablations/laplacian.json")


if __name__ == '__main__':
    main()
