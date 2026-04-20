#!/usr/bin/env python
"""M10: Hyperparameter sensitivity analysis.
Grid search over steps {25, 50, 100} x lr {0.001, 0.005, 0.01} on 20-prompt subset.
"""
import os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        estimate_symmetry_plane, smoothness_reward,
                        compactness_reward)
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS
from _plane_protocol import PlaneStore, make_key


def evaluate_metrics(vertices, faces, sym_n, sym_d):
    with torch.no_grad():
        return {
            'symmetry': symmetry_reward_plane(vertices, sym_n, sym_d).item(),
            'smoothness': smoothness_reward(vertices, faces).item(),
            'compactness': compactness_reward(vertices, faces).item(),
        }


def main():
    device = 'cuda:0'
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = 'results/ablations'
    os.makedirs(out_dir, exist_ok=True)
    plane_store = PlaneStore.load_or_new(os.path.join(out_dir, 'plane_cache.json'))

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device)

    subset = SYMMETRY_PROMPTS[:7] + SMOOTHNESS_PROMPTS[:7] + COMPACTNESS_PROMPTS[:6]
    weights = torch.tensor([0.33, 0.33, 0.34])

    step_options = [25, 50, 100]
    lr_options = [0.001, 0.005, 0.01]

    results = []
    total = len(subset) * len(step_options) * len(lr_options)
    idx = 0

    for prompt in subset:
        print(f"\n=== {prompt[:50]}...")
        meshes = generate_mesh(xm, model, diffusion, prompt, device=device)
        verts, faces, _ = meshes[0]

        # Skip degenerate baseline (Shap-E sometimes emits point clouds with 0 faces).
        if faces.shape[0] == 0 or verts.shape[0] == 0:
            print(f"  SKIP degenerate baseline (point cloud)")
            continue

        # Estimate symmetry plane once on the baseline mesh; share across all variants
        # of this (prompt, seed) for paired protocol compliance.
        sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=verts)

        bl = evaluate_metrics(verts, faces, sym_n, sym_d)
        results.append({
            'prompt': prompt, 'steps': 0, 'lr': 0, 'seed': seed, 'config': 'baseline', **bl
        })

        for steps in step_options:
            for lr in lr_options:
                idx += 1
                config = f"steps{steps}_lr{lr}"
                print(f"  [{idx}/{total}] {config}...")
                refined, _ = refine_with_geo_reward(verts, faces, weights, steps=steps, lr=lr,
                                                    sym_normal=sym_n, sym_offset=sym_d)
                m = evaluate_metrics(refined, faces, sym_n, sym_d)
                results.append({
                    'prompt': prompt, 'steps': steps, 'lr': lr,
                    'seed': seed, 'config': config, **m
                })

    plane_store.save()

    os.makedirs('results/ablations', exist_ok=True)
    with open('results/ablations/hyperparam.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary table
    print("\n" + "=" * 80)
    print("Hyperparameter Sensitivity")
    print("=" * 80)
    print(f"{'Config':>20s} {'Symmetry':>12s} {'Smoothness':>12s} {'Compactness':>12s}")
    print("-" * 60)

    baselines = [r for r in results if r['config'] == 'baseline']
    bl_sym = np.mean([b['symmetry'] for b in baselines])
    bl_smo = np.mean([b['smoothness'] for b in baselines])
    bl_com = np.mean([b['compactness'] for b in baselines])
    print(f"{'baseline':>20s} {bl_sym:>12.6f} {bl_smo:>12.6f} {bl_com:>12.2f}")

    for steps in step_options:
        for lr in lr_options:
            config = f"steps{steps}_lr{lr}"
            entries = [r for r in results if r['config'] == config]
            if not entries:
                continue
            sym = np.mean([e['symmetry'] for e in entries])
            smo = np.mean([e['smoothness'] for e in entries])
            com = np.mean([e['compactness'] for e in entries])
            sym_pct = (sym - bl_sym) / abs(bl_sym) * 100
            smo_pct = (smo - bl_smo) / abs(bl_smo) * 100
            print(f"{config:>20s} {sym:>12.6f} ({sym_pct:+.0f}%) {smo:>12.6f} ({smo_pct:+.0f}%) {com:>12.2f}")

    print(f"\nSaved to results/ablations/hyperparam.json")


if __name__ == '__main__':
    main()
