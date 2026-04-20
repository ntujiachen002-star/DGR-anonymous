"""
Main experiment runner: runs all 3 methods across all prompts and seeds.

Usage:
    # Run DiffGeoReward experiments on GPU 0
    python src/run_experiment.py --method diffgeoreward --device cuda:0

    # Run VLM baseline on GPU 1
    python src/run_experiment.py --method vlm_baseline --device cuda:1

    # Run all methods
    python src/run_experiment.py --method all --device cuda:0
"""

import argparse
import json
import os
import time
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from shape_gen import load_shap_e, run_single_experiment
from lang2comp import Lang2Comp


# Test prompts organized by dominant property
PROMPTS = {
    'symmetry': [
        "a symmetric vase",
        "a perfectly balanced chair",
        "a symmetric butterfly sculpture",
        "an hourglass shape",
    ],
    'smoothness': [
        "a smooth organic blob",
        "a polished sphere",
        "a smooth river stone",
        "a smooth dolphin",
    ],
    'compactness': [
        "a compact cube",
        "a tight ball",
        "a dense solid shape",
        "a minimal round object",
    ],
}

SEEDS = [42, 123, 456, 789, 1024]


def get_weights_for_prompt(prompt, lang2comp_model=None):
    """Get geometric reward weights for a prompt.

    Uses Lang2Comp model if available, otherwise heuristic mapping.
    """
    if lang2comp_model is not None:
        result = lang2comp_model.predict(prompt)
        w = result['weights']
        return torch.tensor([w['symmetry'], w['smoothness'], w['compactness']])

    # Fallback heuristic
    prompt_lower = prompt.lower()
    if 'symmetric' in prompt_lower or 'balanced' in prompt_lower or 'hourglass' in prompt_lower or 'butterfly' in prompt_lower:
        return torch.tensor([0.7, 0.15, 0.15])
    elif 'smooth' in prompt_lower or 'polished' in prompt_lower or 'organic' in prompt_lower or 'dolphin' in prompt_lower:
        return torch.tensor([0.15, 0.7, 0.15])
    elif 'compact' in prompt_lower or 'dense' in prompt_lower or 'tight' in prompt_lower or 'minimal' in prompt_lower:
        return torch.tensor([0.15, 0.15, 0.7])
    else:
        return torch.tensor([0.33, 0.33, 0.34])


def main(args):
    print(f"{'=' * 60}")
    print(f"DiffGeoReward Experiment — method={args.method}, device={args.device}")
    print(f"{'=' * 60}\n")

    # Load Shap-E
    print("Loading Shap-E model...")
    xm, model, diffusion = load_shap_e(device=args.device)
    print("Shap-E loaded.\n")

    # Load Lang2Comp (if checkpoint exists)
    lang2comp = None
    ckpt_path = 'checkpoints/lang2comp_best.pt'
    if os.path.exists(ckpt_path):
        print("Loading Lang2Comp model...")
        lang2comp = Lang2Comp()
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        lang2comp.load_state_dict(state['model_state_dict'])
        lang2comp.eval()
        print(f"Lang2Comp loaded (val_loss={state['val_loss']:.4f}, acc={state['dom_acc']:.2%})\n")

    # Determine methods to run
    if args.method == 'all':
        methods = ['baseline', 'diffgeoreward', 'vlm_baseline']
    else:
        methods = [args.method]

    # Flatten prompts
    all_prompts = []
    for category, prompts in PROMPTS.items():
        for p in prompts:
            all_prompts.append((category, p))

    total_runs = len(methods) * len(all_prompts) * len(SEEDS)
    print(f"Total runs: {len(methods)} methods x {len(all_prompts)} prompts x {len(SEEDS)} seeds = {total_runs}")
    print()

    all_metrics = []
    run_idx = 0

    for method in methods:
        print(f"\n{'=' * 40}")
        print(f"Method: {method}")
        print(f"{'=' * 40}")

        for category, prompt in all_prompts:
            weights = get_weights_for_prompt(prompt, lang2comp)

            for seed in SEEDS:
                run_idx += 1
                output_dir = f"results/{method}/{category}"

                print(f"\n[{run_idx}/{total_runs}] {method} | '{prompt}' | seed={seed}")
                print(f"  weights: sym={weights[0]:.2f} smooth={weights[1]:.2f} compact={weights[2]:.2f}")

                try:
                    metrics = run_single_experiment(
                        prompt=prompt,
                        method=method,
                        seed=seed,
                        weights=weights,
                        xm=xm, model=model, diffusion=diffusion,
                        output_dir=output_dir,
                        device=args.device,
                    )

                    metrics['category'] = category
                    all_metrics.append(metrics)

                    print(f"  sym={metrics['symmetry']:.6f} "
                          f"smooth={metrics['smoothness']:.6f} "
                          f"compact={metrics['compactness']:.6f} "
                          f"time={metrics['total_time']:.1f}s")

                    if 'reward_improvement' in metrics:
                        print(f"  reward: {metrics['initial_reward']:.6f} -> {metrics['final_reward']:.6f} "
                              f"(+{metrics['reward_improvement']:.6f}) "
                              f"avg_grad={metrics['avg_grad_norm']:.6f}")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Save all metrics
    output_path = f"results/{args.method}_all_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n\nAll metrics saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for method in methods:
        method_metrics = [m for m in all_metrics if m['method'] == method]
        if not method_metrics:
            continue

        avg_sym = np.mean([m['symmetry'] for m in method_metrics])
        avg_smooth = np.mean([m['smoothness'] for m in method_metrics])
        avg_compact = np.mean([m['compactness'] for m in method_metrics])
        avg_time = np.mean([m['total_time'] for m in method_metrics])

        print(f"\n{method}:")
        print(f"  Avg Symmetry:    {avg_sym:.6f}")
        print(f"  Avg Smoothness:  {avg_smooth:.6f}")
        print(f"  Avg Compactness: {avg_compact:.6f}")
        print(f"  Avg Time:        {avg_time:.1f}s")

        if method == 'diffgeoreward':
            avg_improvement = np.mean([m.get('reward_improvement', 0) for m in method_metrics])
            avg_grad = np.mean([m.get('avg_grad_norm', 0) for m in method_metrics])
            print(f"  Avg Reward Improvement: {avg_improvement:.6f}")
            print(f"  Avg Gradient Norm:      {avg_grad:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['baseline', 'diffgeoreward', 'vlm_baseline', 'all'],
                        default='all')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Override default seeds')
    args = parser.parse_args()

    if args.seeds:
        SEEDS = args.seeds

    main(args)
