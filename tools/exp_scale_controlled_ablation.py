"""Scale-Controlled Ablation: Rule out scale mismatch as cause of catastrophic degradation.

The key reviewer concern: "Is 2886% degradation just because rewards have different scales?"

This experiment runs single-reward ablations with gradient normalization:
  1. Standard single-reward (as in paper)
  2. Gradient-normalized single-reward: normalize gradient to unit norm at each step

If degradation persists under gradient normalization, the interference is
fundamental (frequency-dependent), not a scale artifact.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/exp_scale_controlled_ablation.py
"""

import os
import sys
import json
import time
import numpy as np
import torch

# Force offline mode
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shape_gen import load_shap_e, generate_mesh, save_mesh
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        estimate_symmetry_plane, smoothness_reward,
                        compactness_reward)
from _plane_protocol import PlaneStore, make_key


def refine_gradient_normalized(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    reward_fn,
    reward_kwargs: dict,
    sym_n: torch.Tensor,
    sym_d: torch.Tensor,
    steps: int = 50,
    lr: float = 0.005,
    target_grad_norm: float = 0.01,
) -> tuple:
    """Refine mesh with gradient normalization.

    Instead of raw gradients, normalize gradient to a fixed norm at each step.
    This ensures the update magnitude is the same regardless of reward scale.
    """
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    history = []
    for step in range(steps):
        optimizer.zero_grad()
        reward = reward_fn(v_opt, **reward_kwargs)
        loss = -reward
        loss.backward()

        # Normalize gradient to target norm
        grad_norm = v_opt.grad.norm().item()
        if grad_norm > 1e-10:
            v_opt.grad.data = v_opt.grad.data * (target_grad_norm / grad_norm)

        optimizer.step()

        # Evaluate ALL metrics
        with torch.no_grad():
            sym = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
            smooth = smoothness_reward(v_opt, faces).item()
            compact = compactness_reward(v_opt, faces).item()

        history.append({
            'step': step,
            'reward': reward.item(),
            'symmetry': sym,
            'smoothness': smooth,
            'compactness': compact,
            'grad_norm': grad_norm,
        })

    return v_opt.detach(), history


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "analysis_results", "scale_controlled_ablation")
    os.makedirs(output_dir, exist_ok=True)
    plane_store = PlaneStore.load_or_new(os.path.join(output_dir, "plane_cache.json"))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load Shap-E
    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=device)

    # Subset of prompts (30, 10 per category)
    from prompts_gpteval3d import ALL_PROMPTS
    subset_prompts = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        cat_prompts = [{'prompt': p[0], 'category': p[1]} for p in ALL_PROMPTS if p[1] == cat][:10]
        subset_prompts.extend(cat_prompts)

    seeds = [42, 123, 456]

    # Config names and normalization flags (reward_fn built per-mesh since
    # symmetry_reward_plane needs the estimated plane).
    config_specs = [
        ('sym_only_standard',    'sym',     False),
        ('sym_only_gradnorm',    'sym',     True),
        ('smooth_only_standard', 'smooth',  False),
        ('smooth_only_gradnorm', 'smooth',  True),
        ('compact_only_standard','compact', False),
        ('compact_only_gradnorm','compact', True),
    ]

    all_results = []
    total = len(subset_prompts) * len(seeds) * len(config_specs)
    count = 0

    for prompt_info in subset_prompts:
        prompt = prompt_info['prompt']
        category = prompt_info['category']

        for seed in seeds:
            # Generate base mesh
            torch.manual_seed(seed)
            np.random.seed(seed)
            results = generate_mesh(xm, model, diffusion, prompt, device=device)
            verts, faces, _ = results[0]
            if faces.shape[0] < 4:
                print(f"  SKIP {prompt}: {faces.shape[0]} faces")
                continue

            # Skip degenerate baseline (Shap-E sometimes emits point clouds with 0 faces).
            if faces.shape[0] == 0 or verts.shape[0] == 0:
                print(f"  SKIP degenerate baseline (point cloud)")
                continue

            # Estimate symmetry plane once on the baseline mesh; share across all variants
            # of this (prompt, seed) for paired protocol compliance.
            sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=verts)

            # Build configs per-mesh so reward_fn closures capture the correct sym_n, sym_d.
            configs = {
                'sym_only_standard': {
                    'reward_fn': lambda v, sn=sym_n, sd=sym_d, **kw: symmetry_reward_plane(v, sn, sd),
                    'normalized': False,
                },
                'sym_only_gradnorm': {
                    'reward_fn': lambda v, sn=sym_n, sd=sym_d, **kw: symmetry_reward_plane(v, sn, sd),
                    'normalized': True,
                },
                'smooth_only_standard': {
                    'reward_fn': lambda v, faces: smoothness_reward(v, faces),
                    'normalized': False,
                },
                'smooth_only_gradnorm': {
                    'reward_fn': lambda v, faces: smoothness_reward(v, faces),
                    'normalized': True,
                },
                'compact_only_standard': {
                    'reward_fn': lambda v, faces: compactness_reward(v, faces),
                    'normalized': False,
                },
                'compact_only_gradnorm': {
                    'reward_fn': lambda v, faces: compactness_reward(v, faces),
                    'normalized': True,
                },
            }

            # Evaluate baseline
            with torch.no_grad():
                base_sym = symmetry_reward_plane(verts, sym_n, sym_d).item()
                base_smooth = smoothness_reward(verts, faces).item()
                base_compact = compactness_reward(verts, faces).item()

            for config_name, config in configs.items():
                count += 1
                t0 = time.time()

                reward_fn = config['reward_fn']
                # Add faces to kwargs for smooth/compact rewards
                if 'smooth' in config_name or 'compact' in config_name:
                    reward_kwargs = {'faces': faces}
                else:
                    reward_kwargs = {}

                if config['normalized']:
                    refined, history = refine_gradient_normalized(
                        verts, faces, reward_fn, reward_kwargs,
                        sym_n, sym_d,
                        steps=50, lr=0.005, target_grad_norm=0.01,
                    )
                else:
                    # Standard: use refine_with_geo_reward with single weight
                    from shape_gen import refine_with_geo_reward
                    if 'sym' in config_name:
                        weights = torch.tensor([1.0, 0.0, 0.0])
                    elif 'smooth' in config_name:
                        weights = torch.tensor([0.0, 1.0, 0.0])
                    else:
                        weights = torch.tensor([0.0, 0.0, 1.0])
                    refined, history = refine_with_geo_reward(
                        verts, faces, weights, steps=50, lr=0.005,
                        sym_normal=sym_n, sym_offset=sym_d,
                    )

                elapsed = time.time() - t0

                # Final metrics
                final = history[-1] if history else {}
                result = {
                    'prompt': prompt,
                    'category': category,
                    'seed': seed,
                    'config': config_name,
                    'normalized': config['normalized'],
                    'symmetry': final.get('symmetry') or 0,
                    'smoothness': final.get('smoothness') or 0,
                    'compactness': final.get('compactness') or 0,
                    'base_symmetry': base_sym,
                    'base_smoothness': base_smooth,
                    'base_compactness': base_compact,
                    'sym_change_pct': ((final.get('symmetry') or 0) - base_sym) / max(abs(base_sym), 1e-8) * 100,
                    'smo_change_pct': ((final.get('smoothness') or 0) - base_smooth) / max(abs(base_smooth), 1e-8) * 100,
                    'com_change_pct': ((final.get('compactness') or 0) - base_compact) / max(abs(base_compact), 1e-8) * 100,
                    'time': elapsed,
                }
                all_results.append(result)

                print(f"[{count}/{total}] {config_name} | {prompt[:25]}... | "
                      f"sym={result['sym_change_pct']:+.1f}% smo={result['smo_change_pct']:+.1f}% "
                      f"com={result['com_change_pct']:+.1f}%")

    plane_store.save()

    # Save results
    with open(os.path.join(output_dir, "scale_controlled_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Filter out degenerate meshes (all metrics zero = Shap-E generation failure)
    def is_degenerate(r):
        return (abs(r['sym_change_pct']) < 0.1 and
                abs(r['smo_change_pct']) < 0.1 and
                abs(r['com_change_pct']) < 0.1)

    valid_results = [r for r in all_results if not is_degenerate(r)]
    n_degen = len(all_results) - len(valid_results)
    print(f"\nFiltered {n_degen}/{len(all_results)} degenerate runs "
          f"(Shap-E generation failures, all metrics ~0%)")
    print(f"Valid runs: {len(valid_results)}")

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS: Standard vs Gradient-Normalized (excluding degenerate)")
    print("=" * 70)

    for reward_type in ['sym_only', 'smooth_only', 'compact_only']:
        std_results = [r for r in valid_results if r['config'] == f'{reward_type}_standard']
        norm_results = [r for r in valid_results if r['config'] == f'{reward_type}_gradnorm']

        print(f"\n--- {reward_type.upper()} (n_std={len(std_results)}, n_norm={len(norm_results)}) ---")
        for metric in ['sym_change_pct', 'smo_change_pct', 'com_change_pct']:
            std_vals = [r[metric] for r in std_results]
            norm_vals = [r[metric] for r in norm_results]
            metric_name = metric.split('_')[0].upper()
            if std_vals and norm_vals:
                print(f"  {metric_name}: Standard={np.mean(std_vals):+.1f}% ± {np.std(std_vals):.1f}  |  "
                      f"GradNorm={np.mean(norm_vals):+.1f}% ± {np.std(norm_vals):.1f}")

    # Key question: does normalization prevent catastrophic degradation?
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does gradient normalization prevent catastrophic degradation?")
    print("=" * 70)

    for reward_type, target, nontarget in [
        ('sym_only', 'sym_change_pct', 'smo_change_pct'),
        ('smooth_only', 'smo_change_pct', 'com_change_pct'),
        ('compact_only', 'com_change_pct', 'smo_change_pct'),
    ]:
        norm_nontarget = [r[nontarget] for r in valid_results
                          if r['config'] == f'{reward_type}_gradnorm']
        if norm_nontarget:
            mean_deg = np.mean(norm_nontarget)
            print(f"  {reward_type} (grad-norm): non-target degradation = {mean_deg:+.1f}%")
            if mean_deg < -50:
                print(f"    -> STILL CATASTROPHIC even with normalization")
            elif mean_deg < -10:
                print(f"    -> Moderate degradation persists")
            else:
                print(f"    -> Normalization mitigates degradation")

    # Generate visualization (also filter degenerate)
    _plot_comparison(valid_results, output_dir)
    print(f"\nResults saved to: {output_dir}")


def _plot_comparison(results, output_dir):
    """Plot standard vs grad-normalized comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax_idx, (reward_type, title) in enumerate([
        ('sym_only', 'Sym-Only'), ('smooth_only', 'Smooth-Only'), ('compact_only', 'Compact-Only')
    ]):
        ax = axes[ax_idx]
        metrics = ['sym_change_pct', 'smo_change_pct', 'com_change_pct']
        labels = ['Symmetry', 'Smoothness', 'Compactness']

        std_means = []
        norm_means = []
        std_stds = []
        norm_stds = []

        for m in metrics:
            std_vals = [r[m] for r in results if r['config'] == f'{reward_type}_standard']
            norm_vals = [r[m] for r in results if r['config'] == f'{reward_type}_gradnorm']
            std_means.append(np.mean(std_vals))
            norm_means.append(np.mean(norm_vals))
            std_stds.append(np.std(std_vals) / np.sqrt(max(len(std_vals), 1)))
            norm_stds.append(np.std(norm_vals) / np.sqrt(max(len(norm_vals), 1)))

        x = np.arange(3)
        width = 0.35

        ax.bar(x - width/2, std_means, width, yerr=std_stds, label='Standard',
               color='#ef5350', alpha=0.8, capsize=3)
        ax.bar(x + width/2, norm_means, width, yerr=norm_stds, label='Grad-Normalized',
               color='#42a5f5', alpha=0.8, capsize=3)

        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Change vs Baseline (%)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Scale-Controlled Ablation: Standard vs Gradient-Normalized',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_controlled_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'scale_controlled_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
