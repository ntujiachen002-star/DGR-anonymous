"""Anti-Collapse Regularizer Experiment.

Adds a max vertex displacement penalty to DiffGeoReward to reduce failure rate.
Tests whether a simple regularizer can mitigate compactness degradation and
gradient instability.

Regularizer:
    L_reg = lambda * mean(max(||delta_v|| / d_bbox - tau, 0))

where delta_v = v_refined - v_init, d_bbox = diagonal of initial bounding box,
tau = maximum allowed fractional displacement (default 0.05 = 5% of bbox).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/exp_anticollapse.py
"""

import os
import sys
import json
import time
import numpy as np
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shape_gen import load_shap_e, generate_mesh, save_mesh
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        estimate_symmetry_plane, smoothness_reward,
                        compactness_reward)
from _plane_protocol import PlaneStore, make_key


def refine_with_anticollapse(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weights: torch.Tensor,
    sym_n: torch.Tensor,
    sym_d: torch.Tensor,
    steps: int = 50,
    lr: float = 0.005,
    reg_lambda: float = 10.0,
    max_displacement_frac: float = 0.05,
) -> tuple:
    """Refine mesh vertices with anti-collapse displacement regularizer.

    Args:
        vertices: (V, 3) initial vertices
        faces: (F, 3) face indices
        weights: (3,) reward weights
        sym_n: (3,) symmetry plane normal
        sym_d: () symmetry plane offset
        reg_lambda: regularization strength
        max_displacement_frac: max allowed displacement as fraction of bbox diagonal
    """
    v_init = vertices.detach().clone()
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    # Compute bounding box diagonal for normalization
    bbox_min = v_init.min(dim=0).values
    bbox_max = v_init.max(dim=0).values
    d_bbox = (bbox_max - bbox_min).norm().item()
    tau = max_displacement_frac * d_bbox

    # Initial reward values for normalization
    with torch.no_grad():
        sym_init = abs(symmetry_reward_plane(v_opt, sym_n, sym_d).item())
        smooth_init = abs(smoothness_reward(v_opt, faces).item())
        compact_init = abs(compactness_reward(v_opt, faces).item())

    sym_scale = max(sym_init, 1e-6)
    smooth_scale = max(smooth_init, 1e-6)
    compact_scale = max(compact_init, 1e-6)

    history = []
    for step in range(steps):
        optimizer.zero_grad()

        sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
        smooth = smoothness_reward(v_opt, faces)
        compact = compactness_reward(v_opt, faces)

        reward = (weights[0] * sym / sym_scale
                  + weights[1] * smooth / smooth_scale
                  + weights[2] * compact / compact_scale)

        # Anti-collapse regularizer: penalize large vertex displacements
        displacement = (v_opt - v_init).norm(dim=1)  # (V,)
        violation = torch.relu(displacement - tau)  # only penalize beyond threshold
        reg_loss = reg_lambda * violation.mean()

        loss = -reward + reg_loss

        loss.backward()
        grad_norm = v_opt.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()

        history.append({
            'step': step,
            'reward': reward.item(),
            'symmetry': sym.item(),
            'smoothness': smooth.item(),
            'compactness': compact.item(),
            'grad_norm': grad_norm,
            'reg_loss': reg_loss.item(),
            'max_displacement': displacement.max().item(),
            'mean_displacement': displacement.mean().item(),
        })

    return v_opt.detach(), history


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "analysis_results", "anticollapse")
    os.makedirs(output_dir, exist_ok=True)
    plane_store = PlaneStore.load_or_new(os.path.join(output_dir, "plane_cache.json"))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=device)

    from prompts_gpteval3d import ALL_PROMPTS
    seeds = [42, 123, 456]

    # Configs: standard DiffGeoReward vs with anti-collapse
    configs = {
        'dgr_standard': {'reg_lambda': 0.0, 'max_disp': 1.0},  # no regularization
        'dgr_anticollapse_soft': {'reg_lambda': 5.0, 'max_disp': 0.05},
        'dgr_anticollapse_medium': {'reg_lambda': 10.0, 'max_disp': 0.03},
        'dgr_anticollapse_strong': {'reg_lambda': 20.0, 'max_disp': 0.02},
    }

    weights = torch.tensor([0.33, 0.33, 0.34])

    all_results = []
    total = len(ALL_PROMPTS) * len(seeds) * len(configs)
    count = 0

    for prompt_tuple in ALL_PROMPTS:
        prompt = prompt_tuple[0]
        category = prompt_tuple[1]

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                results = generate_mesh(xm, model, diffusion, prompt, device=device)
                verts, faces, _ = results[0]
            except Exception as e:
                print(f"  SKIP {prompt}: generation failed ({e})")
                continue

            # Skip degenerate baseline (Shap-E sometimes emits point clouds with 0 faces).
            if faces.shape[0] == 0 or verts.shape[0] == 0:
                print(f"  SKIP degenerate baseline (point cloud)")
                continue

            # Estimate symmetry plane once on the baseline mesh; share across all variants
            # of this (prompt, seed) for paired protocol compliance.
            sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=verts)

            # Baseline metrics
            with torch.no_grad():
                base_sym = symmetry_reward_plane(verts, sym_n, sym_d).item()
                base_smooth = smoothness_reward(verts, faces).item()
                base_compact = compactness_reward(verts, faces).item()

            for config_name, config in configs.items():
                count += 1
                t0 = time.time()

                try:
                    if config['reg_lambda'] == 0:
                        from shape_gen import refine_with_geo_reward
                        refined, history = refine_with_geo_reward(
                            verts, faces, weights, steps=50, lr=0.005,
                            sym_normal=sym_n, sym_offset=sym_d,
                        )
                    else:
                        refined, history = refine_with_anticollapse(
                            verts, faces, weights, sym_n, sym_d,
                            steps=50, lr=0.005,
                            reg_lambda=config['reg_lambda'],
                            max_displacement_frac=config['max_disp'],
                        )
                except Exception as e:
                    print(f"  ERROR {config_name} | {prompt[:25]}...: {e}")
                    continue

                elapsed = time.time() - t0
                final = history[-1] if history else {}

                result = {
                    'prompt': prompt,
                    'category': category,
                    'seed': seed,
                    'config': config_name,
                    'reg_lambda': config['reg_lambda'],
                    'max_disp_frac': config['max_disp'],
                    'symmetry': final.get('symmetry', 0),
                    'smoothness': final.get('smoothness', 0),
                    'compactness': final.get('compactness', 0),
                    'base_symmetry': base_sym,
                    'base_smoothness': base_smooth,
                    'base_compactness': base_compact,
                    'sym_change_pct': (final.get('symmetry', 0) - base_sym) / max(abs(base_sym), 1e-8) * 100,
                    'smo_change_pct': (final.get('smoothness', 0) - base_smooth) / max(abs(base_smooth), 1e-8) * 100,
                    'com_change_pct': (final.get('compactness', 0) - base_compact) / max(abs(base_compact), 1e-8) * 100,
                    'max_displacement': final.get('max_displacement', 0),
                    'mean_displacement': final.get('mean_displacement', 0),
                    'reg_loss': final.get('reg_loss', 0),
                    'time': elapsed,
                }
                all_results.append(result)

                if count % 20 == 0:
                    print(f"[{count}/{total}] {config_name} | {prompt[:25]}... | "
                          f"sym={result['sym_change_pct']:+.1f}% smo={result['smo_change_pct']:+.1f}% "
                          f"com={result['com_change_pct']:+.1f}%")

    plane_store.save()

    # Save results
    with open(os.path.join(output_dir, "anticollapse_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Filter out degenerate meshes (Shap-E generation failures)
    def is_degenerate(r):
        return (abs(r['sym_change_pct']) < 0.1 and
                abs(r['smo_change_pct']) < 0.1 and
                abs(r['com_change_pct']) < 0.1)

    valid_results = [r for r in all_results if not is_degenerate(r)]
    n_degen = len(all_results) - len(valid_results)
    print(f"\nFiltered {n_degen}/{len(all_results)} degenerate runs "
          f"(Shap-E generation failures)")
    print(f"Valid runs: {len(valid_results)}")

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS: Anti-Collapse Regularizer (excluding degenerate)")
    print("=" * 70)

    for config_name in configs:
        cr = [r for r in valid_results if r['config'] == config_name]
        if not cr:
            continue

        sym_imps = [r['sym_change_pct'] for r in cr]
        smo_imps = [r['smo_change_pct'] for r in cr]
        com_imps = [r['com_change_pct'] for r in cr]

        # Failure rate: any metric worsens by >5%
        n_fail_strict = sum(1 for r in cr if r['sym_change_pct'] < -5 or r['smo_change_pct'] < -5)
        n_fail_broad = sum(1 for r in cr if r['com_change_pct'] < -5)

        print(f"\n--- {config_name} (n={len(cr)}) ---")
        print(f"  Symmetry:    {np.mean(sym_imps):+.1f}% ± {np.std(sym_imps):.1f}%")
        print(f"  Smoothness:  {np.mean(smo_imps):+.1f}% ± {np.std(smo_imps):.1f}%")
        print(f"  Compactness: {np.mean(com_imps):+.1f}% ± {np.std(com_imps):.1f}%")
        print(f"  Failure rate (sym/smo worsens): {n_fail_strict}/{len(cr)} = {n_fail_strict/len(cr)*100:.1f}%")
        print(f"  Compactness worsens:            {n_fail_broad}/{len(cr)} = {n_fail_broad/len(cr)*100:.1f}%")

    # Generate comparison plot (using valid results only)
    _plot_results(valid_results, configs, output_dir)
    print(f"\nResults saved to: {output_dir}")


def _plot_results(results, configs, output_dir):
    """Plot anti-collapse comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_names = ['sym_change_pct', 'smo_change_pct', 'com_change_pct']
    metric_labels = ['Symmetry Δ%', 'Smoothness Δ%', 'Compactness Δ%']
    colors = ['#66bb6a', '#42a5f5', '#ffa726', '#ef5350']

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        means = []
        stds = []
        names = []
        for i, config_name in enumerate(configs):
            cr = [r[metric] for r in results if r['config'] == config_name]
            if cr:
                means.append(np.mean(cr))
                stds.append(np.std(cr) / np.sqrt(len(cr)))
                names.append(config_name.replace('dgr_', '').replace('_', '\n'))

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, color=colors[:len(means)], alpha=0.8, capsize=4)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Anti-Collapse Regularizer: Effect on Geometric Metrics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anticollapse_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'anticollapse_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
