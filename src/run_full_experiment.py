"""
Full-scale DiffGeoReward experiment runner.

Methods:
  - baseline: vanilla Shap-E (no refinement)
  - diffgeoreward: full system (Lang2Comp + combined reward)
  - handcrafted: equal weights (0.33, 0.33, 0.34), no Lang2Comp
  - sym_only: symmetry reward only (weights=[1, 0, 0])
  - smooth_only: smoothness reward only (weights=[0, 1, 0])
  - compact_only: compactness reward only (weights=[0, 0, 1])

Usage:
    # Run specific method
    CUDA_VISIBLE_DEVICES=1 python src/run_full_experiment.py --method baseline --seeds 42 123 456

    # Run all methods sequentially
    CUDA_VISIBLE_DEVICES=1 python src/run_full_experiment.py --method all --seeds 42 123 456

    # Run a quick sanity check (6 prompts)
    CUDA_VISIBLE_DEVICES=1 python src/run_full_experiment.py --method baseline --sanity
"""

import argparse
import json
import os
import sys
import time
import traceback

# Force offline mode for HuggingFace (use cached models)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from shape_gen import load_shap_e, run_single_experiment
from lang2comp import Lang2Comp
from prompts_gpteval3d import ALL_PROMPTS

# Optional: CLIP scoring
CLIP_AVAILABLE = False
try:
    import clip
    from PIL import Image
    import trimesh
    CLIP_AVAILABLE = True
except ImportError:
    pass


METHODS_ALL = ['baseline', 'diffgeoreward', 'handcrafted', 'sym_only', 'smooth_only', 'compact_only']

FIXED_WEIGHTS = {
    'handcrafted': torch.tensor([0.33, 0.33, 0.34]),
    'sym_only':    torch.tensor([1.0, 0.0, 0.0]),
    'smooth_only': torch.tensor([0.0, 1.0, 0.0]),
    'compact_only':torch.tensor([0.0, 0.0, 1.0]),
}


def render_mesh_matplotlib(mesh_path, angle_deg=0, resolution=224):
    """Render mesh to image using matplotlib (headless-compatible)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import io

    mesh = trimesh.load(mesh_path)
    verts = np.array(mesh.vertices)
    faces_arr = np.array(mesh.faces)

    # Center and normalize
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale

    # Subsample faces for rendering speed
    max_faces = 5000
    if len(faces_arr) > max_faces:
        idx = np.random.choice(len(faces_arr), max_faces, replace=False)
        faces_arr = faces_arr[idx]

    fig = plt.figure(figsize=(resolution/72, resolution/72), dpi=72)
    ax = fig.add_subplot(111, projection='3d')

    # Create polygon collection
    polys = verts[faces_arr]
    pc = Poly3DCollection(polys, alpha=0.8, facecolor='lightblue', edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(pc)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=angle_deg)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close(fig)
    buf.seek(0)
    return buf


def compute_clip_score(mesh_path, prompt, clip_model, clip_preprocess, device):
    """Compute CLIP similarity between rendered mesh views and text prompt.

    Uses matplotlib for headless rendering (no display required).
    """
    if not CLIP_AVAILABLE or clip_model is None:
        return 0.0

    try:
        scores = []
        for angle in [0, 90, 180, 270]:
            try:
                buf = render_mesh_matplotlib(mesh_path, angle_deg=angle)
                image = Image.open(buf).convert('RGB')
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                text_input = clip.tokenize([prompt]).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features = clip_model.encode_text(text_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).item()
                    scores.append(similarity)
            except Exception:
                continue

        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def get_weights(prompt, method, lang2comp=None):
    """Get reward weights based on method."""
    if method in FIXED_WEIGHTS:
        return FIXED_WEIGHTS[method]
    elif method == 'diffgeoreward' and lang2comp is not None:
        result = lang2comp.predict(prompt)
        w = result['weights']
        return torch.tensor([w['symmetry'], w['smoothness'], w['compactness']])
    elif method == 'baseline':
        return torch.tensor([0.33, 0.33, 0.34])
    else:
        return torch.tensor([0.33, 0.33, 0.34])


def determine_run_method(method):
    """Map our method names to shape_gen's expected method names."""
    if method == 'baseline':
        return 'baseline'
    else:
        return 'diffgeoreward'


def main(args):
    device = 'cuda:0'

    print("=" * 70)
    print("DiffGeoReward Full Experiment")
    print("Method: {} | Seeds: {} | Device: {}".format(args.method, args.seeds, device))
    print("=" * 70)
    print()

    # Load Shap-E
    print("Loading Shap-E model...")
    t0 = time.time()
    xm, model, diffusion = load_shap_e(device=device)
    print("Shap-E loaded in {:.1f}s\n".format(time.time() - t0))

    # Load Lang2Comp
    lang2comp = None
    ckpt_path = 'checkpoints/lang2comp_best.pt'
    if os.path.exists(ckpt_path):
        print("Loading Lang2Comp model...")
        lang2comp = Lang2Comp()
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        lang2comp.load_state_dict(state['model_state_dict'])
        lang2comp.eval()
        print("Lang2Comp loaded (acc={:.2%})\n".format(state['dom_acc']))

    # Load CLIP (optional — skip if --no-clip or if GPU memory is tight)
    clip_model, clip_preprocess = None, None
    if CLIP_AVAILABLE and not args.no_clip:
        try:
            print("Loading CLIP model...")
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            print("CLIP loaded.\n")
        except Exception as e:
            print("CLIP load failed: {}, skipping CLIP scores\n".format(e))

    # Select prompts
    if args.sanity:
        prompts = [p for p in ALL_PROMPTS if p[1] == 'symmetry'][:2] + \
                   [p for p in ALL_PROMPTS if p[1] == 'smoothness'][:2] + \
                   [p for p in ALL_PROMPTS if p[1] == 'compactness'][:2]
        print("SANITY MODE: {} prompts\n".format(len(prompts)))
    else:
        prompts = ALL_PROMPTS
        print("Full run: {} prompts\n".format(len(prompts)))

    # Select methods
    if args.method == 'all':
        methods = METHODS_ALL
    elif args.method == 'ablations':
        methods = ['handcrafted', 'sym_only', 'smooth_only', 'compact_only']
    else:
        methods = [args.method]

    seeds = args.seeds
    total_runs = len(methods) * len(prompts) * len(seeds)
    print("Total runs: {} methods x {} prompts x {} seeds = {}".format(
        len(methods), len(prompts), len(seeds), total_runs))
    print("Estimated time: ~{:.0f} min\n".format(total_runs * 11 / 60))

    os.makedirs('results/full', exist_ok=True)

    # Check for existing results to enable resume
    results_file = 'results/full/{}_all_metrics.json'.format(args.method)
    existing_metrics = []
    completed_keys = set()
    if os.path.exists(results_file) and args.resume:
        with open(results_file) as f:
            existing_metrics = json.load(f)
        for m in existing_metrics:
            key = "{}_{}_{}" .format(m['method'], m['prompt'], m['seed'])
            completed_keys.add(key)
        print("Resuming: {} runs already completed\n".format(len(existing_metrics)))

    all_metrics = list(existing_metrics)
    run_idx = len(existing_metrics)
    errors = []
    start_time = time.time()

    for method in methods:
        print("\n" + "=" * 50)
        print("Method: {}".format(method))
        print("=" * 50)

        method_metrics = []

        for prompt_text, category in prompts:
            weights = get_weights(prompt_text, method, lang2comp)
            gen_method = determine_run_method(method)

            for seed in seeds:
                run_key = "{}_{}_{}" .format(method, prompt_text, seed)
                if run_key in completed_keys:
                    continue

                run_idx += 1
                output_dir = "results/full/{}/{}".format(method, category)

                elapsed = time.time() - start_time
                runs_done = run_idx - len(existing_metrics)
                if runs_done > 0:
                    rate = elapsed / runs_done
                    remaining = (total_runs - run_idx) * rate
                    eta = "ETA: {:.0f}min".format(remaining / 60)
                else:
                    eta = ""

                short_prompt = prompt_text[:40]
                print("\n[{}/{}] {} | '{}' | seed={} | {}".format(
                    run_idx, total_runs, method, short_prompt, seed, eta))
                print("  weights: [{:.2f}, {:.2f}, {:.2f}]".format(
                    weights[0], weights[1], weights[2]))

                try:
                    metrics = run_single_experiment(
                        prompt=prompt_text,
                        method=gen_method,
                        seed=seed,
                        weights=weights,
                        xm=xm, model=model, diffusion=diffusion,
                        output_dir=output_dir,
                        device=device,
                    )

                    metrics['category'] = category
                    metrics['method'] = method
                    metrics['weights_used'] = weights.tolist()

                    # CLIP score
                    if clip_model is not None and 'mesh_path' in metrics:
                        clip_score = compute_clip_score(
                            metrics['mesh_path'], prompt_text,
                            clip_model, clip_preprocess, device
                        )
                        metrics['clip_score'] = clip_score

                    all_metrics.append(metrics)
                    method_metrics.append(metrics)

                    sym_val = metrics['symmetry']
                    smooth_val = metrics['smoothness']
                    compact_val = metrics['compactness']
                    time_val = metrics['total_time']
                    print("  sym={:.6f} smooth={:.6f} compact={:.6f} time={:.1f}s".format(
                        sym_val, smooth_val, compact_val, time_val))

                    if 'reward_improvement' in metrics:
                        print("  reward: {:.4f} -> {:.4f} (+{:.4f})".format(
                            metrics['initial_reward'],
                            metrics['final_reward'],
                            metrics['reward_improvement']))

                except Exception as e:
                    print("  ERROR: {}".format(e))
                    traceback.print_exc()
                    errors.append({
                        'method': method, 'prompt': prompt_text,
                        'seed': seed, 'error': str(e)
                    })

                # Save incrementally every 10 runs
                if run_idx % 10 == 0:
                    with open(results_file, 'w') as f:
                        json.dump(all_metrics, f, indent=2, default=str)

        # Print method summary
        if method_metrics:
            print("\n--- {} Summary ---".format(method))
            print("  Runs: {}".format(len(method_metrics)))
            print("  Avg Symmetry:    {:.6f}".format(
                np.mean([m['symmetry'] for m in method_metrics])))
            print("  Avg Smoothness:  {:.6f}".format(
                np.mean([m['smoothness'] for m in method_metrics])))
            print("  Avg Compactness: {:.6f}".format(
                np.mean([m['compactness'] for m in method_metrics])))
            if any('clip_score' in m for m in method_metrics):
                clips = [m['clip_score'] for m in method_metrics if 'clip_score' in m]
                print("  Avg CLIP:        {:.4f}".format(np.mean(clips)))

    # Final save
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("\nAll metrics saved to {}".format(results_file))

    if errors:
        err_file = 'results/full/{}_errors.json'.format(args.method)
        with open(err_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print("Errors saved: {} failures".format(len(errors)))

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("Total time: {:.1f} min".format(total_time / 60))
    print("Total runs: {} successful, {} errors".format(len(all_metrics), len(errors)))
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffGeoReward Full Experiment')
    parser.add_argument('--method', choices=METHODS_ALL + ['all', 'ablations'],
                        default='all', help='Method to run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--sanity', action='store_true',
                        help='Quick sanity check with 6 prompts')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results file')
    parser.add_argument('--no-clip', action='store_true',
                        help='Skip CLIP scoring to save GPU memory')
    args = parser.parse_args()
    main(args)
