"""Adversarial asymmetric prompts: does DGR break intentional asymmetry?

Response to external reviewer (GPT-5.4 xhigh): "What fraction of prompts
actually contain objects for which bilateral symmetry is semantically
appropriate, and how does DGR behave on intentionally asymmetric prompts?"

We run Shap-E + DGR (equal weights) on 20 adversarial prompts where
symmetry/compactness/smoothness is semantically inappropriate, and measure:
    - reward improvement (R_sym, R_HNC, R_compact): DGR should still
      optimize these even when it shouldn't
    - semantic preservation (CLIP against the ADVERSARIAL prompt): if DGR
      symmetrizes a "one-winged bird" into a two-winged one, CLIP vs "one
      winged" should drop
    - shape-preservation: Chamfer-L1 between refined and baseline meshes

Run on GPU (V100). ~30 min total for 20 prompts x 3 seeds.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import (
    symmetry_reward_plane, smoothness_reward, compactness_reward,
    estimate_symmetry_plane,
)
from compute_baseline_clip import compute_clip_score

# ============================================================
# Adversarial prompt set
# ============================================================
ADVERSARIAL_PROMPTS = {
    # Asymmetric structural (10) — symmetry is semantically wrong
    'asymmetric_structural': [
        "a broken chair",
        "a leaning tower",
        "a bent tree",
        "a half-eaten apple",
        "a crooked house",
        "a cracked egg",
        "a tilted vase",
        "a crashed car",
        "a slanted mushroom",
        "a melting candle",
    ],
    # Pose-asymmetric (5) — enforcing symmetry would undo the pose
    'pose_asymmetric': [
        "a person raising one arm",
        "a running figure",
        "a sitting dog with one paw up",
        "a waving person",
        "a kneeling statue",
    ],
    # Thin structures (5) — compactness would blob them
    'thin_structure': [
        "a sword",
        "a fishing rod",
        "a spear",
        "a long staff",
        "an axe with a long handle",
    ],
}

SEEDS = [42, 123, 456]
WEIGHTS = [1.0/3, 1.0/3, 1.0/3]  # equal weights

OUT_DIR = os.path.join(ROOT, 'analysis_results/adversarial_asymmetric')
MESH_DIR = os.path.join(OUT_DIR, 'meshes')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MESH_DIR, exist_ok=True)


def score_mesh(V, F, plane_normal, plane_offset, device, dtype):
    """Return (sym, hnc, com) rewards for a torch mesh."""
    pn = torch.as_tensor(plane_normal, dtype=dtype, device=device)
    po = torch.tensor(float(plane_offset), dtype=dtype, device=device)
    with torch.no_grad():
        sym = symmetry_reward_plane(V, pn, po).item()
        hnc = smoothness_reward(V, F).item()
        com = compactness_reward(V, F).item()
    return sym, hnc, com


def chamfer_l1(A, B, n=4096):
    """A, B: numpy (V,3). Returns symmetric Chamfer-L1."""
    from scipy.spatial import cKDTree
    if len(A) > n:
        idx = np.random.default_rng(0).choice(len(A), n, replace=False)
        A = A[idx]
    if len(B) > n:
        idx = np.random.default_rng(0).choice(len(B), n, replace=False)
        B = B[idx]
    ta, tb = cKDTree(A), cKDTree(B)
    d1, _ = tb.query(A); d2, _ = ta.query(B)
    return float(0.5 * (d1.mean() + d2.mean()))


def save_obj(verts, faces, path):
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def to_flat_prompt_tag(prompt):
    return prompt.replace(' ', '_').replace('/', '_')


def run_one(prompt, seed, xm, model, diffusion, device, dtype, clip_model, clip_preprocess):
    """Generate + refine + score for one (prompt, seed)."""
    torch.manual_seed(seed); np.random.seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 1. Generate baseline mesh
    t0 = time.time()
    try:
        gens = generate_mesh(xm, model, diffusion, prompt, device=device)
    except Exception as e:
        print(f'    GEN FAIL ({e})')
        return None
    V0, F0, _ = gens[0]
    gen_s = time.time() - t0

    if len(V0) < 50 or len(F0) < 100:
        print('    skip (degenerate baseline)')
        return None

    # 2. Estimate symmetry plane on baseline (signature: vertices only)
    try:
        pn, po = estimate_symmetry_plane(V0)
    except Exception as e:
        print(f'    plane-est fail: {e}')
        return None
    plane_n = pn.detach().cpu().numpy().tolist()
    plane_d = float(po.detach().cpu().item())

    # 3. Score baseline
    bl_sym, bl_hnc, bl_com = score_mesh(V0, F0, plane_n, plane_d, device, dtype)

    # 4. Refine with DGR (equal weights)
    weights = torch.tensor(WEIGHTS, dtype=dtype, device=device)
    t0 = time.time()
    V1, history = refine_with_geo_reward(V0, F0, weights, steps=50, lr=0.005,
                                          sym_normal=pn, sym_offset=po)
    ref_s = time.time() - t0

    # 5. Score refined
    rf_sym, rf_hnc, rf_com = score_mesh(V1, F0, plane_n, plane_d, device, dtype)

    # 6. Shape drift
    V0_np = V0.detach().cpu().numpy()
    V1_np = V1.detach().cpu().numpy()
    shape_cd = chamfer_l1(V0_np, V1_np)

    # 7. Save meshes to disk
    tag = f'{to_flat_prompt_tag(prompt)}_seed{seed}'
    bl_path = os.path.join(MESH_DIR, f'baseline_{tag}.obj')
    rf_path = os.path.join(MESH_DIR, f'refined_{tag}.obj')
    save_obj(V0_np, F0.detach().cpu().numpy(), bl_path)
    save_obj(V1_np, F0.detach().cpu().numpy(), rf_path)

    # 8. CLIP scores against the ADVERSARIAL prompt
    bl_clip = compute_clip_score(bl_path, prompt, clip_model, clip_preprocess, device)
    rf_clip = compute_clip_score(rf_path, prompt, clip_model, clip_preprocess, device)

    return {
        'prompt': prompt, 'seed': seed,
        'n_verts': int(V0.shape[0]),
        'plane_n': plane_n, 'plane_d': plane_d,
        'bl_sym': bl_sym, 'bl_hnc': bl_hnc, 'bl_com': bl_com,
        'rf_sym': rf_sym, 'rf_hnc': rf_hnc, 'rf_com': rf_com,
        'bl_clip': bl_clip, 'rf_clip': rf_clip,
        'shape_cd': shape_cd,
        'gen_s': gen_s, 'ref_s': ref_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--subset', default=None,
                        help='Comma list of category keys to run (default: all)')
    args = parser.parse_args()

    device = args.device
    dtype = torch.float32
    print('=== Adversarial Asymmetric Prompts ===')
    print(f'Device: {device}')

    # Pick subset
    if args.subset:
        keys = [k.strip() for k in args.subset.split(',')]
    else:
        keys = list(ADVERSARIAL_PROMPTS.keys())
    all_prompts = []
    for k in keys:
        for p in ADVERSARIAL_PROMPTS[k]:
            all_prompts.append((k, p))
    print(f'Prompts: {len(all_prompts)}, seeds: {len(SEEDS)}, '
          f'total runs: {len(all_prompts) * len(SEEDS)}\n')

    # Load Shap-E + CLIP
    print('Loading Shap-E...')
    xm, model, diffusion = load_shap_e(device=device)
    print('Loading CLIP...')
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    results = []
    t0 = time.time()
    for i, (cat, prompt) in enumerate(all_prompts):
        for seed in SEEDS:
            print(f'[{i+1}/{len(all_prompts)}] {prompt} seed={seed}', flush=True)
            r = run_one(prompt, seed, xm, model, diffusion, device, dtype,
                         clip_model, clip_preprocess)
            if r is None:
                continue
            r['category'] = cat
            results.append(r)
            print(f'    sym {r["bl_sym"]:+.4f} -> {r["rf_sym"]:+.4f}  '
                  f'CLIP {r["bl_clip"]:.3f} -> {r["rf_clip"]:.3f}  '
                  f'shape-CD {r["shape_cd"]:.4f}', flush=True)
        # Save checkpoint after each prompt
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # ============================================================
    # Aggregate
    # ============================================================
    el = (time.time() - t0) / 60
    print(f'\nDone in {el:.1f} min. n_valid = {len(results)}')

    if not results:
        return

    def agg(values):
        return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)

    summary = {'by_category': {}, 'overall': {}}
    for cat in list(ADVERSARIAL_PROMPTS.keys()) + ['OVERALL']:
        if cat == 'OVERALL':
            rows = results
        else:
            rows = [r for r in results if r['category'] == cat]
        if not rows:
            continue
        d = {'n': len(rows)}
        for metric in ['sym', 'hnc', 'com', 'clip']:
            bl = [r[f'bl_{metric}'] for r in rows]
            rf = [r[f'rf_{metric}'] for r in rows]
            d[f'bl_{metric}_mean'], _ = agg(bl)
            d[f'rf_{metric}_mean'], _ = agg(rf)
            # Delta (rf - bl)
            delta = np.array(rf) - np.array(bl)
            d[f'delta_{metric}_mean'], _ = agg(list(delta))
            # Win rate: for sym/hnc/com higher is better (less negative);
            # for clip higher is better.
            d[f'{metric}_improved_pct'] = float((delta > 0).mean() * 100)
        sd = [r['shape_cd'] for r in rows]
        d['shape_cd_mean'], _ = agg(sd)
        if cat == 'OVERALL':
            summary['overall'] = d
        else:
            summary['by_category'][cat] = d

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print('\n=== Summary ===')
    print(f"{'Category':28s} | {'n':>3s} | {'ΔR_sym':>10s} | {'ΔR_hnc':>10s} | {'ΔR_com':>10s} | {'ΔCLIP':>8s} | {'shape-CD':>8s}")
    for cat, d in summary['by_category'].items():
        print(f"{cat:28s} | {d['n']:>3d} | "
              f"{d['delta_sym_mean']:+10.4f} | {d['delta_hnc_mean']:+10.4f} | "
              f"{d['delta_com_mean']:+10.4f} | {d['delta_clip_mean']:+8.4f} | "
              f"{d['shape_cd_mean']:8.4f}")
    d = summary['overall']
    print(f"{'OVERALL':28s} | {d['n']:>3d} | "
          f"{d['delta_sym_mean']:+10.4f} | {d['delta_hnc_mean']:+10.4f} | "
          f"{d['delta_com_mean']:+10.4f} | {d['delta_clip_mean']:+8.4f} | "
          f"{d['shape_cd_mean']:8.4f}")

    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
