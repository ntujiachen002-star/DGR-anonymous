"""Recompute exp_c Laplacian baseline metrics with volume_constraint=False.

Reads the 20 DreamCS comparison prompts from exp_c's existing output to get
the exact same prompt set and SEED, then:
  1. Loads each baseline mesh from results/mesh_validity_objs/baseline/
  2. Estimates the plane once on the baseline (shared paired protocol)
  3. Applies trimesh.smoothing.filter_laplacian with volume_constraint=False
  4. Computes sym (new plane) + HNC + com
  5. Merges with existing DGR records

Output: analysis_results/dreamcs_comparison/laplacian_baseline_fixed.json
"""
import sys
import os
import re
import json
import glob
import numpy as np
import torch
import trimesh
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from geo_reward import symmetry_reward_plane, smoothness_reward, compactness_reward, estimate_symmetry_plane

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXISTING_JSON = f'{ROOT}/analysis_results/dreamcs_comparison/laplacian_baseline.json'
OUT_JSON = f'{ROOT}/analysis_results/dreamcs_comparison/laplacian_baseline_fixed.json'
MESH_ROOT = f'{ROOT}/results/mesh_validity_objs/baseline'
LAPLACIAN_STEPS = [10, 20, 50]
SEED = 42


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def find_mesh(prompt, seed=SEED):
    name = f'{slug(prompt)}_seed{seed}.obj'
    for cat in ['symmetry', 'smoothness', 'compactness']:
        p = f'{MESH_ROOT}/{cat}/{name}'
        if os.path.exists(p):
            return p
    return None


def compute_metrics(v_np, f_np, sym_n, sym_d):
    v = torch.tensor(np.array(v_np), dtype=torch.float32)
    f = torch.tensor(np.array(f_np), dtype=torch.long)
    if v.numel() == 0 or f.shape[0] == 0:
        return None
    try:
        with torch.no_grad():
            sym = symmetry_reward_plane(v, sym_n, sym_d).item()
            smo = smoothness_reward(v, f).item()
            com = compactness_reward(v, f).item()
        return {'symmetry': sym, 'smoothness': smo, 'compactness': com}
    except Exception as e:
        return None


def main():
    # Load existing exp_c JSON to get the prompt list and the DGR records
    existing = json.load(open(EXISTING_JSON))
    prompts = sorted({r['prompt'] for r in existing if r.get('method') == 'baseline'})
    print(f'prompts in exp_c: {len(prompts)}')

    # Build the new records list
    new_records = []
    n_skip = 0
    n_nan = 0

    for prompt in prompts:
        mesh_path = find_mesh(prompt, SEED)
        if mesh_path is None:
            print(f'  SKIP (no mesh): {prompt}')
            n_skip += 1
            new_records.append({'prompt': prompt, 'method': 'baseline',
                                'error': 'no_local_baseline'})
            continue

        mesh = trimesh.load(mesh_path, process=False)
        v_np = np.array(mesh.vertices)
        f_np = np.array(getattr(mesh, 'faces', np.zeros((0, 3), dtype=np.int64)))

        if f_np.ndim != 2 or f_np.shape[0] == 0 or len(v_np) == 0:
            print(f'  SKIP (degenerate / point cloud): {prompt}')
            n_skip += 1
            new_records.append({'prompt': prompt, 'method': 'baseline',
                                'error': 'degenerate_baseline'})
            continue

        # Estimate plane ONCE (paired protocol)
        v_t = torch.tensor(v_np, dtype=torch.float32)
        sym_n, sym_d = estimate_symmetry_plane(v_t.detach())

        # Baseline metrics
        base_m = compute_metrics(v_np, f_np, sym_n, sym_d)
        new_records.append({'prompt': prompt, 'method': 'baseline', **base_m})
        print(f'  [{prompt[:40]:<40}] base sym={base_m["symmetry"]:+.5f}')

        # Laplacian variants with volume_constraint=False
        for lap_steps in LAPLACIAN_STEPS:
            smoothed = mesh.copy()
            trimesh.smoothing.filter_laplacian(smoothed, iterations=lap_steps,
                                               volume_constraint=False)
            smoothed_v = np.array(smoothed.vertices)
            smoothed_f = np.array(smoothed.faces)
            if np.isnan(smoothed_v).any() or np.isinf(smoothed_v).any():
                print(f'    lap{lap_steps}: NaN/Inf (unexpected with volume_constraint=False)')
                n_nan += 1
                new_records.append({'prompt': prompt, 'method': f'laplacian_{lap_steps}',
                                    'error': 'numerical'})
                continue
            m = compute_metrics(smoothed_v, smoothed_f, sym_n, sym_d)
            if m is None:
                new_records.append({'prompt': prompt, 'method': f'laplacian_{lap_steps}',
                                    'error': 'metric_fail'})
                continue
            new_records.append({'prompt': prompt, 'method': f'laplacian_{lap_steps}', **m})
            print(f'    lap{lap_steps:<2}: sym={m["symmetry"]:+.5f}  smo={m["smoothness"]:+.5f}  com={m["compactness"]:+.2f}')

    # Merge with existing DGR records (keep them — they were already good)
    for r in existing:
        if r.get('method') == 'diffgeoreward':
            new_records.append(r)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(new_records, f, indent=2)

    # Summary
    from collections import Counter
    methods = Counter(r['method'] for r in new_records)
    errors = sum(1 for r in new_records if r.get('error'))
    print(f'\nSaved {len(new_records)} records (errors: {errors}) to {OUT_JSON}')
    print(f'methods: {dict(methods)}')


if __name__ == '__main__':
    main()
