"""2-reward combo ablation.

Reruns DGR's 50-step vertex optimization with each pair of rewards enabled
(weights normalized to 0.5 each, third reward zeroed). This tests whether
any 2 of the 3 rewards alone reach DGR's full operating point.

Variants:
    sym_HNC:  weights = [0.5, 0.5, 0.0]
    sym_com:  weights = [0.5, 0.0, 0.5]
    HNC_com:  weights = [0.0, 0.5, 0.5]

Baseline meshes are loaded from results/mesh_validity_objs/baseline/,
ensuring direct pairing with the existing single-reward and full-DGR variants.
Symmetry planes are read from the existing plane_cache.

Output meshes: results/mesh_validity_objs/{variant}/{category}/*.obj
Matches the layout expected by tools/exp_pareto_component_ablation.py so we
can drop the new points straight into the Pareto plot.

Runtime: ~5-8s per mesh, ~210 meshes * 3 variants = ~630 optimizations,
approximately 30-40 min on a single V100.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from shape_gen import refine_with_geo_reward

BASELINE_DIR = Path(ROOT) / 'results' / 'mesh_validity_objs' / 'baseline'
OUT_ROOT     = Path(ROOT) / 'results' / 'mesh_validity_objs'
PLANE_CACHE  = Path(ROOT) / 'analysis_results' / 'mesh_validity_full' / 'plane_cache.json'

COMBOS = {
    'sym_HNC': [0.5, 0.5, 0.0],
    'sym_com': [0.5, 0.0, 0.5],
    'HNC_com': [0.0, 0.5, 0.5],
}

STEPS = 50
LR = 0.005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_key(fname):
    stem = fname[:-4] if fname.endswith('.obj') else fname
    if '_seed' in stem:
        base, seed = stem.rsplit('_seed', 1)
        try:
            seed = int(seed)
        except ValueError:
            seed = None
    else:
        base, seed = stem, None
    return base.replace('_', ' '), seed


def save_obj(vertices, faces, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v_np = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else vertices
    f_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces
    with open(path, 'w') as f:
        for v in v_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in f_np:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=list(COMBOS.keys()),
                    help='Subset of combos to run')
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit number of baseline meshes per category (debug)')
    args = ap.parse_args()

    with open(PLANE_CACHE) as f:
        plane_cache = json.load(f)

    categories = ['symmetry', 'smoothness', 'compactness']
    targets = []
    for cat in categories:
        cdir = BASELINE_DIR / cat
        if not cdir.is_dir():
            continue
        fnames = sorted(p.name for p in cdir.iterdir() if p.suffix == '.obj')
        if args.limit:
            fnames = fnames[:args.limit]
        for fn in fnames:
            targets.append((cat, fn, cdir / fn))
    print(f'Found {len(targets)} baseline meshes')

    t0 = time.time()
    n_done = 0
    n_skip_cache = 0
    n_skip_exist = 0
    n_err = 0
    for variant in args.variants:
        if variant not in COMBOS:
            print(f'  skipping unknown variant: {variant}')
            continue
        w_list = COMBOS[variant]
        w = torch.tensor(w_list, dtype=torch.float32, device=DEVICE)
        out_dir = OUT_ROOT / variant
        print(f'\n=== variant {variant}  weights={w_list}  -> {out_dir} ===')

        for (cat, fname, bl_path) in targets:
            out_path = out_dir / cat / fname
            if out_path.exists():
                n_skip_exist += 1
                continue
            prompt, seed = parse_key(fname)
            pkey = f'{prompt}|seed={seed}'
            p = plane_cache.get(pkey)
            if p is None:
                n_skip_cache += 1
                continue
            try:
                m = trimesh.load(str(bl_path), force='mesh', process=False)
                V = torch.as_tensor(m.vertices, dtype=torch.float32, device=DEVICE)
                F = torch.as_tensor(m.faces, dtype=torch.long, device=DEVICE)
                sym_n = torch.as_tensor(p['normal'], dtype=torch.float32, device=DEVICE)
                sym_d = torch.tensor(float(p['offset']), dtype=torch.float32, device=DEVICE)
                refined, _ = refine_with_geo_reward(
                    V, F, w, steps=STEPS, lr=LR,
                    sym_normal=sym_n, sym_offset=sym_d,
                )
                save_obj(refined, F, out_path)
                n_done += 1
                if n_done % 30 == 0:
                    el = time.time() - t0
                    rate = n_done / el if el > 0 else 0
                    print(f'  done {n_done}  rate {rate:.1f}/s  elapsed {el:.0f}s', flush=True)
            except Exception as e:
                n_err += 1
                print(f'  ERR {variant}/{cat}/{fname}: {type(e).__name__} {e}')

    el = time.time() - t0
    print(f'\ndone {n_done}  err {n_err}  skipped-exist {n_skip_exist}  skipped-no-plane {n_skip_cache}  total {el:.0f}s')


if __name__ == '__main__':
    main()
