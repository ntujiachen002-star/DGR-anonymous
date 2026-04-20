"""Parallel (multiprocessing) plane cache builder.

Replaces build_plane_cache.py for CPU workloads where we want to scale across
cores. Each mesh is dispatched to a worker process; each worker runs the
estimator on its assigned mesh.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from geo_reward import estimate_symmetry_plane, symmetry_reward_plane


_ARGS = None


def _init(args_dict):
    global _ARGS
    _ARGS = args_dict
    torch.set_num_threads(1)  # avoid thread contention between workers


def _process(args_pair):
    baseline_dir, path = args_pair
    try:
        m = trimesh.load(path, process=False)
        v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32)
        if v.shape[0] == 0:
            return {'path': path, 'error': 'empty verts'}
        n, d = estimate_symmetry_plane(
            v.detach(),
            n_sphere_candidates=_ARGS['n_sphere'],
            top_k_refine=_ARGS['top_k'],
            refine_steps=_ARGS['refine_steps'],
        )
        score = symmetry_reward_plane(v, n, d).item()
        rel = os.path.relpath(path, baseline_dir).replace('\\', '/')
        return {
            'key': rel,
            'normal': n.cpu().tolist(),
            'offset': float(d.cpu().item()),
            'init_score': score,
            'n_vertices': int(v.shape[0]),
        }
    except Exception as e:
        return {'path': path, 'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', required=True)
    ap.add_argument('--pattern', default='**/*.obj')
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--n-sphere', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--refine-steps', type=int, default=50)
    args = ap.parse_args()

    baseline_dir = os.path.abspath(args.baseline_dir)
    paths = sorted(glob.glob(os.path.join(baseline_dir, args.pattern), recursive=True))
    for excl in args.exclude:
        paths = [p for p in paths if excl not in p]
    print(f'found {len(paths)} mesh files under {baseline_dir}')

    jobs = [(baseline_dir, p) for p in paths]
    init_args = {
        'n_sphere': args.n_sphere,
        'top_k': args.top_k,
        'refine_steps': args.refine_steps,
    }

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(init_args,)) as pool:
        results = pool.map(_process, jobs, chunksize=4)
    elapsed = time.time() - t0
    print(f'pool done in {elapsed:.1f}s ({len(results)/max(elapsed,0.1):.1f} meshes/s)')

    cache = {}
    errors = []
    for r in results:
        if 'error' in r:
            errors.append(r)
        else:
            cache[r['key']] = {
                'normal': r['normal'],
                'offset': r['offset'],
                'init_score': r['init_score'],
                'n_vertices': r['n_vertices'],
                'source_path': r['key'],
            }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'wrote {len(cache)} entries to {args.output} ({len(errors)} errors)')
    if errors:
        for e in errors[:5]:
            print(f'  [err] {e}')


if __name__ == '__main__':
    main()
