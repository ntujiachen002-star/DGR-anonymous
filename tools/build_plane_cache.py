"""Build a plane_cache.json for a directory of baseline mesh OBJ files.

Following STRONGER_SYMMETRY_VARIANT_CHECKLIST.md §0: the plane is estimated
once on the *initial* (unrefined) mesh and held fixed across every paired
method variant for a given (prompt, seed) pair. This script produces the
canonical plane cache that every downstream experiment then reads.

Key invariant: only BASELINE meshes are inputs. Refined meshes must not be
used to build the cache — doing so would let the plane shift with the method
and break paired statistics.

Run locally on any mesh subset, or on the server on the full set:
    python tools/build_plane_cache.py \
        --baseline-dir results/full/baseline \
        --output results/plane_cache.json

    python tools/build_plane_cache.py \
        --baseline-dir /path/to/server/baselines \
        --output /path/to/server/plane_cache.json

The cache key is a relative path ("category/file.obj") or a content hash; pass
--key-mode to choose. Default is "relpath" relative to --baseline-dir so the
same cache is portable across machines.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geo_reward import estimate_symmetry_plane, symmetry_reward_plane


def mesh_key(path, baseline_dir, mode):
    if mode == 'relpath':
        return os.path.relpath(path, baseline_dir).replace('\\', '/')
    elif mode == 'basename':
        return os.path.basename(path)
    elif mode == 'sha1':
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    raise ValueError(mode)


def load_mesh(path, device):
    m = trimesh.load(path, process=False)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', required=True,
                    help='Directory containing baseline OBJ files (searched recursively)')
    ap.add_argument('--pattern', default='**/*.obj',
                    help='Glob pattern for baseline meshes relative to --baseline-dir')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='Substrings to exclude (e.g. "refined" "dgr" to keep only baselines)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--key-mode', default='relpath', choices=['relpath', 'basename', 'sha1'])
    ap.add_argument('--n-sphere', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--refine-steps', type=int, default=50)
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip meshes already present in the output cache')
    args = ap.parse_args()

    baseline_dir = os.path.abspath(args.baseline_dir)
    if not os.path.isdir(baseline_dir):
        print(f'baseline-dir not found: {baseline_dir}')
        sys.exit(1)

    search = os.path.join(baseline_dir, args.pattern)
    paths = sorted(glob.glob(search, recursive=True))
    for excl in args.exclude:
        paths = [p for p in paths if excl not in p]
    print(f'found {len(paths)} mesh files under {baseline_dir}')

    cache = {}
    if args.skip_existing and os.path.exists(args.output):
        with open(args.output) as f:
            cache = json.load(f)
        print(f'loaded {len(cache)} existing cache entries')

    n_new = 0
    n_skip = 0
    t_start = time.time()
    for i, p in enumerate(paths):
        key = mesh_key(p, baseline_dir, args.key_mode)
        if key in cache:
            n_skip += 1
            continue
        try:
            verts = load_mesh(p, args.device)
        except Exception as e:
            print(f'[ERR] load {p}: {e}')
            continue
        t0 = time.time()
        n, d = estimate_symmetry_plane(
            verts.detach(),
            n_sphere_candidates=args.n_sphere,
            top_k_refine=args.top_k,
            refine_steps=args.refine_steps,
        )
        score = symmetry_reward_plane(verts, n, d).item()
        dt = time.time() - t0
        cache[key] = {
            'normal': n.cpu().tolist(),
            'offset': float(d.cpu().item()),
            'init_score': score,
            'n_vertices': int(verts.shape[0]),
            'estimator': {
                'n_sphere': args.n_sphere,
                'top_k': args.top_k,
                'refine_steps': args.refine_steps,
            },
            'source_path': os.path.relpath(p, baseline_dir).replace('\\', '/'),
        }
        n_new += 1
        if (i + 1) % 20 == 0 or i == len(paths) - 1:
            elapsed = time.time() - t_start
            print(f'  [{i+1}/{len(paths)}] {key}  score={score:+.5f}  ({dt:.2f}s; '
                  f'{n_new} new, {n_skip} skipped, total {elapsed:.1f}s)')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'\nWrote {len(cache)} entries to {args.output} ({n_new} new)')


if __name__ == '__main__':
    main()
