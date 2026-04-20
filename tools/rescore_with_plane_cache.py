"""Recompute symmetry metrics for existing meshes using a fixed plane cache.

Reads a plane_cache.json (from build_plane_cache.py) and a mesh directory,
then for each mesh:
  - looks up its plane (keyed by relpath or basename)
  - loads the OBJ
  - computes symmetry_reward_plane with the CACHED plane

Produces two outputs:
  1. per-mesh rescored JSON with sym_old / sym_new for side-by-side comparison
  2. summary statistics (mean, median, paired delta)

Typical workflow:
    # On the server, after the cache is built:
    python tools/rescore_with_plane_cache.py \
        --plane-cache results/plane_cache.json \
        --mesh-dirs archive_old_metric/diffgeoreward_old_objs results/qualitative_meshes \
        --key-mode basename \
        --output results/rescore_report.json

Key lookup:
  The plane cache is keyed by BASELINE mesh. Refined meshes of the same
  (prompt, seed) pair share the baseline's plane. To resolve a refined mesh to
  its baseline plane, we strip common suffixes like "_huber_nc", "_dgr",
  "_refined", "_seedXX", and search by a normalized base name. Override by
  passing --suffixes-to-strip.

The old symmetry score is also recomputed via fixed xz plane for side-by-side
reference.
"""
import argparse
import glob
import json
import os
import re
import sys
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geo_reward import symmetry_reward_plane, symmetry_reward


DEFAULT_STRIP_SUFFIXES = [
    '_huber_nc', '_dgr', '_refined', '_dgr_refined',
    '_equal_wt', '_two_reward', '_pcgrad', '_sym_only',
    '_lang2comp', '_base', '_baseline',
]


def normalize_key(name, strip_suffixes):
    base = os.path.splitext(os.path.basename(name))[0]
    for suf in strip_suffixes:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def resolve_plane(mesh_path, cache, key_mode, strip_suffixes, mesh_roots):
    if key_mode == 'basename':
        basename = os.path.basename(mesh_path)
        if basename in cache:
            return cache[basename]
        norm = normalize_key(mesh_path, strip_suffixes)
        matches = [k for k in cache if normalize_key(k, strip_suffixes) == norm]
        if len(matches) == 1:
            return cache[matches[0]]
        if len(matches) > 1:
            # Disambiguate by matching parent directory component (e.g. "symmetry")
            parent = os.path.basename(os.path.dirname(mesh_path))
            for k in matches:
                if parent and parent in k:
                    return cache[k]
            return cache[matches[0]]
        return None
    elif key_mode == 'relpath':
        for root in mesh_roots:
            try:
                rel = os.path.relpath(mesh_path, root).replace('\\', '/')
            except ValueError:
                continue
            if rel in cache:
                return cache[rel]
        return None
    else:
        raise ValueError(key_mode)


def load_mesh(path, device):
    """Load OBJ file. Handles both Trimesh (with faces) and PointCloud
    (verts only, e.g. degenerate Shap-E outputs). Faces default to empty
    tensor when absent — only the symmetry metric is rescored, which does
    not require faces."""
    m = trimesh.load(path, process=False)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    faces_attr = getattr(m, 'faces', None)
    if faces_attr is None or len(faces_attr) == 0:
        f = torch.zeros((0, 3), dtype=torch.long, device=device)
    else:
        f = torch.tensor(np.asarray(faces_attr), dtype=torch.long, device=device)
    return v, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plane-cache', required=True)
    ap.add_argument('--mesh-dirs', nargs='+', required=True)
    ap.add_argument('--pattern', default='**/*.obj')
    ap.add_argument('--key-mode', default='basename', choices=['basename', 'relpath'])
    ap.add_argument('--strip-suffixes', nargs='*', default=DEFAULT_STRIP_SUFFIXES)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    with open(args.plane_cache) as f:
        cache = json.load(f)
    print(f'loaded {len(cache)} plane entries from {args.plane_cache}')

    all_meshes = []
    for d in args.mesh_dirs:
        hits = sorted(glob.glob(os.path.join(d, args.pattern), recursive=True))
        all_meshes.extend(hits)
    print(f'found {len(all_meshes)} mesh files across {len(args.mesh_dirs)} dir(s)')

    mesh_roots = [os.path.abspath(d) for d in args.mesh_dirs]
    records = []
    n_resolved = 0
    n_unresolved = 0
    for path in all_meshes:
        plane = resolve_plane(os.path.abspath(path), cache,
                               args.key_mode, args.strip_suffixes, mesh_roots)
        if plane is None:
            n_unresolved += 1
            continue
        n_resolved += 1
        try:
            verts, faces = load_mesh(path, args.device)
        except Exception as e:
            print(f'[ERR] {path}: {e}')
            continue
        normal = torch.tensor(plane['normal'], dtype=torch.float32, device=args.device)
        offset = torch.tensor(plane['offset'], dtype=torch.float32, device=args.device)

        with torch.no_grad():
            sym_new = symmetry_reward_plane(verts, normal, offset).item()
            sym_xz_old = symmetry_reward(verts, axis=1).item()

        records.append({
            'mesh': os.path.relpath(path).replace('\\', '/'),
            'sym_xz_old': sym_xz_old,
            'sym_new': sym_new,
            'delta_pct': (sym_new - sym_xz_old) / max(abs(sym_xz_old), 1e-9) * 100.0,
            'plane_normal': plane['normal'],
        })

    print(f'resolved {n_resolved} meshes, {n_unresolved} unresolved (no matching plane)')

    if records:
        xz_vals = np.array([r['sym_xz_old'] for r in records])
        new_vals = np.array([r['sym_new'] for r in records])
        summary = {
            'n': len(records),
            'sym_xz_mean': float(xz_vals.mean()),
            'sym_new_mean': float(new_vals.mean()),
            'sym_xz_median': float(np.median(xz_vals)),
            'sym_new_median': float(np.median(new_vals)),
            'n_new_better': int((new_vals > xz_vals).sum()),
            'n_new_worse': int((new_vals < xz_vals).sum()),
            'n_tied': int((new_vals == xz_vals).sum()),
        }
        print('\n=== Summary ===')
        for k, v in summary.items():
            print(f'  {k}: {v}')
    else:
        summary = {'n': 0}

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'summary': summary, 'records': records}, f, indent=2)
    print(f'\nWrote rescore report to {args.output}')


if __name__ == '__main__':
    main()
