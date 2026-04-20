"""Plane-selection ablation: replacement for the coordinate-axis ablation.

Per the STRONGER_SYMMETRY_VARIANT_CHECKLIST.md §0 protocol, the current
tab:axis_ablation (fixed-xz vs fixed-yz vs fixed-xy) should be replaced by a
plane-selection ablation that compares four protocols for picking the symmetry
plane used in the DGR refinement:

    1. fixed_xz     — legacy fixed y-axis reflection (axis=1)
    2. best_of_3    — pick best scoring coordinate plane (3 axis-aligned)
    3. pca_single   — single-PCA-best-of-3 eigenvectors + Adam refine
    4. multi_start  — PCA + 3 coordinate axes + Fibonacci-sphere + top-k refine (default)

For every (prompt, seed) pair:
  - load the baseline mesh
  - select the plane under each protocol (all four)
  - run 50 steps of DGR with DGR weights (sym=0.7, smooth=0.15, compact=0.15)
  - record the final symmetry metric (under each plane, and under a reference
    plane = the multi-start winner, for paired comparison)

Runs on CPU for small mesh subsets. On the server, pass the full baseline
directory as --baseline-dir for the production ablation.
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geo_reward import (
    estimate_symmetry_plane,
    estimate_symmetry_plane_pca,
    symmetry_reward_plane,
    symmetry_reward,
    smoothness_reward,
    compactness_reward,
    compute_initial_huber_delta,
    _build_face_adjacency,
)


DGR_WEIGHTS = torch.tensor([0.7, 0.15, 0.15])


def load(path, device):
    m = trimesh.load(path, process=False)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    faces_attr = getattr(m, 'faces', None)
    if faces_attr is None or len(faces_attr) == 0:
        return None  # caller handles PointCloud-only meshes
    f = torch.tensor(np.asarray(faces_attr), dtype=torch.long, device=device)
    return v, f


def best_of_three_axes(verts):
    """Select whichever coordinate plane (yz, xz, xy) gives the best symmetry score."""
    best_score = float('-inf')
    best_axis = 1
    for ax in range(3):
        s = symmetry_reward(verts, axis=ax).item()
        if s > best_score:
            best_score = s
            best_axis = ax
    n = torch.zeros(3, device=verts.device, dtype=verts.dtype)
    n[best_axis] = 1.0
    d = torch.tensor(0.0, device=verts.device, dtype=verts.dtype)
    return n, d, best_axis


def plane_from_protocol(name, verts):
    """Return (normal, offset, metadata) for one of the four protocols."""
    if name == 'fixed_xz':
        n = torch.tensor([0.0, 1.0, 0.0], device=verts.device, dtype=verts.dtype)
        d = torch.tensor(0.0, device=verts.device, dtype=verts.dtype)
        return n, d, {'axis': 1}
    if name == 'best_of_3':
        n, d, axis = best_of_three_axes(verts)
        return n, d, {'axis': axis}
    if name == 'pca_single':
        n, d = estimate_symmetry_plane_pca(verts.detach())
        return n, d, {}
    if name == 'multi_start':
        n, d = estimate_symmetry_plane(verts.detach())
        return n, d, {}
    raise ValueError(name)


def refine(verts, faces, n, d, steps=50, lr=0.005, weights=DGR_WEIGHTS):
    v_opt = verts.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=lr)
    adj = _build_face_adjacency(faces)
    delta = compute_initial_huber_delta(v_opt, faces)
    with torch.no_grad():
        s0 = symmetry_reward_plane(v_opt, n, d).item()
        sm0 = smoothness_reward(v_opt, faces, delta=delta, _adj=adj).item()
        cp0 = compactness_reward(v_opt, faces).item()
    ss = max(abs(s0), 1e-6); sms = max(abs(sm0), 1e-6); cps = max(abs(cp0), 1e-6)
    for _ in range(steps):
        opt.zero_grad()
        r = weights[0] * symmetry_reward_plane(v_opt, n, d) / ss \
            + weights[1] * smoothness_reward(v_opt, faces, delta=delta, _adj=adj) / sms \
            + weights[2] * compactness_reward(v_opt, faces) / cps
        (-r).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


PROTOCOLS = ['fixed_xz', 'best_of_3', 'pca_single', 'multi_start']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', required=True)
    ap.add_argument('--pattern', default='**/*.obj')
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--refine-steps', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit number of meshes (for smoke tests)')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.baseline_dir, args.pattern), recursive=True))
    if args.limit:
        paths = paths[:args.limit]
    print(f'found {len(paths)} baseline meshes')

    records = []
    t_start = time.time()
    skipped_pc = 0
    for i, path in enumerate(paths):
        try:
            loaded = load(path, args.device)
        except Exception as e:
            print(f'[ERR] {path}: {e}')
            continue
        if loaded is None:
            skipped_pc += 1
            continue
        verts, faces = loaded

        # Reference plane = multi_start winner; used to compare final meshes in a common frame
        ref_n, ref_d = estimate_symmetry_plane(verts.detach())

        rec = {'mesh': os.path.relpath(path, args.baseline_dir).replace('\\', '/'),
               'n_vertices': int(verts.shape[0])}
        for proto in PROTOCOLS:
            t0 = time.time()
            n, d, meta = plane_from_protocol(proto, verts)
            t_est = time.time() - t0

            sym_init_own = symmetry_reward_plane(verts, n, d).item()
            t0 = time.time()
            refined = refine(verts, faces, n, d,
                             steps=args.refine_steps, lr=args.lr)
            t_ref = time.time() - t0

            sym_final_own = symmetry_reward_plane(refined, n, d).item()
            sym_final_ref = symmetry_reward_plane(refined, ref_n, ref_d).item()

            rec[proto] = {
                'normal': n.cpu().tolist(),
                'offset': float(d.cpu().item()),
                'sym_init_own': sym_init_own,
                'sym_final_own': sym_final_own,
                'sym_final_vs_ref_plane': sym_final_ref,
                't_est_s': t_est,
                't_refine_s': t_ref,
                **meta,
            }
        records.append(rec)
        if (i + 1) % 5 == 0 or i == len(paths) - 1:
            print(f'  [{i+1}/{len(paths)}] {rec["mesh"]}  elapsed {time.time()-t_start:.1f}s')

    # Summary stats
    stats = {}
    for proto in PROTOCOLS:
        init = np.array([r[proto]['sym_init_own'] for r in records])
        final_own = np.array([r[proto]['sym_final_own'] for r in records])
        final_ref = np.array([r[proto]['sym_final_vs_ref_plane'] for r in records])
        stats[proto] = {
            'init_mean': float(init.mean()),
            'init_median': float(np.median(init)),
            'final_own_mean': float(final_own.mean()),
            'final_own_median': float(np.median(final_own)),
            'final_vs_ref_mean': float(final_ref.mean()),
            'final_vs_ref_median': float(np.median(final_ref)),
        }

    print('\n=== Plane Selection Ablation Summary ===')
    print(f'{"protocol":<15} {"init_mean":>12} {"final_mean":>14} {"final_vs_ref":>14}')
    for p in PROTOCOLS:
        s = stats[p]
        print(f'{p:<15} {s["init_mean"]:>+12.5f} {s["final_own_mean"]:>+14.5f} '
              f'{s["final_vs_ref_mean"]:>+14.5f}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'summary': stats, 'records': records,
                    'skipped_pointcloud': skipped_pc}, f, indent=2)
    print(f'\nWrote ablation report to {args.output}')
    if skipped_pc:
        print(f'Skipped {skipped_pc} PointCloud-only (faceless) meshes')


if __name__ == '__main__':
    main()
