"""DiffGeoReward — minimal single-mesh demo.

Usage:
    python demo.py --input examples/vase.obj --output refined.obj
    python demo.py --input your_mesh.obj --weights 0.4 0.4 0.2
    python demo.py --input your_mesh.obj --steps 100 --lr 0.005

Loads an input .obj, runs the 50-step DGR refinement, and writes the refined
mesh. Prints baseline and refined reward values to stdout so you can verify the
method is working end-to-end (all three rewards should move toward zero).

This script does NOT need Shap-E; it operates directly on whatever mesh you
supply. For generating baseline meshes from text, see tools/exp_k_full_mesh_validity.py.
"""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from geo_reward import (
    symmetry_reward_plane, smoothness_reward, compactness_reward,
    estimate_symmetry_plane,
)
from shape_gen import refine_with_geo_reward


def save_obj(vertices, faces, path):
    v = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else vertices
    f = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces
    with open(path, 'w') as out:
        for vi in v:
            out.write(f"v {vi[0]:.6f} {vi[1]:.6f} {vi[2]:.6f}\n")
        for fi in f:
            out.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")


def evaluate(V, F, plane_n, plane_d):
    with torch.no_grad():
        return {
            'R_sym':     symmetry_reward_plane(V, plane_n, plane_d).item(),
            'R_smooth':  smoothness_reward(V, F).item(),
            'R_compact': compactness_reward(V, F).item(),
        }


def main():
    ap = argparse.ArgumentParser(description='Refine a single mesh with DGR.')
    ap.add_argument('--input', required=True, help='Path to input .obj')
    ap.add_argument('--output', default='refined.obj', help='Output .obj path')
    ap.add_argument('--weights', nargs=3, type=float, default=[1/3, 1/3, 1/3],
                    metavar=('W_SYM', 'W_HNC', 'W_COM'),
                    help='Reward weights (default: 1/3 1/3 1/3)')
    ap.add_argument('--steps', type=int, default=50, help='Adam steps (default: 50)')
    ap.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 5e-3)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    print(f'=== DiffGeoReward demo ===')
    print(f'  input:  {args.input}')
    print(f'  output: {args.output}')
    print(f'  weights (sym, HNC, com): {args.weights}')
    print(f'  steps: {args.steps}  lr: {args.lr}  device: {args.device}\n')

    mesh = trimesh.load(args.input, force='mesh', process=False)
    V0 = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=args.device)
    F  = torch.as_tensor(mesh.faces,    dtype=torch.long,    device=args.device)
    print(f'Loaded mesh: {len(V0)} vertices, {len(F)} faces.')

    # Multi-start symmetry plane (takes ~200 ms).
    plane_n, plane_d = estimate_symmetry_plane(V0)
    print(f'Estimated symmetry plane normal: {plane_n.cpu().numpy().round(3)}\n')

    before = evaluate(V0, F, plane_n, plane_d)
    print('Baseline rewards:')
    for k, v in before.items():
        print(f'  {k:>10s}: {v:+.4f}')

    # Refinement
    weights = torch.tensor(args.weights, dtype=torch.float32, device=args.device)
    V_ref, _ = refine_with_geo_reward(
        V0, F, weights,
        steps=args.steps, lr=args.lr,
        sym_normal=plane_n, sym_offset=plane_d,
    )

    after = evaluate(V_ref, F, plane_n, plane_d)
    print('\nRefined rewards:')
    for k, v in after.items():
        delta = v - before[k]
        arrow = 'up  ' if delta > 0 else 'down'  # reward values are negative; up = closer to 0 = better
        print(f'  {k:>10s}: {v:+.4f}  (delta {delta:+.4f} {arrow})')

    save_obj(V_ref, F, args.output)
    print(f'\nWrote refined mesh to {args.output}.')


if __name__ == '__main__':
    main()
