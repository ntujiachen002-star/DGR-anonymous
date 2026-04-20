"""
Quick validation: test differentiable geometric rewards on simple shapes.
Verifies gradients are meaningful and rewards correlate with ground truth.

Usage:
    python src/test_geo_reward.py --device cuda:0
"""

import argparse
import torch
import torch.nn.functional as F
import trimesh
import numpy as np

from geo_reward import (
    symmetry_reward, smoothness_reward, compactness_reward,
    DiffGeoReward, chamfer_distance
)


def make_sphere(n_vertices=500, device='cuda:0'):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    v = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return v, f, "sphere"


def make_box(device='cuda:0'):
    mesh = trimesh.creation.box()
    v = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return v, f, "box"


def make_asymmetric(device='cuda:0'):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    verts = mesh.vertices.copy()
    # Stretch one side
    mask = verts[:, 0] > 0
    verts[mask, 0] *= 2.0
    v = torch.tensor(verts, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return v, f, "asymmetric"


def make_noisy_sphere(noise_std=0.1, device='cuda:0'):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    verts = mesh.vertices + np.random.randn(*mesh.vertices.shape) * noise_std
    v = torch.tensor(verts, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return v, f, f"noisy_sphere(σ={noise_std})"


def make_elongated(device='cuda:0'):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    verts = mesh.vertices.copy()
    verts[:, 1] *= 5.0  # elongate along y
    v = torch.tensor(verts, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return v, f, "elongated"


def test_rewards(device='cuda:0'):
    """Test that rewards are ordered correctly for known shapes."""
    print("=" * 60)
    print("TEST 1: Reward Values on Known Shapes")
    print("=" * 60)

    shapes = [
        make_sphere(device=device),
        make_box(device=device),
        make_asymmetric(device=device),
        make_noisy_sphere(0.05, device=device),
        make_noisy_sphere(0.2, device=device),
        make_elongated(device=device),
    ]

    print(f"\n{'Shape':<25} {'Symmetry':>12} {'Smoothness':>12} {'Compactness':>12}")
    print("-" * 65)

    for v, f, name in shapes:
        sym = symmetry_reward(v, axis=0).item()
        smooth = smoothness_reward(v, f).item()
        compact = compactness_reward(v, f).item()
        print(f"{name:<25} {sym:>12.6f} {smooth:>12.6f} {compact:>12.6f}")

    print("\nExpected ordering:")
    print("  Symmetry:   sphere > box > noisy > asymmetric")
    print("  Smoothness: sphere > noisy(0.05) > noisy(0.2)")
    print("  Compactness: sphere > box > elongated")


def test_gradients(device='cuda:0'):
    """Test gradient quality: norm, direction, finite-difference validation."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Quality")
    print("=" * 60)

    shapes = [
        make_sphere(device=device),
        make_box(device=device),
        make_asymmetric(device=device),
    ]

    for v, f, name in shapes:
        print(f"\n--- {name} ---")

        for reward_fn, rname in [
            (lambda v, f: symmetry_reward(v, axis=0), "symmetry"),
            (lambda v, f: smoothness_reward(v, f), "smoothness"),
            (lambda v, f: compactness_reward(v, f), "compactness"),
        ]:
            v_test = v.detach().clone().requires_grad_(True)
            r = reward_fn(v_test, f)
            r.backward()
            grad = v_test.grad

            grad_norm = grad.norm().item()
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            nonzero_ratio = (grad.abs() > 1e-8).float().mean().item()

            # Finite difference validation (random direction)
            eps = 1e-4
            direction = torch.randn_like(v_test)
            direction = direction / direction.norm()

            v_plus = v_test.detach() + eps * direction
            v_plus.requires_grad_(False)
            v_minus = v_test.detach() - eps * direction
            v_minus.requires_grad_(False)

            r_plus = reward_fn(v_plus, f)
            r_minus = reward_fn(v_minus, f)
            fd_grad = (r_plus - r_minus) / (2 * eps)
            analytic_grad = (grad * direction).sum()

            # Both are scalars — just check sign agreement
            if fd_grad.abs() > 1e-10 and analytic_grad.abs() > 1e-10:
                cos_sim = (fd_grad * analytic_grad / (fd_grad.abs() * analytic_grad.abs())).item()
            else:
                cos_sim = 0.0

            print(f"  {rname:<12} | norm={grad_norm:.6f} mean={grad_mean:.6f} "
                  f"max={grad_max:.6f} nonzero={nonzero_ratio:.2%} "
                  f"FD_cos_sim={cos_sim:.4f}")


def test_optimization(device='cuda:0', steps=100):
    """Test that gradient descent on rewards actually improves the property."""
    print("\n" + "=" * 60)
    print("TEST 3: Optimization (gradient descent improves property)")
    print("=" * 60)

    # Start with asymmetric shape, optimize for symmetry
    v, f, _ = make_asymmetric(device=device)
    v_opt = v.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=0.01)

    print("\nOptimizing asymmetric shape → symmetric (100 steps)")
    for i in range(steps):
        optimizer.zero_grad()
        loss = -symmetry_reward(v_opt, axis=0)  # maximize symmetry
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f"  Step {i+1:3d} | symmetry_reward = {-loss.item():.6f}")

    final_sym = symmetry_reward(v_opt, axis=0).item()
    initial_sym = symmetry_reward(v.detach(), axis=0).item()
    improvement = final_sym - initial_sym
    print(f"\n  Initial: {initial_sym:.6f} → Final: {final_sym:.6f} | "
          f"Improvement: {improvement:.6f} ({'✅ PASS' if improvement > 0 else '❌ FAIL'})")

    # Start with noisy sphere, optimize for smoothness
    v2, f2, _ = make_noisy_sphere(0.2, device=device)
    v2_opt = v2.detach().clone().requires_grad_(True)
    optimizer2 = torch.optim.Adam([v2_opt], lr=0.005)

    print("\nOptimizing noisy sphere → smooth (100 steps)")
    for i in range(steps):
        optimizer2.zero_grad()
        loss = -smoothness_reward(v2_opt, f2)
        loss.backward()
        optimizer2.step()
        if (i + 1) % 20 == 0:
            print(f"  Step {i+1:3d} | smoothness_reward = {-loss.item():.6f}")

    final_smooth = smoothness_reward(v2_opt, f2).item()
    initial_smooth = smoothness_reward(v2.detach(), f2).item()
    improvement2 = final_smooth - initial_smooth
    print(f"\n  Initial: {initial_smooth:.6f} → Final: {final_smooth:.6f} | "
          f"Improvement: {improvement2:.6f} ({'✅ PASS' if improvement2 > 0 else '❌ FAIL'})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    test_rewards(args.device)
    test_gradients(args.device)
    test_optimization(args.device)
    print("\n✅ All tests complete.")
