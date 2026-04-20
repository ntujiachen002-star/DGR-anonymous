"""
Experiment B (LOCAL CPU variant): Gradient Explosion Root Cause Analysis
under the new multi-start symmetry plane protocol.

Reads existing baseline meshes from results/mesh_validity_objs/baseline/
(no Shap-E generation, no GPU). Estimates the symmetry plane on each
loaded baseline using the new multi-start estimator and uses that plane
throughout both gradient decomposition and the clip threshold ablation.

Output: analysis_results_newproto/grad_explosion/
  - per_step_grad.json
  - clip_ablation.json
"""
import os
import sys
import re
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import (symmetry_reward_plane, estimate_symmetry_plane,
                        smoothness_reward, compactness_reward)


FAILURE_PROMPTS = [
    "a symmetric vase",
    "a perfectly balanced chair",
    "an hourglass shape",
    "a symmetric wine glass",
    "a balanced chess piece, a king",
]

SUCCESS_PROMPTS = [
    "a smooth sphere",
    "a compact stone",
    "a polished marble",
    "a round pebble",
    "a smooth egg",
]

SEED = 42
STEPS = 50
LR = 0.005
WEIGHTS = torch.tensor([0.33, 0.33, 0.34])

# Local mesh paths (CPU-only, no Shap-E generation)
MESH_ROOT = Path('results/mesh_validity_objs/baseline')


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def find_mesh(prompt, seed=SEED):
    """Locate a baseline OBJ file for `(prompt, seed)` under any category dir."""
    name = f'{slug(prompt)}_seed{seed}.obj'
    for cat in ['symmetry', 'smoothness', 'compactness']:
        candidate = MESH_ROOT / cat / name
        if candidate.exists():
            return candidate
    return None


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def decompose_gradients(vertices, faces, weights, sym_n, sym_d, steps=50, lr=0.005, clip_norm=1.0):
    """Run refinement with per-step gradient decomposition by reward component."""
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    with torch.no_grad():
        sym_init = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        smo_init = smoothness_reward(v_opt, faces).item()
        com_init = compactness_reward(v_opt, faces).item()
    sym_scale = max(abs(sym_init), 1e-6)
    smo_scale = max(abs(smo_init), 1e-6)
    com_scale = max(abs(com_init), 1e-6)

    records = []
    for step in range(steps):
        sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
        smo = smoothness_reward(v_opt, faces)
        com = compactness_reward(v_opt, faces)

        grad_norms = {}
        for name, reward, scale in [("sym", sym, sym_scale),
                                    ("smo", smo, smo_scale),
                                    ("com", com, com_scale)]:
            optimizer.zero_grad()
            (-reward / scale).backward(retain_graph=True)
            grad_norms[f"grad_norm_{name}"] = v_opt.grad.norm().item()

        optimizer.zero_grad()
        combined = (weights[0] * sym / sym_scale
                    + weights[1] * smo / smo_scale
                    + weights[2] * com / com_scale)
        (-combined).backward()
        grad_norm_combined = v_opt.grad.norm().item()
        clip_triggered = grad_norm_combined > clip_norm

        torch.nn.utils.clip_grad_norm_([v_opt], clip_norm)
        optimizer.step()

        records.append({
            "step": step,
            **grad_norms,
            "grad_norm_combined": grad_norm_combined,
            "clip_triggered": clip_triggered,
            "symmetry": sym.item(),
            "smoothness": smo.item(),
            "compactness": com.item(),
        })

    return records


def clip_ablation(vertices, faces, weights, sym_n, sym_d, clip_norms=(1.0, 0.1, 0.01)):
    """Test different clip thresholds on the same mesh."""
    results = {}
    for clip_val in clip_norms:
        v_opt = vertices.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([v_opt], lr=LR)

        with torch.no_grad():
            sym_init = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
            smo_init = smoothness_reward(v_opt, faces).item()
            com_init = compactness_reward(v_opt, faces).item()
        sym_scale = max(abs(sym_init), 1e-6)
        smo_scale = max(abs(smo_init), 1e-6)
        com_scale = max(abs(com_init), 1e-6)

        clip_count = 0
        for step in range(STEPS):
            optimizer.zero_grad()
            sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
            smo = smoothness_reward(v_opt, faces)
            com = compactness_reward(v_opt, faces)
            combined = (weights[0] * sym / sym_scale
                        + weights[1] * smo / smo_scale
                        + weights[2] * com / com_scale)
            (-combined).backward()
            gn = v_opt.grad.norm().item()
            if gn > clip_val:
                clip_count += 1
            torch.nn.utils.clip_grad_norm_([v_opt], clip_val)
            optimizer.step()

        with torch.no_grad():
            final_sym = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
            final_smo = smoothness_reward(v_opt, faces).item()
            final_com = compactness_reward(v_opt, faces).item()

        results[f"clip_{clip_val}"] = {
            "clip_norm": clip_val,
            "symmetry": final_sym,
            "smoothness": final_smo,
            "compactness": final_com,
            "clip_triggered_count": clip_count,
            "clip_triggered_pct": clip_count / STEPS * 100,
        }

    return results


def main():
    out_dir = Path('analysis_results_newproto/grad_explosion')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Gradient Decomposition (NEW PLANE) ===")
    grad_data = {"failure": {}, "success": {}}
    skipped = []

    for group_name, prompts in [("failure", FAILURE_PROMPTS), ("success", SUCCESS_PROMPTS)]:
        for prompt in prompts:
            mesh_path = find_mesh(prompt)
            if mesh_path is None:
                print(f"  [{group_name}] {prompt}: SKIP (no local baseline mesh)")
                skipped.append(prompt)
                continue

            print(f"\n  [{group_name}] {prompt}")
            print(f"    {mesh_path}")

            verts, faces = load_obj(str(mesh_path))
            print(f"    Mesh: {verts.shape[0]}v, {faces.shape[0]}f")
            if faces.shape[0] == 0 or verts.shape[0] == 0:
                print(f"    SKIP degenerate")
                skipped.append(prompt)
                continue

            sym_n, sym_d = estimate_symmetry_plane(verts.detach())
            print(f"    Plane: n=[{sym_n[0]:+.3f},{sym_n[1]:+.3f},{sym_n[2]:+.3f}], d={sym_d.item():+.4f}")

            records = decompose_gradients(verts, faces, WEIGHTS, sym_n, sym_d, steps=STEPS)
            grad_data[group_name][prompt] = records

            avg_gn = float(np.mean([r["grad_norm_combined"] for r in records]))
            max_gn = float(max(r["grad_norm_combined"] for r in records))
            clips = sum(1 for r in records if r["clip_triggered"])
            print(f"    avg_grad={avg_gn:.2f}, max_grad={max_gn:.2f}, clips={clips}/{STEPS}")

    with open(out_dir / 'per_step_grad.json', 'w') as f:
        json.dump(grad_data, f, indent=2)

    print("\n=== Step 2: Clip Threshold Ablation ===")
    clip_data = {}

    for prompt in FAILURE_PROMPTS:
        mesh_path = find_mesh(prompt)
        if mesh_path is None:
            print(f"  {prompt}: SKIP (no mesh)")
            continue
        print(f"\n  {prompt}")
        verts, faces = load_obj(str(mesh_path))
        if faces.shape[0] == 0:
            print(f"    SKIP degenerate")
            continue
        sym_n, sym_d = estimate_symmetry_plane(verts.detach())
        clip_results = clip_ablation(verts, faces, WEIGHTS, sym_n, sym_d)
        clip_data[prompt] = clip_results
        for k, v in clip_results.items():
            print(f"    {k}: sym={v['symmetry']:.6f}, smo={v['smoothness']:.6f}, "
                  f"com={v['compactness']:.2f}, clips={v['clip_triggered_pct']:.0f}%")

    with open(out_dir / 'clip_ablation.json', 'w') as f:
        json.dump(clip_data, f, indent=2)

    print("\n=== CLIP ABLATION SUMMARY ===")
    header = "{:>8} {:>12} {:>12} {:>10} {:>8}".format(
        "Clip", "Sym", "Smo", "Com", "Clip%")
    print(header)
    for clip_val in [1.0, 0.1, 0.01]:
        syms, smos, coms, clips = [], [], [], []
        for prompt in FAILURE_PROMPTS:
            if prompt not in clip_data:
                continue
            d = clip_data[prompt][f"clip_{clip_val}"]
            syms.append(d["symmetry"])
            smos.append(d["smoothness"])
            coms.append(d["compactness"])
            clips.append(d["clip_triggered_pct"])
        if not syms:
            continue
        print("{:>8.2f} {:>12.6f} {:>12.6f} {:>10.2f} {:>7.1f}%".format(
            clip_val, np.mean(syms), np.mean(smos),
            np.mean(coms), np.mean(clips)))

    if skipped:
        print(f"\n[note] skipped {len(skipped)} prompts (no local baseline): {skipped}")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
