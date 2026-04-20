"""
Experiment C: Laplacian Smoothing Baseline + DreamCS Check
Tests whether DiffGeoReward is "just mesh smoothing" by comparing with trimesh Laplacian baseline.
Also checks DreamCS open-source availability.
"""
import os, sys, json, torch, trimesh, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        smoothness_reward, compactness_reward)

sys.path.insert(0, os.path.dirname(__file__))
from _plane_protocol import PlaneStore, make_key

# 20 prompts: ~7 per category
PROMPTS = [
    # Symmetry (7)
    "a symmetric vase", "a perfectly balanced chair", "a symmetric wine glass",
    "a balanced chess piece, a king", "an hourglass shape",
    "a symmetrical temple", "a symmetric butterfly",
    # Smoothness (7)
    "a smooth sphere", "a polished marble", "a smooth egg",
    "a round pebble", "a smooth teapot", "a sleek car", "a smooth dolphin",
    # Compactness (6)
    "a compact stone", "a solid cube", "a dense ball",
    "a compact turtle shell", "a chunky robot", "a thick coin",
]

LAPLACIAN_STEPS = [10, 20, 50]
SEED = 42
DEVICE = 'cuda:0'

def compute_metrics(vertices, faces, sym_plane=None, device='cuda:0'):
    """Compute all three geometric metrics under an optional fixed plane.

    If sym_plane=(normal, offset) is provided, symmetry is computed via the new
    plane protocol. Otherwise falls back to the legacy fixed xz axis.

    Returns a dict with sym/smo/com or {error: 'degenerate_mesh'} if the input
    is a point cloud (0 faces) or empty — Shap-E occasionally emits these and
    they crash _build_face_adjacency.
    """
    v = torch.tensor(np.array(vertices), dtype=torch.float32, device=device) if not isinstance(vertices, torch.Tensor) else vertices
    f = torch.tensor(np.array(faces), dtype=torch.long, device=device) if not isinstance(faces, torch.Tensor) else faces
    if v.numel() == 0 or f.numel() == 0 or f.shape[0] == 0:
        return {"symmetry": None, "smoothness": None, "compactness": None,
                "error": "degenerate_mesh"}
    try:
        if sym_plane is not None:
            sym = symmetry_reward_plane(v, sym_plane[0], sym_plane[1]).item()
        else:
            sym = symmetry_reward(v, axis=1).item()
        return {
            "symmetry": sym,
            "smoothness": smoothness_reward(v, f).item(),
            "compactness": compactness_reward(v, f).item(),
        }
    except Exception as e:
        return {"symmetry": None, "smoothness": None, "compactness": None,
                "error": str(e)}

def main():
    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)

    out_dir = "analysis_results/dreamcs_comparison"
    os.makedirs(out_dir, exist_ok=True)
    plane_store = PlaneStore.load_or_new(os.path.join(out_dir, "plane_cache.json"))

    all_results = []

    for prompt in PROMPTS:
        print(f"\n=== {prompt} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        results = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        verts, faces, _ = results[0]
        print(f"  Mesh: {verts.shape[0]}v, {faces.shape[0]}f")

        # Skip degenerate baselines (Shap-E sometimes emits point clouds).
        if faces.shape[0] == 0 or verts.shape[0] == 0:
            print(f"  SKIP degenerate baseline")
            all_results.append({"prompt": prompt, "method": "baseline",
                                "error": "degenerate_baseline"})
            continue

        # Estimate plane once on the baseline; share across Laplacian/DGR variants
        sym_n, sym_d = plane_store.get(make_key(prompt, SEED), verts=verts)

        # 1. Baseline metrics (under new plane)
        baseline_m = compute_metrics(verts, faces, sym_plane=(sym_n, sym_d), device=DEVICE)
        row = {"prompt": prompt, "method": "baseline", **baseline_m}
        all_results.append(row)
        if baseline_m.get("error"):
            print(f"  Baseline: ERROR {baseline_m['error']}")
            continue
        print(f"  Baseline: sym={baseline_m['symmetry']:.6f}, smo={baseline_m['smoothness']:.6f}, com={baseline_m['compactness']:.2f}")

        # 2. Laplacian smoothing baselines (under the same baseline plane)
        v_np = verts.detach().cpu().numpy()
        f_np = faces.detach().cpu().numpy()
        base_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)

        for lap_steps in LAPLACIAN_STEPS:
            smoothed = base_mesh.copy()
            trimesh.smoothing.filter_laplacian(smoothed, iterations=lap_steps, volume_constraint=False)
            lap_m = compute_metrics(
                torch.tensor(np.array(smoothed.vertices), dtype=torch.float32, device=DEVICE),
                torch.tensor(np.array(smoothed.faces), dtype=torch.long, device=DEVICE),
                sym_plane=(sym_n, sym_d), device=DEVICE,
            )
            row = {"prompt": prompt, "method": f"laplacian_{lap_steps}", **lap_m}
            all_results.append(row)
            print(f"  Lap({lap_steps:>2}): sym={lap_m['symmetry']:.6f}, smo={lap_m['smoothness']:.6f}, com={lap_m['compactness']:.2f}")

        # 3. DiffGeoReward refinement with the cached plane
        weights = torch.tensor([0.33, 0.33, 0.34], device=DEVICE)
        refined_verts, _ = refine_with_geo_reward(
            verts, faces, weights, steps=50, lr=0.005,
            sym_normal=sym_n, sym_offset=sym_d,
        )
        dgr_m = compute_metrics(refined_verts, faces, sym_plane=(sym_n, sym_d), device=DEVICE)
        row = {"prompt": prompt, "method": "diffgeoreward", **dgr_m}
        all_results.append(row)
        print(f"  DGR:     sym={dgr_m['symmetry']:.6f}, smo={dgr_m['smoothness']:.6f}, com={dgr_m['compactness']:.2f}")

    plane_store.save()

    # Save
    with open(os.path.join(out_dir, "laplacian_baseline.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print("\n=== AGGREGATE SUMMARY ===")
    methods = ["baseline"] + [f"laplacian_{s}" for s in LAPLACIAN_STEPS] + ["diffgeoreward"]
    print(f"{'Method':<20} {'Symmetry':>12} {'Smoothness':>12} {'Compactness':>12}")
    for method in methods:
        rows = [r for r in all_results if r["method"] == method]
        if rows:
            sym_vals = [r["symmetry"] for r in rows if r.get("symmetry") is not None]
            smo_vals = [r["smoothness"] for r in rows if r.get("smoothness") is not None]
            com_vals = [r["compactness"] for r in rows if r.get("compactness") is not None]
            if not sym_vals:
                continue
            sym_avg = np.mean(sym_vals)
            smo_avg = np.mean(smo_vals)
            com_avg = np.mean(com_vals)
            print(f"{method:<20} {sym_avg:>12.6f} {smo_avg:>12.6f} {com_avg:>12.2f}")

    # DreamCS check note
    print("\n=== DreamCS Open Source Check ===")
    print("DreamCS (Yang et al., 2025) - no public repository found as of 2026-04.")
    print("Using methodological comparison table (Route 2) in Related Work.")

    # Write methodological table
    table = """| Dimension | DreamCS | DiffGeoReward (ours) |
|-----------|---------|---------------------|
| Optimization stage | Generation process | Post-hoc refinement |
| Backbone dependency | Tightly coupled | Any backbone |
| Requires retraining | Yes | No |
| Multi-objective reward | Not explicit | Core design |
| Catastrophic single-reward analysis | None | Primary contribution |
| Training-free | No | Yes |
"""
    with open(os.path.join(out_dir, "methodological_table.md"), 'w') as f:
        f.write(table)

    print(f"\nResults saved to {out_dir}/")

if __name__ == "__main__":
    main()
