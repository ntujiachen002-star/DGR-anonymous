"""
Experiment N: Laplacian Smoothing Baseline Comparison.
Classic mesh processing method vs DiffGeoReward.
Applies iterative Laplacian smoothing to baseline meshes,
then evaluates all geometric metrics + CLIP.
110 prompts x 3 seeds x 4 smoothing strengths = 1320 runs.
GPU required for Shap-E generation. ~3h on V100.
"""
import os, sys, json, torch, numpy as np, time, re
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

DEVICE = 'cuda:0'
SEEDS = [42, 123, 456]
STEPS = 50
LR = 0.005

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CATEGORIES = {}
for p in SYMMETRY_PROMPTS:
    PROMPT_CATEGORIES[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "smoothness"
for p in COMPACTNESS_PROMPTS:
    PROMPT_CATEGORIES[p] = "compactness"


def laplacian_smooth(vertices, faces, iterations=10, lamb=0.5):
    """Iterative Laplacian smoothing on GPU tensors.

    Classic mesh smoothing: move each vertex toward the average of its neighbors.
    This is the standard baseline in geometry processing.

    Args:
        vertices: (V, 3) float tensor
        faces: (F, 3) long tensor
        iterations: number of smoothing passes
        lamb: smoothing factor (0=no change, 1=full average)
    Returns:
        Smoothed vertices (V, 3)
    """
    V = vertices.shape[0]
    device = vertices.device

    # Build adjacency via edges
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]
    ], dim=0)
    src, dst = edges[:, 0], edges[:, 1]

    v = vertices.clone()
    for _ in range(iterations):
        neighbor_sum = torch.zeros_like(v)
        neighbor_count = torch.zeros(V, 1, device=device)
        neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), v[src])
        neighbor_count.scatter_add_(0, dst.unsqueeze(1),
                                    torch.ones(dst.shape[0], 1, device=device))
        neighbor_count = neighbor_count.clamp(min=1)
        neighbor_avg = neighbor_sum / neighbor_count
        v = (1 - lamb) * v + lamb * neighbor_avg

    return v


# Laplacian configs: (iterations, lambda, name)
LAPLACIAN_CONFIGS = [
    (5, 0.3, "lap_light"),       # light smoothing
    (10, 0.5, "lap_medium"),     # medium smoothing (standard)
    (20, 0.5, "lap_strong"),     # strong smoothing
    (50, 0.5, "lap_extreme"),    # extreme smoothing (over-smoothed)
]

# DiffGeoReward configs for comparison
DGR_CONFIGS = [
    ("dgr_equal", [0.33, 0.33, 0.34]),
    ("dgr_smooth", [0.0, 1.0, 0.0]),
]


def main():
    out_dir = Path("analysis_results/laplacian_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "checkpoint.json"
    results = []
    done_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["seed"], r["method"]) for r in results}
        print(f"Resuming: {len(results)} done")

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)
    t0 = time.time()

    total_prompts = len(ALL_PROMPTS)
    for pi, prompt in enumerate(ALL_PROMPTS):
        cat = PROMPT_CATEGORIES[prompt]

        for seed in SEEDS:
            # Check if we need any methods for this prompt+seed
            needed = []
            for _, _, name in LAPLACIAN_CONFIGS:
                if (prompt, seed, name) not in done_keys:
                    needed.append(("lap", name))
            for name, _ in DGR_CONFIGS:
                if (prompt, seed, name) not in done_keys:
                    needed.append(("dgr", name))
            if (prompt, seed, "baseline") not in done_keys:
                needed.append(("bl", "baseline"))

            if not needed:
                continue

            # Generate base mesh once
            torch.manual_seed(seed)
            np.random.seed(seed)
            mesh_list = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
            base_verts, base_faces, _ = mesh_list[0]

            # Baseline metrics
            with torch.no_grad():
                sym0 = symmetry_reward(base_verts).item()
                smo0 = smoothness_reward(base_verts, base_faces).item()
                com0 = compactness_reward(base_verts, base_faces).item()

            if (prompt, seed, "baseline") not in done_keys:
                results.append({
                    "prompt": prompt, "seed": seed, "method": "baseline",
                    "category": cat,
                    "symmetry": sym0, "smoothness": smo0, "compactness": com0,
                    "n_vertices": int(base_verts.shape[0]),
                    "n_faces": int(base_faces.shape[0]),
                })
                done_keys.add((prompt, seed, "baseline"))

            # Laplacian smoothing variants
            for iters, lamb, name in LAPLACIAN_CONFIGS:
                if (prompt, seed, name) not in done_keys:
                    smoothed = laplacian_smooth(base_verts, base_faces,
                                                iterations=iters, lamb=lamb)
                    with torch.no_grad():
                        sym1 = symmetry_reward(smoothed).item()
                        smo1 = smoothness_reward(smoothed, base_faces).item()
                        com1 = compactness_reward(smoothed, base_faces).item()

                    results.append({
                        "prompt": prompt, "seed": seed, "method": name,
                        "category": cat,
                        "symmetry": sym1, "smoothness": smo1, "compactness": com1,
                        "symmetry_baseline": sym0, "smoothness_baseline": smo0,
                        "compactness_baseline": com0,
                        "lap_iters": iters, "lap_lambda": lamb,
                        "n_vertices": int(smoothed.shape[0]),
                        "n_faces": int(base_faces.shape[0]),
                    })
                    done_keys.add((prompt, seed, name))

            # DiffGeoReward variants
            for name, weights in DGR_CONFIGS:
                if (prompt, seed, name) not in done_keys:
                    weights_t = torch.tensor(weights, dtype=torch.float32,
                                             device=DEVICE)
                    refined, _ = refine_with_geo_reward(
                        base_verts, base_faces, weights_t,
                        steps=STEPS, lr=LR
                    )
                    with torch.no_grad():
                        sym1 = symmetry_reward(refined).item()
                        smo1 = smoothness_reward(refined, base_faces).item()
                        com1 = compactness_reward(refined, base_faces).item()

                    results.append({
                        "prompt": prompt, "seed": seed, "method": name,
                        "category": cat,
                        "symmetry": sym1, "smoothness": smo1, "compactness": com1,
                        "symmetry_baseline": sym0, "smoothness_baseline": smo0,
                        "compactness_baseline": com0,
                        "weights": weights,
                        "n_vertices": int(refined.shape[0]),
                        "n_faces": int(base_faces.shape[0]),
                    })
                    done_keys.add((prompt, seed, name))

            n_done = len(results)
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{pi+1}/{total_prompts}] {n_done} records ({elapsed:.0f}s)")
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f)

    with open(checkpoint_path, 'w') as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nTotal: {len(results)} records in {elapsed:.0f}s")

    # Save full results
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # === ANALYSIS ===
    all_methods = ["baseline"] + [n for _, _, n in LAPLACIAN_CONFIGS] + [n for n, _ in DGR_CONFIGS]

    print(f"\n=== OVERALL COMPARISON ===")
    print(f"{'Method':<16s} | {'Symmetry':>10s} | {'Smoothness':>12s} | {'Compactness':>12s} | {'N':>5s}")
    print("-" * 65)
    for method in all_methods:
        data = [r for r in results if r["method"] == method]
        if not data:
            continue
        sym = np.mean([r["symmetry"] for r in data])
        smo = np.mean([r["smoothness"] for r in data])
        com = np.mean([r["compactness"] for r in data])
        print(f"{method:<16s} | {sym:>10.6f} | {smo:>12.8f} | {com:>12.2f} | {len(data):>5d}")

    # Improvement over baseline
    print(f"\n=== IMPROVEMENT OVER BASELINE ===")
    bl_data = {(r["prompt"], r["seed"]): r for r in results if r["method"] == "baseline"}
    for method in all_methods:
        if method == "baseline":
            continue
        m_data = [r for r in results if r["method"] == method]
        sym_imp, smo_imp, com_imp = [], [], []
        for r in m_data:
            key = (r["prompt"], r["seed"])
            if key in bl_data:
                bl = bl_data[key]
                if abs(bl["symmetry"]) > 1e-10:
                    sym_imp.append((r["symmetry"] - bl["symmetry"]) / abs(bl["symmetry"]) * 100)
                if abs(bl["smoothness"]) > 1e-10:
                    smo_imp.append((r["smoothness"] - bl["smoothness"]) / abs(bl["smoothness"]) * 100)
                if abs(bl["compactness"]) > 1e-10:
                    com_imp.append((r["compactness"] - bl["compactness"]) / abs(bl["compactness"]) * 100)
        if sym_imp:
            print(f"  {method}: sym={np.mean(sym_imp):+.1f}%, smo={np.mean(smo_imp):+.1f}%, com={np.mean(com_imp):+.1f}%")

    # Per-category for key methods
    print(f"\n=== SMOOTHNESS BY CATEGORY (key comparison) ===")
    for method in ["baseline", "lap_medium", "lap_strong", "dgr_equal", "dgr_smooth"]:
        for cat in ["symmetry", "smoothness", "compactness"]:
            data = [r for r in results
                    if r["method"] == method and r["category"] == cat]
            if data:
                smo = np.mean([r["smoothness"] for r in data])
                print(f"  {method} / {cat}: smoothness={smo:.8f}")

    # Statistical significance: DGR vs best Laplacian
    print(f"\n=== STATISTICAL TESTS: DGR vs LAPLACIAN ===")
    from scipy import stats as sp
    for dgr_name, _ in DGR_CONFIGS:
        dgr_dict = {(r["prompt"], r["seed"]): r
                    for r in results if r["method"] == dgr_name}
        for _, _, lap_name in LAPLACIAN_CONFIGS:
            lap_dict = {(r["prompt"], r["seed"]): r
                        for r in results if r["method"] == lap_name}
            common = set(dgr_dict.keys()) & set(lap_dict.keys())
            if len(common) < 10:
                continue
            for metric in ["symmetry", "smoothness", "compactness"]:
                dgr_vals = [dgr_dict[k][metric] for k in common]
                lap_vals = [lap_dict[k][metric] for k in common]
                t, p = sp.ttest_rel(dgr_vals, lap_vals)
                d = np.mean(np.array(dgr_vals) - np.array(lap_vals))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {dgr_name} vs {lap_name} ({metric}): diff={d:+.6f}, p={p:.4e} {sig}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
