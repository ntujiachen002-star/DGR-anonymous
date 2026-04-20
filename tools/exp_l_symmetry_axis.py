"""
Experiment L: Symmetry Axis Sensitivity.
Compare YZ-plane (axis=1), XZ-plane (axis=0), XY-plane (axis=2), and PCA-best-axis.
37 symmetry prompts, seed=42.
GPU required. ~2h on V100.
"""
import os, sys, json, torch, numpy as np, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS

DEVICE = 'cuda:0'
SEED = 42
STEPS = 50
LR = 0.005

# symmetry_reward(vertices, axis=int) uses axis index:
# axis=0 -> mirror across YZ (reflect X)
# axis=1 -> mirror across XZ (reflect Y) — default
# axis=2 -> mirror across XY (reflect Z)

AXIS_CONFIGS = {
    'yz': {'axis': 0, 'desc': 'YZ-plane (mirror X)'},
    'xz': {'axis': 1, 'desc': 'XZ-plane (mirror Y, default)'},
    'xy': {'axis': 2, 'desc': 'XY-plane (mirror Z)'},
}

def pca_best_axis(vertices):
    """Find axis index whose reflection gives best symmetry."""
    v = vertices.detach()
    best_axis = 1
    best_score = float('-inf')
    for ax in [0, 1, 2]:
        score = symmetry_reward(v, axis=ax).item()
        if score > best_score:
            best_score = score
            best_axis = ax
    return best_axis, best_score

def main():
    out_dir = Path("analysis_results/symmetry_axis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)

    results = []
    t0 = time.time()

    for pi, prompt in enumerate(SYMMETRY_PROMPTS):
        print(f"\n[{pi+1}/{len(SYMMETRY_PROMPTS)}] {prompt}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        mesh_list = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        base_verts, base_faces, _ = mesh_list[0]

        # Find PCA-best axis
        pca_ax, pca_score = pca_best_axis(base_verts)
        pca_name = ['yz', 'xz', 'xy'][pca_ax]

        record = {
            "prompt": prompt,
            "n_vertices": int(base_verts.shape[0]),
            "n_faces": int(base_faces.shape[0]),
            "pca_best_axis": pca_name,
            "pca_best_axis_idx": pca_ax,
        }

        # For each axis: baseline score + refine + final scores
        for axis_name, cfg in AXIS_CONFIGS.items():
            ax = cfg['axis']

            # Baseline symmetry
            with torch.no_grad():
                sym_bl = symmetry_reward(base_verts, axis=ax).item()

            # Refine using this axis for symmetry
            # We need custom refinement since refine_with_geo_reward uses sym_axis param
            weights = torch.tensor([0.33, 0.33, 0.34], device=DEVICE)
            refined_verts, history = refine_with_geo_reward(
                base_verts, base_faces, weights,
                steps=STEPS, lr=LR, sym_axis=ax
            )

            # Final metrics
            with torch.no_grad():
                sym_ref = symmetry_reward(refined_verts, axis=ax).item()
                # Cross-eval: also measure on default axis (1=xz)
                sym_ref_default = symmetry_reward(refined_verts, axis=1).item()
                smo_ref = smoothness_reward(refined_verts, base_faces).item()
                com_ref = compactness_reward(refined_verts, base_faces).item()

            sym_imp = (sym_ref - sym_bl) / abs(sym_bl) * 100 if abs(sym_bl) > 1e-10 else 0

            record[f"{axis_name}_sym_baseline"] = sym_bl
            record[f"{axis_name}_sym_refined"] = sym_ref
            record[f"{axis_name}_sym_refined_default"] = sym_ref_default
            record[f"{axis_name}_smo_refined"] = smo_ref
            record[f"{axis_name}_com_refined"] = com_ref
            record[f"{axis_name}_sym_imp"] = sym_imp

            print(f"  {axis_name} (axis={ax}): bl={sym_bl:.6f}, ref={sym_ref:.6f} ({sym_imp:+.1f}%)")

        # Also refine with PCA-best axis
        with torch.no_grad():
            sym_bl_pca = symmetry_reward(base_verts, axis=pca_ax).item()
        weights = torch.tensor([0.33, 0.33, 0.34], device=DEVICE)
        refined_pca, _ = refine_with_geo_reward(
            base_verts, base_faces, weights,
            steps=STEPS, lr=LR, sym_axis=pca_ax
        )
        with torch.no_grad():
            sym_ref_pca = symmetry_reward(refined_pca, axis=pca_ax).item()
            sym_ref_pca_default = symmetry_reward(refined_pca, axis=1).item()
            smo_ref_pca = smoothness_reward(refined_pca, base_faces).item()
            com_ref_pca = compactness_reward(refined_pca, base_faces).item()

        pca_imp = (sym_ref_pca - sym_bl_pca) / abs(sym_bl_pca) * 100 if abs(sym_bl_pca) > 1e-10 else 0
        record["pca_sym_baseline"] = sym_bl_pca
        record["pca_sym_refined"] = sym_ref_pca
        record["pca_sym_refined_default"] = sym_ref_pca_default
        record["pca_smo_refined"] = smo_ref_pca
        record["pca_com_refined"] = com_ref_pca
        record["pca_sym_imp"] = pca_imp
        print(f"  pca ({pca_name}): bl={sym_bl_pca:.6f}, ref={sym_ref_pca:.6f} ({pca_imp:+.1f}%)")

        results.append(record)

        if (pi + 1) % 5 == 0:
            with open(out_dir / "checkpoint.json", 'w') as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - t0

    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n=== SUMMARY ({elapsed:.0f}s) ===")
    all_axes = ['yz', 'xz', 'xy', 'pca']
    print(f"{'Axis':<8s} | {'Mean Sym BL':>12s} | {'Mean Sym Ref':>12s} | {'Mean Imp%':>10s} | {'Mean Smo':>10s} | {'Mean Com':>10s}")
    print("-" * 75)
    for ax in all_axes:
        bl = [r[f"{ax}_sym_baseline"] for r in results]
        ref = [r[f"{ax}_sym_refined"] for r in results]
        imp = [r[f"{ax}_sym_imp"] for r in results]
        smo = [r[f"{ax}_smo_refined"] for r in results]
        com = [r[f"{ax}_com_refined"] for r in results]
        print(f"{ax:<8s} | {np.mean(bl):>12.6f} | {np.mean(ref):>12.6f} | {np.mean(imp):>+9.1f}% | {np.mean(smo):>10.6f} | {np.mean(com):>10.2f}")

    # Cross-eval: measure all on default axis
    print(f"\n=== CROSS-EVAL (all measured on default xz axis) ===")
    for ax in all_axes:
        vals = [r[f"{ax}_sym_refined_default"] for r in results]
        print(f"  Refined with {ax}: mean sym (xz) = {np.mean(vals):.6f}")

    # PCA axis distribution
    pca_dist = {}
    for r in results:
        ax = r["pca_best_axis"]
        pca_dist[ax] = pca_dist.get(ax, 0) + 1
    print(f"\n=== PCA BEST AXIS DISTRIBUTION ===")
    for ax, count in sorted(pca_dist.items()):
        print(f"  {ax}: {count}/{len(results)} ({100*count/len(results):.0f}%)")

    # Paired t-test: default (xz) vs others
    print(f"\n=== PAIRED T-TEST: xz (default) vs others ===")
    from scipy import stats as sp
    xz_vals = [r["xz_sym_refined"] for r in results]
    for ax in ['yz', 'xy', 'pca']:
        other = [r[f"{ax}_sym_refined"] for r in results]
        t, p = sp.ttest_rel(xz_vals, other)
        d = np.mean(np.array(xz_vals) - np.array(other))
        print(f"  xz vs {ax}: diff={d:.6f}, t={t:.3f}, p={p:.4e}")

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
