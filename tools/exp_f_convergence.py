"""
Experiment F: Convergence Curves (Step Selection Validation)
Runs 200-step optimization recording per-step metrics to validate the 50-step choice.
GPU required. ~2h on V100.
"""
import os, sys, json, torch, numpy as np, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        estimate_symmetry_plane, smoothness_reward,
                        compactness_reward)
from _plane_protocol import PlaneStore, make_key

DEVICE = 'cuda:0'
SEED = 42
MAX_STEPS = 200
LR = 0.005

# 15 prompts: 5 per category (mix of success/failure/boundary)
PROMPTS = [
    # Symmetry (5)
    ("a symmetric vase", "symmetry"),
    ("a symmetric butterfly sculpture", "symmetry"),
    ("an hourglass shape", "symmetry"),
    ("a symmetric wine glass", "symmetry"),
    ("a symmetric trophy", "symmetry"),
    # Smoothness (5)
    ("a smooth river stone", "smoothness"),
    ("a polished marble egg", "smoothness"),
    ("a smooth banana", "smoothness"),
    ("a polished wooden spoon", "smoothness"),
    ("a sleek sports car body", "smoothness"),
    # Compactness (5)
    ("a compact cube", "compactness"),
    ("a dense bowling ball", "compactness"),
    ("a compact robot", "compactness"),
    ("a dense rock", "compactness"),
    ("a compact toolbox", "compactness"),
]

# Two weight conditions
CONDITIONS = {
    "handcrafted": [0.33, 0.33, 0.34],
    "lang2comp_sym": [0.814, 0.100, 0.086],  # typical symmetry-focused
}


def refine_with_logging(vertices, faces, weights, sym_n, sym_d, steps=200, lr=0.005):
    """Refine mesh vertices with per-step metric logging.

    Uses the new multi-start symmetry plane (sym_n, sym_d) for both the
    refinement reward and per-step evaluation.
    """
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    with torch.no_grad():
        sym_init = symmetry_reward_plane(v_opt, sym_n, sym_d).item()
        smo_init = smoothness_reward(v_opt, faces).item()
        com_init = compactness_reward(v_opt, faces).item()

    sym_scale = max(abs(sym_init), 1e-6)
    smo_scale = max(abs(smo_init), 1e-6)
    com_scale = max(abs(com_init), 1e-6)

    # Record step 0 (before optimization)
    records = [{
        "step": 0,
        "symmetry": sym_init,
        "smoothness": smo_init,
        "compactness": com_init,
        "combined_reward": 0.0,
        "grad_norm": 0.0,
        "grad_norm_clipped": 0.0,
    }]

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        sym = symmetry_reward_plane(v_opt, sym_n, sym_d)
        smo = smoothness_reward(v_opt, faces)
        com = compactness_reward(v_opt, faces)

        reward = (weights[0] * sym / sym_scale
                  + weights[1] * smo / smo_scale
                  + weights[2] * com / com_scale)
        loss = -reward
        loss.backward()

        grad_norm = v_opt.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        grad_norm_clipped = v_opt.grad.norm().item()
        optimizer.step()

        records.append({
            "step": step,
            "symmetry": sym.item(),
            "smoothness": smo.item(),
            "compactness": com.item(),
            "combined_reward": reward.item(),
            "grad_norm": grad_norm,
            "grad_norm_clipped": grad_norm_clipped,
        })

    return records


def main():
    out_dir = Path("analysis_results/convergence")
    out_dir.mkdir(parents=True, exist_ok=True)
    plane_store = PlaneStore.load_or_new(str(out_dir / "plane_cache.json"))

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)

    all_results = {}
    t0 = time.time()

    for prompt, category in PROMPTS:
        print(f"\n=== {prompt} ({category}) ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        results = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        verts, faces, _ = results[0]
        print(f"  Mesh: {verts.shape[0]}v, {faces.shape[0]}f")
        if faces.shape[0] == 0 or verts.shape[0] == 0:
            print(f"  SKIP degenerate baseline (point cloud)")
            continue

        # Estimate the symmetry plane once on the baseline mesh and reuse it
        # across all conditions (paired protocol).
        sym_n, sym_d = plane_store.get(make_key(prompt, SEED), verts=verts)
        print(f"  Plane: normal=[{sym_n[0]:.3f},{sym_n[1]:.3f},{sym_n[2]:.3f}], offset={sym_d.item():.4f}")

        for cond_name, weights in CONDITIONS.items():
            w_tensor = torch.tensor(weights, device=DEVICE)
            print(f"  Condition: {cond_name} w={weights}")

            records = refine_with_logging(verts, faces, w_tensor, sym_n, sym_d,
                                          steps=MAX_STEPS, lr=LR)

            key = f"{prompt}__{cond_name}"
            all_results[key] = {
                "prompt": prompt,
                "category": category,
                "condition": cond_name,
                "weights": weights,
                "n_vertices": verts.shape[0],
                "n_faces": faces.shape[0],
                "records": records,
            }

            # Summary: metrics at key steps
            for s in [0, 25, 50, 100, 150, 200]:
                rec = records[s] if s < len(records) else records[-1]
                print(f"    step={s:3d}: sym={rec['symmetry']:.6f} smo={rec['smoothness']:.6f} com={rec['compactness']:.2f} grad={rec['grad_norm']:.2f}")

    plane_store.save()

    # Save all results
    with open(out_dir / "all_convergence.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Compute convergence summary: what fraction of final improvement is achieved at step T?
    print("\n=== CONVERGENCE SUMMARY ===")
    print(f"{'Step':>5} {'Sym % of final':>16} {'Smo % of final':>16} {'Com % of final':>16}")

    for step in [10, 25, 50, 75, 100, 150, 200]:
        sym_fracs, smo_fracs, com_fracs = [], [], []
        for key, data in all_results.items():
            records = data["records"]
            if step >= len(records):
                continue
            r0 = records[0]
            r_final = records[-1]
            r_t = records[step]

            for metric, frac_list in [("symmetry", sym_fracs), ("smoothness", smo_fracs), ("compactness", com_fracs)]:
                total_change = r_final[metric] - r0[metric]
                if abs(total_change) > 1e-10:
                    partial_change = r_t[metric] - r0[metric]
                    frac_list.append(partial_change / total_change)

        if sym_fracs:
            print(f"{step:>5} {np.mean(sym_fracs)*100:>15.1f}% {np.mean(smo_fracs)*100:>15.1f}% {np.mean(com_fracs)*100:>15.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
