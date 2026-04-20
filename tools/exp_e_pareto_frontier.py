"""
Experiment E: Weight Simplex Sweep (Pareto Frontier)
Sweeps 66+ weight configurations on the simplex to map the multi-objective trade-off structure.
GPU required. ~6-8h on V100.
"""
import os, sys, json, torch, numpy as np, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        smoothness_reward, compactness_reward)

sys.path.insert(0, os.path.dirname(__file__))
from _plane_protocol import PlaneStore, make_key, eval_symmetry

DEVICE = 'cuda:0'
SEED = 42
STEPS = 50
LR = 0.005

# 20-prompt representative subset (balanced)
PROMPTS = [
    # Symmetry (7)
    "a symmetric vase", "a perfectly balanced chair", "a symmetric wine glass",
    "a balanced chess piece, a king", "an hourglass shape",
    "a symmetrical temple", "a symmetric butterfly sculpture",
    # Smoothness (7)
    "a smooth organic blob", "a polished sphere", "a smooth river stone",
    "a smooth dolphin", "a sleek sports car body", "a smooth pebble",
    "a polished marble egg",
    # Compactness (6)
    "a compact cube", "a tight ball", "a dense solid shape",
    "a dense rock", "a compact robot", "a dense bowling ball",
]

# Generate simplex grid (step 0.1)
def generate_simplex_grid(step=0.1):
    configs = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            w1 = round(i * step, 2)
            w2 = round(j * step, 2)
            w3 = round(1.0 - w1 - w2, 2)
            configs.append([w1, w2, w3])
    return configs

def main():
    out_dir = Path("analysis_results/simplex_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Weight configs: simplex grid + special points
    configs = generate_simplex_grid(0.1)
    special = [
        [0.33, 0.33, 0.34],       # handcrafted
        [0.814, 0.100, 0.086],     # lang2comp symmetry-typical
        [0.164, 0.614, 0.222],     # lang2comp smoothness-typical
        [0.139, 0.133, 0.728],     # lang2comp compactness-typical
    ]
    # Add special configs if not already present
    existing = set(tuple(c) for c in configs)
    for s in special:
        key = tuple(round(x, 3) for x in s)
        if key not in existing:
            configs.append(s)

    print(f"Total weight configs: {len(configs)}")
    print(f"Total prompts: {len(PROMPTS)}")
    print(f"Total runs: {len(configs) * len(PROMPTS)}")

    # Load Shap-E
    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)

    # Pre-generate all meshes (generate once, refine many times)
    print("\nPre-generating meshes...")
    mesh_cache = {}
    for prompt in PROMPTS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        results = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        verts, faces, _ = results[0]
        mesh_cache[prompt] = (verts, faces)
        print(f"  {prompt}: {verts.shape[0]}v, {faces.shape[0]}f")

    # Estimate and cache one symmetry plane per (prompt, seed) on the baseline.
    # Shared across every weight config for paired protocol compliance.
    plane_store = PlaneStore.load_or_new(str(out_dir / 'plane_cache.json'))
    plane_cache = {}
    print("\nEstimating symmetry planes on baselines...")
    for prompt, (verts, faces) in mesh_cache.items():
        key = make_key(prompt, SEED)
        plane_cache[prompt] = plane_store.get(key, verts=verts)
    plane_store.save()

    # Baseline metrics under the NEW protocol.
    # Skip degenerate baselines (Shap-E sometimes emits point clouds with 0
    # faces; computing smoothness on them crashes _build_face_adjacency).
    print("\nComputing baseline metrics...")
    baseline_metrics = {}
    degenerate_prompts = set()
    for prompt, (verts, faces) in mesh_cache.items():
        if faces.numel() == 0 or verts.numel() == 0:
            print(f"  SKIP degenerate {prompt}: {verts.shape[0]}v, "
                  f"{faces.shape[0] if faces.dim() > 0 else 0}f")
            degenerate_prompts.add(prompt)
            continue
        sym_n, sym_d = plane_cache[prompt]
        try:
            with torch.no_grad():
                sym = symmetry_reward_plane(verts, sym_n, sym_d).item()
                smo = smoothness_reward(verts, faces).item()
                com = compactness_reward(verts, faces).item()
            baseline_metrics[prompt] = {"symmetry": sym, "smoothness": smo, "compactness": com}
        except Exception as e:
            print(f"  ERROR baseline metrics {prompt}: {e}")
            degenerate_prompts.add(prompt)

    print(f"  Valid baselines: {len(baseline_metrics)}/{len(mesh_cache)} "
          f"(skipped {len(degenerate_prompts)} degenerate)")

    # Run simplex sweep
    all_results = []
    valid_prompts = [p for p in PROMPTS if p not in degenerate_prompts]
    total = len(configs) * len(valid_prompts)
    done = 0
    t0 = time.time()

    # Add baseline as a "config"
    for prompt in valid_prompts:
        bm = baseline_metrics[prompt]
        all_results.append({
            "prompt": prompt,
            "weights": [0, 0, 0],
            "method": "baseline",
            "symmetry": bm["symmetry"],
            "smoothness": bm["smoothness"],
            "compactness": bm["compactness"],
        })

    for ci, weights in enumerate(configs):
        w_tensor = torch.tensor(weights, device=DEVICE)
        w_str = f"[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"

        for prompt in valid_prompts:
            verts, faces = mesh_cache[prompt]
            sym_n, sym_d = plane_cache[prompt]

            try:
                refined_verts, history = refine_with_geo_reward(
                    verts, faces, w_tensor, steps=STEPS, lr=LR,
                    sym_normal=sym_n, sym_offset=sym_d,
                )
                with torch.no_grad():
                    sym = symmetry_reward_plane(refined_verts, sym_n, sym_d).item()
                    smo = smoothness_reward(refined_verts, faces).item()
                    com = compactness_reward(refined_verts, faces).item()

                all_results.append({
                    "prompt": prompt,
                    "weights": weights,
                    "method": "simplex",
                    "symmetry": sym,
                    "smoothness": smo,
                    "compactness": com,
                })
            except Exception as e:
                print(f"  ERROR: {prompt} w={w_str}: {e}")
                all_results.append({
                    "prompt": prompt,
                    "weights": weights,
                    "method": "simplex",
                    "symmetry": None,
                    "smoothness": None,
                    "compactness": None,
                    "error": str(e),
                })

            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] config={ci+1}/{len(configs)} w={w_str} | elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

        # Save checkpoint every 5 configs
        if (ci + 1) % 5 == 0:
            with open(out_dir / "all_results_checkpoint.json", 'w') as f:
                json.dump(all_results, f)
            print(f"  Checkpoint saved ({ci+1}/{len(configs)} configs)")

    # Final save
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Compute per-category oracle weights
    print("\n=== ORACLE WEIGHTS ===")
    categories = {
        "symmetry": [p for p in PROMPTS if any(kw in p.lower() for kw in ["symmetric", "balanced", "hourglass"])],
        "smoothness": [p for p in PROMPTS if any(kw in p.lower() for kw in ["smooth", "polished", "sleek"])],
        "compactness": [p for p in PROMPTS if any(kw in p.lower() for kw in ["compact", "dense", "tight"])],
    }

    oracle_weights = {}
    for cat, cat_prompts in categories.items():
        target_metric = cat  # "symmetry", "smoothness", "compactness"
        best_score = -float('inf')
        best_w = None

        for weights in configs:
            scores = []
            for prompt in cat_prompts:
                rows = [r for r in all_results if r["prompt"] == prompt
                        and r["weights"] == weights and r.get(target_metric) is not None]
                if rows:
                    scores.append(rows[0][target_metric])
            if scores:
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_w = weights

        oracle_weights[cat] = {
            "weights": best_w,
            "best_score": float(best_score),
            "n_prompts": len(cat_prompts),
        }
        print(f"  {cat}: w={best_w}, score={best_score:.6f}")

    with open(out_dir / "oracle_weights.json", 'w') as f:
        json.dump(oracle_weights, f, indent=2)

    # Aggregate summary per config
    print("\n=== TOP 10 CONFIGS BY COMBINED SCORE ===")
    config_scores = []
    for weights in configs:
        rows = [r for r in all_results if r["weights"] == weights and r.get("symmetry") is not None]
        if rows:
            sym_avg = np.mean([r["symmetry"] for r in rows])
            smo_avg = np.mean([r["smoothness"] for r in rows])
            com_avg = np.mean([r["compactness"] for r in rows])
            # Normalize to baseline for combined score
            bl_rows = [r for r in all_results if r["method"] == "baseline"]
            bl_sym = np.mean([r["symmetry"] for r in bl_rows])
            bl_smo = np.mean([r["smoothness"] for r in bl_rows])
            bl_com = np.mean([r["compactness"] for r in bl_rows])

            # Improvement ratios (higher = better, since metrics are negative)
            sym_imp = (sym_avg - bl_sym) / abs(bl_sym) if bl_sym != 0 else 0
            smo_imp = (smo_avg - bl_smo) / abs(bl_smo) if bl_smo != 0 else 0
            com_imp = (com_avg - bl_com) / abs(bl_com) if bl_com != 0 else 0

            config_scores.append({
                "weights": weights,
                "sym_avg": sym_avg, "smo_avg": smo_avg, "com_avg": com_avg,
                "sym_imp": sym_imp, "smo_imp": smo_imp, "com_imp": com_imp,
                "combined_imp": sym_imp + smo_imp + com_imp,
            })

    config_scores.sort(key=lambda x: x["combined_imp"], reverse=True)
    for i, cs in enumerate(config_scores[:10]):
        w = cs["weights"]
        print(f"  #{i+1}: w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]  sym_imp={cs['sym_imp']:+.1%}  smo_imp={cs['smo_imp']:+.1%}  com_imp={cs['com_imp']:+.1%}  combined={cs['combined_imp']:+.1%}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
