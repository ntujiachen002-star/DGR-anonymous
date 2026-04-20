"""
Experiment D: Keyword-Match and Oracle baselines.
110 prompts x 3 seeds = 330 runs for each condition.
GPU required. ~3h on V100.
"""
import os, sys, json, torch, numpy as np, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        smoothness_reward, compactness_reward)
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

sys.path.insert(0, os.path.dirname(__file__))
from _plane_protocol import PlaneStore, make_key

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

def keyword_weights(prompt):
    p = prompt.lower()
    sym_kw = ["symmetric", "symmetry", "balanced", "mirror",
              "identical", "bilateral", "uniform", "even"]
    smo_kw = ["smooth", "polished", "sleek", "rounded", "soft",
              "curved", "flowing", "glossy", "refined"]
    com_kw = ["compact", "dense", "solid", "cube", "sphere",
              "chunky", "thick", "squat", "round", "ball"]

    s = sum(1 for w in sym_kw if w in p)
    m = sum(1 for w in smo_kw if w in p)
    c = sum(1 for w in com_kw if w in p)

    if s >= m and s >= c and s > 0:
        return [0.80, 0.10, 0.10]
    elif m >= s and m >= c and m > 0:
        return [0.10, 0.80, 0.10]
    elif c >= s and c >= m and c > 0:
        return [0.10, 0.10, 0.80]
    else:
        return [0.33, 0.33, 0.34]

ORACLE_WEIGHTS = {
    "symmetry": [0.8, 0.0, 0.2],
    "smoothness": [0.0, 0.6, 0.4],
    "compactness": [0.0, 0.0, 1.0],
}

def oracle_weights(prompt):
    cat = PROMPT_CATEGORIES.get(prompt, None)
    if cat and cat in ORACLE_WEIGHTS:
        return ORACLE_WEIGHTS[cat]
    return [0.33, 0.33, 0.34]

def run_condition(condition_name, weight_fn, xm, model, diffusion, out_dir,
                   plane_store=None):
    results = []
    checkpoint_path = out_dir / f"{condition_name}_checkpoint.json"

    done_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["seed"]) for r in results}
        print(f"  Resuming: {len(results)} done")

    total = len(ALL_PROMPTS) * len(SEEDS)
    for pi, prompt in enumerate(ALL_PROMPTS):
        for seed in SEEDS:
            if (prompt, seed) in done_keys:
                continue

            weights = weight_fn(prompt)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            category = PROMPT_CATEGORIES.get(prompt, "unknown")

            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                # generate_mesh returns list of (verts, faces, latent)
                mesh_list = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
                base_verts, base_faces, _ = mesh_list[0]

                # Estimate plane once on the baseline mesh and share it with
                # the refinement + both metric evaluations (paired protocol).
                sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=base_verts)

                # Baseline metrics (under new plane)
                with torch.no_grad():
                    sym0 = symmetry_reward_plane(base_verts, sym_n, sym_d).item()
                    smo0 = smoothness_reward(base_verts, base_faces).item()
                    com0 = compactness_reward(base_verts, base_faces).item()

                # Refine with the cached plane
                refined_verts, history = refine_with_geo_reward(
                    base_verts, base_faces, weights_tensor,
                    steps=STEPS, lr=LR,
                    sym_normal=sym_n, sym_offset=sym_d,
                )

                # Final metrics (under the same plane)
                with torch.no_grad():
                    sym1 = symmetry_reward_plane(refined_verts, sym_n, sym_d).item()
                    smo1 = smoothness_reward(refined_verts, base_faces).item()
                    com1 = compactness_reward(refined_verts, base_faces).item()

                record = {
                    "prompt": prompt,
                    "seed": seed,
                    "method": condition_name,
                    "category": category,
                    "weights": weights,
                    "symmetry": sym1,
                    "smoothness": smo1,
                    "compactness": com1,
                    "symmetry_baseline": sym0,
                    "smoothness_baseline": smo0,
                    "compactness_baseline": com0,
                    "n_vertices": int(refined_verts.shape[0]),
                    "n_faces": int(base_faces.shape[0]),
                }
                results.append(record)
                done_keys.add((prompt, seed))

                n_done = len(results)
                if n_done % 10 == 0:
                    print(f"  [{condition_name}] {n_done}/{total}")

                if n_done % 30 == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump(results, f)

            except Exception as e:
                print(f"  ERROR: {prompt} seed={seed}: {e}")

    with open(checkpoint_path, 'w') as f:
        json.dump(results, f)
    return results

def main():
    out_dir = Path("analysis_results/keyword_oracle")
    out_dir.mkdir(parents=True, exist_ok=True)
    plane_store = PlaneStore.load_or_new(str(out_dir / "plane_cache.json"))

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)
    t0 = time.time()

    print(f"\n=== KEYWORD-MATCH: {len(ALL_PROMPTS)} x {len(SEEDS)} = {len(ALL_PROMPTS)*len(SEEDS)} ===")
    keyword_results = run_condition("keyword", keyword_weights, xm, model, diffusion, out_dir, plane_store=plane_store)

    print(f"\n=== ORACLE: {len(ALL_PROMPTS)} x {len(SEEDS)} = {len(ALL_PROMPTS)*len(SEEDS)} ===")
    oracle_results = run_condition("oracle", oracle_weights, xm, model, diffusion, out_dir, plane_store=plane_store)
    plane_store.save()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    all_results = keyword_results + oracle_results
    with open(out_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n=== SUMMARY ===")
    for cond_name in ["keyword", "oracle"]:
        cond_data = [r for r in all_results if r["method"] == cond_name]
        print(f"\n{cond_name} ({len(cond_data)} records):")
        for cat in ["symmetry", "smoothness", "compactness"]:
            cat_data = [r for r in cond_data if r["category"] == cat]
            if cat_data:
                vals = [r[cat] for r in cat_data]
                bl = [r[f"{cat}_baseline"] for r in cat_data]
                imp = (np.mean(vals) - np.mean(bl)) / abs(np.mean(bl)) * 100
                print(f"  {cat}: {np.mean(vals):.6f} (bl={np.mean(bl):.6f}, imp={imp:+.1f}%)")

    # Keyword classification accuracy
    print("\n=== KEYWORD MISMATCHES ===")
    mismatches = 0
    for prompt in ALL_PROMPTS:
        kw = keyword_weights(prompt)
        cat = PROMPT_CATEGORIES[prompt]
        dom = "symmetry" if kw[0] > 0.5 else ("smoothness" if kw[1] > 0.5 else ("compactness" if kw[2] > 0.5 else "equal"))
        if dom != cat and dom != "equal":
            mismatches += 1
            print(f"  '{prompt}' (cat={cat}) -> predicted {dom}")
        elif dom == "equal":
            mismatches += 1
            print(f"  '{prompt}' (cat={cat}) -> fallback equal")
    print(f"Mismatches: {mismatches}/{len(ALL_PROMPTS)} ({100*mismatches/len(ALL_PROMPTS):.1f}%)")

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
