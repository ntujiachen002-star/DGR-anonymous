"""Generate high-resolution meshes for qualitative comparison figure.

Generates baseline and DiffGeoReward-refined meshes for selected prompts,
saving OBJ files at full resolution (up to 10K faces) for paper figures.

Selects 3 prompts per category (9 total), each with the seed that showed
the strongest improvement in the full experiment.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/generate_qualitative_meshes.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
import trimesh

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shape_gen import load_shap_e, generate_mesh, refine_with_geo_reward, save_mesh
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward


def select_best_cases(n_per_category=3):
    """Select best success cases from full experiment results.

    Criteria: both symmetry and smoothness improve, prefer large total improvement.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(project_root, 'results/full/baseline_all_metrics.json')) as f:
        bl = json.load(f)
    with open(os.path.join(project_root, 'results/full/diffgeoreward_all_metrics.json')) as f:
        dgr = json.load(f)

    bl_map = {(d['prompt'], d['seed']): d for d in bl}
    dgr_map = {(d['prompt'], d['seed']): d for d in dgr}

    # Categorize prompts
    categories = {'symmetry': [], 'smoothness': [], 'compactness': []}

    for key, bm in bl_map.items():
        dm = dgr_map.get(key)
        if not dm:
            continue

        sym_imp = (dm['symmetry'] - bm['symmetry']) / max(abs(bm['symmetry']), 1e-8) * 100
        smo_imp = (dm['smoothness'] - bm['smoothness']) / max(abs(bm['smoothness']), 1e-8) * 100
        com_imp = (dm['compactness'] - bm['compactness']) / max(abs(bm['compactness']), 1e-8) * 100

        # Only cases where sym AND smo improve
        if sym_imp < 10 or smo_imp < 10:
            continue

        prompt = bm['prompt']
        prompt_lower = prompt.lower()

        # Determine category from prompt keywords
        if any(w in prompt_lower for w in ['symmetric', 'balanced', 'mirror']):
            cat = 'symmetry'
        elif any(w in prompt_lower for w in ['smooth', 'polished', 'sleek', 'glossy']):
            cat = 'smoothness'
        elif any(w in prompt_lower for w in ['compact', 'dense', 'solid', 'tight']):
            cat = 'compactness'
        else:
            continue

        categories[cat].append({
            'prompt': prompt,
            'seed': bm['seed'],
            'sym_imp': sym_imp,
            'smo_imp': smo_imp,
            'com_imp': com_imp,
            'total': sym_imp + smo_imp,
        })

    # Select top N per category, preferring unique prompts
    selected = []
    for cat in ['symmetry', 'smoothness', 'compactness']:
        items = categories[cat]
        items.sort(key=lambda x: x['total'], reverse=True)

        seen_prompts = set()
        count = 0
        for item in items:
            if item['prompt'] in seen_prompts:
                continue
            seen_prompts.add(item['prompt'])
            selected.append({**item, 'category': cat})
            count += 1
            if count >= n_per_category:
                break

    return selected


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results", "qualitative_meshes")
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Select cases
    cases = select_best_cases(n_per_category=3)
    print(f"\nSelected {len(cases)} cases for qualitative figure:")
    for c in cases:
        print(f"  [{c['category']:12s}] {c['prompt']:40s} seed={c['seed']} "
              f"sym={c['sym_imp']:+.0f}% smo={c['smo_imp']:+.0f}% com={c['com_imp']:+.0f}%")

    # Save selection
    with open(os.path.join(output_dir, "selected_cases.json"), 'w') as f:
        json.dump(cases, f, indent=2)

    # Load Shap-E
    print("\nLoading Shap-E...")
    xm, model, diffusion = load_shap_e(device=device)

    # Lang2Comp for DiffGeoReward weights (fallback to equal weights if ckpt unavailable)
    use_lang2comp = False
    try:
        from lang2comp import Lang2Comp
        lang2comp = Lang2Comp()
        ckpt_path = os.path.join(project_root, "checkpoints", "lang2comp_best.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state, dict):
                lang2comp.load_state_dict(state)
                lang2comp.eval()
                use_lang2comp = True
                print("Loaded Lang2Comp checkpoint")
    except Exception as e:
        print(f"Lang2Comp not available ({e}), using equal weights")

    if not use_lang2comp:
        print("Using handcrafted equal weights [0.33, 0.33, 0.34]")

    results = []

    for i, case in enumerate(cases):
        prompt = case['prompt']
        seed = case['seed']
        category = case['category']

        print(f"\n[{i+1}/{len(cases)}] {prompt} (seed={seed}, {category})")

        # Generate mesh
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            gen_results = generate_mesh(xm, model, diffusion, prompt, device=device)
            verts, faces, _ = gen_results[0]
        except Exception as e:
            print(f"  ERROR generating mesh: {e}")
            continue

        print(f"  Mesh: V={verts.shape[0]} F={faces.shape[0]}")

        # Save baseline mesh
        slug = prompt.lower().replace(' ', '_').replace('"', '').replace("'", '')
        baseline_path = os.path.join(output_dir, f"{slug}_seed{seed}_baseline.obj")
        save_mesh(verts, faces, baseline_path)

        # Evaluate baseline
        with torch.no_grad():
            base_sym = symmetry_reward(verts, axis=1).item()
            base_smo = smoothness_reward(verts, faces).item()
            base_com = compactness_reward(verts, faces).item()

        # DiffGeoReward refinement
        if use_lang2comp:
            with torch.no_grad():
                weights = lang2comp.predict_weights(prompt)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            weights_tensor = torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32)

        refined_verts, history = refine_with_geo_reward(
            verts, faces, weights_tensor, steps=50, lr=0.005,
        )

        # Save refined mesh
        refined_path = os.path.join(output_dir, f"{slug}_seed{seed}_dgr.obj")
        save_mesh(refined_verts, faces, refined_path)

        # Evaluate refined
        with torch.no_grad():
            ref_sym = symmetry_reward(refined_verts, axis=1).item()
            ref_smo = smoothness_reward(refined_verts, faces).item()
            ref_com = compactness_reward(refined_verts, faces).item()

        sym_pct = (ref_sym - base_sym) / max(abs(base_sym), 1e-8) * 100
        smo_pct = (ref_smo - base_smo) / max(abs(base_smo), 1e-8) * 100
        com_pct = (ref_com - base_com) / max(abs(base_com), 1e-8) * 100

        result = {
            'prompt': prompt,
            'seed': seed,
            'category': category,
            'slug': slug,
            'n_vertices': verts.shape[0],
            'n_faces': faces.shape[0],
            'baseline_path': baseline_path,
            'refined_path': refined_path,
            'baseline': {'symmetry': base_sym, 'smoothness': base_smo, 'compactness': base_com},
            'refined': {'symmetry': ref_sym, 'smoothness': ref_smo, 'compactness': ref_com},
            'improvement': {'symmetry': sym_pct, 'smoothness': smo_pct, 'compactness': com_pct},
        }
        results.append(result)

        print(f"  Baseline: sym={base_sym:.6f} smo={base_smo:.6f} com={base_com:.2f}")
        print(f"  Refined:  sym={ref_sym:.6f} smo={ref_smo:.6f} com={ref_com:.2f}")
        print(f"  Change:   sym={sym_pct:+.1f}% smo={smo_pct:+.1f}% com={com_pct:+.1f}%")

    # Save all results
    with open(os.path.join(output_dir, "qualitative_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} mesh pairs in {output_dir}")
    print(f"Run visualization: python analysis/viz_success_from_generated.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
