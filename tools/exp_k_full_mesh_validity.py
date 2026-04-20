"""
Experiment K: Full Mesh Validity Check.
Generate .obj for ALL 110 prompts x 3 seeds x 6 methods,
then run trimesh topology analysis.
GPU required. ~4h on V100.
"""
import os, sys, json, torch, numpy as np, time, re
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

METHOD_WEIGHTS = {
    "baseline": None,
    "handcrafted": [0.33, 0.33, 0.34],
    "sym_only": [1.0, 0.0, 0.0],
    "HNC_only": [0.0, 1.0, 0.0],
    "compact_only": [0.0, 0.0, 1.0],
    "diffgeoreward": "lang2comp",
}

def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def save_obj(vertices, faces, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
    f_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces
    with open(path, 'w') as f:
        for v in v_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in f_np:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def check_mesh_topology(obj_path):
    try:
        import trimesh
        mesh = trimesh.load(str(obj_path), force='mesh')
        areas = trimesh.triangles.area(mesh.triangles)
        return {
            "path": str(obj_path),
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.faces),
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "n_degenerate_faces": int((areas < 1e-8).sum()),
            "degenerate_ratio": float((areas < 1e-8).mean()),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area),
            "euler_number": int(mesh.euler_number),
        }
    except Exception as e:
        return {"path": str(obj_path), "error": str(e)}

def main():
    out_dir = Path("analysis_results/mesh_validity_full")
    obj_dir = Path("results/mesh_validity_objs")
    out_dir.mkdir(parents=True, exist_ok=True)
    plane_store = PlaneStore.load_or_new(str(out_dir / "plane_cache.json"))

    checkpoint_path = out_dir / "generation_checkpoint.json"
    done_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done_keys = set(json.load(f))
        print(f"Resuming: {len(done_keys)} already done")

    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)

    # Lang2Comp for DGR weights
    lang2comp_fn = None
    try:
        from lang2comp import Lang2Comp
        lc = Lang2Comp()
        lc.eval()
        lang2comp_fn = lambda p: list(lc.predict(p)['weights'].values())
        print("Lang2Comp loaded")
    except Exception as e:
        print(f"Lang2Comp unavailable: {e}")
        lang2comp_fn = lambda p: [0.33, 0.33, 0.34]

    t0 = time.time()
    total = len(ALL_PROMPTS) * len(SEEDS) * len(METHOD_WEIGHTS)
    n_gen = len(done_keys)

    for pi, prompt in enumerate(ALL_PROMPTS):
        cat = PROMPT_CATEGORIES[prompt]
        ps = slug(prompt)

        for seed in SEEDS:
            # Generate base mesh once
            torch.manual_seed(seed)
            np.random.seed(seed)
            mesh_list = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
            base_verts, base_faces, _ = mesh_list[0]

            # Estimate plane once per (prompt, seed); shared across all methods
            sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=base_verts)

            for method_name, method_config in METHOD_WEIGHTS.items():
                key = f"{method_name}/{ps}_seed{seed}"
                if key in done_keys:
                    continue

                obj_path = obj_dir / method_name / cat / f"{ps}_seed{seed}.obj"

                try:
                    if method_name == "baseline":
                        save_obj(base_verts, base_faces, obj_path)
                    else:
                        if method_config == "lang2comp":
                            weights = lang2comp_fn(prompt)
                        else:
                            weights = method_config
                        weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
                        refined_verts, _ = refine_with_geo_reward(
                            base_verts, base_faces, weights_t, steps=STEPS, lr=LR,
                            sym_normal=sym_n, sym_offset=sym_d,
                        )
                        save_obj(refined_verts, base_faces, obj_path)

                    done_keys.add(key)
                    n_gen += 1

                    if n_gen % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n_gen}/{total} meshes ({elapsed:.0f}s)")
                        with open(checkpoint_path, 'w') as f:
                            json.dump(sorted(done_keys), f)
                except Exception as e:
                    print(f"  ERROR {key}: {e}")

    with open(checkpoint_path, 'w') as f:
        json.dump(sorted(done_keys), f)
    plane_store.save()

    gen_time = time.time() - t0
    print(f"\nGeneration: {n_gen} meshes in {gen_time:.0f}s")

    # Topology analysis
    print("\n=== TOPOLOGY ANALYSIS ===")
    try:
        import trimesh
    except ImportError:
        os.system("pip install trimesh scipy")

    reports = []
    for method_name in METHOD_WEIGHTS:
        method_dir = obj_dir / method_name
        if not method_dir.exists():
            continue
        obj_files = sorted(method_dir.rglob("*.obj"))
        print(f"  {method_name}: {len(obj_files)} files")
        for op in obj_files:
            r = check_mesh_topology(op)
            r["method"] = method_name
            r["category"] = op.parent.name
            # Extract seed
            seed_match = re.search(r'seed(\d+)', op.name)
            if seed_match:
                r["seed"] = int(seed_match.group(1))
            reports.append(r)

    with open(out_dir / "all_reports.json", 'w') as f:
        json.dump(reports, f, indent=2)

    # Summary
    summary = {}
    for mn in METHOD_WEIGHTS:
        mr = [r for r in reports if r["method"] == mn and "error" not in r]
        if not mr:
            continue
        wt = [r for r in mr if r["is_watertight"] and r.get("volume")]
        summary[mn] = {
            "n": len(mr),
            "watertight_pct": np.mean([r["is_watertight"] for r in mr]) * 100,
            "winding_pct": np.mean([r["is_winding_consistent"] for r in mr]) * 100,
            "mean_degen": float(np.mean([r["degenerate_ratio"] for r in mr])),
            "mean_euler": float(np.mean([r["euler_number"] for r in mr])),
            "n_watertight": len(wt),
            "mean_volume": float(np.mean([r["volume"] for r in wt])) if wt else None,
        }

    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Method':<15s} | {'N':>4s} | {'WT%':>6s} | {'Wind%':>6s} | {'Degen':>8s} | {'Euler':>6s}")
    for mn, s in summary.items():
        print(f"{mn:<15s} | {s['n']:>4d} | {s['watertight_pct']:>5.1f}% | {s['winding_pct']:>5.1f}% | {s['mean_degen']:>7.5f} | {s['mean_euler']:>6.1f}")

    # Volume comparison vs baseline
    print("\n=== VOLUME CHANGE ===")
    bl_vols = {}
    for r in reports:
        if r["method"] == "baseline" and r.get("is_watertight") and r.get("volume"):
            ps = slug(r["path"].split("/")[-1].replace(".obj", ""))
            bl_vols[ps] = r["volume"]

    for mn in METHOD_WEIGHTS:
        if mn == "baseline":
            continue
        ratios = []
        for r in reports:
            if r["method"] == mn and r.get("is_watertight") and r.get("volume"):
                ps = slug(r["path"].split("/")[-1].replace(".obj", ""))
                if ps in bl_vols and bl_vols[ps] > 0:
                    ratios.append(r["volume"] / bl_vols[ps])
        if ratios:
            print(f"  {mn}: vol_ratio={np.mean(ratios):.2f}x +/- {np.std(ratios):.2f} (n={len(ratios)})")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/3600:.1f}h)")

if __name__ == "__main__":
    main()
