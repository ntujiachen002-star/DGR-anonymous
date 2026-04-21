"""[P1-2] TRELLIS backbone benchmark — v2 using new multi-start plane protocol.

Fixes the original exp_trellis_backbone.py which:
  - Called the legacy axis-based symmetry_reward (broken for our protocol)
  - Crashed on CUDA tensors in mesh output

Pipeline: text -> TRELLIS-text-large -> mesh (simplified to <=10K faces)
         -> multi-start plane estimation -> DGR refinement (equal weights)
         -> paired scoring under the same plane.

Output: analysis_results/nips_push_trellis/{all_results,summary}.json
Runtime: ~2h on V100 (60 prompts x 3 seeds x [text-to-mesh + DGR]).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

# Must set env BEFORE importing TRELLIS
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from geo_reward import (  # noqa: E402
    compactness_reward,
    estimate_symmetry_plane,
    smoothness_reward,
    symmetry_reward_plane,
)
from shape_gen import refine_with_geo_reward  # noqa: E402
from prompts_gpteval3d import (  # noqa: E402
    SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS,
)


TRELLIS_PATH = os.environ.get("TRELLIS_PATH", os.path.expanduser("~/TRELLIS"))
TRELLIS_MODEL = "microsoft/TRELLIS-text-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
STEPS = 50
LR = 0.005
N_PER_CAT = 20  # 20 prompts per cat x 3 seeds x 3 cats = 180 runs
DGR_W = torch.tensor([1/3, 1/3, 1/3], device=DEVICE)
MAX_FACES = 10_000

OUT_DIR = Path("analysis_results/nips_push_trellis")
MESH_DIR = Path("results/nips_push_trellis_objs")
for d in [OUT_DIR, MESH_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------- TRELLIS ----------

def load_trellis():
    sys.path.insert(0, TRELLIS_PATH)
    from trellis.pipelines import TrellisTextTo3DPipeline
    print(f"[info] loading TRELLIS {TRELLIS_MODEL} ...", flush=True)
    pipeline = TrellisTextTo3DPipeline.from_pretrained(TRELLIS_MODEL)
    pipeline.cuda()
    print("[info] TRELLIS ready", flush=True)
    return pipeline


def trellis_generate(pipeline, prompt, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        outputs = pipeline.run(prompt, seed=seed,
                                formats=["mesh"],
                                sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
                                slat_sampler_params={"steps": 25, "cfg_strength": 7.5})
    except Exception as e:
        print(f"    [GEN FAIL] {e}", flush=True)
        return None
    # Extract mesh
    meshes = outputs.get("mesh") if isinstance(outputs, dict) else getattr(outputs, "mesh", None)
    if meshes is None:
        return None
    m = meshes[0] if isinstance(meshes, list) else meshes
    try:
        V = to_np(m.vertices)
        F = to_np(m.faces)
    except Exception as e:
        print(f"    [MESH FAIL] {e}", flush=True)
        return None
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def simplify(mesh, max_faces=MAX_FACES):
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(max_faces)
    except Exception:
        try:
            return mesh.simplify_quadratic_decimation(max_faces)
        except Exception:
            return mesh


# ---------- Scoring ----------

def score(V, F, pn, pd):
    with torch.no_grad():
        return {
            "sym": float(symmetry_reward_plane(V, pn, pd)),
            "hnc": float(smoothness_reward(V, F)),
            "com": float(compactness_reward(V, F)),
        }


def run_one(pipeline, prompt, category, seed):
    obj_cache = MESH_DIR / f"{slug(prompt)}_s{seed}.obj"
    if obj_cache.exists():
        try:
            m = trimesh.load(str(obj_cache), process=False)
            print(f"  cached {len(m.vertices)}v {len(m.faces)}f", flush=True)
        except Exception:
            m = None
    else:
        m = None

    if m is None or not hasattr(m, "faces") or len(m.faces) < 4:
        t0 = time.time()
        m = trellis_generate(pipeline, prompt, seed)
        if m is None:
            return None, "gen_fail"
        print(f"  gen {len(m.vertices)}v {len(m.faces)}f in {time.time()-t0:.1f}s", flush=True)
        try:
            m.export(str(obj_cache))
        except Exception:
            pass

    # Simplify
    if len(m.faces) > MAX_FACES:
        m = simplify(m, MAX_FACES)
        print(f"  simplified to {len(m.vertices)}v {len(m.faces)}f", flush=True)

    if len(m.vertices) < 10 or len(m.faces) < 10:
        return None, "mesh_too_small"

    V = torch.tensor(m.vertices, dtype=torch.float32, device=DEVICE)
    F = torch.tensor(m.faces, dtype=torch.long, device=DEVICE)

    # Multi-start plane
    try:
        pn, pd = estimate_symmetry_plane(V)
    except Exception as e:
        return None, f"plane_fail: {e}"

    bl = score(V, F, pn, pd)

    # DGR refine
    t0 = time.time()
    try:
        V_ref, _ = refine_with_geo_reward(V, F, DGR_W, steps=STEPS, lr=LR,
                                          sym_normal=pn, sym_offset=pd)
    except Exception as e:
        return None, f"refine_fail: {e}"
    ref_time = time.time() - t0

    rf = score(V_ref, F, pn, pd)

    return {
        "prompt": prompt, "seed": seed, "category": category,
        "n_vertices": int(V.shape[0]), "n_faces": int(F.shape[0]),
        "plane_n": pn.detach().cpu().tolist(), "plane_d": float(pd),
        "bl_sym": bl["sym"], "bl_hnc": bl["hnc"], "bl_com": bl["com"],
        "rf_sym": rf["sym"], "rf_hnc": rf["hnc"], "rf_com": rf["com"],
        "refine_time": ref_time,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cat", type=int, default=N_PER_CAT)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    prompts = []
    for p in SYMMETRY_PROMPTS[: args.n_per_cat]:
        prompts.append((p, "symmetry"))
    for p in SMOOTHNESS_PROMPTS[: args.n_per_cat]:
        prompts.append((p, "smoothness"))
    for p in COMPACTNESS_PROMPTS[: args.n_per_cat]:
        prompts.append((p, "compactness"))
    total = len(prompts) * len(SEEDS)
    print(f"=== TRELLIS backbone ===\nprompts={len(prompts)} seeds={len(SEEDS)} total={total}", flush=True)

    results_path = OUT_DIR / "all_results.json"
    errors_path = OUT_DIR / "errors.json"
    results, errors = [], []
    done = set()
    if args.resume and results_path.exists():
        with open(results_path) as f: results = json.load(f)
        done = {(r["prompt"], r["seed"]) for r in results}
        print(f"Resume: {len(results)} done", flush=True)

    pipeline = load_trellis()

    t0 = time.time()
    idx = 0
    for prompt, cat in prompts:
        for seed in SEEDS:
            idx += 1
            if (prompt, seed) in done:
                continue
            tm0 = time.time()
            print(f"[{idx}/{total}] {cat[:8]}  {prompt[:50]}  seed={seed}", flush=True)
            rec, err = run_one(pipeline, prompt, cat, seed)
            if rec is None:
                errors.append({"prompt": prompt, "seed": seed, "category": cat, "error": err})
                print(f"  [SKIP] {err}  ({time.time()-tm0:.1f}s)", flush=True)
                continue
            results.append(rec)
            print(f"  sym {rec['bl_sym']:+.4f}->{rec['rf_sym']:+.4f} hnc "
                  f"{rec['bl_hnc']:+.4f}->{rec['rf_hnc']:+.4f} com "
                  f"{rec['bl_com']:+.2f}->{rec['rf_com']:+.2f} ({time.time()-tm0:.1f}s)",
                  flush=True)
            if idx % 5 == 0:
                with open(results_path, "w") as f: json.dump(results, f, indent=2)
                with open(errors_path, "w") as f: json.dump(errors, f, indent=2)

    with open(results_path, "w") as f: json.dump(results, f, indent=2)
    with open(errors_path, "w") as f: json.dump(errors, f, indent=2)
    print(f"\n[done] {len(results)}/{total} valid  ({time.time()-t0:.0f}s)", flush=True)

    # Summary
    if results:
        by_cat = {}
        for cat in ["symmetry", "smoothness", "compactness", "OVERALL"]:
            rs = results if cat == "OVERALL" else [r for r in results if r["category"] == cat]
            if not rs:
                by_cat[cat] = {"n": 0}; continue
            d = {"n": len(rs)}
            for k in ["sym", "hnc", "com"]:
                bl = np.array([r[f"bl_{k}"] for r in rs])
                rf = np.array([r[f"rf_{k}"] for r in rs])
                d[f"{k}_bl_mean"] = float(bl.mean())
                d[f"{k}_rf_mean"] = float(rf.mean())
                d[f"{k}_pct"] = float((rf.mean() - bl.mean()) / (abs(bl.mean()) + 1e-8) * 100)
            by_cat[cat] = d
        with open(OUT_DIR / "summary.json", "w") as f: json.dump(by_cat, f, indent=2)
        print("\nPer-category:")
        print(f"{'Category':<16}{'n':<6}{'sym%':<10}{'hnc%':<10}{'com%':<10}")
        for cat, d in by_cat.items():
            if d["n"] == 0: continue
            print(f"{cat:<16}{d['n']:<6}{d['sym_pct']:<+10.1f}{d['hnc_pct']:<+10.1f}{d['com_pct']:<+10.1f}")


if __name__ == "__main__":
    main()
