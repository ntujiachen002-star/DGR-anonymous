"""Cross-backbone volumetric-symmetry downstream test on TRELLIS meshes.

Takes each cached TRELLIS baseline mesh, re-runs DGR refinement with the
same plane used in the main TRELLIS experiment, and measures V-IoU on both
baseline and refined mesh.

Input:
    results/nips_push_trellis_objs/*.obj                (180 cached baselines)
    analysis_results/nips_push_trellis/all_results.json (planes + metadata)

Output:
    analysis_results/nips_push_voxel_sym_trellis/all_results.json
    analysis_results/nips_push_voxel_sym_trellis/summary.json

Runtime (V100): ~20 min (180 x 6-8s per mesh: simplify + refine + voxelize x2).

Launch:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
        python tools/nips_push/exp_voxel_symmetry_trellis.py
"""
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "nips_push"))
os.chdir(ROOT)

from shape_gen import refine_with_geo_reward  # noqa: E402

# Reuse the voxel-symmetry helpers from the Shap-E script
from exp_voxel_symmetry_downstream import (  # noqa: E402
    voxel_symmetry_iou,
)


TRELLIS_MESH_DIR = Path("results/nips_push_trellis_objs")
TRELLIS_META = Path("analysis_results/nips_push_trellis/all_results.json")
OUT_DIR = Path("analysis_results/nips_push_voxel_sym_trellis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DGR_W = torch.tensor([1/3, 1/3, 1/3], device=DEVICE)
STEPS = 50
LR = 0.005
MAX_FACES = 10_000


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def simplify(m):
    if len(m.faces) <= MAX_FACES:
        return m
    try:
        return m.simplify_quadric_decimation(MAX_FACES)
    except Exception:
        try:
            return m.simplify_quadratic_decimation(MAX_FACES)
        except Exception:
            return m


def run_one(rec):
    prompt, seed = rec["prompt"], rec["seed"]
    obj_cache = TRELLIS_MESH_DIR / f"{slug(prompt)}_s{seed}.obj"
    if not obj_cache.exists():
        return None, "missing_obj"
    try:
        m_raw = trimesh.load(str(obj_cache), force="mesh", process=False)
    except Exception as e:
        return None, f"load_fail: {e}"
    if not hasattr(m_raw, "faces") or len(m_raw.faces) < 4:
        return None, "invalid_mesh"

    # Simplify to <= 10K faces (matches main TRELLIS experiment protocol)
    m_bl = simplify(m_raw)
    if len(m_bl.vertices) < 20:
        return None, "mesh_too_small_after_simplify"

    # Plane from main TRELLIS experiment record
    pn = np.asarray(rec["plane_n"], dtype=np.float64)
    pn = pn / (np.linalg.norm(pn) + 1e-12)
    pd = float(rec["plane_d"])

    # V-IoU on baseline
    r_bl = voxel_symmetry_iou(m_bl, pn, pd, debug=False)
    if r_bl is None:
        return None, "voxelize_fail_baseline"

    # DGR refinement
    V = torch.tensor(m_bl.vertices, dtype=torch.float32, device=DEVICE)
    F = torch.tensor(m_bl.faces, dtype=torch.long, device=DEVICE)
    pn_t = torch.tensor(pn, dtype=torch.float32, device=DEVICE)
    pd_t = torch.tensor(pd, dtype=torch.float32, device=DEVICE)
    try:
        V_ref, _ = refine_with_geo_reward(
            V, F, DGR_W, steps=STEPS, lr=LR,
            sym_normal=pn_t, sym_offset=pd_t,
        )
    except Exception as e:
        return None, f"refine_fail: {e}"

    # Build refined mesh
    m_dgr = trimesh.Trimesh(vertices=V_ref.detach().cpu().numpy(),
                            faces=F.detach().cpu().numpy(), process=False)

    # V-IoU on refined
    r_dgr = voxel_symmetry_iou(m_dgr, pn, pd, debug=False)
    if r_dgr is None:
        return None, "voxelize_fail_refined"

    return {
        "prompt": prompt, "seed": seed, "category": rec["category"],
        "plane_n": pn.tolist(), "plane_d": pd,
        "n_vertices": int(V.shape[0]), "n_faces": int(F.shape[0]),
        "baseline": {"success": True, **r_bl},
        "dgr": {"success": True, **r_dgr},
    }, None


def main():
    if not TRELLIS_META.exists():
        sys.exit(f"Missing {TRELLIS_META}; run exp_trellis_v2.py first.")
    with open(TRELLIS_META) as f:
        records = json.load(f)
    print(f"[info] {len(records)} TRELLIS records", flush=True)
    print(f"[info] device: {DEVICE}", flush=True)

    results_path = OUT_DIR / "all_results.json"
    errors_path = OUT_DIR / "errors.json"
    results, errors = [], []
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done = {(r["prompt"], r["seed"]) for r in results}
        print(f"[resume] {len(results)} done", flush=True)

    t0 = time.time()
    for idx, rec in enumerate(records, 1):
        key = (rec["prompt"], rec["seed"])
        if key in done:
            continue
        tm0 = time.time()
        print(f"[{idx}/{len(records)}] {rec['prompt'][:50]} s{rec['seed']}",
              flush=True)
        try:
            out, err = run_one(rec)
        except Exception as e:
            print(f"  [EXC] {e}\n{traceback.format_exc()}", flush=True)
            errors.append({"prompt": rec["prompt"], "seed": rec["seed"],
                           "error": str(e)})
            continue
        if out is None:
            errors.append({"prompt": rec["prompt"], "seed": rec["seed"],
                           "error": err})
            print(f"  [SKIP] {err}", flush=True)
            continue
        results.append(out)
        bl = out["baseline"]; dg = out["dgr"]
        print(f"  V-IoU {bl['iou']:.3f}->{dg['iou']:.3f}  "
              f"XOR {bl['xor']:.3f}->{dg['xor']:.3f}  "
              f"({time.time()-tm0:.1f}s)", flush=True)
        if idx % 15 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            with open(errors_path, "w") as f:
                json.dump(errors, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)

    # Summary
    from scipy import stats
    valid = [r for r in results
             if r["baseline"]["success"] and r["dgr"]["success"]]
    print(f"\n[done] {len(valid)}/{len(records)} valid in {time.time()-t0:.0f}s",
          flush=True)
    if valid:
        bl_iou = np.array([r["baseline"]["iou"] for r in valid])
        dg_iou = np.array([r["dgr"]["iou"] for r in valid])
        bl_xor = np.array([r["baseline"]["xor"] for r in valid])
        dg_xor = np.array([r["dgr"]["xor"] for r in valid])
        pct_iou = (dg_iou.mean() - bl_iou.mean()) / (abs(bl_iou.mean()) + 1e-12) * 100
        pct_xor = (dg_xor.mean() - bl_xor.mean()) / (abs(bl_xor.mean()) + 1e-12) * 100
        t, p_t = stats.ttest_rel(dg_iou, bl_iou)
        d_iou = (dg_iou - bl_iou).mean() / ((dg_iou - bl_iou).std(ddof=1) + 1e-12)
        wins = int((dg_iou > bl_iou).sum())

        # Per category
        by_cat = {}
        for r in valid:
            by_cat.setdefault(r["category"], []).append(r)

        summary = {
            "n_valid": len(valid),
            "n_total": len(records),
            "overall": {
                "iou_bl": float(bl_iou.mean()), "iou_dgr": float(dg_iou.mean()),
                "iou_pct": float(pct_iou), "iou_p": float(p_t),
                "iou_d": float(d_iou),
                "iou_win_rate": 100.0 * wins / len(valid),
                "xor_bl": float(bl_xor.mean()), "xor_dgr": float(dg_xor.mean()),
                "xor_pct": float(pct_xor),
            },
            "by_category": {},
        }
        for cat, rs in by_cat.items():
            iou_b = np.array([r["baseline"]["iou"] for r in rs])
            iou_d = np.array([r["dgr"]["iou"] for r in rs])
            summary["by_category"][cat] = {
                "n": len(rs),
                "iou_bl": float(iou_b.mean()), "iou_dgr": float(iou_d.mean()),
                "iou_pct": float((iou_d.mean() - iou_b.mean()) /
                                 (abs(iou_b.mean()) + 1e-12) * 100),
            }
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nV-IoU (TRELLIS, n={len(valid)})")
        print(f"  overall: bl={bl_iou.mean():.4f} dgr={dg_iou.mean():.4f}  "
              f"Δ={pct_iou:+.1f}%  p={p_t:.2e}  d={d_iou:+.2f}  "
              f"win={100*wins/len(valid):.1f}%")
        print(f"\nBy category:")
        for cat, rs in by_cat.items():
            d = summary["by_category"][cat]
            print(f"  {cat:<12s} n={d['n']:3d}  "
                  f"IoU {d['iou_bl']:.3f}->{d['iou_dgr']:.3f}  "
                  f"Δ={d['iou_pct']:+.1f}%")


if __name__ == "__main__":
    main()
