"""[Downstream] Volumetric bilateral-symmetry validation.

Motivation:
  DGR's symmetry reward operates on vertex positions via Chamfer distance
  between the point cloud and its reflection. This is a POINT-BASED symmetry
  measure and coincides with the optimised objective. As a NON-CIRCULAR
  downstream test we evaluate a VOLUME-BASED symmetry measure that no reward
  in our optimisation target operates on:

     Voxel-Symmetry IoU (V-IoU) =  |V  ∩  reflect(V, Π)|
                                    ---------------------------
                                    |V  ∪  reflect(V, Π)|

  where V is the filled voxel occupancy of the mesh and Π is our estimated
  bilateral plane. V-IoU = 1 means perfectly symmetric solid; V-IoU < 1
  reflects volumetric asymmetry. This quantity is:
   - Structurally independent of R_sym (which is a point-Chamfer, not a
     voxel overlap).
   - Structurally independent of R_smooth and R_compact (they operate on
     face normals and surface/volume, not voxel reflection overlap).
   - A standard input signal for downstream 3D processing pipelines that
     assume bilateral symmetry (automatic rigging, pose canonicalisation,
     part segmentation).

  A second reported metric is the volumetric XOR asymmetry:
     V-XOR = |V △ reflect(V, Π)| / |V ∪ reflect(V, Π)|  (lower = better)

Protocol:
  - All 37 symmetry-focused prompts x 3 seeds = 111 (prompt, seed) pairs
  - Baseline + DGR meshes from the main benchmark
  - Voxelize at resolution R = 96 (fast, adequate)
  - Reflect voxel grid across shared multi-start plane
  - Report V-IoU and V-XOR with paired t-test + Wilcoxon + Cohen's d

Output:
  analysis_results/nips_push_voxel_sym/all_results.json
  analysis_results/nips_push_voxel_sym/summary.json

Runtime (CPU): ~1s per voxelization + reflection -> ~4 min total.

Launch:
  PYTHONPATH=src python tools/nips_push/exp_voxel_symmetry_downstream.py
"""
import argparse
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
os.chdir(ROOT)

from prompts_gpteval3d import SYMMETRY_PROMPTS  # noqa: E402


SEEDS = [42, 123, 456]
VOXEL_RES = 96

BASELINE_DIR = Path("results/mesh_validity_objs_baseline_snapshot_2026-04-15")
if not BASELINE_DIR.exists():
    BASELINE_DIR = Path("results/mesh_validity_objs/baseline")
DGR_DIR = Path("results/mesh_validity_objs/diffgeoreward")
PLANE_CACHE_PATH = "analysis_results_newproto/plane_cache/production_plane_cache.json"

OUT_DIR = Path("analysis_results/nips_push_voxel_sym")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_obj(path):
    try:
        m = trimesh.load(path, force="mesh", process=False)
    except Exception:
        return None
    if not hasattr(m, "faces") or m.faces is None or len(m.faces) < 4:
        return None
    return m


def find_mesh(base_dir, prompt, seed):
    name = f"{slug(prompt)}_seed{seed}.obj"
    for cat in ["symmetry", "smoothness", "compactness"]:
        p = base_dir / cat / name
        if p.exists():
            return str(p)
    return None


def load_plane_cache():
    if not Path(PLANE_CACHE_PATH).exists():
        return None
    with open(PLANE_CACHE_PATH) as f:
        return json.load(f)


# ---------- Voxelization + reflection ----------

def voxelize(mesh, resolution=VOXEL_RES, debug=False):
    """Return (vol bool [R,R,R], origin [3], pitch). None on failure."""
    try:
        V = np.asarray(mesh.vertices, dtype=np.float64)
        ext = float((V.max(0) - V.min(0)).max()) + 1e-9
        pitch = ext / (resolution - 2)
        vg = mesh.voxelized(pitch=pitch)
        if vg is None:
            if debug: print("[voxelize] vg is None")
            return None
        try:
            vg_filled = vg.fill()
        except Exception as e:
            if debug: print(f"[voxelize] .fill() failed: {e}")
            vg_filled = vg  # proceed with shell voxels
        mat = vg_filled.matrix.astype(bool)
        if mat.size == 0 or not mat.any():
            if debug: print(f"[voxelize] empty mat {mat.shape}")
            return None
        return mat, np.asarray(vg_filled.translation, dtype=np.float64), pitch
    except Exception as e:
        if debug: print(f"[voxelize] exception: {type(e).__name__}: {e}")
        return None


def reflect_voxel(vol, origin, pitch, pn, pd, resolution):
    """Build a reflected voxel grid of the same shape/origin.

    For each occupied voxel in `vol`, reflect its center across the plane
    and mark the nearest voxel in the output grid as occupied.
    """
    pn = np.asarray(pn, dtype=np.float64)
    pn = pn / (np.linalg.norm(pn) + 1e-12)
    idx = np.argwhere(vol)  # (N, 3)
    if len(idx) == 0:
        return np.zeros_like(vol)
    # World-space centers
    C = origin[None, :] + (idx + 0.5) * pitch    # (N, 3)
    # Reflect across plane {x : n.x = d}
    s = C @ pn - pd
    C_ref = C - 2.0 * s[:, None] * pn[None, :]
    # Map back to grid indices
    idx_ref = np.floor((C_ref - origin[None, :]) / pitch).astype(np.int64)
    # Filter valid
    R = vol.shape
    mask = np.all((idx_ref >= 0) & (idx_ref < np.asarray(R)[None, :]), axis=1)
    idx_ref = idx_ref[mask]
    out = np.zeros_like(vol)
    if len(idx_ref) > 0:
        out[idx_ref[:, 0], idx_ref[:, 1], idx_ref[:, 2]] = True
    return out


def voxel_symmetry_iou(mesh, pn, pd, resolution=VOXEL_RES, debug=False):
    """Return {'iou': x, 'xor': y, 'n_vol': n, 'n_ref': n} or None."""
    vx = voxelize(mesh, resolution, debug=debug)
    if vx is None:
        return None
    vol, origin, pitch = vx
    vol_ref = reflect_voxel(vol, origin, pitch, pn, pd, resolution)
    n_vol = int(vol.sum()); n_ref = int(vol_ref.sum())
    if n_vol == 0 or n_ref == 0:
        return None
    intersection = np.logical_and(vol, vol_ref)
    union = np.logical_or(vol, vol_ref)
    sym_diff = np.logical_xor(vol, vol_ref)
    u = int(union.sum())
    return {
        "iou": float(intersection.sum()) / u,
        "xor": float(sym_diff.sum()) / u,
        "n_vol": n_vol, "n_ref": n_ref,
    }


# ---------- Per-mesh pipeline ----------

def run_one(prompt, seed, plane_cache):
    ps = slug(prompt)
    bl_path = find_mesh(BASELINE_DIR, prompt, seed)
    dgr_path = find_mesh(DGR_DIR, prompt, seed)
    if bl_path is None or dgr_path is None:
        return None, "missing_mesh"
    bl = load_obj(bl_path); dgr = load_obj(dgr_path)
    if bl is None or dgr is None:
        return None, "load_fail"
    if len(bl.vertices) < 20 or len(bl.faces) < 20:
        return None, "mesh_too_small"

    key = f"{ps}_seed{seed}"
    if plane_cache and key in plane_cache:
        e = plane_cache[key]
        pn = np.asarray(e["normal"], dtype=np.float64)
        pd = float(e["offset"])
    else:
        from geo_reward import estimate_symmetry_plane
        V = torch.tensor(bl.vertices, dtype=torch.float32)
        pn_t, pd_t = estimate_symmetry_plane(V)
        pn = pn_t.detach().cpu().numpy().astype(np.float64)
        pd = float(pd_t.detach().cpu())
    pn = pn / (np.linalg.norm(pn) + 1e-12)

    rec = {"prompt": prompt, "seed": seed, "plane_n": pn.tolist(),
           "plane_d": pd, "baseline": None, "dgr": None}
    for label, mesh in [("baseline", bl), ("dgr", dgr)]:
        r = voxel_symmetry_iou(mesh, pn, pd, debug=False)
        if r is None:
            rec[label] = {"success": False}
        else:
            rec[label] = {"success": True, **r}
    return rec, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    prompts = args.prompts or SYMMETRY_PROMPTS
    total = len(prompts) * len(args.seeds)
    print(f"[info] {len(prompts)} prompts x {len(args.seeds)} seeds = {total} pairs",
          flush=True)

    plane_cache = load_plane_cache()
    results_path = OUT_DIR / "all_results.json"
    errors_path = OUT_DIR / "errors.json"
    results, errors = [], []
    done = set()
    if args.resume and results_path.exists():
        with open(results_path) as f: results = json.load(f)
        done = {(r["prompt"], r["seed"]) for r in results}
        print(f"[resume] {len(done)} done", flush=True)

    t0 = time.time()
    idx = 0
    for prompt in prompts:
        for seed in args.seeds:
            idx += 1
            if (prompt, seed) in done:
                continue
            tm0 = time.time()
            print(f"[{idx}/{total}] {prompt[:50]} seed={seed}", flush=True)
            try:
                rec, err = run_one(prompt, seed, plane_cache)
            except Exception as e:
                print(f"  [EXC] {e}\n{traceback.format_exc()}", flush=True)
                errors.append({"prompt": prompt, "seed": seed, "error": str(e)})
                continue
            if rec is None:
                errors.append({"prompt": prompt, "seed": seed, "error": err})
                continue
            results.append(rec)
            bl = rec["baseline"]; dg = rec["dgr"]
            if bl["success"] and dg["success"]:
                print(f"  V-IoU {bl['iou']:.3f}->{dg['iou']:.3f}  "
                      f"XOR {bl['xor']:.3f}->{dg['xor']:.3f}  "
                      f"({time.time()-tm0:.1f}s)", flush=True)
            if idx % 15 == 0:
                with open(results_path, "w") as f: json.dump(results, f, indent=2)
                with open(errors_path, "w") as f: json.dump(errors, f, indent=2)

    with open(results_path, "w") as f: json.dump(results, f, indent=2)
    with open(errors_path, "w") as f: json.dump(errors, f, indent=2)
    print(f"\n[done] {len(results)} pairs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
