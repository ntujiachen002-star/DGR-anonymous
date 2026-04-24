"""[Downstream] RigNet rigging quality — does DGR's geometric improvement
translate to better automatic rigging?

Runs RigNet (Xu et al. SIGGRAPH 2020) on baseline and DGR-refined meshes and
measures three NON-CIRCULAR skeleton-level metrics (none of which is one of
DGR's optimized rewards):

    M1  Joint Symmetry Error (JSE)    -- predicted joint symmetry under our
                                         estimated bilateral plane
    M2  Bone-Plane Angular Deviation  -- how close bone directions are to the
                                         0/90 axes of the bilateral plane
    M3  Root-Joint Plane Offset       -- how far the root joint is from the
                                         bilateral plane (should be ~0 for
                                         symmetric objects)
    M4  RigNet Coverage Rate          -- did RigNet produce a valid skeleton
                                         at all? (binary per mesh)

This experiment is cited in the paper as direct downstream-task validation
(rigging is the canonical downstream application we motivate DGR for; see
Section 1 citing RigNet [Xu 2020] and ASMR [Hong 2025]).

Protocol:
    - Iterate over the 37 symmetry-focused prompts x 3 seeds = 111 (prompt, seed)
      pairs
    - Load baseline OBJ and DGR-refined OBJ (produced by exp_k_full_mesh_validity.py)
    - Reuse the cached multi-start bilateral plane for metric evaluation
    - Run RigNet.run_pipeline() on each mesh
    - Compute M1/M2/M3/M4

Output:
    analysis_results/nips_push_rignet/all_results.json
    analysis_results/nips_push_rignet/summary.json

Runtime (V100):
    RigNet inference ~5-10s per mesh; 111 pairs x 2 methods x 7s = ~26 min.
    Plus ~1 min RigNet model loading.

Prerequisites:
    git clone https://github.com/zhan-xu/RigNet.git $HOME/RigNet
    pip install torch_geometric binvox_rw rtree
    # download RigNet pretrained checkpoints into $HOME/RigNet/checkpoints/
    # (see upstream README)

Launch:
    CUDA_VISIBLE_DEVICES=0 RIGNET_PATH=$HOME/RigNet \
        PYTHONPATH=src python tools/nips_push/exp_rignet_downstream.py
"""
import argparse
import json
import math
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

from prompts_gpteval3d import (SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS)  # noqa: E402
PROMPT_CATEGORIES = {}
for _p in SYMMETRY_PROMPTS: PROMPT_CATEGORIES[_p] = "symmetry"
for _p in SMOOTHNESS_PROMPTS: PROMPT_CATEGORIES[_p] = "smoothness"
for _p in COMPACTNESS_PROMPTS: PROMPT_CATEGORIES[_p] = "compactness"


# ---------- Configuration ----------

RIGNET_PATH = os.environ.get("RIGNET_PATH", os.path.expanduser("~/RigNet"))
SEEDS = [42, 123, 456]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Baseline + DGR mesh directories (produced by the main benchmark)
BASELINE_DIR = Path("results/mesh_validity_objs_baseline_snapshot_2026-04-15")
if not BASELINE_DIR.exists():
    BASELINE_DIR = Path("results/mesh_validity_objs/baseline")
DGR_DIR = Path("results/mesh_validity_objs/diffgeoreward")
# Plane cache (shared multi-start plane per (prompt, seed))
PLANE_CACHE_PATH = "analysis_results_newproto/plane_cache/production_plane_cache.json"

OUT_DIR = Path("analysis_results/nips_push_rignet")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---------- Mesh IO ----------

def load_obj(path: str):
    try:
        m = trimesh.load(path, force="mesh", process=False)
    except Exception as e:
        print(f"  [load fail] {path}: {e}", flush=True)
        return None
    if not hasattr(m, "faces") or m.faces is None or len(m.faces) < 4:
        return None
    return m


def find_mesh(base_dir: Path, prompt: str, seed: int) -> str:
    """Find the OBJ file for a (prompt, seed). Tries each category subdir."""
    name = f"{slug(prompt)}_seed{seed}.obj"
    for cat in ["symmetry", "smoothness", "compactness"]:
        p = base_dir / cat / name
        if p.exists():
            return str(p)
    return None


def load_plane_cache():
    if not Path(PLANE_CACHE_PATH).exists():
        print(f"[warn] plane cache missing: {PLANE_CACHE_PATH}")
        return None
    with open(PLANE_CACHE_PATH) as f:
        return json.load(f)


# ---------- RigNet wrapper ----------

_RIGNET = {"loaded": False, "models": None}


def load_rignet():
    """Load RigNet models once, cache in module global.

    Wraps upstream quick_start.py's run_pipeline.
    """
    if _RIGNET["loaded"]:
        return _RIGNET["models"]
    if not Path(RIGNET_PATH).exists():
        raise RuntimeError(
            f"RigNet not found at {RIGNET_PATH}. "
            f"Set RIGNET_PATH env var or clone RigNet there."
        )
    sys.path.insert(0, RIGNET_PATH)
    sys.path.insert(0, str(Path(RIGNET_PATH) / "utils"))
    try:
        # Import the pipeline from upstream quick_start.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rignet_quick_start", Path(RIGNET_PATH) / "quick_start.py")
        qs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qs)
        _RIGNET["models"] = qs
        _RIGNET["loaded"] = True
        print("[info] RigNet loaded", flush=True)
        return qs
    except Exception as e:
        raise RuntimeError(f"Failed to import RigNet: {e}")


_RIGNET_TIMEOUT = 150  # seconds per mesh

def _rignet_timeout_handler(signum, frame):
    raise TimeoutError('RigNet inference exceeded ' + str(_RIGNET_TIMEOUT) + 's')

def rignet_predict(mesh: trimesh.Trimesh):
    """Run RigNet inference on a trimesh; return (joints_Nx3, parents_len_N)
    or None on failure.

    Uses upstream RigNet's run_pipeline which writes/reads a few temp files.
    """
    qs = load_rignet()
    import signal as _sig
    _prev = _sig.signal(_sig.SIGALRM, _rignet_timeout_handler)
    _sig.alarm(_RIGNET_TIMEOUT)
    try:
        # Save mesh to temp OBJ because RigNet expects a file path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "mesh.obj")
            mesh.export(obj_path)
            try:
                info = qs.predict_rig(obj_path, device=DEVICE)
            except AttributeError:
                # Older RigNet layout: call run_pipeline directly
                info = qs.run_pipeline(obj_path, device=DEVICE)
        # RigInfo object has .joint_pos (Nx3 np.array) and .hierarchy (parent list)
        if info is None:
            return None
        if hasattr(info, "joint_pos"):
            J = np.asarray(info.joint_pos)
            P = np.asarray(info.hierarchy) if hasattr(info, "hierarchy") else None
        elif isinstance(info, dict):
            J = np.asarray(info["joints"])
            P = np.asarray(info.get("parents"))
        elif isinstance(info, tuple) and len(info) >= 2:
            J = np.asarray(info[0])
            P = np.asarray(info[1])
        else:
            return None
        _sig.alarm(0); _sig.signal(_sig.SIGALRM, _prev)
        if J.ndim != 2 or J.shape[1] != 3 or J.shape[0] < 2:
            return None
        return J, P
    except TimeoutError as _te:
        _sig.alarm(0); _sig.signal(_sig.SIGALRM, _prev)
        print('  [RigNet timeout] ' + str(_te)[:100], flush=True)
        return None
    except Exception as e:
        # Log but don't raise — we report coverage rate as a metric
        print(f"  [RigNet fail] {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


# ---------- Metrics ----------

def normalize_plane(pn, pd):
    pn = np.asarray(pn, dtype=np.float64)
    n = pn / (np.linalg.norm(pn) + 1e-12)
    return n, float(pd)


def reflect_points(P, n, d):
    """Reflect points P (Nx3) across plane {x : n.x = d}."""
    # signed distance
    s = (P @ n - d)
    return P - 2.0 * s[:, None] * n[None, :]


def bbox_diagonal(P):
    return float(np.linalg.norm(P.max(0) - P.min(0)) + 1e-12)


def joint_symmetry_error(J, n, d, mesh_diagonal):
    """M1: mean nearest-mirror distance across joints, normalized by bbox.

    Each joint j is reflected across the plane; we find the closest joint
    in J to the reflected point. The mean over all joints is the JSE.
    """
    Jr = reflect_points(J, n, d)
    # pairwise distances between reflected and original
    d2 = ((Jr[:, None, :] - J[None, :, :]) ** 2).sum(-1)
    min_d = np.sqrt(d2.min(1))
    return float(min_d.mean() / mesh_diagonal)


def bone_plane_angular_dev(J, P, n):
    """M2: mean angular deviation of bones from {parallel, perpendicular} to plane.

    For each bone, compute angle between bone direction and plane normal. For
    symmetric objects, bones should be either parallel (angle 90 deg) or
    perpendicular (angle 0 deg) to the plane. We report the mean of
    min(theta, pi/2 - theta) -- distance to nearest canonical direction.

    If P is unavailable, use nearest-neighbor chain as a fallback.
    """
    if P is None or len(P) == 0 or P.ndim != 1 or len(P) != J.shape[0]:
        # Fallback: connect each joint to its nearest neighbor
        d2 = ((J[:, None, :] - J[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        P = d2.argmin(1)
    bones = []
    for i, par in enumerate(P):
        par = int(par)
        if par < 0 or par >= J.shape[0] or par == i:
            continue
        v = J[i] - J[par]
        nrm = np.linalg.norm(v)
        if nrm < 1e-8:
            continue
        v = v / nrm
        theta = math.acos(abs(float(v @ n)))  # angle to normal, in [0, pi/2]
        # distance to nearest canonical: min(theta, pi/2 - theta)
        bones.append(min(theta, math.pi / 2 - theta))
    if not bones:
        return None
    return float(np.degrees(np.mean(bones)))


def root_joint_offset(J, P, n, d, mesh_diagonal):
    """M3: distance from root joint to plane, normalized by bbox.

    Root is joint with no parent (parent == -1 or self) or joint 0 as fallback.
    """
    root_idx = 0
    if P is not None and P.ndim == 1 and len(P) == J.shape[0]:
        for i, par in enumerate(P):
            if int(par) < 0 or int(par) == i:
                root_idx = i
                break
    dist = abs(float(J[root_idx] @ n - d)) / (np.linalg.norm(n) + 1e-12)
    return float(dist / mesh_diagonal)


# ---------- Per-mesh pipeline ----------

def run_one(prompt: str, seed: int, plane_cache: dict):
    ps = slug(prompt)
    bl_path = find_mesh(BASELINE_DIR, prompt, seed)
    dgr_path = find_mesh(DGR_DIR, prompt, seed)
    if bl_path is None or dgr_path is None:
        return None, f"missing_mesh (bl={bool(bl_path)} dgr={bool(dgr_path)})"

    bl = load_obj(bl_path); dgr = load_obj(dgr_path)
    if bl is None or dgr is None:
        return None, "load_fail"
    if len(bl.vertices) < 20 or len(bl.faces) < 20:
        return None, "mesh_too_small_baseline"

    # Plane: prefer cache, else quick re-estimate
    cat_key = PROMPT_CATEGORIES.get(prompt, "symmetry")
    key = f"{cat_key}/{ps}_seed{seed}.obj"
    if plane_cache and key in plane_cache:
        e = plane_cache[key]
        pn, pd = normalize_plane(e["normal"], e["offset"])
    else:
        # fallback: re-estimate on baseline mesh
        from geo_reward import estimate_symmetry_plane
        V = torch.tensor(bl.vertices, dtype=torch.float32)
        pn_t, pd_t = estimate_symmetry_plane(V)
        pn, pd = normalize_plane(pn_t.detach().cpu().numpy(),
                                 float(pd_t.detach().cpu()))

    result = {"prompt": prompt, "seed": seed, "plane_n": pn.tolist(),
              "plane_d": pd, "bl_path": bl_path, "dgr_path": dgr_path,
              "baseline": None, "dgr": None}

    for label, mesh in [("baseline", bl), ("dgr", dgr)]:
        t0 = time.time()
        out = rignet_predict(mesh)
        rt = time.time() - t0
        if out is None:
            result[label] = {"success": False, "runtime_s": rt}
            continue
        J, P = out
        diag = bbox_diagonal(np.asarray(mesh.vertices))
        m1 = joint_symmetry_error(J, pn, pd, diag)
        m2 = bone_plane_angular_dev(J, P, pn)
        m3 = root_joint_offset(J, P, pn, pd, diag)
        result[label] = {
            "success": True, "n_joints": int(J.shape[0]),
            "JSE": m1, "angular_dev_deg": m2, "root_offset": m3,
            "runtime_s": rt,
        }
    return result, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Override prompt list; default is all 37 symmetry prompts")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    prompts = args.prompts or SYMMETRY_PROMPTS
    print(f"[info] {len(prompts)} prompts x {len(args.seeds)} seeds = "
          f"{len(prompts)*len(args.seeds)} pairs", flush=True)

    plane_cache = load_plane_cache()
    results_path = OUT_DIR / "all_results.json"
    errors_path = OUT_DIR / "errors.json"
    results, errors = [], []
    done = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done = {(r["prompt"], r["seed"]) for r in results}
        print(f"[resume] {len(done)} already done", flush=True)

    # Pre-load RigNet
    try:
        load_rignet()
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        print("\nRigNet install steps (if missing):")
        print("  git clone https://github.com/zhan-xu/RigNet.git $HOME/RigNet")
        print("  pip install torch_geometric==2.0.4 binvox_rw rtree")
        print("  # download pretrained checkpoints per RigNet README")
        sys.exit(1)

    t0 = time.time()
    idx = 0
    total = len(prompts) * len(args.seeds)
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
                print(f"  [SKIP] {err}", flush=True)
                continue
            results.append(rec)
            bl = rec["baseline"]; dg = rec["dgr"]
            bl_s = f"cov={'Y' if bl['success'] else 'N'}"
            dg_s = f"cov={'Y' if dg['success'] else 'N'}"
            if bl["success"] and dg["success"]:
                bl_s += f" JSE={bl['JSE']:.4f}"
                dg_s += f" JSE={dg['JSE']:.4f}"
            print(f"  bl: {bl_s}   dgr: {dg_s}   ({time.time()-tm0:.1f}s)",
                  flush=True)
            if idx % 10 == 0:
                with open(results_path, "w") as f: json.dump(results, f, indent=2)
                with open(errors_path, "w") as f: json.dump(errors, f, indent=2)

    with open(results_path, "w") as f: json.dump(results, f, indent=2)
    with open(errors_path, "w") as f: json.dump(errors, f, indent=2)
    print(f"\n[done] {len(results)} pairs in {time.time()-t0:.0f}s. "
          f"Run analyze_rignet.py for summary.", flush=True)


if __name__ == "__main__":
    main()
