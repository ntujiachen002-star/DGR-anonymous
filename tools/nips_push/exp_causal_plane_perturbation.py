"""[P0 / Round-2 fix #2] Controlled plane-perturbation causal experiment.

Strengthens the causal claim from exp_causal_plane.py. The 4-estimator design
there compares planes that differ in MORE than angular error (coordinate-axis
bias, PCA structure dependence, discrete vs continuous search). Round-2 review
noted this cannot cleanly isolate plane error from other estimator-specific
properties.

Here we hold EVERYTHING constant except the angular deviation. For each mesh:
    1. Take multi_start plane (n*, d*) as ground-truth reference.
    2. For each angle in {0, 5, 10, 20, 45, 90} deg:
         - Generate N_DIRS random unit axes perpendicular to n*.
         - Rotate n* by the angle around each axis (Rodrigues, offset d* kept).
       (0 deg uses a single replicate; nonzero angles get N_DIRS replicates.)
    3. For each perturbed plane, run equal-weight AND PCGrad for STEPS steps.

This is a true within-mesh controlled trial: same mesh, same everything, only
the plane angle varies. Supports a mixed-effects regression
    PCGradBenefit ~ log1p(angle_deg) + (1 | mesh_id)
that isolates the causal effect of plane error from between-mesh variation.

Output: analysis_results/nips_push_causal_plane_perturbation/all_results.json
        analysis_results/nips_push_causal_plane_perturbation/per_step_cosines.json

Runtime (V100, n_meshes=100, 6 angles x 2 dirs => ~11 planes/mesh x 2 optim):
    ~100 meshes * 22 optim * ~3s/optim ~= 2 hours.
CPU: ~10x slower. Start small with --n-meshes 5 to verify.

Launch:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \\
        python tools/nips_push/exp_causal_plane_perturbation.py \\
        --n-meshes 100 --angles 0 5 10 20 45 90 --n-dirs 2 \\
        --mesh-dir results/mesh_validity_objs_baseline_snapshot_2026-04-15
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from geo_reward import (  # noqa: E402
    _build_face_adjacency,
    compactness_reward,
    compute_initial_huber_delta,
    estimate_symmetry_plane,
    smoothness_reward,
    symmetry_reward_plane,
)


DEFAULT_BASELINE_DIR_CANDIDATES = [
    "results/mesh_validity_objs_baseline_snapshot_2026-04-15",
    "results/mesh_validity_objs/baseline",
    "ablation_meshes/baseline",
]
DEFAULT_OUT_DIR = "analysis_results/nips_push_causal_plane_perturbation"
STEPS = 50
LR = 0.005
CATEGORIES = ["symmetry", "smoothness", "compactness"]
EPS = 1e-8


# ---------- Plane perturbation ----------

def _random_perp_axis(n: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Return a random unit vector perpendicular to n."""
    n_np = n.detach().cpu().numpy().astype(np.float64)
    n_np = n_np / (np.linalg.norm(n_np) + 1e-12)
    # Sample a random vector, project out n component, renormalise.
    for _ in range(20):
        r = rng.standard_normal(3)
        r = r - (r @ n_np) * n_np
        nr = np.linalg.norm(r)
        if nr > 1e-6:
            return torch.tensor(r / nr, device=n.device, dtype=n.dtype)
    # Fallback: orthogonal to n via cross with world up / right.
    for world in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        r = np.cross(n_np, world)
        nr = np.linalg.norm(r)
        if nr > 1e-6:
            return torch.tensor(r / nr, device=n.device, dtype=n.dtype)
    raise RuntimeError("failed to sample perp axis")


def rotate_plane(n: torch.Tensor, d: torch.Tensor, angle_deg: float,
                 axis: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Rotate plane normal n around unit axis by angle_deg (Rodrigues).

    Offset d is kept unchanged. If axis is perpendicular to n, the rotated
    normal satisfies angle(n', n) = angle_deg exactly.
    """
    if angle_deg == 0.0:
        return n.clone(), d.clone()
    theta = math.radians(angle_deg)
    ct, st = math.cos(theta), math.sin(theta)
    axis = axis / (torch.linalg.norm(axis) + EPS)
    # Rodrigues: n' = ct*n + st*(a x n) + (1-ct)*(a . n) a
    cross = torch.linalg.cross(axis, n)
    dot = (axis * n).sum()
    n_rot = ct * n + st * cross + (1.0 - ct) * dot * axis
    n_rot = n_rot / (torch.linalg.norm(n_rot) + EPS)
    return n_rot.detach(), d.detach().clone()


# ---------- Optimizers (same as exp_causal_plane.py) ----------

def _normalized_rewards(v_opt, faces, n, d, adj, huber_delta, init_scales):
    rs = symmetry_reward_plane(v_opt, n, d)
    rm = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=adj)
    rc = compactness_reward(v_opt, faces)
    return rs / init_scales[0], rm / init_scales[1], rc / init_scales[2], rs, rm, rc


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def refine_equal_weight(v, f, n, d, record_cosines: bool):
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=LR)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)
    with torch.no_grad():
        s0 = float(symmetry_reward_plane(v_opt, n, d))
        m0 = float(smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj))
        c0 = float(compactness_reward(v_opt, f))
    init_scales = (abs(s0) + EPS, abs(m0) + EPS, abs(c0) + EPS)
    w = [1.0 / 3.0] * 3

    per_step = []
    for step in range(STEPS):
        opt.zero_grad()
        rs_n, rm_n, rc_n, _, _, _ = _normalized_rewards(v_opt, f, n, d, adj, huber_delta, init_scales)
        loss = -(w[0] * rs_n + w[1] * rm_n + w[2] * rc_n)
        if record_cosines:
            gs = torch.autograd.grad(rs_n, v_opt, retain_graph=True)[0].flatten()
            gm = torch.autograd.grad(rm_n, v_opt, retain_graph=True)[0].flatten()
            gc = torch.autograd.grad(rc_n, v_opt, retain_graph=True)[0].flatten()
            per_step.append({
                "step": step,
                "cos_sym_smo": _cos(gs, gm),
                "cos_sym_com": _cos(gs, gc),
                "cos_smo_com": _cos(gm, gc),
            })
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach(), per_step


def refine_pcgrad(v, f, n, d):
    v_opt = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=LR)
    adj = _build_face_adjacency(f)
    huber_delta = compute_initial_huber_delta(v_opt, f)
    with torch.no_grad():
        s0 = float(symmetry_reward_plane(v_opt, n, d))
        m0 = float(smoothness_reward(v_opt, f, delta=huber_delta, _adj=adj))
        c0 = float(compactness_reward(v_opt, f))
    init_scales = (abs(s0) + EPS, abs(m0) + EPS, abs(c0) + EPS)
    w = [1.0 / 3.0] * 3

    for _ in range(STEPS):
        opt.zero_grad()
        rs_n, rm_n, rc_n, _, _, _ = _normalized_rewards(v_opt, f, n, d, adj, huber_delta, init_scales)
        gs = torch.autograd.grad(rs_n, v_opt, retain_graph=True)[0]
        gm = torch.autograd.grad(rm_n, v_opt, retain_graph=True)[0]
        gc = torch.autograd.grad(rc_n, v_opt, retain_graph=False)[0]
        grads = [gs, gm, gc]
        mod = [g.clone() for g in grads]
        for i in range(3):
            gi = mod[i].reshape(-1)
            for j in range(3):
                if i == j:
                    continue
                gj = grads[j].reshape(-1)
                dot = gi @ gj
                if dot < 0:
                    gi = gi - dot / (gj @ gj + 1e-12) * gj
            mod[i] = gi.reshape(grads[i].shape)
        v_opt.grad = -sum(w[i] * mod[i] for i in range(3))
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


# ---------- Mesh IO ----------

def load_obj(path: str, device, dtype):
    m = trimesh.load(path, process=False)
    if not hasattr(m, "faces") or m.faces is None or len(m.faces) == 0:
        return None
    v = torch.tensor(np.asarray(m.vertices), dtype=dtype, device=device)
    f = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=device)
    return v, f


def find_baseline_dir(arg_dir):
    if arg_dir is not None:
        p = Path(arg_dir)
        if p.exists():
            return p
        sys.exit(f"--mesh-dir {arg_dir} does not exist")
    for cand in DEFAULT_BASELINE_DIR_CANDIDATES:
        p = Path(cand)
        if p.exists():
            print(f"[info] Using baseline mesh dir: {p}")
            return p
    sys.exit(f"No baseline mesh dir found. Tried: {DEFAULT_BASELINE_DIR_CANDIDATES}")


def collect_meshes(base_dir: Path, seeds, n_per_cat):
    out = []
    for cat in CATEGORIES:
        cdir = base_dir / cat
        if not cdir.exists():
            continue
        for seed in seeds:
            cands = sorted(glob.glob(str(cdir / f"*_seed{seed}.obj")))
            out.extend([(cat, seed, p) for p in cands])
    by_cat = {c: [x for x in out if x[0] == c] for c in CATEGORIES}
    sampled = []
    rng = random.Random(0)
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])
        sampled.extend(by_cat[c][:n_per_cat])
    return sampled


# ---------- Main loop ----------

def run_one_mesh(path, device, dtype, angles, n_dirs, mesh_rng, record_cosines):
    out = {"mesh": path, "skipped": False}
    loaded = load_obj(path, device, dtype)
    if loaded is None:
        out["skipped"] = True
        out["reason"] = "pointcloud_or_empty"
        return out, None
    V, F = loaded
    if V.shape[0] < 10 or F.shape[0] < 10:
        out["skipped"] = True
        out["reason"] = f"too_small V={V.shape[0]} F={F.shape[0]}"
        return out, None

    # Reference plane: multi-start
    with torch.enable_grad():
        ref_n, ref_d = estimate_symmetry_plane(V.detach())
    ref_n = ref_n.to(device=device, dtype=dtype).detach()
    ref_d = ref_d.to(device=device, dtype=dtype).detach()
    with torch.no_grad():
        ref_sym_init = float(symmetry_reward_plane(V, ref_n, ref_d))
        init_smo = float(smoothness_reward(V, F))
        init_com = float(compactness_reward(V, F))

    out.update({
        "n_verts": int(V.shape[0]),
        "n_faces": int(F.shape[0]),
        "ref_sym_init": ref_sym_init,
        "ref_normal": ref_n.detach().cpu().tolist(),
        "ref_offset": float(ref_d),
        "init_smo": init_smo,
        "init_com": init_com,
        "perturbations": [],  # list of dicts
    })

    adj = _build_face_adjacency(F)
    step_records = {}

    for ang in angles:
        # 0 deg uses a single replicate (no direction variation); nonzero uses n_dirs.
        n_reps = 1 if ang == 0 else n_dirs
        for rep in range(n_reps):
            axis = _random_perp_axis(ref_n, mesh_rng)
            pn, pd = rotate_plane(ref_n, ref_d, float(ang), axis)
            # Re-compute actual angular error (sanity: should equal ang).
            cos_angle = torch.clamp(
                (pn / (pn.norm() + EPS)) @ (ref_n / (ref_n.norm() + EPS)), -1.0, 1.0)
            actual_ang = math.degrees(math.acos(min(1.0, abs(float(cos_angle)))))

            with torch.no_grad():
                own_sym_init = float(symmetry_reward_plane(V, pn, pd))

            V_eq, per_step = refine_equal_weight(V, F, pn, pd, record_cosines)
            with torch.no_grad():
                eq_sym_own = float(symmetry_reward_plane(V_eq, pn, pd))
                eq_sym_ref = float(symmetry_reward_plane(V_eq, ref_n, ref_d))
                eq_smo = float(smoothness_reward(V_eq, F, _adj=adj))
                eq_com = float(compactness_reward(V_eq, F))

            V_pc = refine_pcgrad(V, F, pn, pd)
            with torch.no_grad():
                pc_sym_own = float(symmetry_reward_plane(V_pc, pn, pd))
                pc_sym_ref = float(symmetry_reward_plane(V_pc, ref_n, ref_d))
                pc_smo = float(smoothness_reward(V_pc, F, _adj=adj))
                pc_com = float(compactness_reward(V_pc, F))

            entry = {
                "angle_nominal_deg": float(ang),
                "angle_actual_deg": float(actual_ang),
                "axis": axis.detach().cpu().tolist(),
                "normal": pn.detach().cpu().tolist(),
                "offset": float(pd),
                "rep": int(rep),
                "own_sym_init": own_sym_init,
                "equal_sym_own": eq_sym_own, "pcgrad_sym_own": pc_sym_own,
                "equal_sym_ref": eq_sym_ref, "pcgrad_sym_ref": pc_sym_ref,
                "equal_smo": eq_smo, "pcgrad_smo": pc_smo,
                "equal_com": eq_com, "pcgrad_com": pc_com,
                "pcgrad_benefit_sym_own": pc_sym_own - eq_sym_own,
                "pcgrad_benefit_sym_ref": pc_sym_ref - eq_sym_ref,
            }
            if per_step:
                cs_smo = [r["cos_sym_smo"] for r in per_step]
                cs_com = [r["cos_sym_com"] for r in per_step]
                cm_com = [r["cos_smo_com"] for r in per_step]
                entry.update({
                    "mean_cos_sym_smo": float(np.mean(cs_smo)),
                    "mean_cos_sym_com": float(np.mean(cs_com)),
                    "mean_cos_smo_com": float(np.mean(cm_com)),
                    "pct_neg_sym_smo": float(np.mean([x < 0 for x in cs_smo]) * 100),
                    "pct_neg_sym_com": float(np.mean([x < 0 for x in cs_com]) * 100),
                })
                step_records[f"{ang}_{rep}"] = per_step
            out["perturbations"].append(entry)
    return out, step_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", default=None)
    parser.add_argument("--n-meshes", type=int, default=100,
                        help="Total target meshes (split equally across 3 cats)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[0.0, 5.0, 10.0, 20.0, 45.0, 90.0],
                        help="Angular perturbations in degrees")
    parser.add_argument("--n-dirs", type=int, default=2,
                        help="Random directions per nonzero angle (0 deg uses 1)")
    parser.add_argument("--perturb-seed", type=int, default=20260422,
                        help="Seed for perturbation axis sampling (reproducible)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--record-cosines", action="store_true", default=True)
    parser.add_argument("--no-cosines", dest="record_cosines", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    base_dir = find_baseline_dir(args.mesh_dir)
    n_per_cat = max(1, args.n_meshes // len(CATEGORIES))
    picks = collect_meshes(base_dir, args.seeds, n_per_cat)
    n_planes_per_mesh = 1 + (len(args.angles) - 1) * args.n_dirs \
        if 0.0 in args.angles else len(args.angles) * args.n_dirs
    print(f"[info] Selected {len(picks)} meshes from {base_dir}")
    print(f"[info] Device: {device}")
    print(f"[info] Angles: {args.angles}  N_dirs (nonzero): {args.n_dirs}")
    print(f"[info] Planes/mesh: {n_planes_per_mesh}  Optimizers/plane: 2 (equal, pcgrad)")
    print(f"[info] Total optim runs: {len(picks) * n_planes_per_mesh * 2}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "all_results.json"
    step_path = out_dir / "per_step_cosines.json"

    if args.resume and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done = {r["mesh"] for r in results}
        print(f"[info] Resume: {len(done)} meshes already done")
    else:
        results, done = [], set()
    step_results = {}
    if args.resume and step_path.exists():
        with open(step_path) as f:
            step_results = json.load(f)

    # Global deterministic rng; per-mesh rng is derived.
    master_rng = np.random.default_rng(args.perturb_seed)

    t0 = time.time()
    for idx, (cat, seed, path) in enumerate(picks):
        if path in done:
            continue
        # Each mesh gets its own rng, seeded deterministically from master + idx.
        mesh_rng = np.random.default_rng(
            master_rng.integers(0, 2**31 - 1) ^ (idx * 2654435761 & 0xFFFFFFFF))
        tm0 = time.time()
        print(f"[{idx+1}/{len(picks)}] {cat}/{Path(path).name} ... ", end="", flush=True)
        try:
            rec, steps = run_one_mesh(
                path, device, dtype, args.angles, args.n_dirs, mesh_rng,
                args.record_cosines)
        except Exception as e:
            print(f"ERR {e}")
            continue
        rec["category"] = cat
        rec["seed"] = seed
        results.append(rec)
        if steps:
            step_results[path] = steps
        print(f"{time.time()-tm0:.1f}s  n_v={rec.get('n_verts','?')}  "
              f"n_pert={len(rec.get('perturbations', []))}")

        if (idx + 1) % 5 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            with open(step_path, "w") as f:
                json.dump(step_results, f)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(step_path, "w") as f:
        json.dump(step_results, f)
    print(f"\n[done] {len(results)} meshes in {time.time()-t0:.0f}s")
    print(f"  {results_path}")
    print(f"  {step_path}")
    print("\nNext: python tools/nips_push/analyze_causal_plane_perturbation.py")


if __name__ == "__main__":
    main()
