"""[P0-1 / most critical] Causal test: plane quality -> gradient conflict -> PCGrad benefit.

For 100 meshes sampled from the main benchmark (stratified by category), for each mesh run:

    4 plane estimators x 2 optimizers x 50 steps
  = (fixed_xz, best_of_3, pca_single, multi_start) x (equal_weight, pcgrad) x 50

Per (mesh, plane, step) record:
    - pairwise gradient cosines (sym-smo, sym-com, smo-com)
    - reward values under EACH plane's own symmetry plane

Per (mesh, plane) record:
    - plane_error  = angular deviation of plane normal from multi_start reference
    - sym_init_own = symmetry score at init under this plane
    - final rewards (equal, pcgrad) under this plane

Output: analysis_results/nips_push_causal_plane/all_results.json
        analysis_results/nips_push_causal_plane/per_step_cosines.json

Runtime (V100): ~6 hours for 100 meshes. CPU: ~24 hours (start small with --n-meshes 10 to verify).

Launch:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/nips_push/exp_causal_plane.py
    # Or limit scope:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/nips_push/exp_causal_plane.py \
        --n-meshes 100 --seeds 42 123 456 --mesh-dir results/mesh_validity_objs_baseline_snapshot_2026-04-15
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
DEFAULT_OUT_DIR = "analysis_results/nips_push_causal_plane"
STEPS = 50
LR = 0.005
CATEGORIES = ["symmetry", "smoothness", "compactness"]
EPS = 1e-8


# ---------- Plane estimators ----------

def plane_fixed_xz(v: torch.Tensor, device, dtype):
    n = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    d = torch.tensor(0.0, device=device, dtype=dtype)
    return n, d


def plane_best_of_three(v: torch.Tensor, device, dtype):
    """Pick whichever coordinate plane yields the best symmetry score."""
    best_s, best = -float("inf"), None
    for axis in range(3):
        n = torch.zeros(3, device=device, dtype=dtype)
        n[axis] = 1.0
        d = torch.tensor(0.0, device=device, dtype=dtype)
        with torch.no_grad():
            s = float(symmetry_reward_plane(v.detach(), n, d))
        if s > best_s:
            best_s, best = s, (n, d)
    return best


def plane_pca(v: torch.Tensor, device, dtype):
    vv = v.detach().cpu().numpy()
    c = vv.mean(0)
    cov = np.cov((vv - c).T)
    _, evecs = np.linalg.eigh(cov)
    best_s, best = -float("inf"), None
    for i in range(3):
        n = torch.tensor(evecs[:, i], device=device, dtype=dtype)
        d = torch.tensor(float(np.dot(c, evecs[:, i])), device=device, dtype=dtype)
        with torch.no_grad():
            s = float(symmetry_reward_plane(v.detach(), n, d))
        if s > best_s:
            best_s, best = s, (n, d)
    return best


def plane_multistart(v: torch.Tensor, device, dtype):
    with torch.enable_grad():
        n, d = estimate_symmetry_plane(v.detach())
    return n.to(device=device, dtype=dtype).detach(), d.to(device=device, dtype=dtype).detach()


PLANE_FNS = {
    "fixed_xz": plane_fixed_xz,
    "best_of_3": plane_best_of_three,
    "pca_single": plane_pca,
    "multi_start": plane_multistart,
}


# ---------- Optimizers ----------

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
    """Equal-weight refinement. Returns final vertices + per-step records."""
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
                "g_sym_norm": float(torch.linalg.norm(gs)),
                "g_smo_norm": float(torch.linalg.norm(gm)),
                "g_com_norm": float(torch.linalg.norm(gc)),
            })
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach(), per_step


def refine_pcgrad(v, f, n, d):
    """PCGrad refinement. Returns final vertices."""
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
    """Balanced sample across categories."""
    out = []
    for cat in CATEGORIES:
        cdir = base_dir / cat
        if not cdir.exists():
            continue
        for seed in seeds:
            cands = sorted(glob.glob(str(cdir / f"*_seed{seed}.obj")))
            out.extend([(cat, seed, p) for p in cands])
    # Stratified: aim for n_per_cat prompts per category (seeds implicit)
    by_cat = {c: [x for x in out if x[0] == c] for c in CATEGORIES}
    sampled = []
    rng = random.Random(0)
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])
        sampled.extend(by_cat[c][:n_per_cat])
    return sampled


# ---------- Main loop ----------

def run_one_mesh(path, device, dtype, record_cosines=True):
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

    # Reference plane (multi-start) — defines plane_error
    ref_n, ref_d = plane_multistart(V, device, dtype)
    with torch.no_grad():
        ref_sym_init = float(symmetry_reward_plane(V, ref_n, ref_d))

    adj = _build_face_adjacency(F)
    with torch.no_grad():
        init_sym_ref = ref_sym_init
        init_smo = float(smoothness_reward(V, F, _adj=adj))
        init_com = float(compactness_reward(V, F))

    out.update({
        "n_verts": int(V.shape[0]),
        "n_faces": int(F.shape[0]),
        "ref_sym_init": ref_sym_init,
        "init_smo": init_smo,
        "init_com": init_com,
        "plane_methods": {},
    })

    step_records = {}
    for mname, mfn in PLANE_FNS.items():
        if mname == "multi_start":
            pn, pd = ref_n, ref_d
        else:
            pn, pd = mfn(V, device, dtype)
        with torch.no_grad():
            own_sym_init = float(symmetry_reward_plane(V, pn, pd))

        # Plane error: (1) angular to ref_n, (2) sym gap to ref's init symmetry
        cos_angle = torch.clamp((pn / (pn.norm() + EPS)) @ (ref_n / (ref_n.norm() + EPS)),
                                -1.0, 1.0)
        ang_err = math.degrees(math.acos(min(1.0, abs(float(cos_angle)))))
        sym_gap = abs(own_sym_init - ref_sym_init)

        # Equal-weight refinement (record cosines)
        V_eq, per_step = refine_equal_weight(V, F, pn, pd, record_cosines)
        with torch.no_grad():
            eq_sym_own = float(symmetry_reward_plane(V_eq, pn, pd))
            eq_sym_ref = float(symmetry_reward_plane(V_eq, ref_n, ref_d))
            eq_smo = float(smoothness_reward(V_eq, F, _adj=adj))
            eq_com = float(compactness_reward(V_eq, F))

        # PCGrad refinement (no cosine recording for speed)
        V_pc = refine_pcgrad(V, F, pn, pd)
        with torch.no_grad():
            pc_sym_own = float(symmetry_reward_plane(V_pc, pn, pd))
            pc_sym_ref = float(symmetry_reward_plane(V_pc, ref_n, ref_d))
            pc_smo = float(smoothness_reward(V_pc, F, _adj=adj))
            pc_com = float(compactness_reward(V_pc, F))

        out["plane_methods"][mname] = {
            "own_sym_init": own_sym_init,
            "plane_angular_err_deg": ang_err,
            "plane_sym_gap": sym_gap,
            "normal": pn.detach().cpu().tolist(),
            "offset": float(pd),
            # Final rewards under OWN plane
            "equal_sym_own": eq_sym_own, "pcgrad_sym_own": pc_sym_own,
            "equal_sym_ref": eq_sym_ref, "pcgrad_sym_ref": pc_sym_ref,
            "equal_smo": eq_smo, "pcgrad_smo": pc_smo,
            "equal_com": eq_com, "pcgrad_com": pc_com,
            # Benefit of PCGrad over equal-weight on sym (under own plane)
            "pcgrad_benefit_sym_own": pc_sym_own - eq_sym_own,
            "pcgrad_benefit_sym_ref": pc_sym_ref - eq_sym_ref,
        }
        if per_step:
            # Summary: mean/median cosines over 50 steps
            cs_smo = [r["cos_sym_smo"] for r in per_step]
            cs_com = [r["cos_sym_com"] for r in per_step]
            cm_com = [r["cos_smo_com"] for r in per_step]
            out["plane_methods"][mname].update({
                "mean_cos_sym_smo": float(np.mean(cs_smo)),
                "mean_cos_sym_com": float(np.mean(cs_com)),
                "mean_cos_smo_com": float(np.mean(cm_com)),
                "pct_neg_sym_smo": float(np.mean([x < 0 for x in cs_smo]) * 100),
                "pct_neg_sym_com": float(np.mean([x < 0 for x in cs_com]) * 100),
                "mean_neg_mag_sym_smo": float(np.mean([min(0, x) for x in cs_smo])),
                "mean_neg_mag_sym_com": float(np.mean([min(0, x) for x in cs_com])),
            })
            step_records[mname] = per_step
    return out, step_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", default=None,
                        help="Directory with {symmetry,smoothness,compactness}/*.obj")
    parser.add_argument("--n-meshes", type=int, default=100,
                        help="Total target meshes (approx, split equally across 3 cats)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
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
    print(f"[info] Selected {len(picks)} meshes "
          f"(target {args.n_meshes}, n_per_cat={n_per_cat}) from {base_dir}")
    print(f"[info] Device: {device}")
    print(f"[info] Planes: {list(PLANE_FNS)}  Optimizers: equal_weight, pcgrad  Steps: {STEPS}")

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

    t0 = time.time()
    for idx, (cat, seed, path) in enumerate(picks):
        if path in done:
            continue
        tm0 = time.time()
        print(f"[{idx+1}/{len(picks)}] {cat}/{Path(path).name} ... ", end="", flush=True)
        try:
            rec, steps = run_one_mesh(path, device, dtype, args.record_cosines)
        except Exception as e:
            print(f"ERR {e}")
            continue
        rec["category"] = cat
        rec["seed"] = seed
        results.append(rec)
        if steps:
            step_results[path] = steps
        print(f"{time.time()-tm0:.1f}s  n_v={rec.get('n_verts','?')}")

        # Checkpoint every 5 meshes
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
    print("\nNext: python tools/nips_push/analyze_causal_plane.py")


if __name__ == "__main__":
    main()
