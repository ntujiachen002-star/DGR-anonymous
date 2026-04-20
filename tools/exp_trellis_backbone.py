"""
Exp: TRELLIS Backbone — stronger backbone generalization test.
NeurIPS rigor: 3 seeds × 20 prompts per category (60 total) = 180 pairs.

TRELLIS (Xiang et al., CVPR 2025 Spotlight) is the strongest open-source
text-to-3D model available. Testing DiffGeoReward on TRELLIS output
demonstrates generalization to high-quality backbones.

Pipeline: text → TRELLIS-text-large (1.1B) → mesh → [DGR refinement]

Methods:
  trellis_baseline — raw TRELLIS output (no refinement)
  trellis_dgr      — TRELLIS + DiffGeoReward (50 Adam steps, lr=5e-3)

Output: analysis_results/trellis_backbone/all_results.json
        analysis_results/trellis_backbone/stats.json

Hardware: V100 32GB (TRELLIS-text-large fits in ~20GB)
Estimated time: ~2-3h (60 text-to-3D inferences + 60 DGR refinements × 3 seeds)

Setup:
    1. git clone https://github.com/microsoft/TRELLIS.git ~/TRELLIS
    2. cd ~/TRELLIS && pip install -r requirements.txt
    3. pip install trimesh
    4. Run this script from DiffGeoReward root:
       CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/exp_trellis_backbone.py
"""

import os, sys, json, time, gc, re
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS     = [42, 123, 456]
STEPS     = 50
LR        = 0.005
DGR_W     = [1/3, 1/3, 1/3]       # equal weights

TRELLIS_PATH = os.path.expanduser('~/autodl-tmp/TRELLIS')
TRELLIS_MODEL = 'microsoft/TRELLIS-text-large'  # 1.1B, fits V100 32GB

# 20 per category = 60 prompts total
N_PER_CAT = 20
PROMPTS_SYM = SYMMETRY_PROMPTS[:N_PER_CAT]
PROMPTS_SMO = SMOOTHNESS_PROMPTS[:N_PER_CAT]
PROMPTS_COM = COMPACTNESS_PROMPTS[:N_PER_CAT]
ALL_PROMPTS = PROMPTS_SYM + PROMPTS_SMO + PROMPTS_COM
PROMPT_CAT  = {p: "symmetry"    for p in PROMPTS_SYM}
PROMPT_CAT |= {p: "curvature_regularity" for p in PROMPTS_SMO}
PROMPT_CAT |= {p: "compactness" for p in PROMPTS_COM}

OUT_DIR = Path("analysis_results/trellis_backbone")
OBJ_DIR = Path("results/trellis_objs")
for d in [OUT_DIR, OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ["symmetry", "smoothness", "compactness"]
CHECKPOINT = OUT_DIR / "checkpoint.json"


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ── TRELLIS loading & inference ───────────────────────────────────────────────

def load_trellis():
    """Load TRELLIS text-to-3D pipeline."""
    sys.path.insert(0, TRELLIS_PATH)
    try:
        from trellis.pipelines import TrellisTextTo3DPipeline
        print(f"Loading TRELLIS ({TRELLIS_MODEL})...")
        pipeline = TrellisTextTo3DPipeline.from_pretrained(TRELLIS_MODEL)
        pipeline.cuda()
        print("  TRELLIS ready.")
        return pipeline
    except ImportError:
        print(f"ERROR: TRELLIS not found at {TRELLIS_PATH}")
        print("  Please clone: git clone https://github.com/microsoft/TRELLIS.git ~/TRELLIS")
        print("  And install:  cd ~/TRELLIS && pip install -r requirements.txt")
        sys.exit(1)


def trellis_text_to_mesh(pipeline, prompt, seed):
    """Generate mesh from text using TRELLIS. Returns trimesh.Trimesh or None."""
    import trimesh
    try:
        outputs = pipeline.run(prompt, seed=seed)

        # Extract mesh from TRELLIS outputs
        # TRELLIS outputs contain gaussians, radiance_field, and mesh
        if hasattr(outputs, 'mesh') and outputs.mesh is not None:
            mesh = outputs.mesh[0] if isinstance(outputs.mesh, list) else outputs.mesh
        elif isinstance(outputs, dict) and 'mesh' in outputs:
            mesh = outputs['mesh'][0] if isinstance(outputs['mesh'], list) else outputs['mesh']
        else:
            # Try to extract mesh via postprocessing
            try:
                mesh = pipeline.extract_mesh(outputs)
            except:
                print(f"    Cannot extract mesh from TRELLIS output")
                return None

        # Convert to trimesh if needed
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            return trimesh.Trimesh(vertices=vertices, faces=faces)

        # If mesh is already a trimesh object
        if isinstance(mesh, trimesh.Trimesh):
            return mesh

        print(f"    Unknown mesh format: {type(mesh)}")
        return None

    except Exception as e:
        print(f"    TRELLIS failed: {e}")
        return None


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_mesh(mesh):
    """Compute symmetry, smoothness, compactness for a trimesh."""
    import trimesh
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=DEVICE)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=DEVICE)

    sym = symmetry_reward(vertices, faces).item()
    smo = smoothness_reward(vertices, faces).item()
    com = compactness_reward(vertices, faces).item()
    return {"symmetry": sym, "smoothness": smo, "compactness": com}


def simplify_mesh(mesh, max_faces=10000):
    """Simplify mesh to max_faces using quadric decimation."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(max_faces)
        return simplified
    except:
        return mesh


def refine_mesh(mesh, weights=DGR_W, steps=STEPS, lr=LR):
    """Apply DiffGeoReward refinement to a mesh."""
    import trimesh
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=DEVICE, requires_grad=True)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=DEVICE)

    refined_verts = refine_with_geo_reward(
        vertices, faces, weights=weights, steps=steps, lr=lr
    )

    return trimesh.Trimesh(
        vertices=refined_verts.detach().cpu().numpy(),
        faces=mesh.faces
    )


# ── Main experiment loop ──────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT.exists():
        return json.load(open(CHECKPOINT))
    return {"completed": [], "results": []}


def save_checkpoint(state):
    json.dump(state, open(CHECKPOINT, "w"), indent=2)


def main():
    print("=" * 60)
    print("TRELLIS Backbone Experiment")
    print(f"  {len(ALL_PROMPTS)} prompts × {len(SEEDS)} seeds = {len(ALL_PROMPTS) * len(SEEDS)} pairs")
    print(f"  Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    state = load_checkpoint()
    results = state["results"]
    completed = set(tuple(x) for x in state["completed"])

    # Load TRELLIS
    pipeline = load_trellis()

    total = len(ALL_PROMPTS) * len(SEEDS)
    done = len(completed)

    for prompt in ALL_PROMPTS:
        for seed in SEEDS:
            key = (prompt, seed)
            if key in completed:
                continue

            done += 1
            ps = slug(prompt)
            cat = PROMPT_CAT[prompt]
            print(f"\n[{done}/{total}] {prompt[:40]}... seed={seed} ({cat})")

            # 1. Generate mesh with TRELLIS
            obj_path = OBJ_DIR / f"{ps}_s{seed}.obj"
            if obj_path.exists():
                import trimesh
                mesh = trimesh.load(str(obj_path), force='mesh')
                print(f"  Loaded cached mesh: {len(mesh.vertices)} verts")
            else:
                t0 = time.time()
                mesh = trellis_text_to_mesh(pipeline, prompt, seed)
                gen_time = time.time() - t0
                if mesh is None:
                    print(f"  [SKIP] Generation failed")
                    continue
                print(f"  Generated: {len(mesh.vertices)} verts, {len(mesh.faces)} faces ({gen_time:.1f}s)")
                mesh.export(str(obj_path))

            # 2. Simplify if needed
            mesh = simplify_mesh(mesh, max_faces=10000)

            # 3. Evaluate baseline
            try:
                bl_metrics = evaluate_mesh(mesh)
            except Exception as e:
                print(f"  [SKIP] Eval failed: {e}")
                continue

            # 4. Refine with DGR
            try:
                t0 = time.time()
                refined = refine_mesh(mesh)
                refine_time = time.time() - t0
                dgr_metrics = evaluate_mesh(refined)
            except Exception as e:
                print(f"  [SKIP] Refinement failed: {e}")
                continue

            # 5. Record results
            record = {
                "prompt": prompt,
                "seed": seed,
                "category": cat,
                "n_vertices": len(mesh.vertices),
                "n_faces": len(mesh.faces),
                "baseline": bl_metrics,
                "dgr": dgr_metrics,
                "refine_time": refine_time,
            }

            # Print per-metric improvement
            for m in METRICS:
                bl_v = bl_metrics[m]
                dgr_v = dgr_metrics[m]
                pct = (dgr_v - bl_v) / max(abs(bl_v), 1e-8) * 100
                print(f"  {m}: {bl_v:.5f} → {dgr_v:.5f} ({pct:+.1f}%)")

            results.append(record)
            completed.add(key)

            # Save checkpoint every 10 runs
            if len(results) % 10 == 0:
                state["completed"] = [list(x) for x in completed]
                state["results"] = results
                save_checkpoint(state)
                print(f"  [checkpoint saved: {len(results)} results]")

            # Memory cleanup
            torch.cuda.empty_cache()

    # ── Final save & statistics ───────────────────────────────────────────────

    state["completed"] = [list(x) for x in completed]
    state["results"] = results
    save_checkpoint(state)

    # Save all results
    json.dump(results, open(OUT_DIR / "all_results.json", "w"), indent=2)
    print(f"\nSaved {len(results)} results to {OUT_DIR / 'all_results.json'}")

    # Compute paired statistics
    if len(results) < 10:
        print("Too few results for statistics")
        return

    stat_results = {}
    for m in METRICS:
        bl_vals = [r["baseline"][m] for r in results]
        dgr_vals = [r["dgr"][m] for r in results]
        diffs = [d - b for b, d in zip(bl_vals, dgr_vals)]
        t_stat, p_val = stats.ttest_rel(dgr_vals, bl_vals)

        mean_bl = np.mean(bl_vals)
        mean_dgr = np.mean(dgr_vals)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        pct_change = mean_diff / max(abs(mean_bl), 1e-8) * 100
        n_improved = sum(1 for d in diffs if d > 0)

        stat_results[m] = {
            "baseline_mean": mean_bl,
            "dgr_mean": mean_dgr,
            "pct_change": pct_change,
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "n_improved": n_improved,
            "n_total": len(diffs),
        }

        print(f"\n{m}:")
        print(f"  Baseline: {mean_bl:.5f}  DGR: {mean_dgr:.5f}  Δ%: {pct_change:+.1f}%")
        print(f"  t={t_stat:.2f}  p={p_val:.2e}  d={cohens_d:.3f}")
        print(f"  Improved: {n_improved}/{len(diffs)} ({n_improved/len(diffs)*100:.0f}%)")

    json.dump(stat_results, open(OUT_DIR / "stats.json", "w"), indent=2)
    print(f"\nStats saved to {OUT_DIR / 'stats.json'}")
    print("\n=== Experiment complete ===")


if __name__ == "__main__":
    main()
