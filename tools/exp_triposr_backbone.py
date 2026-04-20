"""
Exp T: TripoSR Backbone — second backbone generalization test.
NeurIPS rigor: 3 seeds × 30 prompts × 2 methods, full paired statistics.

Pipeline: text → SDXL-Turbo (image) → rembg (bg removal) → TripoSR → [DGR refinement]
Fully independent of Shap-E.

Design decisions:
- 3 seeds (42, 123, 456): each seed generates a DIFFERENT image via SDXL stochasticity,
  yielding 3 independent TripoSR meshes per prompt → genuine variance estimate
- 30 prompts (10/category): powers paired t-test at d=0.3 with >80% power (n=90 pairs)
- SDXL-Turbo (4 steps, ~5s/image) for fast generation
- Equal weights [1/3,1/3,1/3] for DGR: conservative, same as "handcrafted" in Shap-E exp
- Statistics: paired t-test, Cohen's d, 95% bootstrap CI, BH-FDR

Methods:
  triposr_baseline — raw TripoSR output (no refinement)
  triposr_dgr      — TripoSR + DiffGeoReward (50 Adam steps, lr=5e-3)

Output: analysis_results/triposr_backbone/all_results.json
~1h on V100 (90 images × 5s + 90 meshes × 10s + 90 DGR × 5s)
"""

import os, sys, json, time, gc, re
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        smoothness_reward, compactness_reward)
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

sys.path.insert(0, os.path.dirname(__file__))
from _plane_protocol import PlaneStore, make_key

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS     = [42, 123, 456]        # 3 seeds → 3 distinct images per prompt
STEPS     = 50
LR        = 0.005
DGR_W     = [1/3, 1/3, 1/3]       # equal weights = "handcrafted" setting

TRIPOSR_PATH = os.environ.get('TRIPOSR_PATH', os.path.join(os.path.dirname(ROOT), 'TripoSR'))
TRIPOSR_HF   = 'stabilityai/TripoSR'

N_PER_CAT = 10                      # 10 prompts per category = 30 total
PROMPTS_SYM = SYMMETRY_PROMPTS[:N_PER_CAT]
PROMPTS_SMO = SMOOTHNESS_PROMPTS[:N_PER_CAT]
PROMPTS_COM = COMPACTNESS_PROMPTS[:N_PER_CAT]
ALL_PROMPTS = PROMPTS_SYM + PROMPTS_SMO + PROMPTS_COM
PROMPT_CAT  = {p: "symmetry"    for p in PROMPTS_SYM}
PROMPT_CAT |= {p: "smoothness"  for p in PROMPTS_SMO}
PROMPT_CAT |= {p: "compactness" for p in PROMPTS_COM}

OUT_DIR = Path("analysis_results/triposr_backbone")
IMG_DIR = OUT_DIR / "images"
OBJ_DIR = Path("results/triposr_objs")
for d in [OUT_DIR, IMG_DIR, OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ["symmetry", "smoothness", "compactness"]


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ── Image generation ──────────────────────────────────────────────────────────

def _load_sdxl_pipe():
    """Load SDXL-Turbo (fast) with fallbacks to slower SDXL / SD variants."""
    from diffusers import AutoPipelineForText2Image
    candidates = [
        ("stabilityai/sdxl-turbo",             dict(torch_dtype=torch.float16, variant="fp16")),
        ("stabilityai/stable-diffusion-xl-base-1.0",
                                                dict(torch_dtype=torch.float16, variant="fp16")),
        ("stabilityai/stable-diffusion-2-1",   dict(torch_dtype=torch.float16)),
        ("runwayml/stable-diffusion-v1-5",      dict(torch_dtype=torch.float16)),
    ]
    for model_id, kwargs in candidates:
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, use_safetensors=True, **kwargs
            ).to(DEVICE)
            pipe.set_progress_bar_config(disable=True)
            is_turbo = "turbo" in model_id
            print(f"  Loaded image model: {model_id}")
            return pipe, is_turbo
        except Exception as e:
            print(f"  {model_id} unavailable: {e}")
    return None, False


def generate_images(prompts_seeds: list[tuple[str, int]]) -> dict[tuple[str,int], object]:
    """Generate one PIL image per (prompt, seed) pair. Returns {(prompt,seed): PIL.Image}."""
    from PIL import Image as PILImage

    img_dir = IMG_DIR
    images  = {}

    # Load cached first
    for prompt, seed in prompts_seeds:
        path = img_dir / f"{slug(prompt)}_s{seed}.png"
        if path.exists():
            images[(prompt, seed)] = PILImage.open(str(path)).convert("RGB")

    missing = [(p, s) for (p, s) in prompts_seeds if (p, s) not in images]
    if not missing:
        return images

    print(f"\nGenerating {len(missing)} images via SDXL...")
    pipe, is_turbo = _load_sdxl_pipe()

    if pipe is None:
        # Hard fail: no diffusion model available.
        # Do NOT insert placeholder images — that would produce fake TripoSR meshes
        # and pollute the geometric metrics with data not derived from text.
        print("FATAL: No SDXL model available. Cannot generate images for TripoSR.")
        print("  Install diffusers and ensure a model is downloadable, then rerun.")
        sys.exit(1)

    for i, (prompt, seed) in enumerate(missing):
        try:
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            kw  = dict(prompt=prompt, generator=gen)
            kw |= (dict(num_inference_steps=4, guidance_scale=0.0) if is_turbo
                   else dict(num_inference_steps=20, guidance_scale=7.5))
            img = pipe(**kw).images[0]
            path = img_dir / f"{slug(prompt)}_s{seed}.png"
            img.save(str(path))
            images[(prompt, seed)] = img

        except Exception as e:
            # Single-sample generation error: skip this (prompt, seed) pair entirely.
            # The pair will be absent from `images`, causing it to be skipped in main().
            # This produces a missing record rather than a fake one.
            print(f"  [SKIP] Image failed {prompt[:40]} seed={seed}: {e}")

        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(missing)} images done")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return images


# ── TripoSR loading & inference ───────────────────────────────────────────────

def load_triposr():
    sys.path.insert(0, TRIPOSR_PATH)
    from tsr.system import TSR
    print(f"Loading TripoSR ({TRIPOSR_HF})...")
    model = TSR.from_pretrained(TRIPOSR_HF, config_name="config.yaml",
                                weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(DEVICE)
    print("  TripoSR ready.")

    rembg_session = None
    try:
        import rembg
        rembg_session = rembg.new_session()
        print("  rembg ready.")
    except Exception as e:
        print(f"  rembg unavailable ({e}), using raw image.")

    return model, rembg_session


def triposr_infer(tsr_model, rembg_session, image) -> tuple:
    """PIL image → (verts Tensor, faces Tensor) or (None, None)."""
    from tsr.utils import resize_foreground
    from PIL import Image as PILImage
    import trimesh

    try:
        if rembg_session is not None:
            import rembg
            img_nobg = rembg.remove(image, session=rembg_session)
            img_nobg = resize_foreground(img_nobg, 0.85)
            if img_nobg.mode == "RGBA":
                bg = PILImage.new("RGBA", img_nobg.size, (255, 255, 255, 255))
                bg.paste(img_nobg, mask=img_nobg.split()[3])
                img_in = bg.convert("RGB")
            else:
                img_in = img_nobg.convert("RGB")
        else:
            img_in = image.convert("RGB")

        with torch.no_grad():
            codes = tsr_model([img_in], device=DEVICE)
            mesh_out = tsr_model.extract_mesh(codes, False, resolution=256)[0]

        tmesh = mesh_out if hasattr(mesh_out, 'vertices') else \
                trimesh.Trimesh(vertices=mesh_out[0], faces=mesh_out[1])

        if len(tmesh.faces) > 20000:
            try:
                tmesh = tmesh.simplify_quadratic_decimation(10000)
            except Exception:
                pass

        if len(tmesh.faces) < 4:
            return None, None

        v = torch.tensor(tmesh.vertices, dtype=torch.float32, device=DEVICE)
        f = torch.tensor(tmesh.faces,    dtype=torch.long,    device=DEVICE)
        return v, f

    except Exception as e:
        print(f"    TripoSR infer error: {e}")
        return None, None


# ── Mesh I/O ──────────────────────────────────────────────────────────────────

def save_obj(verts, faces, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    v, f = verts.cpu().numpy(), faces.cpu().numpy()
    with open(path, 'w') as fp:
        for vi in v:  fp.write(f"v {vi[0]:.6f} {vi[1]:.6f} {vi[2]:.6f}\n")
        for fi in f:  fp.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")


def load_obj(path: Path):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    v = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
    f = torch.tensor(faces, dtype=torch.long,    device=DEVICE)
    return v, f


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(verts, faces, sym_plane=None) -> dict:
    """Evaluate all metrics under an optional fixed symmetry plane."""
    try:
        if sym_plane is not None:
            sym = symmetry_reward_plane(verts, sym_plane[0], sym_plane[1]).item()
        else:
            sym = symmetry_reward(verts).item()
        return dict(
            symmetry    = sym,
            smoothness  = smoothness_reward(verts, faces).item(),
            compactness = compactness_reward(verts, faces).item(),
            n_verts     = verts.shape[0],
            n_faces     = faces.shape[0],
        )
    except Exception as e:
        return dict(symmetry=None, smoothness=None, compactness=None, error=str(e))


# ── Statistics ────────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d (difference / SD of differences)."""
    diff = a - b
    sd   = diff.std(ddof=1)
    return diff.mean() / sd if sd > 1e-12 else 0.0


def bootstrap_ci(a: np.ndarray, b: np.ndarray,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Bootstrap 95% CI for mean(a) - mean(b) (paired)."""
    rng  = np.random.default_rng(seed)
    diff = a - b
    boot = [rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(n_boot)]
    lo   = np.percentile(boot, 100 * alpha / 2)
    hi   = np.percentile(boot, 100 * (1 - alpha / 2))
    return lo, hi


def _bh_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    p_adj = np.empty(n)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(float(p_values[idx]) * n / (rank + 1), 1.0)
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        if p_adj[order[i]] > p_adj[order[i + 1]]:
            p_adj[order[i]] = p_adj[order[i + 1]]
    return p_adj.tolist()


def run_stats(results: list[dict]) -> dict:
    """
    Compute paired stats for triposr_dgr vs triposr_baseline.
    BH-FDR correction applied across the 3 metric comparisons (symmetry,
    smoothness, compactness) — 3 simultaneous tests on the same dataset.
    """
    bl  = {(r["prompt"], r["seed"]): r for r in results if r["method"] == "triposr_baseline"}
    dgr = {(r["prompt"], r["seed"]): r for r in results if r["method"] == "triposr_dgr"}
    common = set(bl) & set(dgr)

    raw_entries = []
    for metric in METRICS:
        a_list = [bl[k][metric]  for k in common if bl[k].get(metric)  is not None
                                                  and dgr[k].get(metric) is not None]
        b_list = [dgr[k][metric] for k in common if bl[k].get(metric)  is not None
                                                  and dgr[k].get(metric) is not None]
        if len(a_list) < 5:
            raw_entries.append((metric, None))
            continue
        a, b = np.array(a_list), np.array(b_list)
        t, p = stats.ttest_rel(b, a)
        d    = cohen_d(b, a)
        ci   = bootstrap_ci(b, a)
        raw_entries.append((metric, dict(
            n             = len(a),
            baseline_mean = float(a.mean()),
            dgr_mean      = float(b.mean()),
            delta_pct     = float((b.mean() - a.mean()) / abs(a.mean()) * 100),
            t_stat        = float(t),
            p_raw         = float(p),
            cohens_d      = float(d),
            ci_lo_95      = float(ci[0]),
            ci_hi_95      = float(ci[1]),
        )))

    # BH-FDR across the 3 metric comparisons
    valid_idxs = [i for i, (_, v) in enumerate(raw_entries) if v is not None]
    p_raws = [raw_entries[i][1]["p_raw"] for i in valid_idxs]
    p_adjs = _bh_correct(p_raws)
    for rank, i in enumerate(valid_idxs):
        raw_entries[i][1]["p_adj"]       = p_adjs[rank]
        raw_entries[i][1]["significant"] = bool(p_adjs[rank] < 0.05)

    stats_out = {}
    for metric, v in raw_entries:
        stats_out[metric] = v if v is not None else {"note": "insufficient data"}
    return stats_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ckpt_path = OUT_DIR / "checkpoint.json"
    results, done_keys = [], set()

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["seed"], r["method"]) for r in results}
        print(f"Resuming: {len(results)} records done")

    # All (prompt, seed) pairs
    all_pairs = [(p, s) for p in ALL_PROMPTS for s in SEEDS]

    # Step 1: generate all images (cached)
    images = generate_images(all_pairs)
    print(f"Images ready: {len(images)}/{len(all_pairs)}")

    # Step 2: load TripoSR
    print("\nLoading TripoSR...")
    try:
        tsr_model, rembg_session = load_triposr()
    except Exception as e:
        print(f"FATAL: TripoSR load failed: {e}")
        sys.exit(1)

    dgr_weights = torch.tensor(DGR_W, dtype=torch.float32, device=DEVICE)
    plane_store = PlaneStore.load_or_new(str(OUT_DIR / "plane_cache.json"))
    t0 = time.time()
    n_total = len(all_pairs)

    print(f"\nProcessing {n_total} (prompt, seed) pairs...")
    for idx, (prompt, seed) in enumerate(all_pairs):
        cat = PROMPT_CAT[prompt]
        ps  = slug(prompt)
        bl_obj  = OBJ_DIR / cat / f"{ps}_s{seed}_baseline.obj"
        dgr_obj = OBJ_DIR / cat / f"{ps}_s{seed}_dgr.obj"

        # ── baseline ──
        plane_key = make_key(prompt, seed)
        sym_plane = None
        key_bl = (prompt, seed, "triposr_baseline")
        if key_bl not in done_keys:
            if bl_obj.exists():
                try:    verts, faces = load_obj(bl_obj)
                except: verts, faces = None, None
            else:
                img = images.get((prompt, seed))
                if img is None:
                    verts, faces = None, None
                else:
                    verts, faces = triposr_infer(tsr_model, rembg_session, img)
                    if verts is not None:
                        save_obj(verts, faces, bl_obj)

            if verts is not None:
                sym_plane = plane_store.get(plane_key, verts=verts)
                m = compute_metrics(verts, faces, sym_plane=sym_plane)
            else:
                m = dict(symmetry=None, smoothness=None, compactness=None, error="triposr_failed")
            results.append(dict(prompt=prompt, seed=seed, method="triposr_baseline",
                                category=cat, backbone="triposr", **m))
            done_keys.add(key_bl)

        # ── DGR refinement ──
        key_dgr = (prompt, seed, "triposr_dgr")
        if key_dgr not in done_keys:
            if not bl_obj.exists():
                results.append(dict(prompt=prompt, seed=seed, method="triposr_dgr",
                                    category=cat, backbone="triposr",
                                    symmetry=None, smoothness=None, compactness=None,
                                    error="no_baseline_mesh"))
            else:
                try:
                    verts, faces = load_obj(bl_obj)
                    if sym_plane is None:
                        sym_plane = plane_store.get(plane_key, verts=verts)
                    sym_n, sym_d = sym_plane
                    ref_v, _ = refine_with_geo_reward(
                        verts, faces, dgr_weights, steps=STEPS, lr=LR,
                        sym_normal=sym_n, sym_offset=sym_d,
                    )
                    save_obj(ref_v, faces, dgr_obj)
                    m = compute_metrics(ref_v, faces, sym_plane=sym_plane)
                    results.append(dict(prompt=prompt, seed=seed, method="triposr_dgr",
                                        category=cat, backbone="triposr", **m))
                except Exception as e:
                    print(f"  DGR error {ps} s{seed}: {e}")
                    results.append(dict(prompt=prompt, seed=seed, method="triposr_dgr",
                                        category=cat, backbone="triposr",
                                        symmetry=None, smoothness=None, compactness=None,
                                        error=str(e)))
            done_keys.add(key_dgr)

        # checkpoint every 10 pairs
        if (idx + 1) % 10 == 0:
            with open(ckpt_path, 'w') as f:
                json.dump(results, f)
            print(f"  [{idx+1}/{n_total}] ckpt saved  {time.time()-t0:.0f}s")

    # final save
    with open(ckpt_path, 'w') as f:
        json.dump(results, f)
    with open(OUT_DIR / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    plane_store.save()

    del tsr_model
    torch.cuda.empty_cache()
    print(f"\nDone: {len(results)} records  {time.time()-t0:.0f}s")

    # ── Statistical analysis ───────────────────────────────────────────────────
    stats_result = run_stats(results)

    print("\n" + "="*65)
    print("TRIPOSR BACKBONE — PAIRED STATISTICS (DGR vs baseline)")
    print("="*65)
    print("BH-FDR correction across 3 metrics (q=0.05)")
    print(f"{'Metric':<12} | {'BL mean':>9} | {'DGR mean':>9} | "
          f"{'Δ%':>7} | {'d':>6} | {'p_adj':>10} | {'sig':>4} | 95% CI")
    print("-" * 80)
    for m, s in stats_result.items():
        if "note" in s:
            print(f"{m:<12} | insufficient data")
            continue
        sig = "✓" if s.get("significant") else "✗"
        print(f"{m:<12} | {s['baseline_mean']:>9.5f} | {s['dgr_mean']:>9.5f} | "
              f"{s['delta_pct']:>+6.1f}% | {s['cohens_d']:>6.3f} | "
              f"{s['p_adj']:>10.4e} | {sig:>4} | [{s['ci_lo_95']:+.4f}, {s['ci_hi_95']:+.4f}]")

    # per-category breakdown
    print("\n--- Per-category breakdown ---")
    for cat in ["symmetry", "smoothness", "compactness"]:
        cat_rec = [r for r in results if r.get("category") == cat]
        for metric in METRICS:
            bl_s  = [r[metric] for r in cat_rec
                     if r["method"] == "triposr_baseline" and r.get(metric) is not None]
            dgr_s = [r[metric] for r in cat_rec
                     if r["method"] == "triposr_dgr" and r.get(metric) is not None]
            if bl_s and dgr_s:
                d = (np.mean(dgr_s) - np.mean(bl_s)) / abs(np.mean(bl_s)) * 100
                print(f"  [{cat}] {metric}: BL={np.mean(bl_s):.5f}  "
                      f"DGR={np.mean(dgr_s):.5f}  Δ={d:+.1f}%  n={len(dgr_s)}")

    # save stats
    with open(OUT_DIR / "stats.json", 'w') as f:
        json.dump(stats_result, f, indent=2)
    print(f"\nStats saved: {OUT_DIR}/stats.json")
    print(f"Results:     {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
