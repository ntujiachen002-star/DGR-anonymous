"""Resolution robustness analysis (Appendix sec:resolution_robustness).

Stratifies the main benchmark's baseline vs DGR results by mesh vertex count
and reports DGR improvements within each resolution bin. Directly addresses
whether DGR's gains are an artefact of the coarse Shap-E native resolution.

INPUTS (produced by `tools/exp_full_mgda_classical.py` — run that first):
  analysis_results/full_mgda_classical/all_results.json
  data/baseline/{symmetry,smoothness,compactness}/*.obj

OUTPUT:
  analysis_results/resolution_robustness/summary.json
  stdout — formatted Table 4 from Appendix sec:resolution_robustness.

Runtime: ~30 seconds, CPU only (no GPU, no model loading).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

RESULTS_JSON = Path("analysis_results/full_mgda_classical/all_results.json")
BASELINE_OBJ_DIR = Path("data/baseline")
OUT_DIR = Path("analysis_results/resolution_robustness")


def normalize_prompt(p: str) -> str:
    """Canonicalise prompt → filename stem matching. Matches slug() in other scripts."""
    import re
    return re.sub(r'[^a-z0-9]+', '_', p.lower()).strip('_')


def load_vertex_counts() -> dict:
    """Return {(prompt_slug, seed): (n_verts, n_faces, category)} from baseline OBJs."""
    vmap = {}
    if not BASELINE_OBJ_DIR.exists():
        sys.exit(
            f"ERROR: {BASELINE_OBJ_DIR} not found.\n"
            "Unpack baseline meshes first:\n"
            "    tar xzf data/baseline_meshes.tar.gz -C data/"
        )
    for cat_dir in BASELINE_OBJ_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        for obj in cat_dir.glob("*.obj"):
            base = obj.stem  # e.g. a_symmetric_vase_seed42
            if "_seed" not in base:
                continue
            prompt_slug, seed_str = base.rsplit("_seed", 1)
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            try:
                m = trimesh.load(str(obj), process=False)
                vmap[(prompt_slug, seed)] = (len(m.vertices), len(m.faces), cat)
            except Exception:
                pass
    return vmap


def load_paired_results(vmap: dict) -> list:
    """Join full_mgda_classical per-run results with vertex counts."""
    if not RESULTS_JSON.exists():
        sys.exit(
            f"ERROR: {RESULTS_JSON} not found.\n"
            "Run the main per-run benchmark first:\n"
            "    python tools/exp_full_mgda_classical.py"
        )
    with open(RESULTS_JSON) as f:
        records = json.load(f)

    by_key = defaultdict(dict)
    for r in records:
        key = (normalize_prompt(r["prompt"]), r["seed"])
        by_key[key][r["method"]] = r

    paired = []
    for (prompt, seed), methods in by_key.items():
        if "baseline" not in methods or "handcrafted" not in methods:
            continue
        if (prompt, seed) not in vmap:
            continue
        nv, nf, cat = vmap[(prompt, seed)]
        bl, dgr = methods["baseline"], methods["handcrafted"]
        paired.append({
            "prompt": prompt, "seed": seed, "cat": cat,
            "n_verts": nv, "n_faces": nf,
            "sym_bl": bl["symmetry"], "sym_dgr": dgr["symmetry"],
            "smo_bl": bl["smoothness"], "smo_dgr": dgr["smoothness"],
            "com_bl": bl["compactness"], "com_dgr": dgr["compactness"],
        })
    return paired


BINS = [
    ("coarse (|V|<30)",       lambda v: v < 30),
    ("small (30<=|V|<100)",   lambda v: 30 <= v < 100),
    ("medium (100<=|V|<1K)",  lambda v: 100 <= v < 1000),
    ("dense (|V|>=1K)",       lambda v: v >= 1000),
]


def analyse(paired: list) -> dict:
    verts = np.array([r["n_verts"] for r in paired])
    pctiles = {f"p{p}": int(np.percentile(verts, p)) for p in [10, 25, 50, 75, 90, 95, 99]}

    print(f"\nPaired records: n={len(paired)}")
    print("Vertex count distribution:", ", ".join(f"{k}={v}" for k, v in pctiles.items()))

    print("\n" + "=" * 88)
    print(f"{'Bin':<22}{'n':<6}{'med|V|':<10}{'Sym %':<12}{'HNC %':<12}{'Com %':<12}{'p_sym':<10}")
    print("-" * 88)

    bin_results = []
    for name, pred in BINS:
        bucket = [r for r in paired if pred(r["n_verts"])]
        if len(bucket) < 5:
            print(f"{name:<22}{len(bucket):<6} (too few)")
            continue

        sym_bl = np.array([r["sym_bl"] for r in bucket])
        sym_dgr = np.array([r["sym_dgr"] for r in bucket])
        smo_bl = np.array([r["smo_bl"] for r in bucket])
        smo_dgr = np.array([r["smo_dgr"] for r in bucket])
        com_bl = np.array([r["com_bl"] for r in bucket])
        com_dgr = np.array([r["com_dgr"] for r in bucket])

        sym_pct = (sym_dgr.mean() - sym_bl.mean()) / abs(sym_bl.mean()) * 100
        smo_pct = (smo_dgr.mean() - smo_bl.mean()) / abs(smo_bl.mean()) * 100
        com_pct = (com_dgr.mean() - com_bl.mean()) / abs(com_bl.mean()) * 100
        _, p_sym = stats.ttest_rel(sym_dgr, sym_bl)
        med_v = int(np.median([r["n_verts"] for r in bucket]))

        bin_results.append({
            "bin": name, "n": len(bucket), "median_v": med_v,
            "sym_pct": float(sym_pct), "hnc_pct": float(smo_pct), "com_pct": float(com_pct),
            "p_sym": float(p_sym),
            "sym_bl_mean": float(sym_bl.mean()), "sym_dgr_mean": float(sym_dgr.mean()),
            "smo_bl_mean": float(smo_bl.mean()), "smo_dgr_mean": float(smo_dgr.mean()),
            "com_bl_mean": float(com_bl.mean()), "com_dgr_mean": float(com_dgr.mean()),
        })
        print(f"{name:<22}{len(bucket):<6}{med_v:<10}{sym_pct:<+12.1f}{smo_pct:<+12.1f}{com_pct:<+12.1f}{p_sym:<10.2e}")

    # Spearman correlations: per-mesh delta vs vertex count
    print("\n" + "=" * 88)
    print("Spearman correlation (per-mesh delta vs vertex count):")
    print("=" * 88)
    verts = np.array([r["n_verts"] for r in paired])
    corr = {}
    for metric in ["sym", "smo", "com"]:
        bl = np.array([r[f"{metric}_bl"] for r in paired])
        dgr = np.array([r[f"{metric}_dgr"] for r in paired])
        delta = dgr - bl
        rho, p = stats.spearmanr(verts, delta)
        corr[metric] = {"rho": float(rho), "p": float(p)}
        print(f"  {metric}: rho={rho:+.3f}  p={p:.3e}")

    return {
        "n_total": len(paired),
        "vertex_percentiles": pctiles,
        "bin_results": bin_results,
        "spearman_delta_vs_verts": corr,
    }


def main():
    print("Loading vertex counts from baseline OBJs...")
    vmap = load_vertex_counts()
    print(f"  loaded {len(vmap)} meshes")

    print("Loading per-run results (full_mgda_classical)...")
    paired = load_paired_results(vmap)
    print(f"  joined {len(paired)} baseline/DGR pairs")

    summary = analyse(paired)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")

    # Sanity check against paper numbers
    print("\nReference values from paper (Appendix Table 4):")
    print("  coarse  (<30v, n=79):   +64.2% / +19.8% / +48.4%")
    print("  small   (30-100v,n=79): +84.1% / +21.8% / +56.1%")
    print("  medium  (100-1K, n=43): +82.3% / +21.5% / +42.5%")
    print("  dense   (>=1K,   n=30): +77.4% / +22.6% / +49.9%")


if __name__ == "__main__":
    main()
