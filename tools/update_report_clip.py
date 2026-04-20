#!/usr/bin/env python3
"""
After exp_m completes and results are downloaded, run this script to:
1. Print CLIP stats for all 6 methods
2. Print ready-to-paste Markdown for EXPERIMENT_REPORT.md §3.3 and §5

Usage: python tools/update_report_clip.py
"""
import os, sys, json, numpy as np
from pathlib import Path
from scipy import stats

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, "src")

CLIP_FILE = Path("analysis_results/clip_allmethod/all_results.json")
if not CLIP_FILE.exists():
    print(f"ERROR: {CLIP_FILE} not found. Run exp_m first.")
    sys.exit(1)

with open(CLIP_FILE) as f:
    data = json.load(f)

print(f"Loaded {len(data)} CLIP records\n")

METHODS = ["baseline", "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]

# ── per-method stats ──────────────────────────────────────────────────────────
print("=" * 60)
print("CLIP SCORES BY METHOD")
print("=" * 60)
print(f"{'Method':<16s} | {'Mean':>8s} | {'Std':>8s} | {'N':>5s}")
print("-" * 45)

method_scores = {}
for method in METHODS:
    scores = [r["clip_score"] for r in data if r["method"] == method and r.get("clip_score") is not None]
    if scores:
        method_scores[method] = scores
        print(f"{method:<16s} | {np.mean(scores):>8.4f} | {np.std(scores):>8.4f} | {len(scores):>5d}")

# ── paired t-test: each method vs baseline ───────────────────────────────────
print("\n" + "=" * 60)
print("PAIRED COMPARISON vs BASELINE")
print("=" * 60)

bl_dict = {(r["prompt"], r["seed"]): r["clip_score"]
           for r in data if r["method"] == "baseline" and r.get("clip_score") is not None}

for method in METHODS:
    if method == "baseline":
        continue
    m_dict = {(r["prompt"], r["seed"]): r["clip_score"]
              for r in data if r["method"] == method and r.get("clip_score") is not None}
    common = set(bl_dict.keys()) & set(m_dict.keys())
    if len(common) < 5:
        print(f"  {method}: insufficient paired data (n={len(common)})")
        continue
    a = [bl_dict[k] for k in common]
    b = [m_dict[k]  for k in common]
    diff = np.array(b) - np.array(a)
    t, p = stats.ttest_rel(b, a)
    wins = sum(1 for k in common if m_dict[k] > bl_dict[k])
    print(f"  {method:<16s}: diff={np.mean(diff):+.4f}  t={t:.3f}  p={p:.4e}  wins={wins}/{len(common)}")

# ── per-category breakdown ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLIP BY METHOD × CATEGORY")
print("=" * 60)
print(f"{'Method':<16s} | {'Symmetry':>10s} | {'Smoothness':>10s} | {'Compactness':>10s}")
print("-" * 58)
for method in METHODS:
    row = {}
    for cat in ["symmetry", "smoothness", "compactness"]:
        scores = [r["clip_score"] for r in data
                  if r["method"] == method and r.get("category") == cat
                  and r.get("clip_score") is not None]
        row[cat] = f"{np.mean(scores):.4f}" if scores else "  --  "
    print(f"{method:<16s} | {row['symmetry']:>10s} | {row['smoothness']:>10s} | {row['compactness']:>10s}")

# ── ready-to-paste Markdown ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("READY-TO-PASTE MARKDOWN FOR §3.3 (Semantic Preservation)")
print("=" * 60)

shown = ["baseline", "diffgeoreward", "handcrafted"]
print("\n| Method | CLIP Score | N |")
print("|--------|------------|---|")
for m in shown:
    s = method_scores.get(m, [])
    if s:
        print(f"| {m:<14s} | {np.mean(s):.3f} ± {np.std(s):.3f} | {len(s)} |")
    else:
        print(f"| {m:<14s} | -- | -- |")

# paired significance
if "baseline" in method_scores and "diffgeoreward" in method_scores:
    bl_d = {(r["prompt"], r["seed"]): r["clip_score"]
            for r in data if r["method"] == "baseline" and r.get("clip_score") is not None}
    dgr_d = {(r["prompt"], r["seed"]): r["clip_score"]
             for r in data if r["method"] == "diffgeoreward" and r.get("clip_score") is not None}
    common = set(bl_d.keys()) & set(dgr_d.keys())
    if common:
        t, p = stats.ttest_rel([bl_d[k] for k in common], [dgr_d[k] for k in common])
        print(f"\nBaseline vs DiffGeoReward paired t-test: p = {p:.4e}")
        print("(p > 0.05 → no significant CLIP degradation)")
