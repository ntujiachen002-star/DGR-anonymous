import os
"""Quick check of interim experiment results."""
import json, numpy as np
from collections import Counter

d = json.load(open(os.path.join(ROOT, "analysis_results/full_mgda_classical/checkpoint.json")))
print(f"Total records: {len(d)}")

c = Counter(r["method"] for r in d)
for k,v in sorted(c.items()):
    print(f"  {k}: {v}")

bl = {(r["prompt"], r["seed"]): r for r in d if r["method"] == "baseline"}
methods = ["handcrafted", "mgda_0.05", "laplacian", "normal_consist", "classical_combined"]

print("\nPreliminary results (completed pairs):")
print(f"{'Method':22s} {'Symmetry':>10s} {'Smoothness':>12s} {'Compactness':>12s} {'n':>5s}")
print("-" * 65)
for method in methods:
    mt = {(r["prompt"], r["seed"]): r for r in d if r["method"] == method}
    common = sorted(set(bl) & set(mt))
    if len(common) < 5:
        print(f"  {method}: only {len(common)} pairs")
        continue
    vals = []
    for metric in ["symmetry", "smoothness", "compactness"]:
        a = [bl[k][metric] for k in common]
        b = [mt[k][metric] for k in common]
        delta = (np.mean(b) - np.mean(a)) / (abs(np.mean(a)) + 1e-10) * 100
        vals.append(f"{delta:+.1f}%")
    print(f"  {method:20s} {vals[0]:>10s} {vals[1]:>12s} {vals[2]:>12s} {len(common):>5d}")
