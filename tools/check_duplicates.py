import os
import json, subprocess
from collections import Counter

d = json.load(open(os.path.join(ROOT, "analysis_results/full_mgda_classical/checkpoint.json")))

keys = [(r["prompt"], r["seed"], r["method"]) for r in d]
total = len(keys)
unique = len(set(keys))
dups = total - unique
print(f"Total: {total}, Unique: {unique}, Duplicates: {dups}")

if dups > 0:
    c = Counter(keys)
    dup_count = 0
    for k, v in c.items():
        if v > 1:
            dup_count += 1
            if dup_count <= 10:
                print(f"  DUP x{v}: {k[2]} / {k[0][:40]} / seed={k[1]}")
    if dup_count > 10:
        print(f"  ... and {dup_count - 10} more")

bl = [r for r in d if r["method"] == "baseline"]
bl_keys = [(r["prompt"], r["seed"]) for r in bl]
bl_unique = len(set(bl_keys))
print(f"\nBaseline: {len(bl_keys)} total, {bl_unique} unique, {len(bl_keys) - bl_unique} dups")

obj_count = int(subprocess.check_output(
    "find /root/autodl-tmp/DiffGeoReward/results/full_mgda_classical_objs -name '*.obj' | wc -l",
    shell=True).strip())
print(f"OBJ files on disk: {obj_count}")
print(f"Match: {obj_count == bl_unique}")

cats = Counter(r["category"] for r in bl)
print(f"\nBaseline per category:")
for k, v in sorted(cats.items()):
    print(f"  {k}: {v}")
