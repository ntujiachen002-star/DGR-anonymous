import json, os, sys, time, torch, statistics, re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lang2comp import Lang2Comp
from geo_reward import (symmetry_reward, symmetry_reward_plane,
                        estimate_symmetry_plane, smoothness_reward,
                        compactness_reward)
from _plane_protocol import PlaneStore, make_key
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

ALL_PROMPTS = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
PROMPT_CAT = {}
for p in SYMMETRY_PROMPTS: PROMPT_CAT[p] = "symmetry"
for p in SMOOTHNESS_PROMPTS: PROMPT_CAT[p] = "smoothness"
for p in COMPACTNESS_PROMPTS: PROMPT_CAT[p] = "compactness"
SEEDS = [42, 123, 456]

OBJ_DIR = Path("results/full_mgda_classical_objs/baseline")
OUT_DIR = Path("analysis_results/lang2comp_rerun")
OUT_DIR.mkdir(parents=True, exist_ok=True)
plane_store = PlaneStore.load_or_new(str(OUT_DIR / "plane_cache.json"))

def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]

def load_obj(path, device="cpu"):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                fi = [int(x.split("/")[0])-1 for x in line.split()[1:4]]
                if len(fi) == 3: faces.append(fi)
    return (torch.tensor(verts, dtype=torch.float32, device=device),
            torch.tensor(faces, dtype=torch.long, device=device).reshape(-1, 3))

# Load model
model = Lang2Comp()
model.load_state_dict(torch.load("checkpoints/lang2comp_retrained.pt", map_location="cpu"))
model.eval()
print("Lang2Comp loaded")

# Collect baseline OBJs
tasks = []
for obj_path in OBJ_DIR.rglob("*.obj"):
    parts = obj_path.stem.rsplit("_s", 1)
    if len(parts) == 2:
        ps_name, seed_str = parts
        for prompt in ALL_PROMPTS:
            if slug(prompt) == ps_name:
                tasks.append((prompt, int(seed_str), str(obj_path)))
                break

print(f"Found {len(tasks)} baseline OBJs")

def run_one(args):
    prompt, seed, obj_path = args
    try:
        v, f = load_obj(obj_path)
        if f.shape[0] < 4: return None
        # Skip degenerate baseline (Shap-E sometimes emits point clouds with 0 faces).
        if f.shape[0] == 0 or v.shape[0] == 0:
            return None
        # Estimate symmetry plane once on the baseline mesh; share across all variants
        # of this (prompt, seed) for paired protocol compliance.
        sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=v)
        pred = model.predict(prompt)
        weights = torch.tensor([pred["weights"]["symmetry"], pred["weights"]["smoothness"], pred["weights"]["compactness"]])
        with torch.no_grad():
            bl = {m: eval(f"{m}_reward")(v, **({'axis':1} if m=='symmetry' else {'faces':f} if m=='compactness' else {'vertices':v,'faces':f})) for m in []}
            bl_sym = symmetry_reward_plane(v, sym_n, sym_d).item()
            bl_smo = smoothness_reward(v, f).item()
            bl_com = compactness_reward(v, f).item()
        v_ref, _ = refine_with_geo_reward(v.clone(), f, weights=weights, steps=50, lr=0.005,
                                          sym_normal=sym_n, sym_offset=sym_d)
        with torch.no_grad():
            ref_sym = symmetry_reward_plane(v_ref, sym_n, sym_d).item()
            ref_smo = smoothness_reward(v_ref, f).item()
            ref_com = compactness_reward(v_ref, f).item()
        return {
            "prompt": prompt, "seed": seed, "method": "diffgeoreward",
            "category": PROMPT_CAT.get(prompt, "unknown"),
            "weights": pred["weights"], "dominant": pred["dominant_property"],
            "symmetry": ref_sym, "smoothness": ref_smo, "compactness": ref_com,
            "base_symmetry": bl_sym, "base_smoothness": bl_smo, "base_compactness": bl_com,
        }
    except Exception as e:
        print(f"  ERR {prompt[:30]}: {e}")
        return None

results = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_one, t): t for t in tasks}
    done = 0
    for future in as_completed(futures):
        r = future.result()
        done += 1
        if r: results.append(r)
        if done % 50 == 0:
            print(f"  {done}/{len(tasks)} done, {len(results)} valid, {time.time()-t0:.0f}s")

plane_store.save()

with open(OUT_DIR / "all_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Stats
for metric in ["symmetry", "smoothness", "compactness"]:
    bl = statistics.mean(r[f"base_{metric}"] for r in results)
    mt = statistics.mean(r[metric] for r in results)
    delta = (mt - bl) / abs(bl) * 100 if bl != 0 else 0
    wr = sum(1 for r in results if r[metric] > r[f"base_{metric}"]) / len(results) * 100
    print(f"  {metric}: {delta:+.1f}% (wr={wr:.0f}%)")

print(f"\nDone: {len(results)} results in {time.time()-t0:.0f}s")
print(f"Saved to {OUT_DIR}/all_results.json")
