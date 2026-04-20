
import torch, os, re, sys
sys.path.insert(0, 'src')
from pathlib import Path
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from shape_gen import refine_with_geo_reward
import trimesh

OBJ_ROOT = Path("results/mesh_validity_objs")
SEED = 42
W_EQUAL = torch.tensor([1/3, 1/3, 1/3])

EVAL_PROMPTS = {
    "symmetry": [
        "a symmetric vase", "a perfectly balanced chair", "a symmetric trophy",
        "a symmetric cathedral door", "a symmetric wine glass", "a symmetric crown",
        "a balanced chess piece a king", "a symmetric butterfly sculpture",
        "a symmetrical temple", "a balanced candelabra", "a symmetric bell",
        "a symmetric goblet", "a symmetric pagoda",
    ],
    "smoothness": ["a smooth melting chocolate drop"],
    "compactness": [
        "a compact cube", "a compact birdhouse", "a solid ice cube",
        "a solid metal die", "a tight ball", "a compact backpack",
    ],
}

def load_obj(path, device='cpu'):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                fi = [int(x.split('/')[0])-1 for x in line.split()[1:4]]
                if len(fi) == 3: faces.append(fi)
    return (torch.tensor(verts, dtype=torch.float32, device=device),
            torch.tensor(faces, dtype=torch.long, device=device).reshape(-1, 3))

def save_obj(verts, faces, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    v, f = verts.cpu().numpy(), faces.cpu().numpy()
    with open(path, 'w') as fp:
        for vi in v: fp.write(f"v {vi[0]:.6f} {vi[1]:.6f} {vi[2]:.6f}\n")
        for fi in f: fp.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")

count = 0
total = sum(len(v) for v in EVAL_PROMPTS.values())

for category, prompts in EVAL_PROMPTS.items():
    for prompt in prompts:
        slug = prompt.replace(' ', '_').replace(',', '')

        # Find baseline OBJ
        bl_path = None
        for p in (OBJ_ROOT / "baseline" / category).glob(f"*seed{SEED}.obj"):
            name = p.stem.replace(f"_seed{SEED}", "").replace(f"seed{SEED}", "")
            if name == slug or slug in name or name in slug:
                bl_path = p
                break

        if not bl_path:
            print(f"  SKIP {prompt}: baseline not found")
            continue

        v_bl, f_bl = load_obj(str(bl_path))
        if f_bl.shape[0] < 4:
            print(f"  SKIP {prompt}: {f_bl.shape[0]} faces")
            continue

        # Refine with NEW Huber NC code
        v_ref, _ = refine_with_geo_reward(v_bl.clone(), f_bl, weights=W_EQUAL, steps=50, lr=0.005)

        # Save to handcrafted dir (overwrite old)
        out_path = OBJ_ROOT / "handcrafted" / category / bl_path.name
        save_obj(v_ref, f_bl, str(out_path))

        count += 1
        with torch.no_grad():
            bl_sym = symmetry_reward(v_bl, axis=1).item()
            ref_sym = symmetry_reward(v_ref, axis=1).item()
            imp = (ref_sym - bl_sym) / abs(bl_sym) * 100 if bl_sym != 0 else 0
        print(f"  [{count}/{total}] {prompt}: sym {bl_sym:.4f} -> {ref_sym:.4f} ({imp:+.1f}%)")

print(f"\nDone: {count} OBJs regenerated with Huber NC code")
