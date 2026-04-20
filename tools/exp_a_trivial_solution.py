"""
Experiment A: Symmetry Trivial Solution Verification
Tests whether symmetry improvement comes from mesh collapse (vertices collapsing to symmetry plane).
Runs sym_only at 0/50/200/500 steps and monitors thickness_ratio.
"""
import os, sys, json, torch, trimesh, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from shape_gen import load_shap_e, generate_mesh
from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

PROMPTS = [
    # Top-5 failure prompts (compactness degrades most)
    "a symmetric vase",
    "a perfectly balanced chair",
    "an hourglass shape",
    "a symmetric wine glass",
    "a balanced chess piece, a king",
    # 2 success prompts (control)
    "a smooth sphere",
    "a compact stone",
]

STEP_CONFIGS = [0, 50, 200, 500]
SEED = 42
LR = 0.005
DEVICE = 'cuda:0'

def compute_thickness_ratio(vertices):
    """min_bbox_dim / max_bbox_dim. Near 0 = flat (trivial solution)."""
    v = vertices.detach().cpu().numpy()
    bbox = v.max(axis=0) - v.min(axis=0)
    return float(bbox.min() / (bbox.max() + 1e-8))

def compute_mesh_volume_trimesh(vertices, faces):
    v = vertices.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    return float(abs(mesh.volume))

def refine_sym_only_with_tracking(vertices, faces, max_steps, lr=0.005):
    """Sym-only refinement with per-step metric tracking."""
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)
    v_init = vertices.detach().clone()

    with torch.no_grad():
        sym_init = symmetry_reward(v_opt, axis=1).item()
    sym_scale = max(abs(sym_init), 1e-6)

    records = []
    for step in range(max_steps + 1):
        with torch.no_grad():
            sym_val = symmetry_reward(v_opt, axis=1).item()
            smo_val = smoothness_reward(v_opt, faces).item()
            com_val = compactness_reward(v_opt, faces).item()
            thickness = compute_thickness_ratio(v_opt)
            vol = compute_mesh_volume_trimesh(v_opt, faces)
            disp = (v_opt - v_init).norm(dim=1).mean().item()

        record = {
            "step": step,
            "symmetry": sym_val,
            "smoothness": smo_val,
            "compactness": com_val,
            "thickness_ratio": thickness,
            "volume": vol,
            "vertex_displacement_mean": disp,
        }

        if step > 0:
            record["grad_norm"] = grad_norm_val

        records.append(record)

        if step in STEP_CONFIGS:
            print(f"    step={step}: sym={sym_val:.6f}, thick={thickness:.4f}, vol={vol:.4f}, disp={disp:.6f}")

        if step < max_steps:
            optimizer.zero_grad()
            sym = symmetry_reward(v_opt, axis=1)
            loss = -(sym / sym_scale)
            loss.backward()
            grad_norm_val = v_opt.grad.norm().item()
            torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
            optimizer.step()

    return records

def main():
    print("Loading Shap-E...")
    xm, model, diffusion = load_shap_e(device=DEVICE)

    out_dir = "analysis_results/trivial_solution"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for prompt in PROMPTS:
        print(f"\n=== {prompt} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        results = generate_mesh(xm, model, diffusion, prompt, device=DEVICE)
        verts, faces, _ = results[0]
        print(f"  Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")

        if faces.shape[0] == 0:
            print(f"  [SKIP] 0 faces, skipping")
            continue

        records = refine_sym_only_with_tracking(verts, faces, max_steps=500, lr=LR)
        all_results[prompt] = records

    # Save
    with open(os.path.join(out_dir, "metrics_per_step.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Prompt':<35} {'Steps':>5} {'Symmetry':>12} {'Thickness':>10} {'Volume':>10}")
    for prompt, records in all_results.items():
        for step_val in STEP_CONFIGS:
            r = records[step_val]
            print(f"{prompt[:34]:<35} {r['step']:>5} {r['symmetry']:>12.6f} {r['thickness_ratio']:>10.4f} {r['volume']:>10.4f}")

    print(f"\nResults saved to {out_dir}/metrics_per_step.json")

if __name__ == "__main__":
    main()
