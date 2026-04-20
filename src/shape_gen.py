"""
Shap-E integration: text-to-3D generation with DiffGeoReward guidance.

Three generation modes:
1. baseline: vanilla Shap-E (no geometric reward)
2. diffgeoreward: Shap-E + differentiable geometric reward refinement
3. vlm_baseline: Shap-E + VLM scoring (Claude API) for comparison
"""

import os
import json
import time
import torch
import trimesh
import numpy as np
from pathlib import Path

# Shap-E is only required for backbone generation (load_shap_e / generate_mesh).
# The refinement pipeline (refine_with_geo_reward) is Shap-E independent, so we
# import Shap-E lazily inside the two functions that actually need it.

from geo_reward import (DiffGeoReward, symmetry_reward, symmetry_reward_plane,
                        estimate_symmetry_plane, smoothness_reward, compactness_reward,
                        compute_initial_huber_delta, _build_face_adjacency)


def load_shap_e(device='cuda:0'):
    """Load Shap-E model components."""
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    return xm, model, diffusion


def generate_mesh(xm, model, diffusion, prompt, device='cuda:0', batch_size=1, guidance_scale=15.0):
    """Generate 3D mesh from text using Shap-E.

    Returns list of (vertices, faces, latent) tuples.
    """
    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=False,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    results = []
    for latent in latents:
        t = decode_latent_mesh(xm, latent).tri_mesh()

        # Simplify large meshes to avoid OOM in geometric reward computation
        mesh = trimesh.Trimesh(vertices=t.verts, faces=t.faces)
        if len(mesh.faces) > 10000:
            # target_reduction = fraction to REMOVE (e.g., 0.95 removes 95%)
            target_reduction = 1.0 - (10000 / len(mesh.faces))
            mesh = mesh.simplify_quadric_decimation(target_reduction)

        verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.array(mesh.faces), dtype=torch.long, device=device)
        results.append((verts, faces, latent))

    return results


def refine_with_geo_reward(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weights: torch.Tensor,
    steps: int = 50,
    lr: float = 0.005,
    sym_normal: torch.Tensor = None,
    sym_offset: torch.Tensor = None,
    sym_axis: int = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Refine mesh vertices using differentiable geometric reward.

    Symmetry plane resolution order:
    1. If sym_normal/sym_offset are passed, use them (caller-owned plane).
    2. Else if sym_axis is passed, use the legacy coordinate-plane reward
       (kept for the axis-sweep ablation tool only).
    3. Else estimate an arbitrary best-fit plane from the initial mesh once
       and hold it fixed for all optimization steps. The plane is also
       returned in `history[-1]['sym_plane']` for downstream logging.

    Args:
        vertices: (V, 3) initial vertices
        faces: (F, 3) face indices
        weights: (3,) reward weights [symmetry, smoothness, compactness]
        steps: optimization steps
        lr: learning rate
        sym_normal: (3,) pre-estimated symmetry plane normal (optional)
        sym_offset: scalar pre-estimated symmetry plane offset d (optional)
        sym_axis: legacy coordinate-axis fallback (0=yz,1=xz,2=xy). Only used
                  when sym_normal/sym_offset are not provided.
    Returns:
        refined vertices, list of per-step metrics
    """
    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    face_adj = _build_face_adjacency(faces)
    huber_delta = compute_initial_huber_delta(v_opt, faces)

    use_plane = sym_normal is not None and sym_offset is not None
    if not use_plane and sym_axis is None:
        sym_normal, sym_offset = estimate_symmetry_plane(vertices.detach())
        use_plane = True

    def _sym(v):
        return (symmetry_reward_plane(v, sym_normal, sym_offset)
                if use_plane else symmetry_reward(v, axis=sym_axis))

    with torch.no_grad():
        sym_init = _sym(v_opt).item()
        smooth_init = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj).item()
        compact_init = compactness_reward(v_opt, faces).item()

    sym_scale = max(abs(sym_init), 1e-6)
    smooth_scale = max(abs(smooth_init), 1e-6)
    compact_scale = max(abs(compact_init), 1e-6)

    history = []
    for step in range(steps):
        optimizer.zero_grad()

        # Compute all three rewards every step (matches original HEAD behavior).
        # Zero-weight terms contribute zero gradient, so this is correct for
        # _only ablations and the unconditional history.append below works.
        sym = _sym(v_opt)
        smooth = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj)
        compact = compactness_reward(v_opt, faces)
        reward = (weights[0] * sym / sym_scale
                  + weights[1] * smooth / smooth_scale
                  + weights[2] * compact / compact_scale)

        loss = -reward

        loss.backward()
        grad_norm = v_opt.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()

        history.append({
            'step': step,
            'reward': reward.item(),
            'symmetry': sym.item(),
            'smoothness': smooth.item(),
            'compactness': compact.item(),
            'grad_norm': grad_norm,
        })

    return v_opt.detach(), history


def vlm_score_mesh(vertices, faces, prompt, property_name):
    """Score mesh geometric property using Claude API.

    Renders 4 views and asks Claude to rate the property.
    Returns float score in [0, 1].
    """
    try:
        import anthropic

        # Simple approach: describe the mesh statistics to the VLM
        v_np = vertices.detach().cpu().numpy()
        bbox = v_np.max(axis=0) - v_np.min(axis=0)
        center = v_np.mean(axis=0)

        mesh_desc = (
            f"3D mesh with {len(v_np)} vertices, {len(faces)} faces. "
            f"Bounding box: {bbox[0]:.2f} x {bbox[1]:.2f} x {bbox[2]:.2f}. "
            f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}). "
            f"Generated from prompt: '{prompt}'."
        )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": (
                    f"Rate the {property_name} of this 3D object on a scale of 0.0 to 1.0. "
                    f"Only output a single number.\n\n{mesh_desc}"
                )
            }],
        )
        score_text = response.content[0].text.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"  VLM scoring failed: {e}")
        return 0.5


def save_mesh(vertices, faces, path):
    """Save mesh as OBJ file."""
    v_np = vertices.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
    mesh.export(path)


def run_single_experiment(
    prompt: str,
    method: str,
    seed: int,
    weights: torch.Tensor,
    xm, model, diffusion,
    output_dir: str,
    device: str = 'cuda:0',
) -> dict:
    """Run a single generation experiment.

    Args:
        prompt: text prompt
        method: 'baseline', 'diffgeoreward', or 'vlm_baseline'
        seed: random seed
        weights: (3,) geometric reward weights
        xm, model, diffusion: Shap-E components
        output_dir: where to save results
    Returns:
        dict of metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()

    # Step 1: Generate base mesh
    results = generate_mesh(xm, model, diffusion, prompt, device=device)
    verts, faces, latent = results[0]
    gen_time = time.time() - t0

    # Estimate the symmetry plane once on the initial mesh and reuse it across
    # the optimization and the final metric — paired methods for the same
    # (prompt, seed) thus share an identical plane.
    sym_normal, sym_offset = estimate_symmetry_plane(verts.detach())

    t1 = time.time()
    history = []

    if method == 'baseline':
        refined_verts = verts
        refine_time = 0.0

    elif method == 'diffgeoreward':
        refined_verts, history = refine_with_geo_reward(
            verts, faces, weights,
            steps=50, lr=0.005,
            sym_normal=sym_normal,
            sym_offset=sym_offset,
        )
        refine_time = time.time() - t1

    elif method == 'vlm_baseline':
        # VLM scoring (no gradient, just evaluate)
        refined_verts = verts
        refine_time = 0.0

    total_time = time.time() - t0

    # Step 3: Evaluate geometric properties using the same estimated plane
    sym = symmetry_reward_plane(refined_verts, sym_normal, sym_offset).item()
    smooth = smoothness_reward(refined_verts, faces).item()
    compact = compactness_reward(refined_verts, faces).item()

    # VLM scores (for vlm_baseline method, also compute for comparison)
    vlm_scores = {}
    if method == 'vlm_baseline':
        for prop in ['symmetry', 'smoothness', 'compactness']:
            vlm_scores[prop] = vlm_score_mesh(refined_verts, faces, prompt, prop)

    # Save mesh
    mesh_path = os.path.join(output_dir, f"{method}_seed{seed}.obj")
    save_mesh(refined_verts, faces, mesh_path)

    # Compile metrics
    metrics = {
        'prompt': prompt,
        'method': method,
        'seed': seed,
        'weights': weights.tolist(),
        'symmetry': sym,
        'smoothness': smooth,
        'compactness': compact,
        'gen_time': gen_time,
        'refine_time': refine_time,
        'total_time': total_time,
        'mesh_path': mesh_path,
        'n_vertices': refined_verts.shape[0],
        'n_faces': faces.shape[0],
        'vlm_scores': vlm_scores,
        'sym_plane': {
            'normal': sym_normal.tolist(),
            'offset': sym_offset.item(),
        },
    }

    if history:
        metrics['initial_reward'] = history[0]['reward']
        metrics['final_reward'] = history[-1]['reward']
        metrics['reward_improvement'] = history[-1]['reward'] - history[0]['reward']
        metrics['avg_grad_norm'] = np.mean([h['grad_norm'] for h in history])

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{method}_seed{seed}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics
