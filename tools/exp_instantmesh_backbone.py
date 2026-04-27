"""
Exp: InstantMesh Backbone — modern backbone generalization test.
NeurIPS rigor: 3 seeds × 20 prompts per category (60 total) = 180 pairs.

InstantMesh (Xu et al., 2024) uses Zero123++ multi-view diffusion + FlexiCubes
reconstruction. Produces substantially higher-quality meshes than Shap-E.
V100 compatible (no FlashAttention dependency).

Pipeline: text → SDXL-Turbo (image) → rembg → InstantMesh → [DGR refinement]

Methods:
  instantmesh_baseline — raw InstantMesh output (no refinement)
  instantmesh_dgr      — InstantMesh + DiffGeoReward (50 Adam steps, lr=5e-3)

Output: analysis_results/instantmesh_backbone/all_results.json
        analysis_results/instantmesh_backbone/stats.json

Hardware: V100 32GB (sequential pipeline fits in ~16GB)
Estimated time: ~3-4h (180 image gen + 180 InstantMesh infer + 180 DGR)

Setup:
    1. git clone https://github.com/TencentARC/InstantMesh.git ~/InstantMesh
    2. cd ~/InstantMesh && pip install -r requirements.txt
    3. pip install trimesh rembg
    4. Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/exp_instantmesh_backbone.py
"""

import os, sys, json, time, gc, re
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import (symmetry_reward, smoothness_reward, compactness_reward,
                        compute_initial_huber_delta, _build_face_adjacency)
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

sys.path.insert(0, os.path.dirname(__file__))
from _plane_protocol import PlaneStore, make_key

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS  = [42, 123, 456]
STEPS  = 50
LR     = 0.005
DGR_W  = [1/3, 1/3, 1/3]

# Auto-detect InstantMesh path
INSTANTMESH_PATH = None
for _p in [os.environ.get('INSTANTMESH_PATH', os.path.join(os.path.dirname(ROOT), 'InstantMesh')),
           '~/InstantMesh']:
    _ep = os.path.expanduser(_p)
    if os.path.isdir(_ep):
        INSTANTMESH_PATH = _ep
        break
if INSTANTMESH_PATH is None:
    INSTANTMESH_PATH = os.environ.get('INSTANTMESH_PATH', os.path.expanduser('~/InstantMesh'))

N_PER_CAT = 20  # 20 prompts per category = 60 total
PROMPTS_SYM = SYMMETRY_PROMPTS[:N_PER_CAT]
PROMPTS_SMO = SMOOTHNESS_PROMPTS[:N_PER_CAT]
PROMPTS_COM = COMPACTNESS_PROMPTS[:N_PER_CAT]
ALL_PROMPTS = PROMPTS_SYM + PROMPTS_SMO + PROMPTS_COM
PROMPT_CAT  = {p: "symmetry"    for p in PROMPTS_SYM}
PROMPT_CAT |= {p: "smoothness"  for p in PROMPTS_SMO}
PROMPT_CAT |= {p: "compactness" for p in PROMPTS_COM}

OUT_DIR = Path("analysis_results/instantmesh_backbone")
IMG_DIR = OUT_DIR / "images"
OBJ_DIR = Path("results/instantmesh_objs")
for d in [OUT_DIR, IMG_DIR, OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ["symmetry", "smoothness", "compactness"]
CHECKPOINT = OUT_DIR / "checkpoint.json"


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ── Image generation (SDXL-Turbo) ────────────────────────────────────────────

def _load_sdxl_pipe():
    from diffusers import AutoPipelineForText2Image
    candidates = [
        ("stabilityai/sdxl-turbo", dict(torch_dtype=torch.float16, variant="fp16")),
        ("stabilityai/stable-diffusion-2-1", dict(torch_dtype=torch.float16)),
        ("runwayml/stable-diffusion-v1-5", dict(torch_dtype=torch.float16)),
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


def generate_images(prompts_seeds):
    from PIL import Image as PILImage
    images = {}

    # Load cached
    for prompt, seed in prompts_seeds:
        path = IMG_DIR / f"{slug(prompt)}_s{seed}.png"
        if path.exists():
            images[(prompt, seed)] = PILImage.open(str(path)).convert("RGB")

    missing = [(p, s) for (p, s) in prompts_seeds if (p, s) not in images]
    if not missing:
        return images

    print(f"\nGenerating {len(missing)} images via SDXL...")
    pipe, is_turbo = _load_sdxl_pipe()
    if pipe is None:
        print("FATAL: No SDXL model available.")
        sys.exit(1)

    for i, (prompt, seed) in enumerate(missing):
        try:
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            kw = dict(prompt=prompt, generator=gen)
            kw |= (dict(num_inference_steps=4, guidance_scale=0.0) if is_turbo
                   else dict(num_inference_steps=20, guidance_scale=7.5))
            img = pipe(**kw).images[0]
            path = IMG_DIR / f"{slug(prompt)}_s{seed}.png"
            img.save(str(path))
            images[(prompt, seed)] = img
        except Exception as e:
            print(f"  [SKIP] Image failed {prompt[:40]} seed={seed}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(missing)} images done")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    return images


# ── InstantMesh loading & inference ──────────────────────────────────────────

def load_instantmesh():
    """Load InstantMesh pipeline (diffusion + reconstruction)."""
    sys.path.insert(0, INSTANTMESH_PATH)

    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    print("Loading InstantMesh multiview diffusion...")
    # Use local pipeline file to avoid relying on remote GitHub fetch at runtime
    local_pipeline = os.path.join(INSTANTMESH_PATH, "zero123plus")
    mv_pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline=local_pipeline,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    mv_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        mv_pipeline.scheduler.config, timestep_spacing='trailing'
    )
    unet_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="diffusion_pytorch_model.bin",
        local_files_only=True,
    )
    mv_pipeline.unet.load_state_dict(
        torch.load(unet_path, map_location='cpu'), strict=True
    )
    mv_pipeline = mv_pipeline.to(DEVICE)
    mv_pipeline.set_progress_bar_config(disable=True)
    print("  Multiview diffusion ready.")

    rembg_session = None
    try:
        import rembg
        rembg_session = rembg.new_session()
        print("  rembg ready.")
    except Exception:
        print("  rembg unavailable, using raw images.")

    return mv_pipeline, rembg_session


def load_reconstruction_model():
    """Load InstantMesh reconstruction model (after freeing diffusion model)."""
    sys.path.insert(0, INSTANTMESH_PATH)

    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download

    # Try large first, fall back to base
    for config_name, ckpt_name in [
        ("instant-mesh-large", "instant_mesh_large.ckpt"),
        ("instant-mesh-base", "instant_mesh_base.ckpt"),
    ]:
        try:
            config_path = os.path.join(INSTANTMESH_PATH, "configs", f"{config_name}.yaml")
            config = OmegaConf.load(config_path)

            from src.utils.train_util import instantiate_from_config
            model = instantiate_from_config(config.model_config)

            ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename=ckpt_name,
                local_files_only=True,
            )
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict = {k[14:]: v for k, v in state_dict.items()
                         if k.startswith('lrm_generator.')}
            model.load_state_dict(state_dict, strict=True)
            model = model.to(DEVICE).eval()

            # Init mesh extraction
            model.init_flexicubes_geometry(DEVICE, fovy=30.0)
            print(f"  Reconstruction model ready: {config_name}")
            return model, config
        except Exception as e:
            print(f"  {config_name} failed: {e}")

    print("FATAL: Could not load InstantMesh reconstruction model.")
    sys.exit(1)


def instantmesh_infer(mv_pipeline, rembg_session, recon_model, image):
    """Single image → (verts Tensor, faces Tensor) or (None, None)."""
    import trimesh
    from PIL import Image as PILImage

    try:
        # 1. Background removal
        if rembg_session is not None:
            import rembg
            img_nobg = rembg.remove(image, session=rembg_session)
            if img_nobg.mode == "RGBA":
                bg = PILImage.new("RGBA", img_nobg.size, (255, 255, 255, 255))
                bg.paste(img_nobg, mask=img_nobg.split()[3])
                img_in = bg.convert("RGB")
            else:
                img_in = img_nobg.convert("RGB")
        else:
            img_in = image.convert("RGB")

        # 2. Generate multiview images (6 views)
        with torch.no_grad():
            mv_images = mv_pipeline(
                img_in,
                num_inference_steps=75,
            ).images[0]  # 3x2 grid of 6 views

        # 3. Process multiview for reconstruction
        # Split the 3x2 grid into 6 individual images
        from src.utils.mesh_util import extract_mesh_from_multiview
        mesh_out = extract_mesh_from_multiview(recon_model, mv_images, DEVICE)

        if mesh_out is None:
            return None, None

        # Extract trimesh
        if hasattr(mesh_out, 'vertices'):
            tmesh = mesh_out
        else:
            tmesh = trimesh.Trimesh(vertices=mesh_out[0], faces=mesh_out[1])

        # Simplify if needed
        if len(tmesh.faces) > 20000:
            try:
                tmesh = tmesh.simplify_quadratic_decimation(10000)
            except Exception:
                pass

        if len(tmesh.faces) < 4:
            return None, None

        v = torch.tensor(tmesh.vertices, dtype=torch.float32, device=DEVICE)
        f = torch.tensor(tmesh.faces, dtype=torch.long, device=DEVICE)
        return v, f

    except Exception as e:
        print(f"    InstantMesh infer error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def instantmesh_infer_simple(recon_model, mv_pipeline, rembg_session, image):
    """Simplified inference using InstantMesh's own run.py logic."""
    import trimesh
    from PIL import Image as PILImage

    try:
        # Background removal
        if rembg_session is not None:
            import rembg
            img_nobg = rembg.remove(image, session=rembg_session)
            if img_nobg.mode == "RGBA":
                bg = PILImage.new("RGBA", img_nobg.size, (255, 255, 255, 255))
                bg.paste(img_nobg, mask=img_nobg.split()[3])
                img_in = bg.convert("RGB")
            else:
                img_in = img_nobg.convert("RGB")
        else:
            img_in = image.convert("RGB")

        # Resize to 320x320 (InstantMesh expects this)
        img_in = img_in.resize((320, 320), PILImage.LANCZOS)

        # Generate 6 multiview images
        with torch.no_grad():
            mv_output = mv_pipeline(
                img_in,
                num_inference_steps=75,
            ).images[0]

        # mv_output is a single image containing 3x2 grid of views
        # Each view is 320x320, grid is 960x640
        w, h = mv_output.size
        view_w, view_h = w // 2, h // 3
        views = []
        for row in range(3):
            for col in range(2):
                crop = mv_output.crop((
                    col * view_w, row * view_h,
                    (col + 1) * view_w, (row + 1) * view_h
                ))
                views.append(crop)

        # Convert views to tensor for reconstruction
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        images_tensor = torch.stack([transform(v) for v in views]).unsqueeze(0).to(DEVICE)

        # Get camera poses (InstantMesh uses fixed cameras for Zero123++)
        # These are the 6 canonical views from Zero123++
        input_cameras = get_zero123plus_cameras().to(DEVICE)

        # Run reconstruction
        with torch.no_grad():
            planes = recon_model.forward_planes(images_tensor, input_cameras)
            mesh_raw = recon_model.extract_mesh(
                planes,
                use_texture_map=False,
                texture_resolution=512,
            )

        if mesh_raw is None or len(mesh_raw) == 0:
            return None, None

        # mesh_raw is a list; take first
        mesh_data = mesh_raw[0] if isinstance(mesh_raw, list) else mesh_raw

        if hasattr(mesh_data, 'vertices'):
            tmesh = mesh_data
        elif isinstance(mesh_data, tuple) and len(mesh_data) >= 2:
            verts_np = mesh_data[0].cpu().numpy() if torch.is_tensor(mesh_data[0]) else mesh_data[0]
            faces_np = mesh_data[1].cpu().numpy() if torch.is_tensor(mesh_data[1]) else mesh_data[1]
            tmesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
        else:
            return None, None

        # Simplify
        if len(tmesh.faces) > 20000:
            try:
                tmesh = tmesh.simplify_quadratic_decimation(10000)
            except Exception:
                pass

        if len(tmesh.faces) < 4:
            return None, None

        v = torch.tensor(np.array(tmesh.vertices), dtype=torch.float32, device=DEVICE)
        f = torch.tensor(np.array(tmesh.faces), dtype=torch.long, device=DEVICE)
        return v, f

    except Exception as e:
        print(f"    InstantMesh infer error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_zero123plus_cameras():
    """Get the 6 camera poses used by Zero123++ (elevation=20, 6 azimuths)."""
    import math
    azimuths = [30, 90, 150, 210, 270, 330]
    elevation = 20
    cameras = []
    for az in azimuths:
        az_rad = math.radians(az)
        el_rad = math.radians(elevation)
        x = math.cos(el_rad) * math.cos(az_rad)
        y = math.cos(el_rad) * math.sin(az_rad)
        z = math.sin(el_rad)
        cameras.append([x, y, z, az_rad, el_rad, 0.0])
    return torch.tensor(cameras, dtype=torch.float32).unsqueeze(0)


# ── Mesh I/O ──────────────────────────────────────────────────────────────────

def save_obj(verts, faces, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    v, f = verts.cpu().numpy(), faces.cpu().numpy()
    with open(path, 'w') as fp:
        for vi in v:
            fp.write(f"v {vi[0]:.6f} {vi[1]:.6f} {vi[2]:.6f}\n")
        for fi in f:
            fp.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")


def load_obj(path: Path):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    v = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
    f = torch.tensor(faces, dtype=torch.long, device=DEVICE)
    return v, f


# ── Metrics & DGR ─────────────────────────────────────────────────────────────

def compute_metrics(verts, faces, sym_plane=None):
    """Evaluate the three metrics. If sym_plane=(n,d) is given, symmetry is
    computed under that plane; otherwise falls back to fixed xz (legacy)."""
    try:
        if sym_plane is not None:
            from geo_reward import symmetry_reward_plane
            sym = symmetry_reward_plane(verts, sym_plane[0], sym_plane[1]).item()
        else:
            sym = symmetry_reward(verts).item()
        return dict(
            symmetry=sym,
            smoothness=smoothness_reward(verts, faces).item(),
            compactness=compactness_reward(verts, faces).item(),
            n_verts=verts.shape[0],
            n_faces=faces.shape[0],
        )
    except Exception as e:
        return dict(symmetry=None, smoothness=None, compactness=None, error=str(e))


def refine_with_geo_reward(vertices, faces, weights, steps=50, lr=0.005,
                             sym_normal=None, sym_offset=None, sym_axis=None):
    """Inline DGR refinement with Huber NC + new-protocol symmetry plane.

    Resolution order (matches src/shape_gen.refine_with_geo_reward):
      - if sym_normal/sym_offset are passed, use that fixed plane
      - elif sym_axis (int) is passed, use legacy coordinate-axis symmetry
      - else estimate an arbitrary best-fit plane on the baseline mesh once
    """
    w = torch.tensor(weights, dtype=torch.float32, device=vertices.device) \
        if not isinstance(weights, torch.Tensor) else weights

    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    face_adj = _build_face_adjacency(faces)
    huber_delta = compute_initial_huber_delta(v_opt, faces)

    use_plane = sym_normal is not None and sym_offset is not None
    if not use_plane and sym_axis is None:
        from geo_reward import estimate_symmetry_plane
        sym_normal, sym_offset = estimate_symmetry_plane(vertices.detach())
        use_plane = True

    def _sym(v):
        if use_plane:
            from geo_reward import symmetry_reward_plane
            return symmetry_reward_plane(v, sym_normal, sym_offset)
        return symmetry_reward(v, sym_axis)

    with torch.no_grad():
        r0_sym = _sym(v_opt).item()
        r0_smo = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj).item()
        r0_com = compactness_reward(v_opt, faces).item()
    eps = 1e-8

    for step in range(steps):
        optimizer.zero_grad()
        r_sym = _sym(v_opt)
        r_smo = smoothness_reward(v_opt, faces, delta=huber_delta, _adj=face_adj)
        r_com = compactness_reward(v_opt, faces)

        reward = (w[0] * r_sym / (abs(r0_sym) + eps)
                  + w[1] * r_smo / (abs(r0_smo) + eps)
                  + w[2] * r_com / (abs(r0_com) + eps))

        (-reward).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()

    return v_opt.detach()


# ── Statistics ────────────────────────────────────────────────────────────────

def cohen_d(a, b):
    diff = a - b
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 1e-12 else 0.0


def bootstrap_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    diff = a - b
    boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(boot, 100 * alpha / 2), np.percentile(boot, 100 * (1 - alpha / 2))


def bh_correct(p_values):
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (orig_i, p) in enumerate(indexed, 1):
        adjusted[orig_i] = min(p * n / rank, 1.0)
    for i in range(n - 2, -1, -1):
        sort_idx = [x[0] for x in indexed]
        adjusted[sort_idx[i]] = min(adjusted[sort_idx[i]], adjusted[sort_idx[i + 1]])
    return adjusted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("InstantMesh Backbone Experiment")
    print(f"  {len(ALL_PROMPTS)} prompts × {len(SEEDS)} seeds = {len(ALL_PROMPTS) * len(SEEDS)} runs")
    print(f"  InstantMesh path: {INSTANTMESH_PATH}")
    print("=" * 60)

    # Check InstantMesh exists
    if not os.path.isdir(INSTANTMESH_PATH):
        print(f"\nERROR: InstantMesh not found at {INSTANTMESH_PATH}")
        print("Please clone: git clone https://github.com/TencentARC/InstantMesh.git")
        sys.exit(1)

    # Load checkpoint if exists
    results = []
    completed = set()
    if CHECKPOINT.exists():
        ckpt = json.loads(CHECKPOINT.read_text())
        results = ckpt.get("results", [])
        completed = set(tuple(x) for x in ckpt.get("completed", []))
        print(f"  Resuming: {len(completed)} already done")

    plane_store = PlaneStore.load_or_new(str(OUT_DIR / "plane_cache.json"))

    # Build task list
    tasks = [(p, s) for p in ALL_PROMPTS for s in SEEDS
             if (p, s) not in completed]
    print(f"  {len(tasks)} tasks remaining\n")

    if not tasks:
        print("All tasks already completed. Computing stats...")
        compute_stats(results)
        return

    # Phase 1: Generate images
    print("Phase 1: Image generation")
    all_ps = [(p, s) for p in ALL_PROMPTS for s in SEEDS]
    images = generate_images(all_ps)
    print(f"  {len(images)} images ready\n")

    # Phase 2: Load InstantMesh
    print("Phase 2: Loading InstantMesh...")
    mv_pipeline, rembg_session = load_instantmesh()

    # Phase 3: Generate baseline meshes + DGR refinement
    print("\nPhase 3: InstantMesh inference + DGR refinement")

    # We need to do inference in two stages to manage VRAM:
    # Stage A: Generate all multiview images (using diffusion pipeline)
    # Stage B: Free diffusion, load reconstruction, extract meshes

    # Stage A: Generate multiview images for all tasks
    mv_images_cache = {}
    for i, (prompt, seed) in enumerate(tasks):
        key = (prompt, seed)
        if key not in images:
            print(f"  [SKIP] No image for {prompt[:30]} s={seed}")
            continue

        img = images[key]
        mv_cache_path = OUT_DIR / "mv_cache" / f"{slug(prompt)}_s{seed}.png"
        mv_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if mv_cache_path.exists():
            from PIL import Image as PILImage
            mv_images_cache[key] = PILImage.open(str(mv_cache_path))
        else:
            try:
                # Background removal
                if rembg_session is not None:
                    import rembg
                    from PIL import Image as PILImage
                    img_nobg = rembg.remove(img, session=rembg_session)
                    if img_nobg.mode == "RGBA":
                        bg = PILImage.new("RGBA", img_nobg.size, (255, 255, 255, 255))
                        bg.paste(img_nobg, mask=img_nobg.split()[3])
                        img_in = bg.convert("RGB")
                    else:
                        img_in = img_nobg.convert("RGB")
                else:
                    img_in = img.convert("RGB")

                img_in = img_in.resize((320, 320))

                with torch.no_grad():
                    mv_out = mv_pipeline(img_in, num_inference_steps=75).images[0]
                mv_out.save(str(mv_cache_path))
                mv_images_cache[key] = mv_out
            except Exception as e:
                print(f"  [SKIP] MV gen failed {prompt[:30]} s={seed}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  Stage A: {i+1}/{len(tasks)} multiview images")

    # Free diffusion model
    print("  Freeing diffusion model, loading reconstruction...")
    del mv_pipeline
    torch.cuda.empty_cache()
    gc.collect()

    # Stage B: Load reconstruction model and extract meshes
    recon_model, recon_config = load_reconstruction_model()

    for i, (prompt, seed) in enumerate(tasks):
        key = (prompt, seed)
        if key not in mv_images_cache:
            continue

        cat = PROMPT_CAT[prompt]
        obj_base = OBJ_DIR / "baseline" / cat / f"{slug(prompt)}_s{seed}.obj"
        obj_dgr  = OBJ_DIR / "dgr"      / cat / f"{slug(prompt)}_s{seed}.obj"

        # Check if mesh already exists
        if obj_base.exists() and obj_dgr.exists():
            v_base, f_base = load_obj(obj_base)
            v_dgr, f_dgr = load_obj(obj_dgr)
        else:
            # Extract mesh from multiview
            try:
                mv_img = mv_images_cache[key]
                v_base, f_base = _extract_mesh_from_mv(recon_model, mv_img)
            except Exception as e:
                print(f"  [SKIP] Mesh extraction failed {prompt[:30]} s={seed}: {e}")
                continue

            if v_base is None:
                print(f"  [SKIP] Empty mesh {prompt[:30]} s={seed}")
                continue

            save_obj(v_base, f_base, obj_base)

            # Estimate plane on baseline once; reuse for refinement and both
            # metric evaluations (paired protocol).
            sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=v_base)

            v_dgr = refine_with_geo_reward(
                v_base, f_base, DGR_W, steps=STEPS, lr=LR,
                sym_normal=sym_n, sym_offset=sym_d,
            )
            f_dgr = f_base
            save_obj(v_dgr, f_dgr, obj_dgr)

        # Plane lookup (covers both the reuse and fresh branches)
        sym_n, sym_d = plane_store.get(make_key(prompt, seed), verts=v_base)

        m_base = compute_metrics(v_base, f_base, sym_plane=(sym_n, sym_d))
        m_dgr  = compute_metrics(v_dgr, f_dgr, sym_plane=(sym_n, sym_d))

        if m_base.get("symmetry") is None or m_dgr.get("symmetry") is None:
            continue

        record = dict(
            prompt=prompt, seed=seed, category=cat,
            baseline=m_base, dgr=m_dgr,
        )
        results.append(record)
        completed.add(key)

        # Checkpoint
        if (i + 1) % 10 == 0:
            CHECKPOINT.write_text(json.dumps(dict(
                results=results,
                completed=[list(x) for x in completed],
            ), indent=2))

        if (i + 1) % 20 == 0:
            print(f"  Stage B: {i+1}/{len(tasks)} meshes")

    # Final checkpoint
    CHECKPOINT.write_text(json.dumps(dict(
        results=results,
        completed=[list(x) for x in completed],
    ), indent=2))
    plane_store.save()

    # Compute stats
    compute_stats(results)


def _extract_mesh_from_mv(recon_model, mv_image):
    """Extract mesh from multiview grid image using InstantMesh's own processing.

    mv_image: PIL Image (960x640 or similar 3:2 grid of 6 views)
    Returns: (vertices, faces) tensors or (None, None)
    """
    import trimesh
    from einops import rearrange
    from torchvision.transforms import v2

    try:
        # Convert PIL → numpy → tensor, following InstantMesh run.py exactly
        images = np.asarray(mv_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, H, W)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)    # (6, 3, 320, 320)
        images = images.unsqueeze(0).to(DEVICE)                                   # (1, 6, 3, 320, 320)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        # Get cameras from InstantMesh's own utility
        sys.path.insert(0, INSTANTMESH_PATH)
        from src.utils.camera_util import get_zero123plus_input_cameras
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(DEVICE)

        with torch.no_grad():
            planes = recon_model.forward_planes(images, input_cameras)
            mesh_out = recon_model.extract_mesh(planes, use_texture_map=False)

        if mesh_out is None:
            return None, None

        # mesh_out: (vertices, faces) or (vertices, faces, uvs, ...)
        if isinstance(mesh_out, tuple):
            verts_t, faces_t = mesh_out[0], mesh_out[1]
        else:
            return None, None

        verts_np = verts_t.cpu().numpy() if torch.is_tensor(verts_t) else np.array(verts_t)
        faces_np = faces_t.cpu().numpy() if torch.is_tensor(faces_t) else np.array(faces_t)
        tmesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

        if len(tmesh.faces) > 20000:
            try:
                tmesh = tmesh.simplify_quadratic_decimation(10000)
            except Exception:
                pass

        if len(tmesh.faces) < 4:
            return None, None

        v = torch.tensor(np.array(tmesh.vertices), dtype=torch.float32, device=DEVICE)
        f = torch.tensor(np.array(tmesh.faces), dtype=torch.long, device=DEVICE)
        return v, f

    except Exception as e:
        print(f"    Mesh extraction error: {e}")
        return None, None


def compute_stats(results):
    """Compute paired statistics and save."""
    print(f"\n{'=' * 60}")
    print(f"STATISTICS ({len(results)} valid pairs)")
    print("=" * 60)

    if len(results) < 3:
        print("Too few results for statistics.")
        return

    all_stats = {}
    for metric in METRICS:
        base_vals = np.array([r["baseline"][metric] for r in results if r["baseline"].get(metric) is not None])
        dgr_vals  = np.array([r["dgr"][metric]      for r in results if r["dgr"].get(metric) is not None])

        n = min(len(base_vals), len(dgr_vals))
        base_vals, dgr_vals = base_vals[:n], dgr_vals[:n]

        if n < 3:
            continue

        t_stat, p_val = stats.ttest_rel(dgr_vals, base_vals)
        d = cohen_d(dgr_vals, base_vals)
        ci_lo, ci_hi = bootstrap_ci(dgr_vals, base_vals)

        base_mean = base_vals.mean()
        dgr_mean  = dgr_vals.mean()
        pct_change = ((dgr_mean - base_mean) / abs(base_mean) * 100) if abs(base_mean) > 1e-12 else 0
        win_rate = (dgr_vals > base_vals).mean() * 100 if metric != "compactness" else \
                   (dgr_vals > base_vals).mean() * 100

        all_stats[metric] = dict(
            n=int(n),
            baseline_mean=float(base_mean),
            dgr_mean=float(dgr_mean),
            pct_change=float(pct_change),
            t_stat=float(t_stat),
            p_value=float(p_val),
            cohen_d=float(d),
            ci_95=[float(ci_lo), float(ci_hi)],
            win_rate=float(win_rate),
        )

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"\n  {metric}:")
        print(f"    Baseline: {base_mean:.6f}  →  DGR: {dgr_mean:.6f}  ({pct_change:+.1f}%)")
        print(f"    t={t_stat:.3f}, p={p_val:.2e} {sig}, d={d:.3f}")
        print(f"    95% CI: [{ci_lo:.6f}, {ci_hi:.6f}], Win rate: {win_rate:.1f}%")

    # BH correction
    p_vals = [all_stats[m]["p_value"] for m in METRICS if m in all_stats]
    adjusted = bh_correct(p_vals)
    for i, m in enumerate([m for m in METRICS if m in all_stats]):
        all_stats[m]["p_value_bh"] = float(adjusted[i])

    # Save
    stats_path = OUT_DIR / "stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))

    results_path = OUT_DIR / "all_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n  Stats saved to {stats_path}")
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
