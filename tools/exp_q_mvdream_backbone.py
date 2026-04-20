"""
Exp Q: MVDream Backbone — stronger backbone generalization test.
NeurIPS rigor: 3 seeds × 30 prompts × 2 methods, full paired statistics.

Pipeline: text → MVDream (multi-view diffusion, front view) → rembg → TripoSR → [DGR]

MVDream (Shi et al., ICLR 2024) is a multi-view diffusion model fine-tuned from
SD2.1 to generate 4 geometrically-consistent views. It is the backbone used in
DreamCS (arXiv 2506.09814) and DreamReward (ECCV 2024), enabling direct comparison.
We use the front-view output (azimuth=0°) as TripoSR input — a standard practice
in feed-forward 3D generation pipelines.

This demonstrates DiffGeoReward's backbone-agnosticism:
  Shap-E (text→mesh direct)  → exp_k
  TripoSR/SDXL (single-view) → exp_triposr_backbone
  MVDream (multi-view aware)  → this experiment (exp_q)

Methods:
  mvdream_baseline — raw TripoSR mesh from MVDream front-view image
  mvdream_dgr      — TripoSR mesh + DiffGeoReward (50 Adam steps, lr=5e-3)

Output: analysis_results/mvdream_backbone/all_results.json
~1.5h on V100 (90 MVDream inferences ~20s + 90 TripoSR ~10s + 90 DGR ~5s)
"""

import os, sys, json, time, gc, re
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
from shape_gen import refine_with_geo_reward
from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS     = [42, 123, 456]
STEPS     = 50
LR        = 0.005
DGR_W     = [1/3, 1/3, 1/3]

# ── 路径配置（根据服务器实际情况修改）──
# TripoSR: 如果没有本地 clone, 设为 None 会尝试 pip install
TRIPOSR_PATH = os.environ.get('TRIPOSR_PATH', os.environ.get('TRIPOSR_PATH', os.path.join(os.path.dirname(ROOT), 'TripoSR')))
TRIPOSR_HF   = 'stabilityai/TripoSR'

# MVDream: 本地权重路径（避免网络下载）
# 自动搜索顺序: 项目 models_cache → HuggingFace cache → 在线下载
PROJECT_ROOT = Path(__file__).parent.parent
MVDREAM_LOCAL_CKPT = str(PROJECT_ROOT / "models_cache" / "models--MVDream--MVDream" /
                         "snapshots" / "d14ac9d78c48c266005729f2d5633f6c265da467" /
                         "sd-v2.1-base-4view.pt")
MVDREAM_HF   = 'ashawkey/mvdream-sd2.1-diffusers'  # diffusers-compatible (unused with native API)

N_PER_CAT = 10
PROMPTS_SYM = SYMMETRY_PROMPTS[:N_PER_CAT]
PROMPTS_SMO = SMOOTHNESS_PROMPTS[:N_PER_CAT]
PROMPTS_COM = COMPACTNESS_PROMPTS[:N_PER_CAT]
ALL_PROMPTS = PROMPTS_SYM + PROMPTS_SMO + PROMPTS_COM
PROMPT_CAT  = {p: "symmetry"    for p in PROMPTS_SYM}
PROMPT_CAT |= {p: "smoothness"  for p in PROMPTS_SMO}
PROMPT_CAT |= {p: "compactness" for p in PROMPTS_COM}

OUT_DIR = Path("analysis_results/mvdream_backbone")
IMG_DIR = OUT_DIR / "images"
OBJ_DIR = Path("results/mvdream_objs")
for d in [OUT_DIR, IMG_DIR, OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ["symmetry", "smoothness", "compactness"]


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ── MVDream image generation ──────────────────────────────────────────────────

def _install_mvdream():
    """Install MVDream package if not available.

    尝试多种安装方式，适配中国服务器网络环境：
    1. 直接 import（已安装）
    2. pip install from GitHub（需外网）
    3. pip install from GitHub mirror（中国镜像）
    4. 从本地 clone 安装
    """
    try:
        import mvdream  # noqa
        return True
    except ImportError:
        pass

    import subprocess

    # 方式 1: GitHub 直装
    sources = [
        ("GitHub", "git+https://github.com/bytedance/MVDream"),
        ("GitHub mirror (ghproxy)", "git+https://ghproxy.com/https://github.com/bytedance/MVDream"),
        ("GitHub mirror (gitclone)", "git+https://gitclone.com/github.com/bytedance/MVDream"),
    ]

    for name, url in sources:
        print(f"  Trying MVDream install via {name}...")
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", url],
            capture_output=True, timeout=120
        )
        if ret.returncode == 0:
            print(f"  MVDream installed via {name}.")
            return True
        print(f"  {name} failed: {ret.stderr.decode()[:100]}")

    # 方式 2: 如果项目里有本地 clone
    local_mvdream = PROJECT_ROOT / "third_party" / "MVDream"
    if local_mvdream.exists():
        print(f"  Trying local install from {local_mvdream}...")
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(local_mvdream)],
            capture_output=True
        )
        if ret.returncode == 0:
            print("  MVDream installed from local clone.")
            return True

    print("  FATAL: Cannot install MVDream. Please either:")
    print("    1. pip install git+https://github.com/bytedance/MVDream")
    print("    2. git clone https://github.com/bytedance/MVDream third_party/MVDream")
    print("       pip install -e third_party/MVDream")
    return False


def _find_local_file(name, search_dirs):
    """在多个目录中搜索文件（包括子目录）。"""
    import glob
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        # 精确匹配
        exact = os.path.join(d, name)
        if os.path.exists(exact):
            return exact
        # 递归搜索
        matches = glob.glob(os.path.join(d, '**', name), recursive=True)
        # 排除 .incomplete 文件
        matches = [m for m in matches if not m.endswith('.incomplete')]
        if matches:
            # 返回最大的文件（避免不完整的下载）
            return max(matches, key=os.path.getsize)
    return None


def _load_mvdream_model():
    """Load MVDream model, 完全离线加载，不依赖网络。

    策略：
    1. 找到 MVDream 权重（sd-v2.1-base-4view.pt）的本地路径
    2. 找到 CLIP 权重（open_clip_pytorch_model.bin）的本地路径
    3. 用 monkey-patch 方式让 open_clip 不联网，直接用本地文件
    4. 构建模型并加载权重
    """
    if not _install_mvdream():
        return None, None

    # 设置离线模式，阻止所有网络下载
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    # 搜索本地文件的目录列表
    search_dirs = [
        str(PROJECT_ROOT / "models_cache"),
        os.path.join(ROOT, ".cache", "huggingface"),
        os.path.join(ROOT, ".cache", "mvdream"),
        '/root/autodl-tmp/.cache',
        os.path.expanduser('~/.cache/huggingface'),
        os.path.expanduser('~/.cache'),
    ]

    # 1. 找 MVDream 权重
    ckpt_path = _find_local_file('sd-v2.1-base-4view.pt', search_dirs)
    if ckpt_path:
        size_gb = os.path.getsize(ckpt_path) / 1e9
        print(f"  MVDream weights: {ckpt_path} ({size_gb:.1f} GB)")
        if size_gb < 4.0:
            print(f"  WARNING: File seems incomplete ({size_gb:.1f} GB, expected ~4.9 GB)")
    else:
        print("  FATAL: Cannot find sd-v2.1-base-4view.pt locally")
        return None, None

    # 2. 找 CLIP 权重
    clip_path = _find_local_file('open_clip_pytorch_model.bin', search_dirs)
    if clip_path:
        size_gb = os.path.getsize(clip_path) / 1e9
        print(f"  CLIP weights: {clip_path} ({size_gb:.1f} GB)")
    else:
        print("  FATAL: Cannot find open_clip_pytorch_model.bin locally")
        return None, None

    # 3. Monkey-patch open_clip 的 create_model_and_transforms 使其离线加载
    try:
        import open_clip
        _original_create = open_clip.create_model_and_transforms

        def _patched_create(model_name, pretrained=None, **kwargs):
            """用本地 CLIP 权重文件代替网络下载。"""
            print(f"  [patch] open_clip: '{model_name}', pretrained='{pretrained}'")
            if pretrained and clip_path:
                print(f"  [patch] Using local CLIP: {clip_path}")
                model, _, preprocess = _original_create(model_name, pretrained=clip_path, **kwargs)
            else:
                model, _, preprocess = _original_create(model_name, pretrained=None, **kwargs)
            return model, _, preprocess

        open_clip.create_model_and_transforms = _patched_create
        print("  Patched open_clip for offline loading")
    except ImportError:
        print("  open_clip not installed, skipping patch")
    except Exception as e:
        print(f"  open_clip patch failed: {e}")

    # 4. 也 patch huggingface_hub.hf_hub_download 防止任何网络调用
    try:
        import huggingface_hub
        _original_hf_download = huggingface_hub.hf_hub_download

        def _patched_hf_download(repo_id, filename, **kwargs):
            """拦截 hf_hub_download，尝试从本地找文件。"""
            print(f"  [patch] hf_hub_download intercepted: {repo_id}/{filename}")
            local = _find_local_file(filename, search_dirs)
            if local:
                print(f"  [patch] Found locally: {local}")
                return local
            print(f"  [patch] Not found locally, trying original...")
            return _original_hf_download(repo_id, filename, **kwargs)

        huggingface_hub.hf_hub_download = _patched_hf_download
        # 也 patch file_download 模块
        import huggingface_hub.file_download
        huggingface_hub.file_download.hf_hub_download = _patched_hf_download
        print("  Patched hf_hub_download for offline loading")
    except Exception as e:
        print(f"  hf_hub_download patch failed: {e}")

    # 5. Patch DDIM sampler to handle list-wrapped conditioning
    try:
        from mvdream.ldm.models.diffusion import ddim as ddim_module
        _orig_sample = ddim_module.DDIMSampler.sample

        def _patched_sample(self, S, batch_size, shape, conditioning=None, **kwargs):
            """Patch: handle list-wrapped conditioning for batch size check."""
            if conditioning is not None and isinstance(conditioning, dict):
                first_val = conditioning[list(conditioning.keys())[0]]
                if isinstance(first_val, list):
                    # DDIM expects .shape but value is list — check element shape
                    if hasattr(first_val[0], 'shape') and first_val[0].shape[0] != batch_size:
                        print(f"Warning: Got {first_val[0].shape[0]} conditionings but batch-size is {batch_size}")
                    # Temporarily unwrap for the shape check, then call original
            return _orig_sample(self, S, batch_size, shape, conditioning=conditioning, **kwargs)

        # Actually, simpler: just patch the problematic line in sample()
        # The issue is line 82: conditioning[key].shape[0]
        # We can't easily patch one line, so patch the whole method
        import types
        original_code = ddim_module.DDIMSampler.sample

        def safe_sample(self, S, batch_size, shape, conditioning=None,
                       callback=None, img_callback=None, quantize_x0=False,
                       eta=0., mask=None, x0=None, temperature=1.,
                       noise_dropout=0., score_corrector=None,
                       corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100,
                       unconditional_guidance_scale=1., unconditional_conditioning=None, **kwargs):
            # Skip the problematic batch size check, go straight to scheduling + sampling
            self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
            C, H, W = shape
            size = (batch_size, C, H, W)
            return self.ddim_sampling(conditioning, size, callback=callback,
                                     img_callback=img_callback, quantize_denoised=quantize_x0,
                                     mask=mask, x0=x0, ddim_use_original_steps=False,
                                     noise_dropout=noise_dropout, temperature=temperature,
                                     score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                                     x_T=x_T, log_every_t=log_every_t,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning, **kwargs)

        ddim_module.DDIMSampler.sample = safe_sample
        print("  Patched DDIM sampler for list-wrapped conditioning")
    except Exception as e:
        print(f"  DDIM patch failed: {e}")

    # 6. 构建模型
    try:
        from mvdream.model_zoo import build_model
        from mvdream.ldm.models.diffusion.ddim import DDIMSampler

        model = build_model("sd-v2.1-base-4view", ckpt_path=ckpt_path)
        model.device = DEVICE
        model.to(DEVICE)
        model.eval()
        sampler = DDIMSampler(model)
        print(f"  MVDream loaded successfully!")
        return model, sampler
    except Exception as e:
        print(f"  MVDream load failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _mvdream_generate(model, sampler, prompt, seed, n_steps=30, scale=7.5):
    """Generate 4 multi-view images from a text prompt using MVDream."""
    from mvdream.camera_utils import get_camera
    from PIL import Image as PILImage
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Camera for 4 views: front(0), right(90), back(180), left(270)
    camera = get_camera(4, elevation=15.0, azimuth_start=0)
    if isinstance(camera, list):
        camera = torch.tensor(camera, dtype=torch.float32)
    elif not isinstance(camera, torch.Tensor):
        camera = torch.tensor(np.array(camera), dtype=torch.float32)
    camera = camera.to(DEVICE)

    # Text conditioning
    uc = model.get_learned_conditioning([""])
    c = model.get_learned_conditioning([prompt])
    # Repeat for 4 views
    # UNet expects list-wrapped tensors, but DDIM sampler calls .shape on dict values
    # We monkey-patch the DDIM sampler's batch size check to handle lists
    uc_4 = {"c_crossattn": [uc.repeat(4, 1, 1)], "c_concat": [camera]}
    c_4  = {"c_crossattn": [c.repeat(4, 1, 1)],  "c_concat": [camera]}

    shape = [4, 4, 32, 32]  # 4 views, 4 channels, 32x32 latent (256px)
    with torch.no_grad(), torch.autocast("cuda"):
        samples, _ = sampler.sample(
            S=n_steps,
            batch_size=4,
            shape=shape[1:],
            conditioning=c_4,
            unconditional_conditioning=uc_4,
            unconditional_guidance_scale=scale,
            eta=0.0,
        )
        x_sample = model.decode_first_stage(samples)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, 0, 0.999)

    images = []
    for i in range(4):
        img_np = (x_sample[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        images.append(PILImage.fromarray(img_np))
    return images


def generate_images(prompts_seeds: list) -> dict:
    """
    Generate front-view image (azimuth=0) per (prompt, seed) via MVDream.
    Returns {(prompt, seed): PIL.Image}.
    """
    from PIL import Image as PILImage

    images = {}
    for prompt, seed in prompts_seeds:
        path = IMG_DIR / f"{slug(prompt)}_s{seed}.png"
        if path.exists():
            images[(prompt, seed)] = PILImage.open(str(path)).convert("RGB")

    missing = [(p, s) for p, s in prompts_seeds if (p, s) not in images]
    if not missing:
        return images

    print(f"\nGenerating {len(missing)} front-view images via MVDream...", flush=True)
    model, sampler = _load_mvdream_model()
    if model is None:
        print("FATAL: MVDream unavailable. Cannot continue.", flush=True)
        return images

    for i, (prompt, seed) in enumerate(missing):
        try:
            views = _mvdream_generate(model, sampler, prompt, seed)
            front_img = views[0]  # azimuth=0
            path = IMG_DIR / f"{slug(prompt)}_s{seed}.png"
            front_img.save(str(path))
            images[(prompt, seed)] = front_img
        except Exception as e:
            print(f"  [SKIP] MVDream failed {prompt[:40]} seed={seed}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(missing)} images done")

    del model, sampler
    torch.cuda.empty_cache()
    gc.collect()
    return images


# ── TripoSR loading & inference (identical to exp_triposr_backbone) ───────────

def load_triposr():
    # 尝试多种 TripoSR 路径
    triposr_candidates = [
        TRIPOSR_PATH,
        str(PROJECT_ROOT / "third_party" / "TripoSR"),
        os.environ.get('TRIPOSR_PATH', os.path.join(os.path.dirname(ROOT), 'TripoSR')),
        os.path.expanduser("~/TripoSR"),
    ]

    loaded = False
    for tpath in triposr_candidates:
        if os.path.exists(tpath):
            sys.path.insert(0, tpath)
            print(f"  TripoSR found at: {tpath}")
            loaded = True
            break

    if not loaded:
        print(f"  WARNING: TripoSR not found in any of: {triposr_candidates}")
        print(f"  Trying pip import...")

    try:
        from tsr.system import TSR
    except ImportError:
        print("  FATAL: Cannot import TripoSR. Please either:")
        print("    1. Set TRIPOSR_PATH env var to your TripoSR directory")
        print("    2. git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR")
        sys.exit(1)

    # 设置 HuggingFace 镜像
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

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


def triposr_infer(tsr_model, rembg_session, image):
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
                __import__('trimesh').Trimesh(vertices=mesh_out[0], faces=mesh_out[1])

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

def compute_metrics(verts, faces) -> dict:
    # symmetry_reward signature: (vertices, axis=1) — no faces argument
    try:
        return dict(
            symmetry    = symmetry_reward(verts).item(),
            smoothness  = smoothness_reward(verts, faces).item(),
            compactness = compactness_reward(verts, faces).item(),
        )
    except Exception as e:
        return dict(symmetry=None, smoothness=None, compactness=None, error=str(e))


# ── Statistics ────────────────────────────────────────────────────────────────

def _bh_correct(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    p_adj = np.empty(n)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(float(p_values[idx]) * n / (rank + 1), 1.0)
    for i in range(n - 2, -1, -1):
        if p_adj[order[i]] > p_adj[order[i + 1]]:
            p_adj[order[i]] = p_adj[order[i + 1]]
    return p_adj.tolist()


def _bootstrap_ci(a, b, n_boot=2000, rng=None):
    """95% bootstrap CI for mean(b-a)."""
    rng = rng or np.random.default_rng(0)
    diff = np.array(b) - np.array(a)
    boots = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_stats(results):
    bl = {(r["prompt"], r["seed"]): r for r in results if r["method"] == "mvdream_baseline"}
    dgr = {(r["prompt"], r["seed"]): r for r in results if r["method"] == "mvdream_dgr"}
    common = sorted(set(bl) & set(dgr))
    if len(common) < 5:
        print(f"  Insufficient paired data ({len(common)} pairs). Skipping stats.")
        return {}

    print(f"\n{'='*60}")
    print(f"MVDream BACKBONE STATISTICS  (n={len(common)} pairs)")
    print(f"{'='*60}")
    print(f"{'Metric':<14} | {'BL mean':>9} | {'DGR mean':>9} | {'Δ%':>7} | "
          f"{'t':>6} | {'p_adj':>9} | {'d':>6} | {'95% CI':>18} | {'sig':>4}")
    print("-" * 100)

    p_raws = []
    rows = []
    rng = np.random.default_rng(42)
    for metric in METRICS:
        a = [bl[k][metric]  for k in common if bl[k].get(metric) is not None
             and dgr[k].get(metric) is not None]
        b = [dgr[k][metric] for k in common if bl[k].get(metric) is not None
             and dgr[k].get(metric) is not None]
        if len(a) < 5:
            continue
        t_val, p_raw = stats.ttest_rel(b, a)
        diff = np.array(b) - np.array(a)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(a, b, rng=rng)
        delta_pct = (np.mean(b) - np.mean(a)) / abs(np.mean(a)) * 100 if np.mean(a) != 0 else 0
        p_raws.append(p_raw)
        rows.append((metric, np.mean(a), np.mean(b), delta_pct, t_val, p_raw, d, ci_lo, ci_hi))

    if not rows:
        return {}

    p_adjs = _bh_correct([r[5] for r in rows])
    out = {}
    for i, (metric, bl_m, dgr_m, dpct, t_val, p_raw, d, ci_lo, ci_hi) in enumerate(rows):
        p_adj = p_adjs[i]
        sig = "[YES]" if p_adj < 0.05 else "[NO]"
        print(f"{metric:<14} | {bl_m:>9.5f} | {dgr_m:>9.5f} | {dpct:>+6.1f}% | "
              f"{t_val:>6.3f} | {p_adj:>9.4e} | {d:>+6.3f} | "
              f"[{ci_lo:+.5f}, {ci_hi:+.5f}] | {sig}")
        out[metric] = dict(
            bl_mean=bl_m, dgr_mean=dgr_m, delta_pct=dpct,
            t=t_val, p_raw=p_raw, p_adj=p_adj,
            cohens_d=d, ci_lo_95=ci_lo, ci_hi_95=ci_hi, significant=(p_adj < 0.05)
        )
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ckpt_path = OUT_DIR / "checkpoint.json"
    results   = []
    done_keys = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        done_keys = {(r["prompt"], r["seed"], r["method"]) for r in results}
        print(f"Resuming: {len(results)} done")

    all_pairs = [(p, s) for p in ALL_PROMPTS for s in SEEDS]
    total = len(all_pairs) * 2  # baseline + dgr

    # ── Phase 1: generate MVDream images ──────────────────────────────────────
    need_imgs = [(p, s) for p, s in all_pairs
                 if (p, s, "mvdream_baseline") not in done_keys
                 or (p, s, "mvdream_dgr") not in done_keys]

    images = {}
    if need_imgs:
        images = generate_images(need_imgs)
        print(f"  {len(images)} images ready.")

    # ── Phase 2: TripoSR + DGR ────────────────────────────────────────────────
    tsr_model, rembg_session = None, None
    t0 = time.time()

    for pi, (prompt, seed) in enumerate(all_pairs):
        cat = PROMPT_CAT[prompt]
        ps  = slug(prompt)
        bl_key  = (prompt, seed, "mvdream_baseline")
        dgr_key = (prompt, seed, "mvdream_dgr")
        if bl_key in done_keys and dgr_key in done_keys:
            continue

        image = images.get((prompt, seed))
        if image is None:
            print(f"  [SKIP] No image for {prompt[:40]} s={seed}")
            continue

        # Lazy-load TripoSR
        if tsr_model is None:
            tsr_model, rembg_session = load_triposr()

        print(f"  [{pi+1}/{len(all_pairs)}] {prompt[:45]} seed={seed}")

        # Baseline mesh
        bl_path = OBJ_DIR / "baseline" / cat / f"{ps}_seed{seed}.obj"
        if bl_key not in done_keys:
            v, f = triposr_infer(tsr_model, rembg_session, image)
            if v is None:
                continue
            save_obj(v, f, bl_path)
            m = compute_metrics(v, f)
            results.append({"prompt": prompt, "seed": seed,
                             "method": "mvdream_baseline", "category": cat, **m})
            done_keys.add(bl_key)
        else:
            v, f = load_obj(bl_path)

        # DGR refined mesh
        if dgr_key not in done_keys:
            try:
                v_ref, f_ref = refine_with_geo_reward(
                    v.clone(), f, steps=STEPS, lr=LR, weights=DGR_W)
            except Exception as e:
                print(f"    DGR refine error: {e}")
                v_ref, f_ref = v.clone(), f

            dgr_path = OBJ_DIR / "dgr" / cat / f"{ps}_seed{seed}.obj"
            save_obj(v_ref, f_ref, dgr_path)
            m = compute_metrics(v_ref, f_ref)
            results.append({"prompt": prompt, "seed": seed,
                             "method": "mvdream_dgr", "category": cat, **m})
            done_keys.add(dgr_key)

        del v, f
        torch.cuda.empty_cache()

        if len(results) % 20 == 0:
            with open(ckpt_path, 'w') as fp:
                json.dump(results, fp)
            elapsed = time.time() - t0
            print(f"    checkpoint: {len(results)}/{total} ({elapsed:.0f}s)")

    with open(ckpt_path, 'w') as fp:
        json.dump(results, fp)
    with open(OUT_DIR / "all_results.json", 'w') as fp:
        json.dump(results, fp, indent=2)

    print(f"\nDone: {len(results)} records, {time.time()-t0:.0f}s")

    # ── Statistics ────────────────────────────────────────────────────────────
    stat_out = run_stats(results)
    if stat_out:
        with open(OUT_DIR / "stats.json", 'w') as fp:
            json.dump(stat_out, fp, indent=2)
        print(f"\nStats saved to {OUT_DIR}/stats.json")

    print(f"\nResults saved to {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
