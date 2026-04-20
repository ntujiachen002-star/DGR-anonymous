"""Debug MVDream image generation step by step."""
import os, sys, torch, numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault('HF_HOME', os.path.join(ROOT, '.cache', 'huggingface'))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda:0"

# Load model using our patched loader
exec(open("tools/exp_q_mvdream_backbone.py").read().split("def _mvdream_generate")[0])
model, sampler = _load_mvdream_model()
print(f"Model loaded: {model is not None}")

if model is None:
    sys.exit(1)

# Debug image generation step by step
from mvdream.camera_utils import get_camera

print("\n=== Step 1: Camera ===")
camera = get_camera(4, elevation=15.0, azimuth_start=0)
print(f"  type: {type(camera)}")
if isinstance(camera, list):
    print(f"  len: {len(camera)}")
    camera = torch.tensor(camera, dtype=torch.float32)
elif not isinstance(camera, torch.Tensor):
    camera = torch.tensor(np.array(camera), dtype=torch.float32)
print(f"  tensor shape: {camera.shape}")
camera = camera.to(DEVICE)

print("\n=== Step 2: Text conditioning ===")
prompt = "a symmetric vase"
try:
    uc = model.get_learned_conditioning([""])
    print(f"  uc type: {type(uc)}, ", end="")
    if isinstance(uc, torch.Tensor):
        print(f"shape: {uc.shape}")
    elif isinstance(uc, list):
        print(f"len: {len(uc)}, element types: {[type(x) for x in uc]}")
    else:
        print(f"value: {uc}")
except Exception as e:
    print(f"  uc FAILED: {e}")
    import traceback; traceback.print_exc()

try:
    c = model.get_learned_conditioning([prompt])
    print(f"  c type: {type(c)}, ", end="")
    if isinstance(c, torch.Tensor):
        print(f"shape: {c.shape}")
    elif isinstance(c, list):
        print(f"len: {len(c)}, element types: {[type(x) for x in c]}")
    else:
        print(f"value: {c}")
except Exception as e:
    print(f"  c FAILED: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 3: Build conditioning dict ===")
try:
    if isinstance(uc, torch.Tensor):
        uc_repeat = uc.repeat(4, 1, 1)
    else:
        print(f"  WARNING: uc is {type(uc)}, not Tensor. Trying conversion...")
        uc = torch.tensor(uc) if not isinstance(uc, torch.Tensor) else uc
        uc_repeat = uc.repeat(4, 1, 1)

    if isinstance(c, torch.Tensor):
        c_repeat = c.repeat(4, 1, 1)
    else:
        print(f"  WARNING: c is {type(c)}, not Tensor. Trying conversion...")
        c = torch.tensor(c) if not isinstance(c, torch.Tensor) else c
        c_repeat = c.repeat(4, 1, 1)

    uc_4 = {"c_crossattn": [uc_repeat], "c_concat": [camera]}
    c_4 = {"c_crossattn": [c_repeat], "c_concat": [camera]}
    print(f"  uc_4 keys: {list(uc_4.keys())}")
    print(f"  c_4 keys: {list(c_4.keys())}")
    print("  Conditioning built OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 4: Sample ===")
try:
    shape = [4, 4, 32, 32]
    with torch.no_grad(), torch.autocast("cuda"):
        samples, _ = sampler.sample(
            S=5,  # fewer steps for debug
            batch_size=4,
            shape=shape[1:],
            conditioning=c_4,
            unconditional_conditioning=uc_4,
            unconditional_guidance_scale=7.5,
            eta=0.0,
        )
    print(f"  samples shape: {samples.shape}")
    print("  Sampling OK!")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
