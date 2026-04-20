import os
import torch
ckpt = torch.load(os.path.join(ROOT, ".cache", "mvdream/sd-v2.1-base-4view.pt"), map_location="cpu")
keys = list(ckpt.keys())
print(f"Total keys: {len(keys)}")
clip_keys = [k for k in keys if "cond_stage" in k or "clip" in k.lower() or "text" in k.lower()]
print(f"CLIP/text-related keys: {len(clip_keys)}")
for k in clip_keys[:15]:
    print(f"  {k}: {ckpt[k].shape}")
if not clip_keys:
    print("No CLIP keys! First 20:")
    for k in keys[:20]:
        v = ckpt[k]
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
