"""Test MVDream loading on server."""
import os, sys, glob
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault('HF_HOME', os.path.join(ROOT, '.cache', 'huggingface'))

search_dirs = [
    os.path.join(ROOT, ".cache", "huggingface"),
    os.path.join(ROOT, ".cache", "mvdream"),
    "/root/autodl-tmp/.cache",
]

for name in ["sd-v2.1-base-4view.pt", "open_clip_pytorch_model.bin"]:
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        matches = glob.glob(os.path.join(d, "**", name), recursive=True)
        real_matches = [m for m in matches if not m.endswith(".incomplete")]
        for m in real_matches:
            is_link = os.path.islink(m)
            size = os.path.getsize(m) if not is_link else os.path.getsize(os.path.realpath(m))
            print(f"  {name}: {m} ({size/1e9:.1f} GB) {'[symlink]' if is_link else ''}")

print("\nLoading MVDream...")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import and run _load_mvdream_model
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exec(open("tools/exp_q_mvdream_backbone.py").read().split("def _mvdream_generate")[0])
model, sampler = _load_mvdream_model()
if model is not None:
    print("SUCCESS!")
else:
    print("FAILED")
