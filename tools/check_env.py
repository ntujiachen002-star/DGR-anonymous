"""
部署前环境检查脚本。

在 GPU 服务器上运行此脚本，确认所有依赖就绪后再启动实验。

用法:
  python tools/check_env.py
"""

import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def check(name, condition, fix=""):
    status = "[OK]" if condition else "[FAIL]"
    print(f"  {status} {name}")
    if not condition and fix:
        print(f"        Fix: {fix}")
    return condition


def main():
    print("=" * 60)
    print("DiffGeoReward 环境检查")
    print("=" * 60)
    all_ok = True

    # 1. Python & PyTorch
    print("\n[1] Python & PyTorch")
    all_ok &= check("Python >= 3.8", sys.version_info >= (3, 8))

    try:
        import torch
        all_ok &= check(f"PyTorch {torch.__version__}", True)
        has_cuda = torch.cuda.is_available()
        all_ok &= check(f"CUDA available: {has_cuda}", has_cuda,
                        "需要 NVIDIA GPU + CUDA")
        if has_cuda:
            print(f"        GPU: {torch.cuda.get_device_name(0)}")
            print(f"        VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    except ImportError:
        all_ok &= check("PyTorch", False, "pip install torch")

    # 2. 核心依赖
    print("\n[2] 核心依赖")
    for pkg in ['numpy', 'scipy', 'PIL', 'trimesh']:
        try:
            __import__(pkg)
            check(pkg, True)
        except ImportError:
            all_ok &= check(pkg, False, f"pip install {pkg}")

    # 3. DiffGeoReward 模块
    print("\n[3] DiffGeoReward 模块")
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    try:
        from geo_reward import symmetry_reward, smoothness_reward, compactness_reward
        check("geo_reward", True)
    except Exception as e:
        all_ok &= check(f"geo_reward: {e}", False)

    try:
        from shape_gen import load_shap_e
        check("shape_gen (Shap-E)", True)
    except Exception as e:
        all_ok &= check(f"shape_gen: {e}", False, "pip install shap-e")

    try:
        from prompts_gpteval3d import SYMMETRY_PROMPTS
        check(f"prompts_gpteval3d ({len(SYMMETRY_PROMPTS)} sym prompts)", True)
    except Exception as e:
        all_ok &= check(f"prompts_gpteval3d: {e}", False)

    # 4. MVDream
    print("\n[4] MVDream")
    mvdream_ckpt = PROJECT_ROOT / "models_cache" / "models--MVDream--MVDream" / \
                   "snapshots" / "d14ac9d78c48c266005729f2d5633f6c265da467" / \
                   "sd-v2.1-base-4view.pt"
    all_ok_mvdream = True

    all_ok_mvdream &= check(f"本地权重: {mvdream_ckpt.name}",
                            mvdream_ckpt.exists(),
                            f"需要将权重文件放到 {mvdream_ckpt}")
    if mvdream_ckpt.exists():
        size_gb = mvdream_ckpt.stat().st_size / 1e9
        check(f"  大小: {size_gb:.1f} GB", size_gb > 4.0)

    try:
        import mvdream
        check("mvdream 包已安装", True)
    except ImportError:
        check("mvdream 包未安装", False,
              "pip install git+https://github.com/bytedance/MVDream\n"
              "        或: git clone https://github.com/bytedance/MVDream third_party/MVDream && pip install -e third_party/MVDream")

    # 5. TripoSR
    print("\n[5] TripoSR")
    triposr_candidates = [
        os.environ.get('TRIPOSR_PATH', ''),
        os.environ.get('TRIPOSR_PATH', os.path.join(os.path.dirname(ROOT), 'TripoSR')),
        str(PROJECT_ROOT / 'third_party' / 'TripoSR'),
        os.path.expanduser('~/TripoSR'),
    ]
    triposr_found = False
    for tpath in triposr_candidates:
        if tpath and os.path.exists(tpath):
            check(f"TripoSR 目录: {tpath}", True)
            triposr_found = True
            break
    if not triposr_found:
        all_ok &= check("TripoSR 目录", False,
                        "git clone https://github.com/VAST-AI-Research/TripoSR /root/autodl-tmp/TripoSR\n"
                        "        或设置 TRIPOSR_PATH 环境变量")

    try:
        import rembg
        check("rembg (背景移除)", True)
    except ImportError:
        check("rembg (背景移除)", False, "pip install rembg[gpu]")

    # 6. 输出目录
    print("\n[6] 输出目录")
    for d in ['analysis_results/full_mgda_classical',
              'analysis_results/mvdream_backbone',
              'results/full_mgda_classical_objs',
              'results/mvdream_objs']:
        dpath = PROJECT_ROOT / d
        dpath.mkdir(parents=True, exist_ok=True)
        check(f"{d}/", dpath.exists())

    # 7. HuggingFace 镜像
    print("\n[7] HuggingFace 配置")
    hf_endpoint = os.environ.get('HF_ENDPOINT', '(未设置)')
    check(f"HF_ENDPOINT: {hf_endpoint}", True)
    if hf_endpoint == '(未设置)':
        print("        建议: export HF_ENDPOINT=https://hf-mirror.com")

    # 总结
    print(f"\n{'=' * 60}")
    if all_ok:
        print("[ALL OK] 环境就绪，可以运行实验！")
        print("\n  启动命令:")
        print("    # P0: MGDA + Classical (约 3-4h)")
        print("    python tools/exp_full_mgda_classical.py --device cuda:0 --resume")
        print()
        print("    # P1: MVDream (约 1.5h)")
        print("    python tools/exp_q_mvdream_backbone.py")
        print()
        print("    # 或一键全跑:")
        print("    bash tools/launch_all_gpu_experiments.sh 0")
    else:
        print("[ISSUES FOUND] 请修复上述问题后再运行实验。")
    print("=" * 60)


if __name__ == '__main__':
    main()
