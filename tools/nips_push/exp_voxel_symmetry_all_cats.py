"""V-IoU on all 110 prompts (symmetry + smoothness + compactness), not just
the 37 symmetry-focused subset. Extends Appendix `sec:voxel_symmetry_appendix`.

Uses the main exp_voxel_symmetry_downstream machinery but with a larger
prompt list and a different output directory.
"""
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "nips_push"))
os.chdir(ROOT)

# Import AFTER sys.path/chdir so relative paths work
import exp_voxel_symmetry_downstream as ev  # noqa: E402
from prompts_gpteval3d import (  # noqa: E402
    SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS,
)

# Monkey-patch output directory
ev.OUT_DIR = Path("analysis_results/nips_push_voxel_sym_all")
ev.OUT_DIR.mkdir(parents=True, exist_ok=True)

# Override argv to pass all prompts
all_prompts = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
print(f"[info] V-IoU across {len(all_prompts)} prompts x 3 seeds = "
      f"{len(all_prompts) * 3} pairs", flush=True)
sys.argv = ["exp_voxel_symmetry_downstream.py", "--prompts", *all_prompts]

ev.main()
