"""Analyze the n=200 meshes causal plane experiment (adds statistical power
over the original n=100 run)."""
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools", "nips_push"))
os.chdir(ROOT)

import analyze_causal_plane as ac
ac.OUT_DIR = Path("analysis_results/nips_push_causal_plane_200")
ac.RESULTS_PATH = ac.OUT_DIR / "all_results.json"
ac.main()
