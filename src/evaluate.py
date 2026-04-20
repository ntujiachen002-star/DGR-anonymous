"""
Evaluate experiment results: compare baseline vs DiffGeoReward.

Usage:
    python src/evaluate.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_metrics(method):
    """Load all metrics for a method."""
    path = f"results/{method}_all_metrics.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def analyze():
    baseline = load_metrics('baseline')
    diffgeo = load_metrics('diffgeoreward')

    if not baseline or not diffgeo:
        print("ERROR: Missing experiment results.")
        return

    print("=" * 70)
    print("DiffGeoReward Pilot Experiment — Evaluation Report")
    print("=" * 70)

    # Overall comparison
    print("\n## 1. Overall Results\n")
    print(f"{'Metric':<25} {'Baseline':>12} {'DiffGeoReward':>14} {'Delta':>10} {'Verdict':>10}")
    print("-" * 75)

    metrics_to_compare = [
        ('symmetry', 'Avg Symmetry', True),
        ('smoothness', 'Avg Smoothness', True),
        ('compactness', 'Avg Compactness', True),
        ('total_time', 'Avg Time (s)', False),
    ]

    results = {}
    for key, label, higher_better in metrics_to_compare:
        b_vals = [m[key] for m in baseline]
        d_vals = [m[key] for m in diffgeo]
        b_mean = np.mean(b_vals)
        d_mean = np.mean(d_vals)
        delta = d_mean - b_mean

        if higher_better:
            improved = delta > 0
        else:
            improved = delta < 0  # lower time is better

        verdict = "BETTER" if improved else "WORSE"
        results[key] = {'baseline': b_mean, 'diffgeo': d_mean, 'delta': delta, 'improved': improved}

        print(f"{label:<25} {b_mean:>12.6f} {d_mean:>14.6f} {delta:>+10.6f} {'  ' + verdict:>10}")

    # Per-category analysis
    print("\n\n## 2. Per-Category Results\n")

    categories = set(m.get('category', 'unknown') for m in baseline)
    for cat in sorted(categories):
        print(f"\n### Category: {cat}")
        b_cat = [m for m in baseline if m.get('category') == cat]
        d_cat = [m for m in diffgeo if m.get('category') == cat]

        print(f"{'Metric':<20} {'Baseline':>12} {'DiffGeoReward':>14} {'Delta':>10}")
        print("-" * 60)

        for key in ['symmetry', 'smoothness', 'compactness']:
            b_mean = np.mean([m[key] for m in b_cat])
            d_mean = np.mean([m[key] for m in d_cat])
            delta = d_mean - b_mean
            marker = " *" if delta > 0 else ""
            print(f"{key:<20} {b_mean:>12.6f} {d_mean:>14.6f} {delta:>+10.6f}{marker}")

    # DiffGeoReward specific analysis
    print("\n\n## 3. DiffGeoReward Optimization Analysis\n")

    improvements = [m.get('reward_improvement', 0) for m in diffgeo]
    grad_norms = [m.get('avg_grad_norm', 0) for m in diffgeo]

    print(f"Avg Reward Improvement: {np.mean(improvements):.6f}")
    print(f"Med Reward Improvement: {np.median(improvements):.6f}")
    print(f"Min Reward Improvement: {np.min(improvements):.6f}")
    print(f"Max Reward Improvement: {np.max(improvements):.6f}")
    print(f"% Positive Improvement: {100 * np.mean([i > 0 for i in improvements]):.1f}%")
    print(f"Avg Gradient Norm:      {np.mean(grad_norms):.4f}")

    # Go/No-Go assessment
    print("\n\n## 4. Go/No-Go Assessment\n")

    # Criterion 1: Gradient norms > 0.02
    avg_grad = np.mean(grad_norms)
    grad_ok = avg_grad > 0.02
    print(f"  [{'PASS' if grad_ok else 'FAIL'}] Gradient norm > 0.02: {avg_grad:.4f}")

    # Criterion 2: At least 2/3 properties improved
    props_improved = sum([
        results['symmetry']['improved'],
        results['smoothness']['improved'],
        results['compactness']['improved'],
    ])
    props_ok = props_improved >= 2
    print(f"  [{'PASS' if props_ok else 'FAIL'}] Properties improved >= 2/3: {props_improved}/3")

    # Criterion 3: Speed comparison
    b_time = results['total_time']['baseline']
    d_time = results['total_time']['diffgeo']
    # Note: DiffGeoReward includes refinement time, so compare fairly
    refine_times = [m.get('refine_time', 0) for m in diffgeo]
    avg_refine = np.mean(refine_times)
    print(f"  [INFO] Avg refinement time: {avg_refine:.1f}s (overhead over baseline)")

    # Criterion 4: Reward improvement positive
    reward_ok = np.mean(improvements) > 0
    print(f"  [{'PASS' if reward_ok else 'FAIL'}] Avg reward improvement > 0: {np.mean(improvements):.6f}")

    # Overall verdict
    print("\n" + "=" * 70)
    passed = sum([grad_ok, props_ok, reward_ok])
    if passed >= 3:
        verdict = "GO — Proceed to full paper"
    elif passed >= 2:
        verdict = "ITERATE — Tune hyperparameters, then re-evaluate"
    else:
        verdict = "PIVOT — Consider GeoGAN backup approach"

    print(f"VERDICT: {verdict} ({passed}/3 criteria passed)")
    print("=" * 70)

    # Save report
    # Convert numpy types to native Python for JSON
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer, float, int)):
                clean_results[k][kk] = float(vv)
            elif isinstance(vv, (np.bool_, bool)):
                clean_results[k][kk] = bool(vv)
            else:
                clean_results[k][kk] = vv

    report = {
        'overall': clean_results,
        'improvements': {
            'mean': float(np.mean(improvements)),
            'median': float(np.median(improvements)),
            'positive_pct': float(np.mean([i > 0 for i in improvements])),
        },
        'gradient': {
            'mean': float(np.mean(grad_norms)),
        },
        'verdict': verdict,
        'criteria_passed': int(passed),
    }

    with open('evaluation/pilot_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to evaluation/pilot_report.json")


if __name__ == '__main__':
    os.makedirs('evaluation', exist_ok=True)
    analyze()
