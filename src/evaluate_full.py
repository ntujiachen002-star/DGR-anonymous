"""
Full evaluation: analyze results from run_full_experiment.py.
Produces paper-ready tables and analysis.

Usage:
    python src/evaluate_full.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_all_results():
    """Load all result files from results/full/."""
    all_metrics = []
    results_dir = 'results/full'

    for fname in os.listdir(results_dir):
        if fname.endswith('_all_metrics.json'):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
                all_metrics.extend(data)

    return all_metrics


def print_table_1(metrics):
    """Table 1: Main result — all methods overall comparison."""
    print("\n" + "=" * 80)
    print("TABLE 1: Overall Results (All Methods)")
    print("=" * 80)

    methods = sorted(set(m['method'] for m in metrics))

    header = "{:<15} {:>10} {:>12} {:>12} {:>8} {:>8}".format(
        "Method", "Symmetry", "Smoothness", "Compactness", "Time(s)", "CLIP")
    print(header)
    print("-" * 80)

    baseline_vals = {}
    for method in methods:
        mm = [m for m in metrics if m['method'] == method]
        if not mm:
            continue

        sym = np.mean([m['symmetry'] for m in mm])
        smooth = np.mean([m['smoothness'] for m in mm])
        compact = np.mean([m['compactness'] for m in mm])
        time_avg = np.mean([m['total_time'] for m in mm])
        clips = [m.get('clip_score', 0) for m in mm]
        clip_avg = np.mean(clips) if any(c > 0 for c in clips) else 0

        if method == 'baseline':
            baseline_vals = {'sym': sym, 'smooth': smooth, 'compact': compact, 'clip': clip_avg}

        clip_str = "{:.4f}".format(clip_avg) if clip_avg > 0 else "N/A"
        print("{:<15} {:>10.6f} {:>12.6f} {:>12.4f} {:>8.1f} {:>8}".format(
            method, sym, smooth, compact, time_avg, clip_str))

    # Print deltas
    if baseline_vals:
        print("\n--- Delta vs Baseline ---")
        for method in methods:
            if method == 'baseline':
                continue
            mm = [m for m in metrics if m['method'] == method]
            if not mm:
                continue

            sym = np.mean([m['symmetry'] for m in mm])
            smooth = np.mean([m['smoothness'] for m in mm])
            compact = np.mean([m['compactness'] for m in mm])
            clips = [m.get('clip_score', 0) for m in mm]
            clip_avg = np.mean(clips) if any(c > 0 for c in clips) else 0

            d_sym = sym - baseline_vals['sym']
            d_smooth = smooth - baseline_vals['smooth']
            d_compact = compact - baseline_vals['compact']
            d_clip = clip_avg - baseline_vals['clip'] if clip_avg > 0 and baseline_vals['clip'] > 0 else 0

            sym_pct = abs(d_sym / baseline_vals['sym']) * 100 if baseline_vals['sym'] != 0 else 0
            smooth_pct = abs(d_smooth / baseline_vals['smooth']) * 100 if baseline_vals['smooth'] != 0 else 0

            clip_str = "{:+.4f}".format(d_clip) if d_clip != 0 else "N/A"
            print("{:<15} {:>+10.6f} ({:.0f}%) {:>+10.6f} ({:.0f}%) {:>+10.4f} {:>8}".format(
                method, d_sym, sym_pct, d_smooth, smooth_pct, d_compact, clip_str))


def print_table_2(metrics):
    """Table 2: Per-category results (targeted property improvement)."""
    print("\n" + "=" * 80)
    print("TABLE 2: Per-Category Results (Targeted Property)")
    print("=" * 80)

    categories = ['symmetry', 'smoothness', 'compactness']
    methods = sorted(set(m['method'] for m in metrics))

    for cat in categories:
        print("\n--- Category: {} ---".format(cat.upper()))
        target_metric = cat  # The metric that should improve most for this category

        header = "{:<15} {:>10} {:>12} {:>12}".format(
            "Method", "Symmetry", "Smoothness", "Compactness")
        print(header)
        print("-" * 55)

        baseline_cat = {}
        for method in methods:
            mm = [m for m in metrics if m['method'] == method and m.get('category') == cat]
            if not mm:
                continue

            sym = np.mean([m['symmetry'] for m in mm])
            smooth = np.mean([m['smoothness'] for m in mm])
            compact = np.mean([m['compactness'] for m in mm])

            if method == 'baseline':
                baseline_cat = {'symmetry': sym, 'smoothness': smooth, 'compactness': compact}

            # Bold the target metric (mark with *)
            sym_mark = " *" if target_metric == 'symmetry' and method != 'baseline' else ""
            smooth_mark = " *" if target_metric == 'smoothness' and method != 'baseline' else ""
            compact_mark = " *" if target_metric == 'compactness' and method != 'baseline' else ""

            print("{:<15} {:>10.6f}{} {:>10.6f}{} {:>10.4f}{}".format(
                method, sym, sym_mark, smooth, smooth_mark, compact, compact_mark))


def print_table_3(metrics):
    """Table 3: Ablation — Lang2Comp vs equal weights."""
    print("\n" + "=" * 80)
    print("TABLE 3: Ablation — Lang2Comp vs Equal Weights")
    print("=" * 80)

    categories = ['symmetry', 'smoothness', 'compactness']

    for cat in categories:
        print("\n--- {} prompts ---".format(cat))
        target = cat

        for method in ['diffgeoreward', 'handcrafted']:
            mm = [m for m in metrics if m['method'] == method and m.get('category') == cat]
            if not mm:
                continue

            val = np.mean([m[target] for m in mm])
            print("  {:<15}: {} = {:.6f}".format(method, target, val))


def print_optimization_analysis(metrics):
    """Optimization statistics for reward-based methods."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 80)

    reward_methods = [m for m in metrics if 'reward_improvement' in m]
    methods = sorted(set(m['method'] for m in reward_methods))

    for method in methods:
        mm = [m for m in reward_methods if m['method'] == method]
        if not mm:
            continue

        improvements = [m['reward_improvement'] for m in mm]
        grad_norms = [m.get('avg_grad_norm', 0) for m in mm]

        print("\n--- {} ---".format(method))
        print("  Runs: {}".format(len(mm)))
        print("  Avg reward improvement: {:.6f}".format(np.mean(improvements)))
        print("  Median reward improvement: {:.6f}".format(np.median(improvements)))
        print("  % positive improvement: {:.1f}%".format(
            100 * np.mean([i > 0 for i in improvements])))
        print("  Avg gradient norm: {:.4f}".format(np.mean(grad_norms)))
        print("  Min/Max improvement: {:.6f} / {:.6f}".format(
            np.min(improvements), np.max(improvements)))


def print_failure_analysis(metrics):
    """Analyze cases where DiffGeoReward made things worse."""
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    dgr = [m for m in metrics if m['method'] == 'diffgeoreward' and 'reward_improvement' in m]
    failures = [m for m in dgr if m['reward_improvement'] < 0]

    print("Total DiffGeoReward runs: {}".format(len(dgr)))
    print("Negative reward improvement: {} ({:.1f}%)".format(
        len(failures), 100 * len(failures) / max(len(dgr), 1)))

    if failures:
        print("\nFailure cases:")
        for m in sorted(failures, key=lambda x: x['reward_improvement']):
            print("  prompt='{}' seed={} improvement={:.6f} category={}".format(
                m['prompt'][:50], m['seed'], m['reward_improvement'],
                m.get('category', '?')))


def print_statistical_significance(metrics):
    """Basic statistical tests."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    baseline = [m for m in metrics if m['method'] == 'baseline']
    dgr = [m for m in metrics if m['method'] == 'diffgeoreward']

    if not baseline or not dgr:
        print("Insufficient data for statistical analysis")
        return

    for metric_name in ['symmetry', 'smoothness', 'compactness']:
        b_vals = np.array([m[metric_name] for m in baseline])
        d_vals = np.array([m[metric_name] for m in dgr])

        # Per-prompt paired comparison (average across seeds first)
        b_by_prompt = defaultdict(list)
        d_by_prompt = defaultdict(list)
        for m in baseline:
            b_by_prompt[m['prompt']].append(m[metric_name])
        for m in dgr:
            d_by_prompt[m['prompt']].append(m[metric_name])

        common_prompts = set(b_by_prompt.keys()) & set(d_by_prompt.keys())
        b_means = [np.mean(b_by_prompt[p]) for p in sorted(common_prompts)]
        d_means = [np.mean(d_by_prompt[p]) for p in sorted(common_prompts)]

        diff = np.array(d_means) - np.array(b_means)
        win_rate = np.mean(diff > 0)

        print("\n{}: DiffGeoReward wins on {:.1f}% of prompts ({}/{})".format(
            metric_name, 100 * win_rate, int(win_rate * len(diff)), len(diff)))
        print("  Mean diff: {:.6f}, Std: {:.6f}".format(np.mean(diff), np.std(diff)))

        # Simple paired t-test
        if len(diff) > 1:
            t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
            print("  Paired t-stat: {:.3f} (n={})".format(t_stat, len(diff)))


def save_report(metrics):
    """Save structured report for downstream use."""
    methods = sorted(set(m['method'] for m in metrics))
    report = {}

    for method in methods:
        mm = [m for m in metrics if m['method'] == method]
        report[method] = {
            'n_runs': len(mm),
            'symmetry': float(np.mean([m['symmetry'] for m in mm])),
            'smoothness': float(np.mean([m['smoothness'] for m in mm])),
            'compactness': float(np.mean([m['compactness'] for m in mm])),
            'time': float(np.mean([m['total_time'] for m in mm])),
        }
        clips = [m.get('clip_score', 0) for m in mm if m.get('clip_score', 0) > 0]
        if clips:
            report[method]['clip_score'] = float(np.mean(clips))

        if any('reward_improvement' in m for m in mm):
            improvements = [m.get('reward_improvement', 0) for m in mm]
            report[method]['avg_reward_improvement'] = float(np.mean(improvements))
            report[method]['pct_positive'] = float(np.mean([i > 0 for i in improvements]))

    os.makedirs('results/full', exist_ok=True)
    with open('results/full/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nReport saved to results/full/evaluation_report.json")


def main():
    metrics = load_all_results()
    if not metrics:
        print("ERROR: No results found in results/full/")
        return

    print("Loaded {} total experiment runs".format(len(metrics)))
    methods = sorted(set(m['method'] for m in metrics))
    print("Methods: {}".format(', '.join(methods)))
    print("Categories: {}".format(', '.join(sorted(set(m.get('category', '?') for m in metrics)))))

    print_table_1(metrics)
    print_table_2(metrics)
    print_table_3(metrics)
    print_optimization_analysis(metrics)
    print_failure_analysis(metrics)
    print_statistical_significance(metrics)
    save_report(metrics)


if __name__ == '__main__':
    main()
