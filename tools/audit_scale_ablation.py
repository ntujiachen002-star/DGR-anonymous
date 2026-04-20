import os
"""Audit the scale_controlled_ablation result. Check:

1. What was the baseline sym distribution for compact_only runs?
2. Is the -281% driven by small denominators (near-zero baselines) or
   by genuinely large delta?
3. Is the plane cache consistent with production?
4. Does dgr_standard match the main table's dgr_standard?
"""
import json
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALE = f'{ROOT}/analysis_results/scale_controlled_ablation/scale_controlled_results.json'
ANTI  = f'{ROOT}/analysis_results/anticollapse/anticollapse_results.json'
SCALE_PLANE = f'{ROOT}/analysis_results/scale_controlled_ablation/plane_cache.json'
ANTI_PLANE  = f'{ROOT}/analysis_results/anticollapse/plane_cache.json'
PROD_PLANE  = f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json'


def load(p):
    with open(p) as f: return json.load(f)


def main():
    print('=' * 78)
    print('AUDIT 1: scale_controlled_ablation — per-run baseline distribution')
    print('=' * 78)

    scale = load(SCALE)
    print(f'\ntop-level type: {type(scale).__name__}')
    if isinstance(scale, dict):
        print(f'top-level keys: {list(scale.keys())[:10]}')
        # try to find the records list
        for k in ('results', 'records', 'all_results'):
            if k in scale:
                recs = scale[k]
                print(f'  using key "{k}" -> {len(recs)} records')
                break
        else:
            # maybe flat dict of method -> records
            first_k = list(scale.keys())[0]
            if isinstance(scale[first_k], list):
                recs = []
                for k, v in scale.items():
                    for r in v:
                        r['_method'] = k
                        recs.append(r)
                print(f'  flattened nested -> {len(recs)} records')
            else:
                recs = []
    else:
        recs = scale
        print(f'  flat list -> {len(recs)} records')

    if not recs:
        print('  (no records, trying raw inspection)')
        print(json.dumps(scale, indent=2)[:2000])
        return

    # What fields does a record have?
    print(f'\nsample record keys: {list(recs[0].keys())}')
    print(f'sample record: {json.dumps(recs[0], indent=2, default=str)[:800]}')

    # Group by method
    by_method = defaultdict(list)
    for r in recs:
        m = r.get('method', r.get('_method', r.get('config', 'unknown')))
        by_method[m].append(r)

    print(f'\nmethods in file: {dict((k, len(v)) for k, v in by_method.items())}')


if __name__ == '__main__':
    main()
