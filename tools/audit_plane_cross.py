import os
"""Cross-experiment plane cache consistency check.

For each (prompt, seed) baseline mesh, collect the plane normal from
every experiment's local cache and check if they agree. If planes
differ meaningfully, it could mean:
 1. The multi-start estimator is non-deterministic
 2. Different experiments re-estimated and converged to different local minima
 3. Different experiments were run on different baseline mesh sets
 4. The "production" cache and the other caches are genuinely different

This matters because §4.2's cross-experiment synthesis assumes all
experiments measured the same thing.
"""
import json
import numpy as np
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHES = {
    'production':   f'{ROOT}/analysis_results_newproto/plane_cache/production_plane_cache.json',
    'anticollapse': f'{ROOT}/analysis_results/anticollapse/plane_cache.json',
    'scale_abl':    f'{ROOT}/analysis_results/scale_controlled_ablation/plane_cache.json',
    'l2c_rerun':    f'{ROOT}/analysis_results/lang2comp_rerun/plane_cache.json',
}


def load(p):
    try:
        with open(p) as f: return json.load(f)
    except FileNotFoundError:
        return None


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def normalize_key(k):
    """Return a canonical (prompt, seed) tuple from any key format."""
    if '|seed=' in k:
        prompt, seed_part = k.rsplit('|seed=', 1)
        return (prompt, int(seed_part))
    if k.endswith('.obj'):
        # format: category/slug_seedN.obj
        stem = k.rsplit('/', 1)[-1][:-4]  # remove .obj
        m = re.match(r'(.+?)_seed(\d+)$', stem)
        if m:
            return (m.group(1), int(m.group(2)))  # prompt is slugged
    return None


def main():
    caches = {name: load(path) for name, path in CACHES.items()}
    for name, c in caches.items():
        if c is None:
            print(f'{name}: NOT FOUND ({CACHES[name]})')
        else:
            print(f'{name}: {len(c)} entries, sample key: {list(c)[0][:60]}')

    # Build unified (prompt_slug, seed) -> {cache_name: normal}
    by_ps = defaultdict(dict)

    for name, c in caches.items():
        if c is None: continue
        for k, v in c.items():
            canonical = normalize_key(k)
            if canonical is None: continue
            prompt, seed = canonical
            # slugify prompt for consistency
            if name == 'production':
                key = (prompt, seed)  # already slugged
            else:
                key = (slug(prompt), seed)
            by_ps[key][name] = np.array(v['normal'])

    print(f'\nunified keys: {len(by_ps)}')

    # Count how many are in all 4 caches
    full_overlap = sum(1 for v in by_ps.values() if len(v) == len(caches))
    print(f'in all {len(caches)} caches: {full_overlap}')

    # Pairwise consistency
    names = list(caches.keys())
    print('\nPairwise max/avg normal differences (accounting for sign flip):')
    print(f'{"pair":<30}{"n_common":>10}{"mean_diff":>12}{"max_diff":>12}{"n>0.01":>10}{"n>0.1":>10}')
    for i, a in enumerate(names):
        for b in names[i+1:]:
            diffs = []
            for key, d in by_ps.items():
                if a in d and b in d:
                    na = d[a] / (np.linalg.norm(d[a]) + 1e-12)
                    nb = d[b] / (np.linalg.norm(d[b]) + 1e-12)
                    diff = min(np.linalg.norm(na - nb), np.linalg.norm(na + nb))
                    diffs.append(diff)
            if not diffs:
                continue
            d = np.array(diffs)
            print(f'  {a} vs {b:<20}{len(diffs):>10}{d.mean():>12.6f}{d.max():>12.6f}{(d > 0.01).sum():>10}{(d > 0.1).sum():>10}')

    # Look at the most-mismatched pairs
    print('\n' + '=' * 78)
    print('Worst anticollapse <-> production disagreements (sign-normalized)')
    print('=' * 78)
    worst = []
    for key, d in by_ps.items():
        if 'anticollapse' in d and 'production' in d:
            na = d['anticollapse'] / (np.linalg.norm(d['anticollapse']) + 1e-12)
            np_= d['production']   / (np.linalg.norm(d['production']) + 1e-12)
            diff = min(np.linalg.norm(na - np_), np.linalg.norm(na + np_))
            worst.append((diff, key, na, np_))
    worst.sort(reverse=True)
    for diff, key, na, np_ in worst[:10]:
        print(f'  diff={diff:.4f}  key={key}  anti={na.round(3)}  prod={np_.round(3)}')


if __name__ == '__main__':
    main()
