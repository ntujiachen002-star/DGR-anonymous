"""Summary bar chart of old vs new symmetry plane estimator on all test meshes."""
import json
import os
import sys
import time
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geo_reward import (
    estimate_symmetry_plane,
    estimate_symmetry_plane_pca,
    symmetry_reward_plane,
    smoothness_reward,
    compactness_reward,
    compute_initial_huber_delta,
    _build_face_adjacency,
)


MESHES = [
    ('results/qualitative_meshes/a_mirror-symmetric_face_mask_seed456_baseline.obj', 'face mask'),
    ('results/qualitative_meshes/a_symmetric_butterfly_sculpture_seed123_baseline.obj', 'butterfly'),
    ('results/qualitative_meshes/a_symmetric_vase_seed123_baseline.obj', 'vase'),
    ('results/qualitative_meshes/a_compact_treasure_chest_seed42_baseline.obj', 'treasure chest'),
    ('results/full/baseline/symmetry/baseline_seed42.obj', 'window frame'),
    ('results/full/baseline/symmetry/baseline_seed123.obj', 'torii gate'),
    ('results/baseline/symmetry/baseline_seed42.obj', 'hourglass'),
]


def load(path):
    m = trimesh.load(path, process=False)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32)
    f = torch.tensor(np.asarray(m.faces), dtype=torch.long)
    return v, f


def refine(verts, faces, n, d, steps=50):
    v_opt = verts.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.005)
    adj = _build_face_adjacency(faces)
    delta = compute_initial_huber_delta(v_opt, faces)
    with torch.no_grad():
        s0 = symmetry_reward_plane(v_opt, n, d).item()
        sm0 = smoothness_reward(v_opt, faces, delta=delta, _adj=adj).item()
        cp0 = compactness_reward(v_opt, faces).item()
    ss, sms, cps = max(abs(s0), 1e-6), max(abs(sm0), 1e-6), max(abs(cp0), 1e-6)
    for _ in range(steps):
        opt.zero_grad()
        r = 0.7 * symmetry_reward_plane(v_opt, n, d) / ss \
            + 0.15 * smoothness_reward(v_opt, faces, delta=delta, _adj=adj) / sms \
            + 0.15 * compactness_reward(v_opt, faces) / cps
        (-r).backward()
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        opt.step()
    return v_opt.detach()


def main():
    rows = []
    for path, label in MESHES:
        if not os.path.exists(path):
            continue
        verts, faces = load(path)
        t = time.time()
        n_old, d_old = estimate_symmetry_plane_pca(verts.detach())
        t_old = time.time() - t
        t = time.time()
        n_new, d_new = estimate_symmetry_plane(verts.detach())
        t_new = time.time() - t

        init_old = symmetry_reward_plane(verts, n_old, d_old).item()
        init_new = symmetry_reward_plane(verts, n_new, d_new).item()
        ref_old = refine(verts, faces, n_old, d_old)
        ref_new = refine(verts, faces, n_new, d_new)
        sc_old = symmetry_reward_plane(ref_old, n_old, d_old).item()
        sc_new = symmetry_reward_plane(ref_new, n_new, d_new).item()
        print(f'{label:<18} init {init_old:+.5f}→{init_new:+.5f}  '
              f'refined {sc_old:+.5f}→{sc_new:+.5f}  time {t_old:.1f}s/{t_new:.1f}s')
        rows.append({
            'label': label, 'n_verts': verts.shape[0],
            'init_old': init_old, 'init_new': init_new,
            'ref_old': sc_old, 'ref_new': sc_new,
            't_old': t_old, 't_new': t_new,
        })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [r['label'] for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    init_old = [-r['init_old'] for r in rows]
    init_new = [-r['init_new'] for r in rows]
    axes[0].bar(x - w/2, init_old, w, label='Old: single PCA', color='#4a90d9')
    axes[0].bar(x + w/2, init_new, w, label='New: multi-start', color='#3cb371')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('|symmetry score|  (lower = more symmetric)')
    axes[0].set_title('Plane quality BEFORE DGR refinement\n'
                       '(score from plane estimator alone)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)

    ref_old = [-r['ref_old'] for r in rows]
    ref_new = [-r['ref_new'] for r in rows]
    axes[1].bar(x - w/2, ref_old, w, label='Old: single PCA', color='#4a90d9')
    axes[1].bar(x + w/2, ref_new, w, label='New: multi-start', color='#3cb371')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('|symmetry score|  (lower = more symmetric)')
    axes[1].set_title('Final symmetry AFTER 50 DGR refinement steps\n'
                       '(lower bar = new algorithm wins)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)

    for r, i in zip(rows, range(len(rows))):
        delta_init = (r['init_old'] - r['init_new']) / max(abs(r['init_old']), 1e-9)
        if delta_init > 0.1:
            axes[0].annotate(f'{delta_init*100:.0f}%↓', xy=(i + w/2, -r['init_new']),
                              xytext=(0, 5), textcoords='offset points',
                              ha='center', fontsize=8, color='#1a5c1a', fontweight='bold')
        delta_ref = (r['ref_old'] - r['ref_new']) / max(abs(r['ref_old']), 1e-9)
        if delta_ref > 0.1:
            axes[1].annotate(f'{delta_ref*100:.0f}%↓', xy=(i + w/2, -r['ref_new']),
                              xytext=(0, 5), textcoords='offset points',
                              ha='center', fontsize=8, color='#1a5c1a', fontweight='bold')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/symmetry_plane_summary.png', dpi=140, bbox_inches='tight')
    print('Saved figures/symmetry_plane_summary.png')

    with open('figures/symmetry_plane_summary.json', 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
