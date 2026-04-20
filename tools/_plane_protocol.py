"""Shared helper for the STRONGER_SYMMETRY_VARIANT_CHECKLIST protocol.

Every experiment that reports a symmetry metric under the new protocol must:
  1. Estimate a single plane per (prompt, seed) on the BASELINE mesh.
  2. Share that plane across all paired method variants for that pair.
  3. Use that plane when computing the final symmetry score.

This module centralises the plane lookup/eval so each tool only needs a small
patch to upgrade to the new protocol. A persistent JSON cache is used to
guarantee that parallel jobs on the same server share identical planes and to
avoid re-estimating planes across reruns of the same experiment.

Typical patch to an existing tool:

    from _plane_protocol import PlaneStore, eval_symmetry

    store = PlaneStore.load_or_new('results/plane_cache.json')
    ...
    for prompt, seed in pairs:
        verts, faces = load_baseline(prompt, seed)
        sym_n, sym_d = store.get(f'{prompt}/{seed}', verts)
        refined, _ = refine_with_geo_reward(
            verts, faces, weights,
            sym_normal=sym_n, sym_offset=sym_d,
        )
        sym = eval_symmetry(refined, sym_n, sym_d)
    store.save()
"""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geo_reward import estimate_symmetry_plane, symmetry_reward_plane


class PlaneStore:
    """JSON-backed cache of {key: {normal, offset}} plane estimates.

    Plane tensors are stored as python lists/floats and converted back to
    torch.Tensor on read via `.get()`. Safe to call repeatedly from many
    processes if each process calls `.save()` at the end — last-writer wins,
    but identical keys give identical planes, so contention is harmless.
    """

    def __init__(self, path=None, data=None):
        self.path = path
        self.data = dict(data) if data else {}
        self._dirty = False

    @classmethod
    def load_or_new(cls, path):
        if path and os.path.exists(path):
            with open(path) as f:
                return cls(path=path, data=json.load(f))
        return cls(path=path)

    def has(self, key):
        return key in self.data

    def get(self, key, verts=None, estimator_kwargs=None):
        """Return (normal, offset) tensors for the given key.

        If key is missing and `verts` is given, estimate a new plane, store it,
        and return. If both are missing, raises KeyError.
        """
        if key not in self.data:
            if verts is None:
                raise KeyError(f'plane not cached for key={key!r} and no verts provided')
            kwargs = estimator_kwargs or {}
            n, d = estimate_symmetry_plane(verts.detach(), **kwargs)
            self.data[key] = {
                'normal': n.detach().cpu().tolist(),
                'offset': float(d.detach().cpu().item()),
                'n_vertices': int(verts.shape[0]),
            }
            self._dirty = True

        entry = self.data[key]
        device = verts.device if verts is not None else 'cpu'
        dtype = verts.dtype if verts is not None else torch.float32
        n = torch.tensor(entry['normal'], dtype=dtype, device=device)
        d = torch.tensor(entry['offset'], dtype=dtype, device=device)
        return n, d

    def save(self, path=None):
        target = path or self.path
        if target is None:
            return
        os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
        with open(target, 'w') as f:
            json.dump(self.data, f, indent=2)
        self._dirty = False


def eval_symmetry(verts, normal, offset):
    """Compute symmetry score under a fixed (normal, offset) plane.

    Returns a Python float, no autograd. Use in place of
    `symmetry_reward(axis=1).item()` wherever the new protocol is required.
    """
    with torch.no_grad():
        return symmetry_reward_plane(verts, normal, offset).item()


def make_key(prompt, seed):
    """Canonical cache key for a (prompt, seed) pair."""
    return f'{prompt}|seed={seed}'
