"""
Spectral-Aware Adaptive Weighting (SAAW) for DiffGeoReward.

Instead of fixed weights across all frequency bands, SAAW:
1. Projects reward gradients onto the mesh Laplacian eigenbasis
2. Detects pairwise conflict in each frequency band
3. Suppresses the weaker gradient in conflicting bands
4. Reconstructs a conflict-reduced combined gradient

This upgrades the spectral diagnosis framework into an algorithmic solution.
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


class SpectralWeighting:
    """Spectral-aware adaptive weighting for multi-objective mesh optimization.

    Usage:
        sw = SpectralWeighting(vertices, faces, n_eigenmodes=100, n_bands=3)
        # In optimization loop:
        grads = [g_sym, g_creg, g_com]  # list of (V,3) gradients
        weights = [w1, w2, w3]
        combined = sw.combine_gradients(grads, weights)
        # Use combined as the gradient for Adam step
    """

    def __init__(self, vertices, faces, n_eigenmodes=100, n_bands=3):
        """
        Args:
            vertices: (V, 3) initial vertex positions (numpy or tensor)
            faces: (F, 3) face indices (numpy or tensor)
            n_eigenmodes: number of Laplacian eigenvectors to compute
            n_bands: number of frequency bands (default 3: low/mid/high)
        """
        self.n_bands = n_bands
        self.device = vertices.device if torch.is_tensor(vertices) else 'cpu'

        verts_np = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else vertices
        faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces

        V = verts_np.shape[0]
        K = min(n_eigenmodes, V - 2)
        self.K = K

        # Build cotangent Laplacian
        L, D = self._build_cotangent_laplacian(verts_np, faces_np)

        # Compute eigenbasis
        self.eigenvalues, eigvecs = self._compute_eigenbasis(L, D, K)

        # Store as torch tensor for fast projection
        self.Phi = torch.tensor(eigvecs, dtype=torch.float32, device=self.device)  # (V, K)

        # Precompute band indices
        band_size = K // n_bands
        self.band_slices = []
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else K
            self.band_slices.append(slice(start, end))

    def compute_adaptive_weights(self, grads, base_weights):
        """Compute per-reward adaptive weight scalars based on spectral conflict.

        Instead of projecting and reconstructing gradients (which loses high-freq info),
        we analyze conflicts in spectral space and output adjusted SCALAR weights
        that are applied to the ORIGINAL full-resolution gradients in vertex space.

        Args:
            grads: list of (V, 3) gradient tensors
            base_weights: list of base scalar weights
        Returns:
            list of adjusted scalar weights (same length as grads)
        """
        n_rewards = len(grads)

        # Project each gradient onto eigenbasis: (K, 3) per reward
        coeffs = []
        for g in grads:
            c = self.Phi.t() @ g.detach()  # (K, 3)
            coeffs.append(c)

        # Accumulate per-reward suppression across bands
        # weight_scale[i] starts at 1.0, gets reduced in conflicting bands
        weight_scales = [1.0] * n_rewards

        for band_slice in self.band_slices:
            band_coeffs = [c[band_slice] for c in coeffs]
            band_energy = [bc.norm().item() for bc in band_coeffs]
            total_energy = sum(band_energy) + 1e-12

            # Band importance = fraction of total gradient energy in this band
            band_importance = [e / total_energy for e in band_energy]

            for i in range(n_rewards):
                for j in range(i + 1, n_rewards):
                    if band_energy[i] < 1e-12 or band_energy[j] < 1e-12:
                        continue

                    cos_ij = (band_coeffs[i] * band_coeffs[j]).sum() / (
                        band_coeffs[i].norm() * band_coeffs[j].norm() + 1e-12
                    )

                    if cos_ij < 0:
                        # Conflict detected in this band
                        suppress = max(0.0, 1.0 + cos_ij.item())
                        # Suppress the weaker reward, weighted by band importance
                        if band_energy[i] < band_energy[j]:
                            weight_scales[i] *= (1.0 - band_importance[i] * (1.0 - suppress))
                        else:
                            weight_scales[j] *= (1.0 - band_importance[j] * (1.0 - suppress))

        # Apply scales to base weights
        return [w * s for w, s in zip(base_weights, weight_scales)]

    def combine_gradients(self, grads, weights):
        """Combine gradients with spectral-aware adaptive weights (v2).

        Returns combined gradient in ORIGINAL vertex space (no projection loss).
        """
        adaptive_weights = self.compute_adaptive_weights(grads, weights)
        combined = sum(w * g for w, g in zip(adaptive_weights, grads))
        return combined

    def combine_gradients_surgery(self, grads, weights):
        """Combine gradients with spectral gradient surgery (v3).

        PCGrad-style projection applied per frequency band:
        - Project each gradient onto eigenbasis (K modes)
        - In each band, if two gradients conflict, project out the conflicting component
        - Preserve the residual (modes > K) untouched
        - Recombine modified spectral part + original residual

        This is "surgical" — only removes the specific conflicting components
        in the specific frequency bands where they occur.
        """
        n_rewards = len(grads)

        # 1. Decompose each gradient: g = g_spectral + g_residual
        spectral_coeffs = []  # list of (K, 3) per reward
        residuals = []        # list of (V, 3) per reward
        for g in grads:
            g_det = g.detach()
            coeffs = self.Phi.t() @ g_det           # (K, 3)
            g_spectral = self.Phi @ coeffs           # (V, 3) - reconstructed from K modes
            g_residual = g_det - g_spectral          # (V, 3) - high-freq tail
            spectral_coeffs.append(coeffs)
            residuals.append(g_residual)

        # 2. Per-band PCGrad surgery on spectral coefficients
        modified_coeffs = [c.clone() for c in spectral_coeffs]

        for band_slice in self.band_slices:
            band_vecs = [c[band_slice] for c in modified_coeffs]  # list of (band_size, 3)

            for i in range(n_rewards):
                for j in range(n_rewards):
                    if i == j:
                        continue
                    gi_band = band_vecs[i].reshape(-1)  # flatten to 1D
                    gj_band = band_vecs[j].reshape(-1)

                    gj_norm = gj_band.norm()
                    if gj_norm < 1e-12:
                        continue

                    cos_ij = (gi_band @ gj_band) / (gi_band.norm() * gj_norm + 1e-12)

                    if cos_ij < 0:
                        # PCGrad: project out the conflicting component
                        # g_i -= (g_i · ĝ_j) ĝ_j
                        gj_hat = gj_band / gj_norm
                        projection = (gi_band @ gj_hat) * gj_hat
                        modified_coeffs[i][band_slice] -= projection.reshape(
                            modified_coeffs[i][band_slice].shape
                        )

        # 3. Reconstruct: modified_spectral + original_residual
        modified_grads = []
        for i in range(n_rewards):
            g_spectral_new = self.Phi @ modified_coeffs[i]  # (V, 3)
            g_new = g_spectral_new + residuals[i]           # add back residual
            modified_grads.append(g_new)

        # 4. Weighted combination
        combined = sum(w * g for w, g in zip(weights, modified_grads))
        return combined

    @staticmethod
    def _build_cotangent_laplacian(verts_np, faces_np):
        V = verts_np.shape[0]
        rows, cols, vals = [], [], []

        for f in faces_np:
            for i in range(3):
                i0 = f[i]
                i1 = f[(i + 1) % 3]
                i2 = f[(i + 2) % 3]

                e1 = verts_np[i0] - verts_np[i2]
                e2 = verts_np[i1] - verts_np[i2]

                dot = np.dot(e1, e2)
                cross_norm = np.linalg.norm(np.cross(e1, e2))
                cot_angle = dot / (cross_norm + 1e-10)
                w = 0.5 * max(cot_angle, 1e-4)

                rows.extend([i0, i1])
                cols.extend([i1, i0])
                vals.extend([w, w])

        W = coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
        W = W.tocsc().tocsr()
        D = diags(np.array(W.sum(axis=1)).flatten())
        L = D - W
        return L, D

    @staticmethod
    def _compute_eigenbasis(L, D, k):
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, M=D, sigma=0, which='LM')
        except Exception:
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-10))
            D_inv_sqrt = diags(d_inv_sqrt)
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt
            eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM')

        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]


def refine_with_saaw(vertices, faces, weights, steps=50, lr=0.005,
                     sym_axis=1, n_eigenmodes=100, n_bands=3, method='surgery'):
    """DiffGeoReward refinement with Spectral-Aware Adaptive Weighting.

    Drop-in replacement for refine_with_geo_reward.

    Args:
        vertices: (V, 3) tensor
        faces: (F, 3) tensor
        weights: [w_sym, w_creg, w_com]
        steps: optimization steps
        lr: learning rate
        sym_axis: symmetry reflection axis
        n_eigenmodes: eigenmodes for spectral decomposition
        n_bands: frequency bands
    Returns:
        (V, 3) refined vertices
    """
    from geo_reward import symmetry_reward, smoothness_reward, compactness_reward

    # Initialize spectral weighting (one-time cost)
    sw = SpectralWeighting(vertices, faces, n_eigenmodes, n_bands)

    v_opt = vertices.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v_opt], lr=lr)

    # Initial rewards for normalization
    with torch.no_grad():
        r0_sym = symmetry_reward(v_opt, sym_axis).item()
        r0_smo = smoothness_reward(v_opt, faces).item()
        r0_com = compactness_reward(v_opt, faces).item()
    eps = 1e-8

    for step in range(steps):
        optimizer.zero_grad()

        # Compute per-reward values and gradients
        r_sym = symmetry_reward(v_opt, sym_axis)
        r_smo = smoothness_reward(v_opt, faces)
        r_com = compactness_reward(v_opt, faces)

        # Normalized rewards
        nr_sym = r_sym / (abs(r0_sym) + eps)
        nr_smo = r_smo / (abs(r0_smo) + eps)
        nr_com = r_com / (abs(r0_com) + eps)

        # Get per-reward gradients via autograd
        g_sym = torch.autograd.grad(nr_sym, v_opt, retain_graph=True)[0]
        g_smo = torch.autograd.grad(nr_smo, v_opt, retain_graph=True)[0]
        g_com = torch.autograd.grad(nr_com, v_opt, retain_graph=False)[0]

        # SAAW: combine gradients with spectral conflict resolution
        if method == 'surgery':
            combined_grad = sw.combine_gradients_surgery(
                [g_sym, g_smo, g_com],
                [weights[0], weights[1], weights[2]]
            )
        else:
            combined_grad = sw.combine_gradients(
                [g_sym, g_smo, g_com],
                [weights[0], weights[1], weights[2]]
            )

        # Set the combined gradient (maximize reward = negate for loss)
        v_opt.grad = -combined_grad
        torch.nn.utils.clip_grad_norm_([v_opt], 1.0)
        optimizer.step()

    return v_opt.detach()
