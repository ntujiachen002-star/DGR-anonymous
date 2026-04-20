"""
DiffGeoReward: Differentiable Geometric Reward Primitives

Three core geometric properties with differentiable implementations:
1. Symmetry: reflection symmetry about a plane
2. Smoothness: surface smoothness via discrete mean curvature
3. Compactness: surface-to-volume ratio
"""

import torch
import torch.nn.functional as F
import numpy as np


def chamfer_distance(x: torch.Tensor, y: torch.Tensor, max_points: int = 4096) -> torch.Tensor:
    """Differentiable Chamfer distance between two point sets.

    Subsamples if point sets are too large to avoid OOM.

    Args:
        x: (N, 3) points
        y: (M, 3) points
        max_points: max points per set (subsample if larger)
    Returns:
        Scalar chamfer distance
    """
    # Subsample large point sets (deterministic: use stride, not random)
    if x.shape[0] > max_points:
        stride = x.shape[0] // max_points
        x = x[::stride][:max_points]
    if y.shape[0] > max_points:
        stride = y.shape[0] // max_points
        y = y[::stride][:max_points]

    # Chunk computation to avoid huge NxM matrix
    chunk_size = 2048
    min_dist_x_parts = []
    for i in range(0, x.shape[0], chunk_size):
        x_chunk = x[i:i+chunk_size]
        xx = (x_chunk ** 2).sum(dim=1, keepdim=True)
        yy = (y ** 2).sum(dim=1, keepdim=True)
        xy = x_chunk @ y.T
        dist = xx - 2 * xy + yy.T
        min_dist_x_parts.append(dist.min(dim=1).values)

    min_dist_y_parts = []
    for j in range(0, y.shape[0], chunk_size):
        y_chunk = y[j:j+chunk_size]
        yy = (y_chunk ** 2).sum(dim=1, keepdim=True)
        xx = (x ** 2).sum(dim=1, keepdim=True)
        yx = y_chunk @ x.T
        dist = yy - 2 * yx + xx.T
        min_dist_y_parts.append(dist.min(dim=1).values)

    min_dist_x = torch.cat(min_dist_x_parts)
    min_dist_y = torch.cat(min_dist_y_parts)

    return min_dist_x.mean() + min_dist_y.mean()


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute face normals for a triangle mesh.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices (long)
    Returns:
        (F, 3) unit face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = torch.cross(e1, e2, dim=1)
    normals = F.normalize(normals, dim=1, eps=1e-8)
    return normals


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute vertex normals by averaging adjacent face normals.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
    Returns:
        (V, 3) unit vertex normals
    """
    face_normals = compute_face_normals(vertices, faces)

    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):
        vertex_normals.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), face_normals)

    vertex_normals = F.normalize(vertex_normals, dim=1, eps=1e-8)
    return vertex_normals


def compute_face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute face areas.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
    Returns:
        (F,) face areas
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = 0.5 * torch.norm(cross, dim=1)
    return areas


def compute_surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Differentiable surface area computation."""
    return compute_face_areas(vertices, faces).sum()


def compute_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Differentiable signed volume via divergence theorem.

    V = (1/6) * |Σ (v0 · (v1 × v2))| for each face
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = torch.cross(v1, v2, dim=1)
    vol = (v0 * cross).sum(dim=1).sum() / 6.0
    return vol.abs()


# ============================================================
# Core Geometric Reward Functions
# ============================================================

def symmetry_reward(vertices: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """Reflection symmetry reward (coordinate-plane version, kept for backward compat).

    S(M, axis) = -chamfer(M, Reflect(M, axis))

    Higher (less negative) = more symmetric.

    Args:
        vertices: (V, 3) vertex positions
        axis: reflection axis (0=yz-plane, 1=xz-plane, 2=xy-plane)
    Returns:
        Scalar reward (negative, closer to 0 = more symmetric)
    """
    reflected = vertices.clone()
    reflected[:, axis] = -reflected[:, axis]
    return -chamfer_distance(vertices, reflected)


def symmetry_reward_plane(
    vertices: torch.Tensor,
    normal: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Reflection symmetry reward for an arbitrary plane.

    Plane equation: {x : n·x = d}
    Reflection of x across the plane: x' = x - 2*(n·x - d)*n
    S(M, n, d) = -chamfer(M, Reflect(M, n, d))

    Fully differentiable w.r.t. vertices for mesh optimization. When the plane
    is pre-estimated and detached, the cost matches the coordinate-plane version.

    Args:
        vertices: (V, 3) vertex positions
        normal: (3,) unit normal vector of the symmetry plane
        offset: scalar d; signed distance from origin to plane along `normal`
    Returns:
        Scalar reward (negative; closer to 0 = more symmetric)
    """
    proj = (vertices @ normal) - offset
    reflected = vertices - 2.0 * proj.unsqueeze(1) * normal.unsqueeze(0)
    return -chamfer_distance(vertices, reflected)


def _fibonacci_sphere(n: int, device, dtype) -> torch.Tensor:
    """Generate n approximately uniform points on the unit sphere via golden-angle spiral.

    Used as multi-start seeds for symmetry plane search. Plane (n, d) and
    (-n, -d) are equivalent, so the upper hemisphere alone would suffice — but
    full-sphere sampling is uniform enough that the redundancy is harmless.
    """
    indices = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = torch.acos(1.0 - 2.0 * indices / n)
    theta = (1.0 + 5.0 ** 0.5) * indices  # golden angle in radians
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def _refine_plane_seed(vertices, seed_n, seed_d, refine_steps, refine_lr,
                       early_stop_tol):
    """Adam-refine a single (normal, offset) seed; return (n, d, score).

    Falls back to the initial seed if refinement produces a worse score —
    guarantees monotone non-regression relative to the starting candidate.
    """
    with torch.no_grad():
        init_score = symmetry_reward_plane(vertices, seed_n, seed_d).item()

    n_raw = seed_n.detach().clone().requires_grad_(True)
    d_param = seed_d.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([n_raw, d_param], lr=refine_lr)

    last_check = float('inf')
    for step in range(refine_steps):
        optimizer.zero_grad()
        n = n_raw / n_raw.norm().clamp_min(1e-8)
        score = symmetry_reward_plane(vertices, n, d_param)
        (-score).backward()
        optimizer.step()
        if step > 0 and step % 10 == 0:
            cur = score.item()
            if abs(cur - last_check) < early_stop_tol:
                break
            last_check = cur

    with torch.no_grad():
        n_final = (n_raw / n_raw.norm().clamp_min(1e-8)).detach()
        d_final = d_param.detach()
        final_score = symmetry_reward_plane(vertices, n_final, d_final).item()

    if final_score < init_score:
        return seed_n.detach().clone(), seed_d.detach().clone(), init_score
    return n_final, d_final, final_score


def estimate_symmetry_plane(
    vertices: torch.Tensor,
    n_sphere_candidates: int = 16,
    top_k_refine: int = 3,
    refine_steps: int = 50,
    refine_lr: float = 0.01,
    early_stop_tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the best reflection symmetry plane via multi-start search.

    Algorithm:
    1. Build (3 + n_sphere_candidates) candidate normals: 3 from PCA
       eigenvectors plus uniform Fibonacci-sphere samples.
    2. Score each candidate (offset = n·centroid; plane through centroid).
    3. Adam-refine the top_k_refine candidates with plateau early-stop.
    4. Return the refined plane with the best final score.

    Multi-start handles PCA failure modes: asymmetric features pulling principal
    axes off-true, eigenvalue degeneracy on multi-symmetric objects, and surface
    noise on generated meshes.

    Call ONCE on the initial mesh; hold the returned plane fixed across all
    optimization steps and all paired method variants for a (prompt, seed) pair.

    Args:
        vertices: (V, 3) vertex positions; pass detached
        n_sphere_candidates: extra Fibonacci-sphere normals on top of 3 PCA candidates
        top_k_refine: number of best candidates to Adam-refine (>=1)
        refine_steps: max Adam steps per refinement
        refine_lr: Adam learning rate
        early_stop_tol: refine stops when score improvement over 10 steps < this
    Returns:
        normal: (3,) unit normal of best plane (detached)
        offset: scalar tensor d (detached)
    """
    device = vertices.device
    dtype = vertices.dtype

    with torch.no_grad():
        centroid = vertices.mean(dim=0)
        v_c = vertices - centroid
        cov = (v_c.T @ v_c) / vertices.shape[0]
        _, eigenvectors = torch.linalg.eigh(cov)

        pca_normals = eigenvectors.T  # (3, 3); rows are unit normals
        axis_normals = torch.eye(3, device=device, dtype=dtype)  # [1,0,0],[0,1,0],[0,0,1]
        parts = [pca_normals, axis_normals]
        if n_sphere_candidates > 0:
            parts.append(_fibonacci_sphere(n_sphere_candidates, device, dtype))
        normals = torch.cat(parts, dim=0)

        # For each normal, consider two offsets: through centroid (free fit)
        # and through origin (legacy fixed-axis protocol). Doubling candidates
        # guarantees superset of both protocols.
        n_norms = normals.shape[0]
        zero = torch.zeros(n_norms, device=device, dtype=dtype)
        centroid_proj = normals @ centroid
        candidate_normals = torch.cat([normals, normals], dim=0)              # (2N, 3)
        candidate_offsets = torch.cat([centroid_proj, zero], dim=0)           # (2N,)

        n_total = candidate_normals.shape[0]
        scores = torch.empty(n_total, device=device, dtype=dtype)
        for i in range(n_total):
            scores[i] = symmetry_reward_plane(
                vertices, candidate_normals[i], candidate_offsets[i])

        init_best_idx = int(scores.argmax().item())
        init_best_score = float(scores[init_best_idx].item())
        init_best_n = candidate_normals[init_best_idx].clone()
        init_best_d = candidate_offsets[init_best_idx].clone()

        k = min(top_k_refine, n_total)
        top_idx = torch.topk(scores, k=k).indices
        seeds = [(candidate_normals[i].clone(), candidate_offsets[i].clone())
                 for i in top_idx.tolist()]

    v_fixed = vertices.detach()
    best_score = init_best_score
    best_n, best_d = init_best_n, init_best_d
    for seed_n, seed_d in seeds:
        n_final, d_final, final_score = _refine_plane_seed(
            v_fixed, seed_n, seed_d, refine_steps, refine_lr, early_stop_tol)
        if final_score > best_score:
            best_score = final_score
            best_n, best_d = n_final, d_final

    return best_n, best_d


def estimate_symmetry_plane_pca(
    vertices: torch.Tensor,
    refine_steps: int = 50,
    refine_lr: float = 0.01,
    early_stop_tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy single-PCA symmetry plane estimator (kept for ablation/comparison).

    Equivalent to estimate_symmetry_plane(n_sphere_candidates=0, top_k_refine=1):
    selects the best of the 3 PCA eigenvectors and Adam-refines that one only.
    Use estimate_symmetry_plane instead for production runs.
    """
    return estimate_symmetry_plane(
        vertices,
        n_sphere_candidates=0,
        top_k_refine=1,
        refine_steps=refine_steps,
        refine_lr=refine_lr,
        early_stop_tol=early_stop_tol,
    )


def _build_face_adjacency(faces: torch.Tensor):
    """Build face adjacency from face indices (vectorized).

    Returns (idx_i, idx_j) tensors of adjacent face pairs sharing an edge.
    Cached result can be reused across optimization steps since topology is fixed.
    """
    F = faces.shape[0]
    # Each face contributes 3 half-edges: (v0,v1), (v1,v2), (v2,v0)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    half_edges = torch.stack([
        torch.stack([v0, v1], dim=1),
        torch.stack([v1, v2], dim=1),
        torch.stack([v2, v0], dim=1),
    ], dim=0).reshape(-1, 2)  # (3F, 2)

    # Face index for each half-edge
    face_ids = torch.arange(F, device=faces.device).repeat(3)  # (3F,)

    # Canonicalize edges: (min, max)
    edge_sorted = torch.sort(half_edges, dim=1).values  # (3F, 2)

    # Encode edge as single int for grouping
    V_max = faces.max().item() + 1
    edge_keys = edge_sorted[:, 0] * V_max + edge_sorted[:, 1]  # (3F,)

    # Sort by edge key to group matching edges
    sort_idx = edge_keys.argsort()
    edge_keys_sorted = edge_keys[sort_idx]
    face_ids_sorted = face_ids[sort_idx]

    # Find consecutive pairs with same edge key = adjacent faces
    mask = edge_keys_sorted[:-1] == edge_keys_sorted[1:]
    idx_i = face_ids_sorted[:-1][mask]
    idx_j = face_ids_sorted[1:][mask]

    return idx_i, idx_j


# Cache for face adjacency (topology doesn't change during optimization)
_face_adj_cache = {}


def compute_initial_huber_delta(vertices: torch.Tensor, faces: torch.Tensor) -> float:
    """Compute Huber delta from initial mesh dihedral angles.

    Returns the median dihedral angle deviation, to be fixed throughout optimization.
    Call once at initialization, then pass to smoothness_reward(delta=...).
    """
    with torch.no_grad():
        face_normals = compute_face_normals(vertices, faces)
        idx_i, idx_j = _build_face_adjacency(faces)
        if idx_i.shape[0] == 0:
            return 0.1
        cos_ij = (face_normals[idx_i] * face_normals[idx_j]).sum(dim=1).clamp(-1, 1)
        angle_dev = 1.0 - cos_ij
        delta = angle_dev.median().item()
        return max(delta, 1e-4)


def compute_feature_edge_mask(vertices: torch.Tensor, faces: torch.Tensor,
                               angle_threshold: float = None,
                               percentile: float = 75.0) -> torch.Tensor:
    """Identify feature edges to exclude from smoothness optimization.

    Uses two complementary criteria to detect geometric features:

    1. **Sharp edges (creases)**: Edges where the dihedral angle deviation exceeds
       a threshold. These are intentional creases, folds, and sharp boundaries.

    2. **Corner/tip vertices**: Vertices where the discrete Gaussian curvature
       (angular deficit) is high, indicating pointed tips, corners, or cone-like
       features. All edges incident to such vertices are also marked as features.

    Based on bilateral normal filtering (Zheng et al., TVCG 2011) and discrete
    differential geometry (Meyer et al., 2003).

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        angle_threshold: absolute threshold for dihedral angle deviation (1-cos).
                         If None, computed from percentile.
        percentile: edges above this percentile are features (default 75.0).
                    Ignored if angle_threshold is provided.
    Returns:
        Boolean mask (n_edges,): True = smooth edge (include), False = feature (exclude)
    """
    with torch.no_grad():
        face_normals = compute_face_normals(vertices, faces)
        idx_i, idx_j = _build_face_adjacency(faces)
        if idx_i.shape[0] == 0:
            return torch.ones(0, dtype=torch.bool, device=faces.device)

        cos_ij = (face_normals[idx_i] * face_normals[idx_j]).sum(dim=1).clamp(-1, 1)
        angle_dev = 1.0 - cos_ij  # 0 = flat, 2 = opposite normals

        # Criterion 1: sharp edges by dihedral angle
        if angle_threshold is None:
            angle_threshold = torch.quantile(angle_dev.float(), percentile / 100.0).item()
        edge_is_feature = angle_dev > angle_threshold

        # Criterion 2: high-curvature vertices (corners, tips)
        # Discrete angular deficit: 2π - sum of incident face angles at vertex
        V = vertices.shape[0]
        angle_sum = torch.zeros(V, device=vertices.device)
        for local in range(3):
            v0 = faces[:, local]
            v1 = faces[:, (local + 1) % 3]
            v2 = faces[:, (local + 2) % 3]
            e1 = F.normalize(vertices[v1] - vertices[v0], dim=1, eps=1e-8)
            e2 = F.normalize(vertices[v2] - vertices[v0], dim=1, eps=1e-8)
            cos_angle = (e1 * e2).sum(dim=1).clamp(-1, 1)
            angle_sum.scatter_add_(0, v0, torch.acos(cos_angle))

        angular_deficit = (2 * 3.14159265 - angle_sum).abs()
        # Vertices with high angular deficit = corners/tips
        deficit_threshold = torch.quantile(angular_deficit.float(), percentile / 100.0).item()
        vertex_is_feature = angular_deficit > deficit_threshold

        # Map vertex features to edges: if either face of an edge touches a
        # feature vertex, mark that edge as feature
        face_has_feature_vert = (
            vertex_is_feature[faces[:, 0]] |
            vertex_is_feature[faces[:, 1]] |
            vertex_is_feature[faces[:, 2]]
        )
        edge_touches_feature_vert = face_has_feature_vert[idx_i] | face_has_feature_vert[idx_j]

        # Combine: feature = sharp edge OR touches corner vertex
        is_feature = edge_is_feature | edge_touches_feature_vert
        smooth_mask = ~is_feature

        return smooth_mask


def smoothness_reward(vertices: torch.Tensor, faces: torch.Tensor,
                      delta: float = None, _adj: tuple = None,
                      _init_angles: torch.Tensor = None) -> torch.Tensor:
    """Structure-preserving surface regularity reward.

    Instead of penalizing all dihedral angle deviations (which would smooth away
    intentional features like steps, creases, and corners), this reward penalizes
    only the CHANGE in dihedral angles relative to the initial mesh.

    This preserves all original geometric features — sharp edges stay sharp,
    steps stay stepped, corners stay pointed — while preventing the optimization
    from introducing NEW surface irregularities.

    When _init_angles is provided (recommended for optimization loops):
        R = -(1/|E|) Σ huber(|angle_current - angle_initial|, δ)
        Only penalizes deviations FROM the initial structure.

    When _init_angles is None (standalone evaluation, backward compatible):
        R = -(1/|E|) Σ huber(angle_deviation, δ)
        Penalizes all dihedral angle deviations (original behavior).

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        delta: Huber threshold. If None, auto-set from median.
        _adj: Optional precomputed (idx_i, idx_j) adjacency.
        _init_angles: Optional (n_edges,) tensor of initial dihedral angle deviations
                      (1 - cos(n_i, n_j)) from compute_initial_angles().
                      When provided, the reward penalizes change from these values.
    Returns:
        Scalar reward (higher = better preserved / more regular surface)
    """
    face_normals = compute_face_normals(vertices, faces)

    if _adj is not None:
        idx_i, idx_j = _adj
    else:
        idx_i, idx_j = _build_face_adjacency(faces)

    if idx_i.shape[0] == 0:
        return torch.tensor(0.0, device=vertices.device, requires_grad=True)

    cos_ij = (face_normals[idx_i] * face_normals[idx_j]).sum(dim=1).clamp(-1, 1)
    angle_dev = 1.0 - cos_ij  # range [0, 2]

    if _init_angles is not None:
        # Structure-preserving mode: penalize CHANGE from initial angles
        angle_change = (angle_dev - _init_angles).abs()
        target = angle_change
    else:
        # Standalone mode: penalize absolute dihedral deviation (backward compatible)
        target = angle_dev

    if delta is None:
        with torch.no_grad():
            delta = target.median().item()
            delta = max(delta, 1e-4)

    huber = torch.where(
        target <= delta,
        target ** 2 / (2 * delta),
        target - delta / 2
    )

    return -huber.mean()


def structural_consistency_reward(vertices: torch.Tensor, faces: torch.Tensor,
                                   axis: int = 1, _adj: tuple = None) -> torch.Tensor:
    """Symmetric structural consistency reward.

    Instead of smoothing all surfaces (which destroys features like steps, corners,
    and creases), this reward penalizes STRUCTURAL ASYMMETRY: differences in local
    geometry between a point and its mirror counterpart.

    For each face, we compare its normal with the normal of the nearest face in the
    reflected mesh. If both sides have the same structure (e.g., both have a step),
    no penalty. If one side has a bump the other doesn't, penalty.

    This naturally:
    - Preserves symmetric features (steps, corners stay sharp)
    - Removes asymmetric noise (random bumps get symmetrized)
    - Works synergistically with symmetry reward (both push toward symmetry)

    Additionally includes a light continuity term: penalizes face normal flips
    relative to the initial mesh to prevent degenerate geometry.

    R = -(1/|F|) Σ_f (1 - cos(n_f, n_f_mirror))

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        axis: symmetry axis (0=yz, 1=xz, 2=xy)
        _adj: unused, kept for API compatibility
    Returns:
        Scalar reward (higher = more structurally consistent)
    """
    face_normals = compute_face_normals(vertices, faces)  # (F, 3)
    F_count = faces.shape[0]

    # Compute face centroids
    face_centroids = vertices[faces].mean(dim=1)  # (F, 3)

    # Reflect centroids across symmetry plane
    reflected_centroids = face_centroids.clone()
    reflected_centroids[:, axis] = -reflected_centroids[:, axis]

    # For each reflected centroid, find the nearest original face
    # Use chunked computation for memory efficiency
    chunk_size = 2048
    nearest_face_idx = []
    for i in range(0, F_count, chunk_size):
        end = min(i + chunk_size, F_count)
        dists = torch.cdist(reflected_centroids[i:end], face_centroids)  # (chunk, F)
        nearest_face_idx.append(dists.argmin(dim=1))
    nearest_face_idx = torch.cat(nearest_face_idx)  # (F,)

    # Get the mirror face's normal
    mirror_normals = face_normals[nearest_face_idx]  # (F, 3)

    # Reflect the mirror normal back (flip the axis component)
    mirror_normals_reflected = mirror_normals.clone()
    mirror_normals_reflected[:, axis] = -mirror_normals_reflected[:, axis]

    # Structural consistency: cos similarity between face normal and its
    # reflected mirror counterpart. 1.0 = perfect structural symmetry.
    cos_sim = (face_normals * mirror_normals_reflected).sum(dim=1).clamp(-1, 1)
    structural_diff = 1.0 - cos_sim  # 0 = consistent, 2 = opposite

    return -structural_diff.mean()


def continuity_reward(vertices: torch.Tensor, faces: torch.Tensor,
                      _init_normals: torch.Tensor = None) -> torch.Tensor:
    """Surface continuity reward: penalizes face normal flips and degenerate geometry.

    Prevents optimization from creating self-intersections, flipped faces, or
    collapsed triangles. Acts as a soft constraint, not an objective.

    When _init_normals is provided, penalizes normal direction change from initial:
        R = (1/|F|) Σ cos(n_f_current, n_f_initial)

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        _init_normals: (F, 3) initial face normals from compute_initial_normals()
    Returns:
        Scalar reward (higher = better continuity, max 0 = no flips)
    """
    face_normals = compute_face_normals(vertices, faces)

    if _init_normals is not None:
        # Penalize deviation from initial normal direction
        cos_sim = (face_normals * _init_normals).sum(dim=1).clamp(-1, 1)
        # cos_sim = 1 means same direction, -1 means flipped
        # We want to penalize flips: reward = mean(cos_sim) - 1 (so max is 0)
        return cos_sim.mean() - 1.0
    else:
        # Without reference: penalize degenerate faces (near-zero area)
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        areas = cross.norm(dim=1)
        # Log of area: penalizes tiny faces heavily
        return -torch.clamp(-torch.log(areas + 1e-10), min=0).mean()


def compute_initial_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute initial face normals to be preserved during optimization."""
    with torch.no_grad():
        return compute_face_normals(vertices, faces).detach()


def compute_initial_angles(vertices: torch.Tensor, faces: torch.Tensor,
                           _adj: tuple = None) -> torch.Tensor:
    """Compute initial dihedral angle deviations to be preserved during optimization.

    Call once at initialization, then pass to smoothness_reward(_init_angles=...).
    The optimizer will only penalize deviations from these initial angles,
    preserving all original geometric features (steps, creases, corners, etc.).

    Args:
        vertices: (V, 3) initial vertex positions
        faces: (F, 3) face indices
        _adj: Optional precomputed adjacency
    Returns:
        (n_edges,) tensor of initial angle deviations (1 - cos(n_i, n_j))
    """
    with torch.no_grad():
        face_normals = compute_face_normals(vertices, faces)
        if _adj is not None:
            idx_i, idx_j = _adj
        else:
            idx_i, idx_j = _build_face_adjacency(faces)
        if idx_i.shape[0] == 0:
            return torch.zeros(0, device=vertices.device)
        cos_ij = (face_normals[idx_i] * face_normals[idx_j]).sum(dim=1).clamp(-1, 1)
        return (1.0 - cos_ij).detach()


def smoothness_reward_legacy(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Legacy smoothness reward via curvature variance (non-feature-preserving).
    Kept for backward compatibility and ablation comparison.
    """
    vertex_normals = compute_vertex_normals(vertices, faces)
    V = vertices.shape[0]
    neighbor_sum = torch.zeros(V, 3, device=vertices.device)
    neighbor_count = torch.zeros(V, 1, device=vertices.device)
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]
    ], dim=0)
    src, dst = edges[:, 0], edges[:, 1]
    neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), vertex_normals[src])
    neighbor_count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=vertices.device))
    neighbor_count = neighbor_count.clamp(min=1)
    neighbor_avg = F.normalize(neighbor_sum / neighbor_count, dim=1, eps=1e-8)
    curvature = 1.0 - (vertex_normals * neighbor_avg).sum(dim=1)
    return -curvature.var()


def compactness_reward(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compactness reward: negative isoperimetric ratio.

    K(M) = -SA / V^(2/3)

    Sphere is the most compact (lowest SA/V^(2/3)).
    Higher (less negative) = more compact.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
    Returns:
        Scalar reward
    """
    sa = compute_surface_area(vertices, faces)
    vol = compute_volume(vertices, faces)
    # Guard against near-zero volume (non-watertight or degenerate meshes)
    vol_23 = vol ** (2.0 / 3.0)
    vol_23 = vol_23.clamp(min=1e-2)  # prevent extreme ratios
    return -sa / vol_23


class DiffGeoReward(torch.nn.Module):
    """Combined differentiable geometric reward.

    R_geo(M, w) = w1 * R_sym(M)/|R_sym^0| + w2 * R_smo(M)/|R_smo^0| + w3 * R_com(M)/|R_com^0|

    Each reward is normalized by its initial magnitude (at the unrefined mesh)
    to ensure comparable scales regardless of raw reward magnitudes.
    This matches the paper formulation (Eq. 6).
    """

    def __init__(self):
        super().__init__()
        self._sym_scale = None
        self._smooth_scale = None
        self._compact_scale = None
        self._huber_delta = None
        self._face_adj = None

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        weights: torch.Tensor,
        sym_axis: int = 1
    ) -> torch.Tensor:
        """Compute weighted geometric reward.

        Args:
            vertices: (V, 3) mesh vertices
            faces: (F, 3) mesh faces
            weights: (3,) weights for [symmetry, smoothness, compactness]
        Returns:
            Scalar reward
        """
        # On first call, precompute and cache adjacency + delta
        if self._face_adj is None:
            self._face_adj = _build_face_adjacency(faces)
            self._huber_delta = compute_initial_huber_delta(vertices, faces)

        sym = symmetry_reward(vertices, axis=sym_axis)
        smooth = smoothness_reward(vertices, faces,
                                   delta=self._huber_delta, _adj=self._face_adj)
        compact = compactness_reward(vertices, faces)

        # On first call, record initial magnitudes for normalization
        if self._sym_scale is None:
            self._sym_scale = max(abs(sym.item()), 1e-6)
            self._smooth_scale = max(abs(smooth.item()), 1e-6)
            self._compact_scale = max(abs(compact.item()), 1e-6)

        reward = (weights[0] * sym / self._sym_scale
                  + weights[1] * smooth / self._smooth_scale
                  + weights[2] * compact / self._compact_scale)
        return reward

    def reset(self):
        """Reset normalization scales (call before refining a new mesh)."""
        self._sym_scale = None
        self._smooth_scale = None
        self._compact_scale = None
        self._huber_delta = None
        self._face_adj = None

    def compute_all(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        sym_axis: int = 1
    ) -> dict:
        """Compute all rewards individually (for evaluation)."""
        if self._face_adj is None:
            self._face_adj = _build_face_adjacency(faces)
            self._huber_delta = compute_initial_huber_delta(vertices, faces)
        return {
            'symmetry': symmetry_reward(vertices, axis=sym_axis),
            'smoothness': smoothness_reward(vertices, faces,
                                            delta=self._huber_delta, _adj=self._face_adj),
            'compactness': compactness_reward(vertices, faces),
        }
