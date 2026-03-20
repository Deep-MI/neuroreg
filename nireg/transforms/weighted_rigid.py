"""Weighted rigid alignment solver for IRLS registration.

Implements GPU-accelerated weighted rigid body alignment using Horn's quaternion
method with SVD decomposition. Used in IRLS loops where points are weighted
based on robust M-estimators.
"""

import torch


def _det_3x3_mps_compatible(R: torch.Tensor) -> torch.Tensor:
    """Compute determinant of 3x3 matrix in an MPS-compatible way.
    
    MPS doesn't support torch.linalg.det, so we compute it manually using
    the standard formula for 3x3 matrices:
    det(R) = R[0,0]*(R[1,1]*R[2,2] - R[1,2]*R[2,1])
           - R[0,1]*(R[1,0]*R[2,2] - R[1,2]*R[2,0])
           + R[0,2]*(R[1,0]*R[2,1] - R[1,1]*R[2,0])
    
    This is faster and more compatible than torch.linalg.det for small matrices.
    
    Parameters
    ----------
    R : torch.Tensor
        3x3 matrix, shape [3, 3]
    
    Returns
    -------
    det : torch.Tensor
        Scalar determinant value
    """
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")
    
    # Compute determinant using cofactor expansion along first row
    det = (
        R[0, 0] * (R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1])
        - R[0, 1] * (R[1, 0] * R[2, 2] - R[1, 2] * R[2, 0])
        + R[0, 2] * (R[1, 0] * R[2, 1] - R[1, 1] * R[2, 0])
    )
    return det


def solve_weighted_rigid_gpu(
    src_points: torch.Tensor,
    trg_points: torch.Tensor,
    weights: torch.Tensor,
    center: bool = True,
) -> torch.Tensor:
    """Solve weighted rigid alignment using GPU SVD (Horn's method).

    Finds the optimal rigid transformation (rotation + translation) that
    minimizes the weighted squared distance between point correspondences:

        min_{R,t} Σ w_i * ||R*p_i + t - q_i||²

    where R is a rotation matrix and t is a translation vector.

    This implements the weighted version of Horn's quaternion method, solved
    via SVD decomposition on GPU for efficiency.

    Parameters
    ----------
    src_points : torch.Tensor
        Source points, shape [N, 3] where N is number of points.
    trg_points : torch.Tensor
        Target points, shape [N, 3].
    weights : torch.Tensor
        Point weights, shape [N]. Higher weights give points more influence.
        Should be non-negative; typically from robust M-estimators.
    center : bool, optional
        If True (default), computes weighted centroids and centers the points.
        Should always be True for rigid alignment.

    Returns
    -------
    T : torch.Tensor
        4×4 homogeneous transformation matrix on GPU, shape [4, 4].
        Encodes rotation (top-left 3×3) and translation (top-right 3×1).

    Notes
    -----
    The algorithm:
    1. Compute weighted centroids of source and target
    2. Center both point sets by their weighted means
    3. Build weighted covariance matrix H = Σ w_i * (p_i - p̄) ⊗ (q_i - q̄)
    4. Compute SVD: H = U S V^T
    5. Rotation: R = V^T U^T (with reflection check)
    6. Translation: t = q̄ - R p̄

    The SVD decomposition is performed using PyTorch's `torch.linalg.svd`,
    which is highly optimized on all platforms (CPU via LAPACK, GPU via cuBLAS).

    References
    ----------
    .. [1] Horn, B. K. (1987). Closed-form solution of absolute orientation
           using unit quaternions. JOSA A, 4(4), 629-642.
    .. [2] Umeyama, S. (1991). Least-squares estimation of transformation
           parameters between two point patterns. TPAMI, 13(4), 376-380.

    Examples
    --------
    >>> # Create synthetic rigid transform
    >>> R_true = rotation_matrix_from_euler([0.1, 0.2, 0.0])
    >>> t_true = torch.tensor([5.0, -3.0, 2.0])
    >>> src = torch.randn(100, 3)
    >>> trg = (R_true @ src.T).T + t_true
    >>>
    >>> # Add outliers
    >>> trg[::10] += torch.randn(10, 3) * 10
    >>>
    >>> # Weights (downweight outliers)
    >>> weights = torch.ones(100)
    >>> weights[::10] = 0.1
    >>>
    >>> # Solve
    >>> T = solve_weighted_rigid_gpu(src, trg, weights)
    >>> R_est = T[:3, :3]
    >>> t_est = T[:3, 3]
    >>> print(f"Rotation error: {torch.norm(R_est - R_true):.6f}")
    >>> print(f"Translation error: {torch.norm(t_est - t_true):.6f}")
    """
    # Validate inputs
    if src_points.shape != trg_points.shape:
        raise ValueError(f"Source and target must have same shape, got {src_points.shape} vs {trg_points.shape}")
    if src_points.shape[1] != 3:
        raise ValueError(f"Points must be 3D, got shape {src_points.shape}")
    if weights.shape[0] != src_points.shape[0]:
        raise ValueError(f"Number of weights ({weights.shape[0]}) must match number of points ({src_points.shape[0]})")

    # Ensure all tensors are on the same device
    device = src_points.device
    trg_points = trg_points.to(device)
    weights = weights.to(device)

    # Normalize weights (sum to 1)
    w_sum = weights.sum()
    if w_sum < 1e-10:
        # All weights are zero - return identity
        return torch.eye(4, device=device, dtype=src_points.dtype)

    w_norm = weights / w_sum

    if center:
        # Compute weighted centroids
        src_center = (w_norm[:, None] * src_points).sum(dim=0)
        trg_center = (w_norm[:, None] * trg_points).sum(dim=0)

        # Center point sets
        src_centered = src_points - src_center
        trg_centered = trg_points - trg_center
    else:
        src_centered = src_points
        trg_centered = trg_points
        src_center = torch.zeros(3, device=device, dtype=src_points.dtype)
        trg_center = torch.zeros(3, device=device, dtype=src_points.dtype)

    # Weighted covariance matrix: H = Σ w_i * p_i ⊗ q_i
    # Shape: [3, 3]
    H = (src_centered.T * w_norm) @ trg_centered

    # SVD decomposition (GPU-accelerated)
    U, S, Vh = torch.linalg.svd(H)

    # Rotation matrix: R = V^T U^T
    R = Vh.T @ U.T

    # Check for reflection (determinant should be +1, not -1)
    # Use MPS-compatible determinant calculation
    det = _det_3x3_mps_compatible(R)
    if det < 0:
        # Flip the last column of V (or last row of Vh)
        Vh[-1, :] *= -1
        R = Vh.T @ U.T

    # Translation: t = target_center - R @ source_center
    t = trg_center - R @ src_center

    # Build 4×4 homogeneous transformation matrix
    T = torch.eye(4, device=device, dtype=src_points.dtype)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def sample_weighted_voxel_grid(
    grid_shape: tuple[int, int, int],
    weights: torch.Tensor,
    sample_fraction: float = 0.1,
    min_samples: int = 1000,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample voxel coordinates from a 3D grid based on robust weights.

    Preferentially samples high-weight voxels (inliers) from a 3D voxel grid.
    Used in IRLS registration to select reliable voxels for rigid alignment.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Shape of the 3D voxel grid (H, W, D).
    weights : torch.Tensor
        Per-voxel weights from robust M-estimator, shape [H, W, D].
        Higher weights indicate more reliable voxels (inliers).
    sample_fraction : float, optional
        Fraction of voxels to sample (default 0.1 = 10%).
    min_samples : int, optional
        Minimum number of voxels to sample (default 1000).
    device : str, optional
        Device for tensor operations ('cpu' or 'cuda').

    Returns
    -------
    voxel_coords : torch.Tensor
        Sampled voxel coordinates in grid space, shape [N, 3] where N is
        number of samples. Each row is [i, j, k] in voxel units.
    voxel_weights : torch.Tensor
        Corresponding weights for each sampled voxel, shape [N].

    Notes
    -----
    This function is used in IRLS registration to extract voxel coordinates
    for weighted rigid alignment. By sampling based on weights (top-K), we:

    1. Reduce computation (don't need all voxels)
    2. Emphasize inliers (high-weight voxels)
    3. Downweight outliers (low-weight voxels are less likely to be selected)

    The sampled coordinates can be used directly with `solve_weighted_rigid_gpu`
    for voxel-based rigid alignment.

    Examples
    --------
    >>> # After computing IRLS weights
    >>> weights = tukey_weights(residuals / sigma, c=6.0)  # shape [H, W, D]
    >>> coords, w = sample_weighted_voxel_grid((100, 100, 80), weights)
    >>> # Use in weighted rigid solver (same coords for source and target)
    >>> T = solve_weighted_rigid_gpu(coords, coords, w)
    """
    # Validate inputs
    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must be 3D, got {grid_shape}")
    if weights.shape != grid_shape:
        raise ValueError(f"weights shape {weights.shape} must match grid_shape {grid_shape}")

    # Flatten weights
    weights_flat = weights.flatten()

    # Create voxel grid coordinates [i, j, k] for each voxel
    grid_i, grid_j, grid_k = torch.meshgrid(
        torch.arange(grid_shape[0], device=device),
        torch.arange(grid_shape[1], device=device),
        torch.arange(grid_shape[2], device=device),
        indexing="ij",
    )

    # Flatten and stack: shape [H*W*D, 3]
    grid_flat = torch.stack([grid_i.flatten(), grid_j.flatten(), grid_k.flatten()], dim=1).float()

    # Determine number of samples
    n_total = len(grid_flat)
    n_samples = max(min_samples, int(n_total * sample_fraction))
    n_samples = min(n_samples, n_total)  # Don't exceed total

    # Sample voxels based on weights (top-K by weight)
    # This preferentially selects high-weight (inlier) voxels
    if n_samples < n_total:
        _, top_indices = torch.topk(weights_flat, n_samples)
        voxel_coords = grid_flat[top_indices]
        voxel_weights = weights_flat[top_indices]
    else:
        voxel_coords = grid_flat
        voxel_weights = weights_flat

    return voxel_coords, voxel_weights
