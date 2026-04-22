"""Closed-form point-set registration helpers.

This module implements the small family of transform models used by
``segreg``: rigid, similarity, anisotropic no-shear, and full affine fits
between paired 3-D point sets.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

PointArray = npt.NDArray[np.float64]


def _as_points(points: npt.ArrayLike, *, name: str) -> PointArray:
    """Coerce an array-like object into an ``(N, 3)`` float64 point array."""
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points_array.shape}.")
    return points_array


def _validate_pairwise_points(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> tuple[PointArray, PointArray]:
    """Validate and return paired moving/target point arrays with matching shape."""
    mov = _as_points(p_mov, name="p_mov")
    dst = _as_points(p_dst, name="p_dst")
    if mov.shape != dst.shape:
        raise ValueError(f"Point arrays must have identical shape, got {mov.shape} and {dst.shape}.")
    return mov, dst


def find_rotation(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the proper rotation that best aligns two centered point sets."""
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    h = mov.T @ dst
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def _center_pairwise_points(
    p_mov: npt.ArrayLike,
    p_dst: npt.ArrayLike,
) -> tuple[PointArray, PointArray, PointArray, PointArray]:
    """Return paired point sets together with their mean-centered coordinates."""
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    centroid_mov = mov.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    return mov, dst, mov - centroid_mov, dst - centroid_dst


def find_rigid(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the rigid homogeneous transform that best aligns two point sets."""
    mov, dst, mov_centered, dst_centered = _center_pairwise_points(p_mov, p_dst)
    if mov.shape[0] < 3:
        raise ValueError("Rigid registration requires at least 3 point correspondences.")

    if np.linalg.matrix_rank(mov_centered) < 2 or np.linalg.matrix_rank(dst_centered) < 2:
        raise ValueError("Rigid registration requires non-collinear point correspondences.")

    rotation = find_rotation(mov_centered, dst_centered)
    translation = dst.mean(axis=0) - rotation @ mov.mean(axis=0)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def find_similarity(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the similarity transform with one global scale factor."""
    mov, dst, mov_centered, dst_centered = _center_pairwise_points(p_mov, p_dst)
    if mov.shape[0] < 3:
        raise ValueError("Similarity registration requires at least 3 point correspondences.")
    if np.linalg.matrix_rank(mov_centered) < 2 or np.linalg.matrix_rank(dst_centered) < 2:
        raise ValueError("Similarity registration requires non-collinear point correspondences.")

    rotation = find_rotation(mov_centered, dst_centered)
    numerator = float(np.sum((mov_centered @ rotation.T) * dst_centered))
    denominator = float(np.sum(mov_centered * mov_centered))
    if denominator <= 0.0:
        raise ValueError("Similarity registration requires non-degenerate point correspondences.")
    scale = numerator / denominator
    if scale <= 0.0:
        raise ValueError("Similarity registration requires a positive isotropic scale.")

    linear = scale * rotation
    translation = dst.mean(axis=0) - linear @ mov.mean(axis=0)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = translation
    return transform


def find_rigid_anisotropic_scale(
    p_mov: npt.ArrayLike,
    p_dst: npt.ArrayLike,
    *,
    max_iter: int = 64,
    tol: float = 1e-10,
) -> PointArray:
    """Return a no-shear transform with rotation plus anisotropic scaling.

    The fitted linear part is constrained to ``R @ diag(s)`` with ``R`` a proper
    rotation and ``s`` positive axis scales. The solver alternates exact updates
    for ``R`` and ``diag(s)`` while always rebuilding a single constrained linear
    transform instead of composing incremental updates that would accumulate shear.
    """
    mov, dst, mov_centered, dst_centered = _center_pairwise_points(p_mov, p_dst)
    if mov.shape[0] < 4:
        raise ValueError("Anisotropic-scale registration requires at least 4 point correspondences.")
    if np.linalg.matrix_rank(mov_centered) < 3 or np.linalg.matrix_rank(dst_centered) < 3:
        raise ValueError("Anisotropic-scale registration requires point correspondences that span 3-D space.")

    scales = np.ones(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    min_scale = 1e-12

    for _ in range(max_iter):
        prev_scales = scales.copy()

        scaled_mov = mov_centered * scales
        rotation = find_rotation(scaled_mov, dst_centered)

        aligned_dst = dst_centered @ rotation
        denominator = np.sum(mov_centered * mov_centered, axis=0)
        if np.any(denominator <= 0.0):
            raise ValueError("Anisotropic-scale registration requires non-degenerate point correspondences.")
        scales = np.sum(mov_centered * aligned_dst, axis=0) / denominator
        if np.any(scales <= 0.0):
            raise ValueError("Anisotropic-scale registration requires positive axis scales.")
        scales = np.maximum(scales, min_scale)

        if np.max(np.abs(scales - prev_scales)) <= tol:
            break

    linear = rotation @ np.diag(scales)
    translation = dst.mean(axis=0) - linear @ mov.mean(axis=0)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = translation
    return transform


def find_affine(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the least-squares affine homogeneous transform for paired points."""
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    if mov.shape[0] < 4:
        raise ValueError("Affine registration requires at least 4 point correspondences.")

    design = np.hstack([mov, np.ones((mov.shape[0], 1), dtype=np.float64)])
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError("Affine registration requires point correspondences that span 3-D affine space.")

    coeffs, _, _, _ = np.linalg.lstsq(design, dst, rcond=None)
    transform = np.vstack([coeffs.T, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])
    return transform


def register_points(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike, dof: int = 6) -> PointArray:
    """Dispatch to the closed-form solver matching the requested degrees of freedom."""
    if dof == 6:
        return find_rigid(p_mov, p_dst)
    if dof == 7:
        return find_similarity(p_mov, p_dst)
    if dof == 9:
        return find_rigid_anisotropic_scale(p_mov, p_dst)
    if dof == 12:
        return find_affine(p_mov, p_dst)
    raise ValueError(f"Unsupported dof={dof}. Closed-form point registration supports only 6, 7, 9, or 12 DoF.")
