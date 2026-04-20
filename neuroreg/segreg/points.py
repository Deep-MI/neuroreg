"""Closed-form point-set registration helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

PointArray = npt.NDArray[np.float64]


def _as_points(points: npt.ArrayLike, *, name: str) -> PointArray:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points_array.shape}.")
    return points_array


def _validate_pairwise_points(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> tuple[PointArray, PointArray]:
    mov = _as_points(p_mov, name="p_mov")
    dst = _as_points(p_dst, name="p_dst")
    if mov.shape != dst.shape:
        raise ValueError(f"Point arrays must have identical shape, got {mov.shape} and {dst.shape}.")
    return mov, dst


def find_rotation(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the rotation matrix that best aligns two centered point sets."""
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    h = mov.T @ dst
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def find_rigid(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the rigid homogeneous transform that best aligns two point sets."""
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    if mov.shape[0] < 3:
        raise ValueError("Rigid registration requires at least 3 point correspondences.")

    centroid_mov = mov.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    mov_centered = mov - centroid_mov
    dst_centered = dst - centroid_dst

    if np.linalg.matrix_rank(mov_centered) < 2 or np.linalg.matrix_rank(dst_centered) < 2:
        raise ValueError("Rigid registration requires non-collinear point correspondences.")

    rotation = find_rotation(mov_centered, dst_centered)
    translation = centroid_dst - rotation @ centroid_mov

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def find_affine(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Return the affine homogeneous transform that best aligns two point sets."""
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
    """Register two point sets with a rigid or affine closed-form fit."""
    if dof == 6:
        return find_rigid(p_mov, p_dst)
    if dof == 12:
        return find_affine(p_mov, p_dst)
    raise ValueError(f"Unsupported dof={dof}. Closed-form point registration supports only 6 or 12 DoF.")
