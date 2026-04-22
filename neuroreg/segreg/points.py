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
    """Coerce an array-like object into an ``(N, 3)`` point array.

    Parameters
    ----------
    points : array-like
        Candidate point coordinates.
    name : str
        Argument name used in validation errors.

    Returns
    -------
    np.ndarray
        Float64 array with shape ``(N, 3)``.

    Raises
    ------
    ValueError
        If ``points`` cannot be interpreted as a two-dimensional array with
        three columns.
    """
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points_array.shape}.")
    return points_array


def _validate_pairwise_points(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> tuple[PointArray, PointArray]:
    """Validate paired moving and destination point arrays.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Candidate moving and destination point arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated float64 arrays with identical ``(N, 3)`` shape.

    Raises
    ------
    ValueError
        If either array has the wrong shape or if the shapes do not match.
    """
    mov = _as_points(p_mov, name="p_mov")
    dst = _as_points(p_dst, name="p_dst")
    if mov.shape != dst.shape:
        raise ValueError(f"Point arrays must have identical shape, got {mov.shape} and {dst.shape}.")
    return mov, dst


def find_rotation(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Estimate the best-fit proper rotation between centered point sets.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Mean-centered moving and destination point arrays with matching shape.

    Returns
    -------
    np.ndarray
        ``(3, 3)`` proper rotation matrix minimizing the least-squares error.

    Notes
    -----
    The solution uses the Kabsch SVD update and flips the final singular vector
    when necessary to enforce a positive determinant.
    """
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
    """Return paired point sets together with mean-centered coordinates.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Moving and destination point arrays.

    Returns
    -------
    mov : np.ndarray
        Original moving points.
    dst : np.ndarray
        Original destination points.
    mov_centered : np.ndarray
        Moving points with the centroid removed.
    dst_centered : np.ndarray
        Destination points with the centroid removed.
    """
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    centroid_mov = mov.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    return mov, dst, mov - centroid_mov, dst - centroid_dst


def find_rigid(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> PointArray:
    """Fit a rigid transform between paired 3-D point sets.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Paired moving and destination points with shape ``(N, 3)``.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` homogeneous rigid transform mapping moving points to
        destination points.

    Raises
    ------
    ValueError
        If fewer than three correspondences are provided or if the points are
        degenerate/collinear.
    """
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
    """Fit a similarity transform with one global scale factor.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Paired moving and destination points with shape ``(N, 3)``.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` homogeneous similarity transform.

    Raises
    ------
    ValueError
        If the correspondences are insufficient, degenerate, or imply a
        non-positive isotropic scale.
    """
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
    """Fit a rotation-plus-anisotropic-scale transform without shear.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Paired moving and destination points with shape ``(N, 3)``.
    max_iter : int, default=64
        Maximum number of alternating updates for the rotation and scale terms.
    tol : float, default=1e-10
        Convergence tolerance on the change in axis scales.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` homogeneous transform whose linear part is constrained to
        ``R @ diag(s)``.

    Raises
    ------
    ValueError
        If fewer than four correspondences are provided, if the points do not
        span 3-D space, or if the fitted axis scales become non-positive.

    Notes
    -----
    The solver alternates exact updates for ``R`` and ``diag(s)`` while always
    rebuilding one constrained linear transform instead of composing
    incremental updates that would accumulate shear.
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
    """Fit a full affine transform between paired 3-D point sets.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Paired moving and destination points with shape ``(N, 3)``.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` least-squares affine transform.

    Raises
    ------
    ValueError
        If fewer than four correspondences are provided or if the points do not
        span affine 3-D space.
    """
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
    """Dispatch to the closed-form point-set solver for a requested DoF.

    Parameters
    ----------
    p_mov, p_dst : array-like
        Paired moving and destination points with shape ``(N, 3)``.
    dof : {6, 7, 9, 12}, default=6
        Requested transform family.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` homogeneous transform returned by the selected solver.

    Raises
    ------
    ValueError
        If ``dof`` is not one of the supported closed-form solvers.
    """
    if dof == 6:
        return find_rigid(p_mov, p_dst)
    if dof == 7:
        return find_similarity(p_mov, p_dst)
    if dof == 9:
        return find_rigid_anisotropic_scale(p_mov, p_dst)
    if dof == 12:
        return find_affine(p_mov, p_dst)
    raise ValueError(f"Unsupported dof={dof}. Closed-form point registration supports only 6, 7, 9, or 12 DoF.")
