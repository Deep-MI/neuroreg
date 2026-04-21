"""Point-set registration helpers for fixed-correspondence centroid fits.

The robust rigid path adapts the fixed-correspondence robust weighting idea
from ``head-motion-tools``. That implementation cites:

- Bergstrom and Edlund (2014), *Robust registration of point sets using
  iteratively reweighted least squares*.
- Bergstrom and Edlund (2017), *Robust registration of surfaces using a
  refined iterative closest point algorithm with a trust region approach*.
- Pollak et al. (2023), *Quantifying MR head motion in the Rhineland Study:
  A robust method for population cohorts*.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import numpy.typing as npt

PointArray = npt.NDArray[np.float64]
WeightArray = npt.NDArray[np.float64]
RobustEstimatorName = Literal["tukey", "huber", "cauchy"]

_ROBUST_ESTIMATOR_SCALES: dict[RobustEstimatorName, float] = {
    "tukey": 7.0589,
    "huber": 2.0138,
    "cauchy": 4.304,
}


@dataclass(frozen=True)
class RobustPointRegistrationInfo:
    """Diagnostics returned by :func:`register_points_robust`.

    Attributes
    ----------
    estimator : {'tukey', 'huber', 'cauchy'}
        Robust weighting rule used during the fit.
    iterations : int
        Number of IRLS-style reweighting iterations performed.
    bound_scale : float
        Base scaling applied to the median residual distance before converting
        it to an estimator-specific saturation threshold.
    robust_bound : float
        Final estimator-specific saturation threshold used to compute
        ``weights`` from ``residuals``.
    residuals : np.ndarray
        Final Euclidean residual distance for each fixed correspondence.
    weights : np.ndarray
        Final robust weight for each fixed correspondence.
    """

    estimator: RobustEstimatorName
    iterations: int
    bound_scale: float
    robust_bound: float
    residuals: WeightArray
    weights: WeightArray


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


def _validate_rigid_points(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike) -> tuple[PointArray, PointArray]:
    mov, dst = _validate_pairwise_points(p_mov, p_dst)
    if mov.shape[0] < 3:
        raise ValueError("Rigid registration requires at least 3 point correspondences.")

    mov_centered = mov - mov.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)
    if np.linalg.matrix_rank(mov_centered) < 2 or np.linalg.matrix_rank(dst_centered) < 2:
        raise ValueError("Rigid registration requires non-collinear point correspondences.")
    return mov, dst


def _apply_rigid_transform(points: PointArray, transform: PointArray) -> PointArray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _rigid_residuals(p_mov: PointArray, p_dst: PointArray, transform: PointArray) -> WeightArray:
    return np.linalg.norm(_apply_rigid_transform(p_mov, transform) - p_dst, axis=1)


def _estimator_saturation(distance_scale: float, estimator: RobustEstimatorName) -> float:
    return _ROBUST_ESTIMATOR_SCALES[estimator] * distance_scale


def _estimate_distance_scale(distances: WeightArray, *, bound_scale: float) -> float:
    median_distance = float(np.median(distances))
    return bound_scale * median_distance


def _robust_weights(
    distances: WeightArray,
    *,
    estimator: RobustEstimatorName,
    bound_scale: float,
) -> tuple[WeightArray, float]:
    max_distance = float(np.max(distances))
    if max_distance <= 1e-12:
        return np.ones_like(distances, dtype=np.float64), 1.0

    distance_scale = _estimate_distance_scale(distances, bound_scale=bound_scale)
    robust_bound = _estimator_saturation(distance_scale, estimator)
    if max_distance == 0.0:
        robust_bound = 1.0
    elif robust_bound < 1e-6 * max_distance:
        robust_bound = 0.3 * max_distance
    if estimator == "tukey":
        ratios = distances / robust_bound
        weights = np.where(distances <= robust_bound, (1.0 - ratios**2) ** 2, 0.0)
    elif estimator == "huber":
        safe_distances = np.maximum(distances, np.finfo(np.float64).eps)
        weights = np.where(distances <= robust_bound, 1.0, robust_bound / safe_distances)
    else:
        weights = 1.0 / (1.0 + (distances / robust_bound) ** 2)

    return np.asarray(weights, dtype=np.float64), float(robust_bound)


def _find_weighted_rigid(p_mov: PointArray, p_dst: PointArray, weights: WeightArray) -> PointArray:
    if weights.shape != (p_mov.shape[0],):
        raise ValueError(f"weights must have shape ({p_mov.shape[0]},), got {weights.shape}.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")

    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Robust rigid registration produced zero total weight.")

    support_threshold = np.finfo(np.float64).eps * max(1.0, float(np.max(weights)))
    support_mask = weights > support_threshold
    if int(np.count_nonzero(support_mask)) < 3:
        raise ValueError("Robust rigid registration requires at least 3 supported correspondences.")

    normalized_weights = weights / weight_sum
    centroid_mov = np.sum(p_mov * normalized_weights[:, None], axis=0)
    centroid_dst = np.sum(p_dst * normalized_weights[:, None], axis=0)
    mov_centered = p_mov - centroid_mov
    dst_centered = p_dst - centroid_dst

    if np.linalg.matrix_rank(mov_centered[support_mask]) < 2 or np.linalg.matrix_rank(dst_centered[support_mask]) < 2:
        raise ValueError("Robust rigid registration requires non-collinear supported correspondences.")

    h = (mov_centered * normalized_weights[:, None]).T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = centroid_dst - rotation @ centroid_mov

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _validate_robust_args(
    *,
    estimator: str,
    max_iters: int,
    bound_scale: float,
) -> RobustEstimatorName:
    if estimator not in _ROBUST_ESTIMATOR_SCALES:
        valid = ", ".join(sorted(_ROBUST_ESTIMATOR_SCALES))
        raise ValueError(f"Unknown robust estimator '{estimator}'. Choose from: {valid}.")
    if max_iters < 1:
        raise ValueError(f"robust_max_iters must be >= 1, got {max_iters}.")
    if bound_scale <= 0.0:
        raise ValueError(f"robust_bound_scale must be > 0, got {bound_scale}.")
    return estimator


def _trimmed_residual_score(residuals: WeightArray) -> float:
    trim_count = max(3, int(np.ceil(0.75 * residuals.shape[0])))
    kept = np.partition(residuals, trim_count - 1)[:trim_count]
    return float(np.mean(kept**2))


def _candidate_subset_iter(num_points: int) -> list[tuple[int, ...]]:
    if num_points <= 12:
        return list(combinations(range(num_points), 3))
    return []


def _robust_initial_transform(p_mov: PointArray, p_dst: PointArray) -> PointArray:
    best_transform = find_rigid(p_mov, p_dst)
    best_score = _trimmed_residual_score(_rigid_residuals(p_mov, p_dst, best_transform))
    num_points = p_mov.shape[0]

    candidate_masks: list[np.ndarray] = []
    if num_points > 3:
        for drop_idx in range(num_points):
            keep_mask = np.ones(num_points, dtype=bool)
            keep_mask[drop_idx] = False
            candidate_masks.append(keep_mask)
    for subset in _candidate_subset_iter(num_points):
        keep_mask = np.zeros(num_points, dtype=bool)
        keep_mask[list(subset)] = True
        candidate_masks.append(keep_mask)

    for keep_mask in candidate_masks:
        try:
            candidate = find_rigid(p_mov[keep_mask], p_dst[keep_mask])
        except ValueError:
            continue
        score = _trimmed_residual_score(_rigid_residuals(p_mov, p_dst, candidate))
        if score < best_score:
            best_transform = candidate
            best_score = score

    return best_transform


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
    mov, dst = _validate_rigid_points(p_mov, p_dst)
    centroid_mov = mov.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    mov_centered = mov - centroid_mov
    dst_centered = dst - centroid_dst

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


def register_points_robust(
    p_mov: npt.ArrayLike,
    p_dst: npt.ArrayLike,
    *,
    estimator: RobustEstimatorName = "tukey",
    robust_max_iters: int = 20,
    robust_bound_scale: float = 0.5,
) -> tuple[PointArray, RobustPointRegistrationInfo]:
    """Register two fixed-correspondence point sets with robust rigid reweighting.

    This implements an IRLS-style rigid fit for small labeled point sets. It
    adapts the fixed-correspondence robust weighting idea from
    ``head-motion-tools`` without importing that package's KDTree-based ICP or
    its point-cloud preprocessing pipeline. The underlying method lineage is
    documented in the module-level references above.
    """
    mov, dst = _validate_rigid_points(p_mov, p_dst)
    validated_estimator = _validate_robust_args(
        estimator=estimator,
        max_iters=robust_max_iters,
        bound_scale=robust_bound_scale,
    )

    transform = _robust_initial_transform(mov, dst)
    residuals = _rigid_residuals(mov, dst, transform)
    weights, robust_bound = _robust_weights(
        residuals,
        estimator=validated_estimator,
        bound_scale=robust_bound_scale,
    )

    iterations = 0
    for iteration in range(1, robust_max_iters + 1):
        new_transform = _find_weighted_rigid(mov, dst, weights)
        new_residuals = _rigid_residuals(mov, dst, new_transform)
        new_weights, new_bound = _robust_weights(
            new_residuals,
            estimator=validated_estimator,
            bound_scale=robust_bound_scale,
        )

        transform_delta = float(np.max(np.abs(new_transform - transform)))
        weight_delta = float(np.max(np.abs(new_weights - weights)))
        transform = new_transform
        residuals = new_residuals
        weights = new_weights
        robust_bound = new_bound
        iterations = iteration

        if transform_delta <= 1e-10 and weight_delta <= 1e-10:
            break

    info = RobustPointRegistrationInfo(
        estimator=validated_estimator,
        iterations=iterations,
        bound_scale=float(robust_bound_scale),
        robust_bound=robust_bound,
        residuals=residuals.copy(),
        weights=weights.copy(),
    )
    return transform, info


def register_points(p_mov: npt.ArrayLike, p_dst: npt.ArrayLike, dof: int = 6) -> PointArray:
    """Register two point sets with a rigid or affine closed-form fit."""
    if dof == 6:
        return find_rigid(p_mov, p_dst)
    if dof == 12:
        return find_affine(p_mov, p_dst)
    raise ValueError(f"Unsupported dof={dof}. Closed-form point registration supports only 6 or 12 DoF.")
