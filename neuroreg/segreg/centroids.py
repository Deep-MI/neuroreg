"""Centroid extraction helpers for label images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

from .io import CentroidDict

ImageLike = str | Path | Any


def load_spatial_image(image: ImageLike) -> Any:
    """Load a nibabel-compatible image or return an already-loaded image."""
    if isinstance(image, str | Path):
        return nib.load(str(image))
    return image


def _resolve_label_ids(seg_data: npt.NDArray[np.integer], label_ids: list[int] | None) -> list[int]:
    if label_ids is None:
        labels = np.unique(seg_data)
        labels = labels[labels > 0]
        return [int(label) for label in labels.tolist()]
    return [int(label) for label in label_ids]


def compute_voxel_centroids_from_seg(
    seg_img: ImageLike,
    label_ids: list[int] | None = None,
) -> dict[int, npt.NDArray[np.float64] | None]:
    """Compute voxel-space centroids for segmentation labels."""
    image = load_spatial_image(seg_img)
    seg_data = np.asarray(image.dataobj)
    labels = _resolve_label_ids(seg_data, label_ids)

    centroids: dict[int, npt.NDArray[np.float64] | None] = {}
    for label in labels:
        coords = np.argwhere(seg_data == label)
        centroids[label] = None if coords.size == 0 else coords.mean(axis=0, dtype=np.float64)
    return centroids


def compute_ras_centroids_from_seg(
    seg_img: ImageLike,
    label_ids: list[int] | None = None,
) -> dict[int, npt.NDArray[np.float64] | None]:
    """Compute RAS-space centroids for segmentation labels."""
    image = load_spatial_image(seg_img)
    voxel_centroids = compute_voxel_centroids_from_seg(image, label_ids=label_ids)
    affine = np.asarray(image.affine, dtype=np.float64)

    ras_centroids: dict[int, npt.NDArray[np.float64] | None] = {}
    for label, voxel_point in voxel_centroids.items():
        if voxel_point is None:
            ras_centroids[label] = None
            continue
        voxel_h = np.append(voxel_point, 1.0)
        ras_centroids[label] = (affine @ voxel_h)[:3]
    return ras_centroids


def collect_joint_centroids(
    mov_centroids: dict[int, npt.NDArray[np.float64] | None],
    ref_centroids: CentroidDict | dict[int, npt.NDArray[np.float64] | None],
    *,
    min_common_labels: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[int]]:
    """Collect matched centroid arrays and their shared label IDs."""
    labels = [
        label
        for label, mov_point in mov_centroids.items()
        if mov_point is not None and label in ref_centroids and ref_centroids[label] is not None
    ]
    labels.sort()

    if len(labels) < min_common_labels:
        raise ValueError(
            f"Need at least {min_common_labels} shared labels, but found only {len(labels)}: {labels}."
        )

    mov_points = np.stack([np.asarray(mov_centroids[label], dtype=np.float64) for label in labels], axis=0)
    ref_points = np.stack([np.asarray(ref_centroids[label], dtype=np.float64) for label in labels], axis=0)
    return mov_points, ref_points, labels


def build_flipped_centroid_targets(
    voxel_centroids: dict[int, npt.NDArray[np.float64] | None],
    lr_pairs: tuple[tuple[int, int], ...],
    *,
    mid_slice: float,
    min_common_labels: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[int]]:
    """Build source/target voxel centroids for left-right flipped self-registration."""
    source_points: list[npt.NDArray[np.float64]] = []
    target_points: list[npt.NDArray[np.float64]] = []
    labels: list[int] = []

    for left_label, right_label in lr_pairs:
        left_point = voxel_centroids.get(left_label)
        right_point = voxel_centroids.get(right_label)
        if left_point is None or right_point is None:
            continue

        left_point = np.asarray(left_point, dtype=np.float64)
        right_point = np.asarray(right_point, dtype=np.float64)

        flipped_right = right_point.copy()
        flipped_right[0] = (2.0 * mid_slice) - flipped_right[0]
        flipped_left = left_point.copy()
        flipped_left[0] = (2.0 * mid_slice) - flipped_left[0]

        source_points.extend([left_point, right_point])
        target_points.extend([flipped_right, flipped_left])
        labels.extend([left_label, right_label])

    if len(labels) < min_common_labels:
        raise ValueError(
            f"Need at least {min_common_labels} flipped label correspondences, but found only {len(labels)} labels."
        )

    return np.stack(source_points, axis=0), np.stack(target_points, axis=0), labels
