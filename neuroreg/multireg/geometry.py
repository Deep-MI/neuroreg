"""Geometry helpers for multi-timepoint registration."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..image import compute_centroid
from ..image.map import coerce_image_data_3d
from ..transforms import LTA


def voxel_sizes(image: Any) -> np.ndarray:
    """Return voxel sizes extracted from an image affine.

    Parameters
    ----------
    image : Any
        Image-like object exposing an ``affine`` attribute.

    Returns
    -------
    numpy.ndarray
        Length-3 vector of voxel sizes in millimeters.
    """
    return np.linalg.norm(np.asarray(image.affine, dtype=np.float64)[:3, :3], axis=0)


def direction_cosines(image: Any) -> np.ndarray:
    """Return normalized voxel-axis direction cosines from an image affine.

    Parameters
    ----------
    image : Any
        Image-like object exposing an ``affine`` attribute.

    Returns
    -------
    numpy.ndarray
        ``3 x 3`` direction-cosine matrix.
    """
    sizes = voxel_sizes(image)
    return np.asarray(image.affine, dtype=np.float64)[:3, :3] / sizes[np.newaxis, :]


def ras_center(image: Any) -> np.ndarray:
    """Return the RAS-space center implied by image shape and affine.

    Parameters
    ----------
    image : Any
        Image-like object exposing ``shape`` and ``affine``.

    Returns
    -------
    numpy.ndarray
        Length-3 RAS coordinate of the image center.
    """
    shape = np.asarray(image.shape[:3], dtype=np.float64)
    affine = np.asarray(image.affine, dtype=np.float64)
    return affine[:3, :3] @ (shape / 2.0) + affine[:3, 3]


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a near-rigid matrix onto the closest proper rotation.

    Parameters
    ----------
    matrix : numpy.ndarray
        ``3 x 3`` matrix to project.

    Returns
    -------
    numpy.ndarray
        Proper rotation matrix in ``SO(3)``.
    """
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def reorder_cosines(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Reorder axis directions to match a reference orientation.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference ``3 x 3`` direction-cosine matrix.
    current : numpy.ndarray
        Direction-cosine matrix to reorder.

    Returns
    -------
    numpy.ndarray
        Reordered direction-cosine matrix aligned with ``reference`` without
        introducing a reflection.

    Raises
    ------
    ValueError
        If the input orientations cannot be matched by axis reordering alone or
        if the reordered orientation would be left-handed.
    """
    v2v = reference.T @ current
    axes: list[int] = []
    for column in range(3):
        row = int(np.argmax(np.abs(v2v[:, column])))
        axis = row + 1
        if v2v[row, column] < 0.0:
            axis = -axis
        axes.append(axis)
    if sum(abs(axis) for axis in axes) != 6:
        raise ValueError(
            "Input voxel orientations are not compatible with FreeSurfer-style axis reordering. "
            "Make sure all inputs use consistent voxel orientation."
        )
    if axes == [1, 2, 3]:
        return current
    reorder = np.zeros((3, 3), dtype=np.float64)
    for row, axis in enumerate(axes):
        reorder[row, abs(axis) - 1] = 1.0 if axis > 0 else -1.0
    reordered = current @ reorder
    if np.linalg.det(reordered) < 0:
        raise ValueError("Input voxel orientations would introduce a reflection after axis reordering.")
    return reordered


def mean_direction_cosines(images: Sequence[Any]) -> np.ndarray:
    """Average image orientations after FreeSurfer-style axis reordering.

    Parameters
    ----------
    images : sequence of Any
        Input images whose affine orientations should be averaged.

    Returns
    -------
    numpy.ndarray
        ``3 x 3`` average direction-cosine matrix.

    Raises
    ------
    ValueError
        If any input orientation is incompatible with the FreeSurfer-style axis
        reordering convention.
    """
    cosines = [direction_cosines(image) for image in images]
    first = cosines[0]
    if all(np.allclose(first, cosine, atol=1e-9) for cosine in cosines[1:]):
        return first
    reordered = [first] + [reorder_cosines(first, cosine) for cosine in cosines[1:]]
    return project_to_rotation(np.mean(np.stack(reordered, axis=0), axis=0))


def image_centroid_voxel(image: Any) -> np.ndarray:
    """Compute an intensity-weighted voxel-space centroid.

    Parameters
    ----------
    image : Any
        Image-like object exposing ``get_fdata()``.

    Returns
    -------
    numpy.ndarray
        Length-3 voxel-space centroid.

    Raises
    ------
    ValueError
        If the image data cannot be coerced to a 3-D volume.
    """
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    centroid = compute_centroid(torch.from_numpy(coerce_image_data_3d(data, name="movable image")))
    return centroid.detach().cpu().numpy().astype(np.float64)


def r2r_to_v2v(source_image: Any, target_image: Any, r2r: np.ndarray) -> np.ndarray:
    """Convert an RAS-to-RAS transform into a source-to-target voxel mapping.

    Parameters
    ----------
    source_image : Any
        Source image defining the input voxel lattice.
    target_image : Any
        Target image defining the output voxel lattice.
    r2r : numpy.ndarray
        Source-to-target RAS transform.

    Returns
    -------
    numpy.ndarray
        Source-to-target voxel-to-voxel transform.
    """
    source_affine = np.asarray(source_image.affine, dtype=np.float64)
    target_affine = np.asarray(target_image.affine, dtype=np.float64)
    return np.linalg.inv(target_affine) @ np.asarray(r2r, dtype=np.float64) @ source_affine


def mean_mapped_centroid(
    images: Sequence[Any],
    pairwise_r2r: Sequence[np.ndarray],
    *,
    target_index: int,
    mean_space_from_target_r2r: np.ndarray,
) -> np.ndarray:
    """Compute the mean mapped centroid center in RAS space.

    Parameters
    ----------
    images : sequence of Any
        Input images.
    pairwise_r2r : sequence of numpy.ndarray
        Pairwise RAS transforms mapping each image into the initial target space.
    target_index : int
        Index of the initial target image.
    mean_space_from_target_r2r : numpy.ndarray
        RAS transform from the target image into the unbiased mean space.

    Returns
    -------
    numpy.ndarray
        Length-3 RAS coordinate used to center the template geometry.
    """
    target_image = images[target_index]
    centroid_sum = np.zeros(4, dtype=np.float64)
    for image, r2r in zip(images, pairwise_r2r, strict=False):
        centroid_h = np.ones(4, dtype=np.float64)
        # FreeSurfer's mean-coordinate path maps the centroid through voxel space
        # using 1-based CRS coordinates before converting back to RAS.
        centroid_h[:3] = image_centroid_voxel(image) + 1.0
        centroid_sum += r2r_to_v2v(image, target_image, r2r) @ centroid_h
    centroid_target = centroid_sum / float(len(images))
    target_affine = np.asarray(target_image.affine, dtype=np.float64)
    return (np.asarray(mean_space_from_target_r2r, dtype=np.float64) @ (target_affine @ centroid_target))[:3]


def create_template_geometry(images: Sequence[Any], center_ras: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray]:
    """Create a common template grid covering all input images.

    Parameters
    ----------
    images : sequence of Any
        Input images used to define the template field of view.
    center_ras : numpy.ndarray
        Desired template center in RAS coordinates.

    Returns
    -------
    template_shape : tuple of int
        Template grid shape.
    template_affine : numpy.ndarray
        Template voxel-to-RAS affine.

    Raises
    ------
    ValueError
        If the inputs mix 2-D and 3-D image geometries.
    """
    image_voxel_sizes = [voxel_sizes(image) for image in images]
    conform_size = max(float(np.min(sizes)) for sizes in image_voxel_sizes)
    shapes = [tuple(int(v) for v in image.shape[:3]) for image in images]
    depths = [shape[2] for shape in shapes]
    has_2d = any(depth == 1 for depth in depths)
    if has_2d and any(depth != 1 for depth in depths):
        raise ValueError("Mixing 2-D and 3-D inputs is not supported.")
    eps = 1e-4
    dims = []
    for shape, sizes in zip(shapes, image_voxel_sizes, strict=False):
        fov = np.asarray(shape, dtype=np.float64) * sizes
        dims.append(np.ceil((fov / conform_size) - eps).astype(int))
    template_shape = tuple(int(v) for v in np.max(np.stack(dims, axis=0), axis=0))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = mean_direction_cosines(images) * conform_size
    affine[:3, 3] = np.asarray(center_ras, dtype=np.float64) - affine[:3, :3] @ (np.asarray(template_shape) / 2.0)
    return template_shape, affine


def template_geometry_from_lta(transform: LTA) -> tuple[tuple[int, int, int], np.ndarray]:
    """Extract template geometry from the destination volume info of an LTA.

    Parameters
    ----------
    transform : LTA
        Transform whose destination geometry should define the template.

    Returns
    -------
    template_shape : tuple of int
        Template grid shape.
    template_affine : numpy.ndarray
        Template voxel-to-RAS affine reconstructed from the destination volume
        info.

    Raises
    ------
    ValueError
        If the LTA does not include valid destination geometry.
    """
    info = transform.dst
    if info.get("valid", 1) == 0:
        raise ValueError("init_ltas must include valid destination geometry for template reconstruction.")
    required = ("volume", "voxelsize", "xras", "yras", "zras", "cras")
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"init_ltas destination geometry is missing required fields: {missing}")
    shape = tuple(int(v) for v in info["volume"])
    image_voxel_sizes = np.asarray(info["voxelsize"], dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = np.asarray(info["xras"], dtype=np.float64) * image_voxel_sizes[0]
    affine[:3, 1] = np.asarray(info["yras"], dtype=np.float64) * image_voxel_sizes[1]
    affine[:3, 2] = np.asarray(info["zras"], dtype=np.float64) * image_voxel_sizes[2]
    affine[:3, 3] = np.asarray(info["cras"], dtype=np.float64) - affine[:3, :3] @ (np.asarray(shape) / 2.0)
    return shape, affine


def validate_input_geometries(images: Sequence[Any]) -> None:
    """Validate the cross-timepoint geometry assumptions used by the current MVP.

    Parameters
    ----------
    images : sequence of Any
        Input images to validate.

    Returns
    -------
    None
        This function returns ``None`` when all input geometries are accepted.

    Raises
    ------
    ValueError
        If the images do not share identical voxel sizes.
    RuntimeWarning
        Warns when image orientations differ and will be averaged for the
        template geometry.
    """
    ref_voxel_sizes = voxel_sizes(images[0])
    for idx, image in enumerate(images[1:], start=1):
        sizes = voxel_sizes(image)
        if not np.allclose(sizes, ref_voxel_sizes, atol=1e-6):
            raise ValueError(
                "All input images must have identical voxel sizes for rigid multireg mean-space initialization; "
                f"image 0 has voxel sizes {ref_voxel_sizes.tolist()} and image {idx} has {sizes.tolist()}."
            )
    if any(not np.allclose(direction_cosines(image), direction_cosines(images[0]), atol=1e-9) for image in images[1:]):
        warnings.warn(
            "Input direction cosines differ; multireg will average them when constructing the template geometry.",
            RuntimeWarning,
            stacklevel=2,
        )
