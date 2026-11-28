"""Geometry helpers for multi-timepoint registration."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..image import compute_centroid
from ..image.map import coerce_image_data_3d
from ..transforms import LTA

logger = logging.getLogger(__name__)

#: Voxel sizes closer than this, in millimeters, count as equal; FreeSurfer's
#: ``mri_robust_template`` default.
VOXEL_SIZE_TOLERANCE = 1e-3


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


def rotation_mean(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Average rotation matrices via their axis-angle rotation vectors.

    Each rotation is converted to a rotation vector (axis scaled by angle), the
    vectors are averaged arithmetically, and the mean vector is converted back
    to a matrix. This is the log-Euclidean mean on ``SO(3)`` and matches
    FreeSurfer's ``MyMatrix::RotationMean`` used for the rigid mean-space
    construction in ``mri_robust_template``. It differs from an arithmetic mean
    of the rotation matrices followed by polar/SVD projection, which biases the
    result by a small fixed rotation (~0.1 deg here) when the inputs span
    several degrees.

    Parameters
    ----------
    rotations : sequence of numpy.ndarray
        ``3 x 3`` rotation matrices to average. Must be non-empty.

    Returns
    -------
    numpy.ndarray
        ``3 x 3`` mean rotation matrix.

    Raises
    ------
    ValueError
        If ``rotations`` is empty.
    """
    mats = [project_to_rotation(rotation) for rotation in rotations]
    if not mats:
        raise ValueError("rotation_mean requires at least one rotation.")
    rotvecs = np.stack([Rotation.from_matrix(rotation).as_rotvec() for rotation in mats], axis=0)
    return Rotation.from_rotvec(rotvecs.mean(axis=0)).as_matrix()


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
    conform_size = template_voxel_size(images)
    template_shape = template_grid_shape(images, conform_size)
    return template_shape, grid_affine(mean_direction_cosines(images), conform_size, template_shape, center_ras)


def template_voxel_size(images: Sequence[Any]) -> float:
    """Return the isotropic template voxel size FreeSurfer derives from the inputs.

    Each image contributes its smallest voxel side, and the largest of those is
    taken, as in ``MultiRegistration::createTemplateGeo``: no input is upsampled.

    Parameters
    ----------
    images : sequence of Any
        Input images.

    Returns
    -------
    float
        Template voxel size in millimeters.
    """
    return max(float(np.min(voxel_sizes(image))) for image in images)


def template_grid_shape(images: Sequence[Any], voxel_size: float) -> tuple[int, int, int]:
    """Return template dimensions large enough for every input's field of view.

    Parameters
    ----------
    images : sequence of Any
        Input images.
    voxel_size : float
        Template voxel size in millimeters.

    Returns
    -------
    tuple of int
        Template grid shape, the per-axis maximum over the inputs.

    Raises
    ------
    ValueError
        If the inputs mix 2-D and 3-D image geometries.
    """
    shapes = [tuple(int(v) for v in image.shape[:3]) for image in images]
    depths = [shape[2] for shape in shapes]
    if any(depth == 1 for depth in depths) and any(depth != 1 for depth in depths):
        raise ValueError("Mixing 2-D and 3-D inputs is not supported.")
    eps = 1e-4
    dims = []
    for image, shape in zip(images, shapes, strict=False):
        fov = np.asarray(shape, dtype=np.float64) * voxel_sizes(image)
        dims.append(np.ceil((fov / voxel_size) - eps).astype(int))
    return tuple(int(v) for v in np.max(np.stack(dims, axis=0), axis=0))


def grid_affine(
    cosines: np.ndarray, voxel_size: float, shape: tuple[int, int, int], center_ras: np.ndarray
) -> np.ndarray:
    """Build an isotropic voxel-to-RAS affine whose grid center lies at ``center_ras``.

    Parameters
    ----------
    cosines : numpy.ndarray
        ``3 x 3`` direction cosines, one column per voxel axis.
    voxel_size : float
        Isotropic voxel size in millimeters.
    shape : tuple of int
        Grid shape; its center is taken at ``shape / 2``, the c_ras convention.
    center_ras : numpy.ndarray
        RAS coordinate of the grid center.

    Returns
    -------
    numpy.ndarray
        ``4 x 4`` voxel-to-RAS affine.
    """
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.asarray(cosines, dtype=np.float64) * voxel_size
    affine[:3, 3] = np.asarray(center_ras, dtype=np.float64) - affine[:3, :3] @ (np.asarray(shape) / 2.0)
    return affine


def init_transform_template_geometry(
    images: Sequence[Any],
    transforms_r2r: Sequence[np.ndarray],
    destination: tuple[tuple[int, int, int], np.ndarray],
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return the template grid for inputs given with initial transforms.

    The transforms' shared destination grid is used as it is when its voxel
    size is the one the inputs call for. Otherwise only its orientation is kept:
    an atlas destination is typically 1 mm, and taking it outright would
    resample finer inputs onto it. The voxel size then follows the inputs, as
    for a derived grid, and the grid is centered on the average of the image
    centers mapped into template space. Its extent along each of its own axes
    covers every input's field of view as mapped there, so inputs stored in
    another axis order (a sagittal scan into an LIA grid) are not cropped.

    Parameters
    ----------
    images : sequence of Any
        Input images.
    transforms_r2r : sequence of numpy.ndarray
        Timepoint-to-template RAS transforms aligned with ``images``.
    destination : tuple
        ``(shape, affine)`` of the transforms' shared destination grid.

    Returns
    -------
    template_shape : tuple of int
        Template grid shape.
    template_affine : numpy.ndarray
        Template voxel-to-RAS affine.
    """
    dst_shape, dst_affine = destination
    dst_affine = np.asarray(dst_affine, dtype=np.float64)
    dst_sizes = np.linalg.norm(dst_affine[:3, :3], axis=0)
    size = template_voxel_size(images)
    if np.allclose(dst_sizes, size, atol=VOXEL_SIZE_TOLERANCE):
        return tuple(int(v) for v in dst_shape), dst_affine
    logger.info(
        "Keeping the orientation of the init_ltas' destination grid, but at the inputs' voxel size %g mm "
        "instead of its %s mm.",
        size,
        np.round(dst_sizes, 6).tolist(),
    )
    cosines = dst_affine[:3, :3] / dst_sizes
    center = np.mean(
        np.stack(
            [
                (np.asarray(r2r, dtype=np.float64) @ np.append(ras_center(image), 1.0))[:3]
                for image, r2r in zip(images, transforms_r2r, strict=False)
            ]
        ),
        axis=0,
    )
    # How far each input's field of view reaches below and above the center,
    # in voxels along each grid axis: its mapped corners, projected onto them.
    unit_corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)
    low = np.zeros(3, dtype=np.float64)
    high = np.zeros(3, dtype=np.float64)
    for image, r2r in zip(images, transforms_r2r, strict=False):
        voxel_corners = unit_corners * np.asarray(image.shape[:3], dtype=np.float64) - 0.5
        ras = np.asarray(r2r, dtype=np.float64) @ np.asarray(image.affine, dtype=np.float64)
        mapped = (np.c_[voxel_corners, np.ones(8)] @ ras.T)[:, :3]
        projected = (mapped - center) @ cosines / size
        low = np.minimum(low, projected.min(axis=0))
        high = np.maximum(high, projected.max(axis=0))
    # The center sits at voxel shape / 2, so the grid covers shape / 2 + 0.5
    # voxels below it and shape / 2 - 0.5 above.
    shape = tuple(int(v) for v in np.ceil(np.maximum(-2.0 * low - 1.0, 2.0 * high + 1.0) - 1e-4))
    return shape, grid_affine(cosines, size, shape, center)


#: How to supply a destination geometry when the given transforms carry none.
_MISSING_DST_REMEDY = (
    "Attach one with 'lta convert --dst', building the grid first with 'mri geom' if no "
    "suitable reference image exists."
)


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
        # Transforms from atlas registration that carries no target image are
        # written with valid = 0, for example centroid-based segreg against a
        # target file with no geometry block of its own.
        raise ValueError(
            "init_ltas must include valid destination geometry for template reconstruction, but "
            "this transform has valid = 0 and so does not say where the template lives. "
            f"{_MISSING_DST_REMEDY}"
        )
    required = ("volume", "voxelsize", "xras", "yras", "zras", "cras")
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(
            f"init_ltas destination geometry is missing required fields: {missing}. "
            f"{_MISSING_DST_REMEDY}"
        )
    shape = tuple(int(v) for v in info["volume"])
    image_voxel_sizes = np.asarray(info["voxelsize"], dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = np.asarray(info["xras"], dtype=np.float64) * image_voxel_sizes[0]
    affine[:3, 1] = np.asarray(info["yras"], dtype=np.float64) * image_voxel_sizes[1]
    affine[:3, 2] = np.asarray(info["zras"], dtype=np.float64) * image_voxel_sizes[2]
    affine[:3, 3] = np.asarray(info["cras"], dtype=np.float64) - affine[:3, :3] @ (np.asarray(shape) / 2.0)
    return shape, affine


#: Overlap below which a time point and the template grid are reported as barely
#: related. The measure is the larger of the two containment fractions, so this
#: only fires when little of the template is filled *and* little of the time
#: point is kept: a grid framing one structure inside a larger image scores 1.0,
#: as does a grid padded out around a smaller image.
_SMALL_OVERLAP_FRACTION = 0.5


def check_template_grid_covers_inputs(
    images: Sequence[Any],
    transforms_r2r: Sequence[np.ndarray],
    *,
    template_shape: tuple[int, int, int],
    template_affine: np.ndarray,
) -> None:
    """Check that each mapped time point lands on the template grid.

    A derived template grid covers all inputs by construction, but a grid taken
    from ``init_ltas`` or supplied outright can be placed anywhere, and a wrong
    ``c_ras`` then yields an empty or heavily cropped template with no other
    symptom.

    Coverage is compared between axis-aligned bounding boxes in template voxel
    space, with two consequences. An obliquely mapped extent is overstated, so
    the measure errs toward staying quiet: a reported miss is always a real one,
    but two genuinely disjoint oblique boxes whose bounding boxes overlap go
    unreported. And the measure is purely geometric, so a grid covering half an
    image that is mostly background scores the same as one covering half of
    dense anatomy.

    Parameters
    ----------
    images : sequence of Any
        Input images, exposing ``shape`` and ``affine``.
    transforms_r2r : sequence of numpy.ndarray
        Timepoint-to-template RAS transforms aligned with ``images``.
    template_shape : tuple of int
        Template grid shape.
    template_affine : numpy.ndarray
        Template voxel-to-RAS affine.

    Returns
    -------
    None
        This function returns ``None`` when every time point reaches the grid.

    Raises
    ------
    ValueError
        If a time point maps entirely outside the template grid.

    Warns
    -----
    RuntimeWarning
        Warns once per time point that overlaps the template grid by less than
        50%, measured as the larger of the two containment fractions. That is a
        partial result rather than an impossible one and can be intended.
    """
    unit_corners = np.array(
        [[i, j, k, 1.0] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=np.float64,
    )
    template_low = np.full(3, -0.5)
    template_high = np.asarray(template_shape, dtype=np.float64) - 0.5
    template_volume = float(np.prod(template_high - template_low))
    template_affine = np.asarray(template_affine, dtype=np.float64)
    template_inv = np.linalg.inv(template_affine)
    # Grid center at shape / 2, the c_ras convention used throughout.
    template_center = (template_affine @ np.append(np.asarray(template_shape, dtype=np.float64) / 2.0, 1.0))[:3]
    for index, (image, r2r) in enumerate(zip(images, transforms_r2r, strict=False)):
        image_affine = np.asarray(image.affine, dtype=np.float64)
        image_shape = np.asarray(image.shape[:3], dtype=np.float64)
        mapped_center = (np.asarray(r2r, dtype=np.float64) @ image_affine @ np.append(image_shape / 2.0, 1.0))[:3]
        # Reported because it is the number that identifies a misplaced c_ras,
        # and it stays meaningful whatever the two grids' orientations are.
        center_distance = float(np.linalg.norm(mapped_center - template_center))
        corners = unit_corners.copy()
        # Voxel centers span 0 to shape - 1, so the covered region runs half a
        # voxel further out on each side.
        corners[:, :3] = corners[:, :3] * image_shape - 0.5
        mapped = (corners @ (template_inv @ np.asarray(r2r, dtype=np.float64) @ image_affine).T)[:, :3]
        mapped_low = mapped.min(axis=0)
        mapped_high = mapped.max(axis=0)
        overlap_extent = np.minimum(mapped_high, template_high) - np.maximum(mapped_low, template_low)
        if np.any(overlap_extent <= 0.0):
            raise ValueError(
                f"TP {index + 1} maps entirely outside the template grid, so it would contribute nothing to "
                f"the template. Its center lands {center_distance:.1f} mm from the grid center. Check the "
                "placement (c_ras) of the template geometry and the input transforms."
            )
        overlap_volume = float(np.prod(overlap_extent))
        mapped_volume = max(float(np.prod(mapped_high - mapped_low)), np.finfo(np.float64).tiny)
        covered = max(overlap_volume / template_volume, overlap_volume / mapped_volume)
        if covered < _SMALL_OVERLAP_FRACTION:
            warnings.warn(
                f"TP {index + 1} overlaps the template grid by only {covered:.0%}, so most of the template "
                "will be empty and most of the time point will be cropped away. Its center lands "
                f"{center_distance:.1f} mm from the grid center. Check the placement (c_ras) and extent of "
                "the template geometry.",
                RuntimeWarning,
                stacklevel=2,
            )


def validate_input_geometries(
    images: Sequence[Any], *, derives_geometry: bool = True, allow_diff_vox_size: bool = False
) -> None:
    """Validate the cross-timepoint geometry assumptions.

    Parameters
    ----------
    images : sequence of Any
        Input images to validate.
    derives_geometry : bool, default=True
        Whether the caller will derive the template geometry from these images.
        When ``False`` the geometry comes from elsewhere, so differing input
        orientations are not averaged and are not worth reporting.
    allow_diff_vox_size : bool, default=False
        Accept inputs with different voxel sizes, with a warning, instead of
        rejecting them.

    Returns
    -------
    None
        This function returns ``None`` when all input geometries are accepted.

    Raises
    ------
    ValueError
        If voxel sizes differ by more than :data:`VOXEL_SIZE_TOLERANCE` and
        ``allow_diff_vox_size`` is not set. As in FreeSurfer's
        ``mri_robust_template``, time points are expected to share one
        resolution: the template takes the coarsest, so finer ones would be
        downsampled and differ from the rest in noise and detail.

    Warns
    -----
    RuntimeWarning
        Warns when voxel sizes differ and ``allow_diff_vox_size`` is set, or
        when image orientations differ and will be averaged for the template
        geometry.
    """
    ref_voxel_sizes = voxel_sizes(images[0])
    for idx, image in enumerate(images[1:], start=1):
        sizes = voxel_sizes(image)
        # Sorted, so the same resolution stored in another axis order (an
        # axial and a sagittal scan of one protocol) is not a difference.
        if not np.allclose(np.sort(sizes), np.sort(ref_voxel_sizes), atol=VOXEL_SIZE_TOLERANCE):
            message = (
                f"Input voxel sizes differ: image 1 has {np.round(ref_voxel_sizes, 6).tolist()} mm and image "
                f"{idx + 1} has {np.round(sizes, 6).tolist()} mm."
            )
            if not allow_diff_vox_size:
                raise ValueError(
                    f"{message} Time points should share one resolution, so conform them to a common voxel "
                    "size first, or allow it explicitly (--allow-diff-vox-size); the template then takes the "
                    "coarsest size, and finer time points are downsampled."
                )
            warnings.warn(
                f"{message} The template takes the coarsest size, so finer time points are downsampled.",
                RuntimeWarning,
                stacklevel=2,
            )
            break
    if derives_geometry and any(
        not np.allclose(direction_cosines(image), direction_cosines(images[0]), atol=1e-9) for image in images[1:]
    ):
        warnings.warn(
            "Input direction cosines differ; multireg will average them when constructing the template geometry.",
            RuntimeWarning,
            stacklevel=2,
        )
