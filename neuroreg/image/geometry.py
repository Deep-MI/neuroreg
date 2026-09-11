"""FreeSurfer-style image geometry helpers.

This module centralizes conversions involving voxel coordinates, scanner RAS,
and FreeSurfer tkRAS so that BBR, segmentation-derived surfaces, and transform
format conversions all use the same geometry conventions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import nibabel as nib
import numpy as np

_AnyVolRef: TypeAlias = (
        nib.Nifti1Image
        | nib.MGHImage
        | nib.nifti1.Nifti1Header
        | nib.freesurfer.mghformat.MGHHeader
)

#: RAS unit vector each FreeSurfer orientation letter points along.
_ORIENTATION_AXES: dict[str, tuple[float, float, float]] = {
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
    "A": (0.0, 1.0, 0.0),
    "P": (0.0, -1.0, 0.0),
    "S": (0.0, 0.0, 1.0),
    "I": (0.0, 0.0, -1.0),
}

#: Letters that name the same anatomical axis and so cannot both appear.
_ORIENTATION_PAIRS: tuple[frozenset[str], ...] = (
    frozenset("RL"),
    frozenset("AP"),
    frozenset("SI"),
)


def direction_cosines_from_orientation(code: str) -> np.ndarray:
    """Build axis-aligned direction cosines from an orientation code.

    A FreeSurfer orientation code such as ``"LIA"`` names, for each voxel axis
    in order, the anatomical direction in which that index increases. Letter
    ``n`` therefore becomes column ``n`` of the returned matrix: ``"LIA"`` gives
    columns ``(-1, 0, 0)``, ``(0, 0, -1)``, ``(0, 1, 0)``, which is what
    ``nibabel.aff2axcodes`` reports back as ``('L', 'I', 'A')``.

    The result is a signed permutation of the identity, i.e. a strict
    axis-aligned orientation. Preserving an oblique rotation while reordering
    axes ("soft" orientation) is a different operation and not done here.

    Parameters
    ----------
    code : str
        Three letters, one from each of ``R``/``L``, ``A``/``P``, ``S``/``I``,
        in any order. Case-insensitive.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Direction cosines with unit-length columns.

    Raises
    ------
    ValueError
        If ``code`` is not three letters naming three distinct axes.
    """
    normalized = str(code).strip().upper()
    if len(normalized) != 3 or any(letter not in _ORIENTATION_AXES for letter in normalized):
        raise ValueError(
            f"Orientation must be three letters from R/L, A/P, S/I (e.g. LIA), got: {code!r}"
        )
    for pair in _ORIENTATION_PAIRS:
        if len(pair.intersection(normalized)) != 1:
            raise ValueError(
                f"Orientation must name each of the R/L, A/P and S/I axes exactly once, got: {code!r}"
            )
    # Column n is the RAS direction of voxel axis n.
    return np.column_stack([np.asarray(_ORIENTATION_AXES[letter], dtype=np.float64) for letter in normalized])


def shape_from_fov(fov: np.ndarray, vox_size: np.ndarray) -> tuple[int, int, int]:
    """Derive image dimensions from a field of view and a voxel size.

    A field of view is usually the known quantity (FreeSurfer conforms to
    256 mm) while the dimensions follow from the chosen voxel size, so this
    avoids callers computing ``fov / vox`` themselves.

    ``fov / vox_size`` is an exact integer whenever the field of view is a
    multiple of the voxel size, but float arithmetic makes it only
    approximately so: storing a zoom as float32 leaves a relative residual that
    grows with the voxel count, so ``256 / 0.8`` can land on ``320.0000001``.
    Rounding that up would add a spurious voxel. Counts that are integer within
    a small *relative* tolerance are therefore snapped to the nearest integer,
    and only genuine partial voxels round up, so the field of view is never
    truncated. This matches FastSurfer's ``conform`` so the two agree on the
    same inputs.

    Parameters
    ----------
    fov : numpy.ndarray, shape (3,)
        Field of view per axis, in mm.
    vox_size : numpy.ndarray, shape (3,)
        Voxel size per axis, in mm.

    Returns
    -------
    tuple of int
        Image dimensions in voxels, per axis.

    Raises
    ------
    ValueError
        If any field of view or voxel size is not strictly positive.
    """
    fov = np.asarray(fov, dtype=np.float64)
    vox_size = np.asarray(vox_size, dtype=np.float64)
    if np.any(fov <= 0) or np.any(vox_size <= 0):
        raise ValueError(
            f"Field of view and voxel size must be positive, got fov={fov.tolist()}, vox_size={vox_size.tolist()}."
        )
    counts = fov / vox_size
    rounded = np.rint(counts)
    snapped = np.where(np.isclose(counts, rounded, rtol=1e-6, atol=0.0), rounded, np.ceil(counts))
    sizes = [int(v) for v in snapped]
    return sizes[0], sizes[1], sizes[2]


def place_grid_at_cras(
        affine: np.ndarray,
        shape: tuple[int, int, int],
        cras: np.ndarray,
) -> np.ndarray:
    """Move a voxel grid so its centre sits at ``cras`` in world space.

    ``c_ras`` is the world coordinate of the grid centre, taken at
    ``shape / 2`` exactly rather than ``(shape - 1) / 2``, matching MGH's
    ``Pxyz_c`` field and :func:`neuroreg.image.describe_image`. Only the
    translation column changes, so direction cosines, voxel sizes and matrix
    size are preserved by construction, including for anisotropic and oblique
    grids.

    Parameters
    ----------
    affine : numpy.ndarray, shape (4, 4)
        Voxel-to-RAS affine to reposition.
    shape : tuple of int
        Spatial shape, used to locate the grid centre.
    cras : numpy.ndarray, shape (3,)
        Requested world coordinate of the grid centre.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Copy of ``affine`` whose grid centre lands on ``cras``.
    """
    out = np.array(affine, dtype=np.float64, copy=True)
    center_vox = np.asarray(shape, dtype=np.float64) / 2.0
    out[:3, 3] = np.asarray(cras, dtype=np.float64) - out[:3, :3] @ center_vox
    return out


def build_grid_affine(
        *,
        cosines: np.ndarray,
        vox_size: np.ndarray,
        shape: tuple[int, int, int],
        cras: np.ndarray,
) -> np.ndarray:
    """Assemble a voxel-to-RAS affine from its four independent components.

    Parameters
    ----------
    cosines : numpy.ndarray, shape (3, 3)
        Direction cosines with unit-length columns.
    vox_size : numpy.ndarray, shape (3,)
        Voxel size per axis, in mm.
    shape : tuple of int
        Image dimensions in voxels, per axis.
    cras : numpy.ndarray, shape (3,)
        World coordinate of the grid centre.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Voxel-to-RAS affine describing the requested grid.
    """
    affine = np.eye(4, dtype=np.float64)
    # Scale each direction-cosine column by its own voxel size.
    affine[:3, :3] = np.asarray(cosines, dtype=np.float64) @ np.diag(np.asarray(vox_size, dtype=np.float64))
    return place_grid_at_cras(affine, shape, cras)


def vox2tkras_from_volume_info(info: Mapping[str, Any]) -> np.ndarray:
    """Construct the FreeSurfer voxel-to-tkRAS matrix from volume metadata.

    Parameters
    ----------
    info : mapping
        FreeSurfer-style volume-info mapping containing ``volume`` and
        ``voxelsize`` entries.

    Returns
    -------
    np.ndarray, shape (4, 4)
        Voxel-to-tkRAS matrix.
    """
    dims = np.asarray(info["volume"], dtype=float)
    delta = np.asarray(info["voxelsize"], dtype=float)
    return np.array(
        [
            [-delta[0], 0.0, 0.0, delta[0] * dims[0] / 2.0],
            [0.0, 0.0, delta[2], -delta[2] * dims[2] / 2.0],
            [0.0, -delta[1], 0.0, delta[1] * dims[1] / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def get_vox2tkras(ref_volume: _AnyVolRef) -> np.ndarray:
    """Get the voxel-to-tkRAS matrix for an image or header.

    Parameters
    ----------
    ref_volume : nibabel image or nibabel header
        Reference volume or its header.

    Returns
    -------
    np.ndarray, shape (4, 4)
        Voxel-to-tkRAS matrix.
    """
    header = ref_volume.header if hasattr(ref_volume, "header") else ref_volume
    if hasattr(header, "get_vox2ras_tkr"):
        return np.asarray(header.get_vox2ras_tkr(), dtype=float)
    return vox2tkras_from_volume_info(
        {
            "volume": header.get_data_shape()[:3],
            "voxelsize": header.get_zooms()[:3],
        }
    )


def get_tkras2ras(ref_volume: _AnyVolRef) -> np.ndarray:
    """Get the tkRAS-to-scanner-RAS matrix for an image or header.

    Parameters
    ----------
    ref_volume : nibabel image or nibabel header
        Reference volume or its header.

    Returns
    -------
    np.ndarray, shape (4, 4)
        tkRAS-to-scanner-RAS matrix.
    """
    affine = ref_volume.affine if hasattr(ref_volume, "affine") else ref_volume.get_best_affine()
    return np.asarray(affine, dtype=float) @ np.linalg.inv(get_vox2tkras(ref_volume))


def get_ras2tkras(ref_volume: _AnyVolRef) -> np.ndarray:
    """Get the scanner-RAS-to-tkRAS matrix for an image or header.

    Parameters
    ----------
    ref_volume : nibabel image or nibabel header
        Reference volume or its header.

    Returns
    -------
    np.ndarray, shape (4, 4)
        Scanner-RAS-to-tkRAS matrix.
    """
    return np.linalg.inv(get_tkras2ras(ref_volume))
