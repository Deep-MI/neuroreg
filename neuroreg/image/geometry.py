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
