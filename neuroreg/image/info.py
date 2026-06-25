"""Image header/geometry inspection helpers (analogue of FreeSurfer ``mri_info``)."""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np

from .geometry import get_vox2tkras


def describe_image(img: Any, fname: str | None = None) -> dict[str, Any]:
    """Collect header and geometry information for an image.

    Parameters
    ----------
    img : Any
        Loaded nibabel-like image.
    fname : str or None, optional
        Source filename, recorded under ``"fname"`` for display.

    Returns
    -------
    dict
        Structured fields: ``fname``, ``file_type``, ``shape``, ``nframes``,
        ``voxel_sizes``, ``dtype``, ``fov``, ``voxvol``, ``cras``,
        ``orientation``, ``vox2ras``, ``ras2vox``, ``vox2ras_tkr``, and
        ``determinant``.
    """
    affine = np.asarray(img.affine, dtype=np.float64)
    shape = tuple(int(v) for v in img.shape[:3])
    nframes = int(img.shape[3]) if len(img.shape) > 3 else 1

    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if len(zooms) < 3:
        # Fall back to affine column norms when the header omits zooms.
        zooms = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    voxvol = float(np.prod(zooms))
    fov = float(max(shape[i] * zooms[i] for i in range(3)))

    center_vox = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0, 1.0], dtype=np.float64)
    cras = (affine @ center_vox)[:3]

    return {
        "fname": fname,
        "file_type": type(img).__name__,
        "shape": shape,
        "nframes": nframes,
        "voxel_sizes": zooms,
        "dtype": np.dtype(img.get_data_dtype()),
        "fov": fov,
        "voxvol": voxvol,
        "cras": cras,
        "orientation": "".join(nib.aff2axcodes(affine)),
        "vox2ras": affine,
        "ras2vox": np.linalg.inv(affine),
        "vox2ras_tkr": np.asarray(get_vox2tkras(img), dtype=np.float64),
        "determinant": float(np.linalg.det(affine[:3, :3])),
    }


def image_value_stats(img: Any) -> dict[str, float]:
    """Compute min/max/mean over all finite voxel values.

    Parameters
    ----------
    img : Any
        Loaded nibabel-like image. Voxel data are read lazily here, so callers
        that only need header fields never pay this cost.

    Returns
    -------
    dict
        ``min``, ``max``, and ``mean`` of the finite voxel values (all ``nan``
        when the image has no finite values).
    """
    data = np.asarray(img.dataobj, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}
