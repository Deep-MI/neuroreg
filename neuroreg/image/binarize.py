"""Volume binarization helper (analogue of FreeSurfer ``mri_binarize``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .map import create_image_like


def binarize_image(
    img: Any,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    match: Sequence[float] | None = None,
    binval: int = 1,
    binvalnot: int = 0,
    invert: bool = False,
    use_abs: bool = False,
    frame: int | None = None,
    out_dtype: np.dtype | type = np.int32,
) -> Any:
    """Binarize an image by intensity range or by matching label values.

    A voxel is *selected* when it matches one of ``match`` (exact equality), or
    when it lies in the inclusive range ``[vmin, vmax]`` (either bound may be
    omitted). Selected voxels are set to ``binval`` and the rest to
    ``binvalnot``; ``invert`` swaps that assignment.

    Parameters
    ----------
    img : Any
        Loaded nibabel-like image.
    vmin, vmax : float or None, optional
        Inclusive lower/upper intensity bounds. Ignored in match mode.
    match : sequence of float or None, optional
        Values to match exactly (e.g. segmentation labels). When given, range
        bounds are ignored.
    binval : int, default=1
        Output value for selected voxels.
    binvalnot : int, default=0
        Output value for unselected voxels.
    invert : bool, default=False
        Swap ``binval`` and ``binvalnot`` (matches FreeSurfer's ``--inv``, whose
        default 1/0 values become 0/1).
    use_abs : bool, default=False
        Take the absolute value of the input before thresholding.
    frame : int or None, optional
        For 4D input, operate on this frame only (producing a 3D result). When
        ``None`` the full array is binarized element-wise.
    out_dtype : np.dtype or type, default=np.int32
        Output data type. FreeSurfer's default is ``int32`` (``--uchar`` selects
        ``uint8``).

    Returns
    -------
    Any
        Binarized image in the input geometry.

    Raises
    ------
    ValueError
        If no selection criterion (``vmin``, ``vmax``, or ``match``) is given.
    """
    if vmin is None and vmax is None and match is None:
        raise ValueError("binarize_image requires at least one of vmin, vmax, or match.")

    data = np.asarray(img.dataobj, dtype=np.float64)
    if frame is not None and data.ndim > 3:
        data = data[..., frame]
    if use_abs:
        data = np.abs(data)

    if match is not None:
        selected = np.isin(data, np.asarray(match, dtype=np.float64))
    else:
        selected = np.ones(data.shape, dtype=bool)
        if vmin is not None:
            selected &= data >= vmin
        if vmax is not None:
            selected &= data <= vmax

    true_val, false_val = (binvalnot, binval) if invert else (binval, binvalnot)
    out = np.where(selected, true_val, false_val).astype(out_dtype)
    return create_image_like(img, out, np.asarray(img.affine, dtype=np.float64))
