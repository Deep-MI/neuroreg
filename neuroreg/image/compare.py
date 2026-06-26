"""Volume comparison helpers (analogue of FreeSurfer ``mri_diff``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ImageDiff:
    """Structured comparison of two images.

    Pixel fields (``n_voxels_differ``, ``max_abs_diff``, ``max_diff_loc``) are
    only populated when the two images share the same full shape; otherwise they
    are ``-1``, ``nan``, and ``None`` respectively.
    """

    shape1: tuple[int, ...]
    shape2: tuple[int, ...]
    voxsize1: tuple[float, float, float]
    voxsize2: tuple[float, float, float]
    affine1: np.ndarray
    affine2: np.ndarray
    dtype1: np.dtype
    dtype2: np.dtype
    res_max_diff: float
    geo_max_diff: float
    n_voxels_differ: int
    max_abs_diff: float
    max_diff_loc: tuple[int, ...] | None

    @property
    def shape_match(self) -> bool:
        """Whether the two images share the same full shape (including frames)."""
        return self.shape1 == self.shape2

    @property
    def dtype_match(self) -> bool:
        """Whether the two images share the same stored data dtype."""
        return self.dtype1 == self.dtype2


def _voxsize(img: Any) -> tuple[float, float, float]:
    """Return the first three voxel sizes from an image header."""
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if len(zooms) == 3:
        return zooms
    affine = np.asarray(img.affine, dtype=np.float64)
    return tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))


def compare_images(
    a: Any,
    b: Any,
    *,
    pix_thresh: float = 0.0,
    compare_pixels: bool = True,
) -> ImageDiff:
    """Compare two images and return a structured difference report.

    Parameters
    ----------
    a, b : Any
        Loaded nibabel-like images.
    pix_thresh : float, default=0.0
        Voxel differences with absolute value at or below this are not counted
        in ``n_voxels_differ``. Only used when the shapes match and
        ``compare_pixels`` is ``True``.
    compare_pixels : bool, default=True
        When ``False`` skip pixel-data comparison entirely (no ``.dataobj``
        access). The pixel fields of the returned ``ImageDiff`` will be
        ``-1``, ``nan``, and ``None`` respectively. Useful for fast header-only
        checks before deciding whether to pay the cost of loading voxel data.

    Returns
    -------
    ImageDiff
        Structured comparison result.
    """
    affine1 = np.asarray(a.affine, dtype=np.float64)
    affine2 = np.asarray(b.affine, dtype=np.float64)
    shape1 = tuple(int(v) for v in a.shape)
    shape2 = tuple(int(v) for v in b.shape)
    vs1 = _voxsize(a)
    vs2 = _voxsize(b)

    res_max_diff = float(max(abs(vs1[i] - vs2[i]) for i in range(3)))
    geo_max_diff = float(np.max(np.abs(affine1 - affine2)))

    n_voxels_differ = -1
    max_abs_diff = float("nan")
    max_diff_loc: tuple[int, ...] | None = None
    if compare_pixels and shape1 == shape2:
        d1 = np.asarray(a.dataobj, dtype=np.float64)
        d2 = np.asarray(b.dataobj, dtype=np.float64)
        absdiff = np.abs(d1 - d2)
        # Build absdiff_safe with explicit NaN semantics:
        #   NaN vs NaN  → 0.0 (both undefined — treat as "same", matching FS)
        #   NaN vs finite or Inf vs finite → inf (always counts as differing)
        #   finite diff → absdiff value (normal)
        both_nan = np.isnan(d1) & np.isnan(d2)
        absdiff_safe = np.where(both_nan, 0.0, np.where(np.isfinite(absdiff), absdiff, np.inf))
        n_voxels_differ = int(np.count_nonzero(absdiff_safe > pix_thresh))
        max_abs_diff = float(absdiff_safe.max()) if absdiff_safe.size else 0.0
        if absdiff_safe.size:
            max_diff_loc = tuple(int(v) for v in np.unravel_index(int(np.argmax(absdiff_safe)), absdiff_safe.shape))

    return ImageDiff(
        shape1=shape1,
        shape2=shape2,
        voxsize1=vs1,
        voxsize2=vs2,
        affine1=affine1,
        affine2=affine2,
        dtype1=np.dtype(a.get_data_dtype()),
        dtype2=np.dtype(b.get_data_dtype()),
        res_max_diff=res_max_diff,
        geo_max_diff=geo_max_diff,
        n_voxels_differ=n_voxels_differ,
        max_abs_diff=max_abs_diff,
        max_diff_loc=max_diff_loc,
    )
