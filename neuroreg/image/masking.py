"""Shared utilities for loading, validating, and resampling registration masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .io import load_image

MaskLike = str | Path | Any | Tensor


def coerce_binary_mask_data(data: Any, *, name: str = "mask") -> np.ndarray:
    """Return a 3-D float mask array with nonzero values mapped to 1.

    Parameters
    ----------
    data : Any
        Mask-like array data. Singleton extra dimensions are squeezed using the
        same logic as image loading.
    name : str, default="mask"
        Human-readable label used in validation errors.

    Returns
    -------
    np.ndarray
        Float32 binary mask array with shape ``(D, H, W)``.
    """
    from .map import coerce_image_data_3d  # Lazy import avoids a circular dependency.

    mask = np.asarray(coerce_image_data_3d(data, name=name), dtype=np.float32)
    return (mask > 0).astype(np.float32, copy=False)


def load_mask(mask: str | Path | Any | None) -> Any | None:
    """Load a path-based mask and otherwise return the input unchanged."""
    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        return cast(Any, load_image(mask))
    return mask


def as_mask_tensor_and_affine(
    mask: MaskLike | None,
    *,
    affine: Tensor | None = None,
    name: str = "mask",
) -> tuple[Tensor | None, Tensor | None]:
    """Convert a supported mask input into tensor data and an affine.

    Parameters
    ----------
    mask : MaskLike or None
        Mask specification. This may be a filesystem path, a nibabel-like image
        object exposing ``get_fdata()`` and ``affine``, a ``torch.Tensor``, or
        ``None``.
    affine : Tensor, optional
        Explicit affine used when ``mask`` is already a tensor. If omitted, an
        identity affine is assumed.
    name : str, default="mask"
        Human-readable label used in validation errors.

    Returns
    -------
    mask_tensor : Tensor or None
        Float32 binary mask tensor, or ``None`` when no mask was supplied.
    mask_affine : Tensor or None
        Corresponding voxel-to-RAS affine, or ``None`` when no mask was supplied.

    Raises
    ------
    TypeError
        If ``mask`` is not one of the supported input types.
    """
    if mask is None:
        return None, None

    if isinstance(mask, torch.Tensor):
        return (mask > 0).float(), (affine.float() if affine is not None else torch.eye(4, dtype=torch.float32))

    loaded = load_mask(mask)
    if hasattr(loaded, "get_fdata") and hasattr(loaded, "affine"):
        data = torch.from_numpy(coerce_binary_mask_data(loaded.get_fdata(), name=name)).float()
        return data, torch.from_numpy(np.asarray(loaded.affine, dtype=np.float32)).float()

    raise TypeError(f"Unsupported mask type: {type(mask)!r}")


def build_binary_mask_pyramid(mask: Tensor, shapes: list[tuple[int, int, int]]) -> list[Tensor]:
    """Resample a binary mask onto a list of pyramid shapes with nearest neighbor.

    Parameters
    ----------
    mask : Tensor
        Finest-resolution binary mask tensor, shape ``(D, H, W)``.
    shapes : list of tuple[int, int, int]
        Pyramid level shapes in the same order as the image pyramid.

    Returns
    -------
    list of Tensor
        Binary mask tensors aligned with ``shapes``.
    """
    base = (mask > 0).float().unsqueeze(0).unsqueeze(0)
    levels: list[Tensor] = []
    for shape in shapes:
        if tuple(int(v) for v in shape) == tuple(int(v) for v in mask.shape):
            levels.append(base[0, 0].clone())
        else:
            levels.append(F.interpolate(base, size=shape, mode="nearest")[0, 0])
    return levels
