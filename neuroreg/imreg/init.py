"""Initialization helpers shared across image-registration entry points."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ..image.centroid import compute_centroid

InitType = Literal["header", "centroid", "image_center"]
ResolvedInitType = InitType


def normalize_init_type(init_type: InitType) -> ResolvedInitType:
    """Validate and return the requested initialization mode.

    Parameters
    ----------
    init_type : {"header", "centroid", "image_center"}
        Requested initialization strategy.

    Returns
    -------
    ResolvedInitType
        The validated initialization mode.
    """
    return init_type


def resolve_init_type(
        init_type: InitType | None = None,
        *,
        default_init_type: ResolvedInitType,
) -> ResolvedInitType:
    """Resolve the effective initialization mode.

    Parameters
    ----------
    init_type : {"header", "centroid", "image_center"} or None, optional
        Explicit user-requested initialization mode.
    default_init_type : {"header", "centroid", "image_center"}
        Backend-specific default to use when ``init_type`` is omitted.

    Returns
    -------
    ResolvedInitType
        The explicit mode when provided, otherwise ``default_init_type``.
    """
    if init_type is None:
        return default_init_type
    return normalize_init_type(init_type)


def _normalize_affine(affine: Tensor | None, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return a 4x4 voxel-to-RAS affine on the requested dtype/device."""
    if affine is None:
        return torch.eye(4, dtype=dtype, device=device)
    return affine.to(device=device, dtype=dtype)


def _point_to_ras(point: Tensor, affine: Tensor) -> Tensor:
    """Map a voxel-space point into RAS coordinates."""
    point_h = torch.ones(4, dtype=affine.dtype, device=affine.device)
    point_h[:3] = point.to(device=affine.device, dtype=affine.dtype)
    return (affine @ point_h)[:3]


def _point_init_transform(
        spoint: Tensor,
        tpoint: Tensor,
        saffine: Tensor | None,
        taffine: Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
) -> Tensor:
    """Return a translation-only vox2vox transform that aligns two voxel-space points in RAS.

    The input points are interpreted in voxel coordinates of their respective
    images, converted into RAS through the supplied affines, aligned with a
    pure RAS translation, and then converted back into a voxel-to-voxel matrix.
    """
    src_affine = _normalize_affine(saffine, dtype=dtype, device=device)
    trg_affine = _normalize_affine(taffine, dtype=dtype, device=device)
    src_ras = _point_to_ras(spoint, src_affine)
    trg_ras = _point_to_ras(tpoint, trg_affine)
    ras_translation = torch.eye(4, dtype=dtype, device=device)
    ras_translation[:3, 3] = trg_ras - src_ras
    return torch.inverse(trg_affine) @ ras_translation @ src_affine


def _image_center(shape: torch.Size | tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return the geometric center voxel of a 3-D image."""
    dims = torch.as_tensor(shape[-3:], dtype=dtype, device=device)
    return (dims - 1.0) / 2.0


def get_ixform_centroids(
        simg: Tensor,
        timg: Tensor,
        saffine: Tensor | None = None,
        taffine: Tensor | None = None,
) -> Tensor:
    """
    Compute a translation-only voxel-to-voxel transform from intensity centroids.

    The centroid is computed in voxel coordinates, converted to RAS through each
    image affine, and then aligned with a pure RAS translation. This respects
    differing headers, voxel sizes, and orientation matrices.
    """
    return _point_init_transform(
        compute_centroid(simg),
        compute_centroid(timg),
        saffine,
        taffine,
        dtype=simg.dtype,
        device=simg.device,
    )



def get_ixform_image_centers(
        simg: Tensor,
        timg: Tensor,
        saffine: Tensor | None = None,
        taffine: Tensor | None = None,
) -> Tensor:
    """Compute a translation-only voxel-to-voxel transform by aligning image centers in RAS.

    Unlike centroid initialization, this mode ignores image intensities and uses
    only the geometric center of each voxel grid. This matches the intent of the
    FreeSurfer-style image-center start used by the Powell coreg path.
    """
    return _point_init_transform(
        _image_center(simg.shape, dtype=simg.dtype, device=simg.device),
        _image_center(timg.shape, dtype=timg.dtype, device=timg.device),
        saffine,
        taffine,
        dtype=simg.dtype,
        device=simg.device,
    )



def get_vox2vox_from_header(saffine: Tensor, taffine: Tensor) -> Tensor:
    """Compute the header-derived voxel-to-voxel initialization transform.

    Parameters
    ----------
    saffine : Tensor
        Source voxel-to-RAS affine.
    taffine : Tensor
        Target voxel-to-RAS affine.

    Returns
    -------
    Tensor
        The voxel-to-voxel transform implied by the two headers alone.
    """
    return torch.inverse(taffine) @ saffine



def get_init_vox2vox(
        simg: Tensor,
        timg: Tensor,
        *,
        saffine: Tensor | None = None,
        taffine: Tensor | None = None,
        init_type: InitType = "centroid",
) -> Tensor:
    """Compute a shared initial voxel-to-voxel transform for image registration.

    Parameters
    ----------
    simg, timg : Tensor
        Source and target image tensors.
    saffine, taffine : Tensor or None, optional
        Source and target voxel-to-RAS affines. When omitted, identity affines
        are assumed.
    init_type : {"header", "centroid", "image_center"}, default="centroid"
        Initialization strategy to apply.

    Returns
    -------
    Tensor
        A 4x4 voxel-to-voxel transform suitable as the starting point for a
        registration optimizer.
    """
    normalized_init_type = normalize_init_type(init_type)
    src_affine = _normalize_affine(saffine, dtype=simg.dtype, device=simg.device)
    trg_affine = _normalize_affine(taffine, dtype=simg.dtype, device=simg.device)
    if normalized_init_type == "header":
        return get_vox2vox_from_header(src_affine, trg_affine)
    if normalized_init_type == "centroid":
        return get_ixform_centroids(simg, timg, src_affine, trg_affine)
    if normalized_init_type == "image_center":
        return get_ixform_image_centers(simg, timg, src_affine, trg_affine)
    raise ValueError(
        f"Unknown init_type '{init_type}'. Choose from: 'header', 'centroid', 'image_center'."
    )
