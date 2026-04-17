"""Utilities for mapping (resampling) 3-D images via affine transforms."""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

import neuroreg.transforms.matrices as trans


def map(
        image: torch.Tensor,
        transform: torch.Tensor,
        is_torch_mat: bool = True,
        target_shape: tuple[int, int, int] | None = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
) -> torch.Tensor:
    """Map an input image to another space using the inverse transformation matrix.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor, shape ``(D, H, W)``.
    transform : torch.Tensor
        4 × 4 transformation matrix.
    is_torch_mat : bool, optional
        If ``True`` (default), *transform* is already in PyTorch grid-sample
        format (3 × 4 or 4 × 4).  If ``False``, it is treated as a vox-to-vox
        matrix and converted via :func:`~neuroreg.transforms.matrices.convert_v2v_to_torch`.
    target_shape : tuple[int, int, int], optional
        Shape of the output grid ``(D, H, W)``.  Only used when
        ``is_torch_mat=False``.  Defaults to the shape of *image*.
    mode : {'bilinear', 'nearest'}, optional
        Interpolation mode passed to :func:`torch.nn.functional.grid_sample`.
        Default is ``'bilinear'``.
    padding_mode : {'zeros', 'border', 'reflection'}, optional
        Padding strategy for out-of-bounds coordinates, passed directly to
        :func:`torch.nn.functional.grid_sample`.  Default is ``'zeros'``.

    Returns
    -------
    torch.Tensor
        Resampled image with shape *target_shape* (or the shape of *image* if
        *target_shape* is ``None``).

    Raises
    ------
    ValueError
        If *mode* is not ``'bilinear'`` or ``'nearest'``, or if *padding_mode*
        is not one of ``'zeros'``, ``'border'``, or ``'reflection'``.
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"mode must be 'bilinear' or 'nearest', got '{mode}'.")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(f"padding_mode must be 'zeros', 'border', or 'reflection', got '{padding_mode}'.")
    if not is_torch_mat:
        torch_transform = trans.convert_v2v_to_torch(transform, image.shape, target_shape)
    else:
        torch_transform = transform[:3, :]
    torch_transform = torch_transform.to(device=image.device)
    out_shape = target_shape if target_shape is not None else image.shape
    grid_size = (1, 1) + tuple(out_shape)
    grid = nn.functional.affine_grid(torch_transform.unsqueeze(0).float(), grid_size, align_corners=False)
    return nn.functional.grid_sample(
        image.unsqueeze(0).unsqueeze(0), grid, mode=mode, padding_mode=padding_mode, align_corners=False
    ).squeeze()


def map_r2r(
        image: torch.Tensor,
        r2r: torch.Tensor,
        source_affine: torch.Tensor,
        target_affine: torch.Tensor,
        target_shape: tuple[int, int, int] | None = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
) -> torch.Tensor:
    """Map an image using a RAS-to-RAS transform without a v2v intermediate.

    Wrapper around :func:`map` that calls
    :func:`~neuroreg.transforms.matrices.convert_r2r_to_torch` to build the
    PyTorch grid transform directly from the physical-space (RAS) chain::

        trg_norm → trg_vox → trg_RAS → src_RAS → src_vox → src_norm

    This avoids creating the intermediate vox-to-vox matrix
    ``inv(target_affine) @ r2r @ source_affine``, whose off-diagonal elements
    reflect genuine anisotropy but look like shear and can mislead callers.

    Parameters
    ----------
    image : torch.Tensor
        Source image tensor, shape ``(D, H, W)``.
    r2r : torch.Tensor
        4 × 4 RAS-to-RAS transform (source_RAS → target_RAS).
    source_affine : torch.Tensor
        4 × 4 voxel-to-RAS affine of the source image.
    target_affine : torch.Tensor
        4 × 4 voxel-to-RAS affine of the target image.
    target_shape : tuple[int, int, int], optional
        Output shape ``(D, H, W)``.  Defaults to the shape of *image*.
    mode : {'bilinear', 'nearest'}, optional
        Interpolation mode.  Default is ``'bilinear'``.
    padding_mode : {'zeros', 'border', 'reflection'}, optional
        Out-of-bounds padding.  Default is ``'zeros'``.

    Returns
    -------
    torch.Tensor
        Resampled image with shape *target_shape* (or source shape).
    """
    if target_shape is None:
        target_shape = image.shape
    torch_mat = trans.convert_r2r_to_torch(r2r, image.shape, source_affine, target_shape, target_affine)
    return map(image, torch_mat, is_torch_mat=True, target_shape=target_shape, mode=mode, padding_mode=padding_mode)


def resample_isotropic(
        img: nib.Nifti1Image,
        iso: float,
        out_shape: tuple[int, int, int] | None = None,
        mode: str = "bilinear",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a NIfTI image to an isotropic grid.

    Creates an isotropic resampled version of the input image where all voxels
    have the same physical size (*iso* mm) in all three dimensions. The
    isotropic affine preserves the original image origin while rescaling the
    column vectors to have uniform length.

    This function is commonly used as a preprocessing step before multi-scale
    registration to ensure both images share a consistent voxel grid.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image to resample.
    iso : float
        Target isotropic voxel size in millimeters.
    out_shape : tuple[int, int, int], optional
        Output image shape ``(D, H, W)``.  If ``None``, the output shape is
        computed automatically to cover the entire field of view of the
        original image at the specified isotropic resolution.
    mode : {'bilinear', 'nearest'}, optional
        Interpolation mode.  Default is ``'bilinear'``.

    Returns
    -------
    data : torch.Tensor
        Resampled image data, shape *out_shape*.
    iso_affine : torch.Tensor, dtype float32
        Isotropic voxel-to-RAS affine (4 × 4).
    Rvox : torch.Tensor, dtype float32
        Voxel-to-voxel transform from the isotropic grid back to the original
        grid (4 × 4), computed as ``inv(orig_affine) @ iso_affine``.

    Notes
    -----
    The isotropic affine is constructed by normalizing each column of the
    original 3 × 3 rotation/scale block to unit length, then scaling by *iso*.
    The origin (fourth column) is preserved.

    Examples
    --------
    >>> img = nib.load("example.mgz")
    >>> data_iso, aff_iso, Rvox = resample_isotropic(img, iso=1.0)
    >>> print(data_iso.shape, aff_iso.shape)
    """
    orig_affine = torch.from_numpy(img.affine).double()

    # Build isotropic affine: same origin, isotropic voxels
    iso_affine = orig_affine.clone()
    for i in range(3):
        col_norm = orig_affine[:3, i].norm()
        if col_norm > 0:
            iso_affine[:3, i] = orig_affine[:3, i] / col_norm * iso

    # Compute output shape if not provided
    if out_shape is None:
        zooms = np.linalg.norm(img.affine[:3, :3], axis=0)  # column norms = voxel sizes
        shape = np.array(img.shape[:3])
        out_shape = tuple(max(1, int(np.ceil(s * z / iso))) for s, z in zip(shape, zooms, strict=False))

    # Resample using identity RAS-to-RAS transform
    identity_r2r = torch.eye(4, dtype=torch.float64)
    orig_data = torch.from_numpy(img.get_fdata()).float()

    resampled = map_r2r(
        orig_data,
        identity_r2r.float(),
        source_affine=orig_affine.float(),
        target_affine=iso_affine.float(),
        target_shape=out_shape,
        mode=mode,
    )

    # Rvox: isotropic vox → original vox
    Rvox = torch.inverse(orig_affine) @ iso_affine

    return resampled, iso_affine.float(), Rvox.float()


def resample_isotropic_tensor(
        img: torch.Tensor,
        affine: np.ndarray,
        iso: float,
        out_shape: tuple[int, int, int] | None = None,
        mode: str = "bilinear",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a torch tensor to an isotropic grid.

    Like resample_isotropic but works with torch tensors instead of nibabel images.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor, shape (D, H, W).
    affine : np.ndarray
        4×4 voxel-to-RAS affine matrix.
    iso : float
        Target isotropic voxel size in millimeters.
    out_shape : tuple[int, int, int], optional
        Output image shape (D, H, W). If None, computed automatically.
    mode : {'bilinear', 'nearest'}, optional
        Interpolation mode. Default is 'bilinear'.

    Returns
    -------
    data : torch.Tensor
        Resampled image data.
    iso_affine : torch.Tensor
        Isotropic voxel-to-RAS affine (4×4).
    Rvox : torch.Tensor
        Voxel-to-voxel transform from isotropic grid to original grid (4×4).
    """
    orig_affine = torch.from_numpy(affine).double()

    # Build isotropic affine: same origin, isotropic voxels
    iso_affine = orig_affine.clone()
    for i in range(3):
        col_norm = orig_affine[:3, i].norm()
        if col_norm > 0:
            iso_affine[:3, i] = orig_affine[:3, i] / col_norm * iso

    # Compute output shape if not provided
    if out_shape is None:
        zooms = np.linalg.norm(affine[:3, :3], axis=0)
        shape = np.array(img.shape[:3])
        out_shape = tuple(max(1, int(np.ceil(s * z / iso))) for s, z in zip(shape, zooms, strict=False))

    # Resample using identity RAS-to-RAS transform
    identity_r2r = torch.eye(4, dtype=torch.float64)

    resampled = map_r2r(
        img,
        identity_r2r.float(),
        source_affine=orig_affine.float(),
        target_affine=iso_affine.float(),
        target_shape=out_shape,
        mode=mode,
    )

    # Rvox: isotropic vox → original vox
    Rvox = torch.inverse(orig_affine) @ iso_affine

    return resampled, iso_affine.float(), Rvox.float()
