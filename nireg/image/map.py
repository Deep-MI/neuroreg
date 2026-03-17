"""Utilities for mapping (resampling) 3-D images via affine transforms."""

import torch
import torch.nn as nn

import nireg.transforms.matrices as trans


def map(
        image: torch.Tensor,
        transform: torch.Tensor,
        is_torch_mat: bool = True,
        target_shape: tuple[int, int, int] | None = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros'
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
        matrix and converted via :func:`~nireg.transforms.matrices.convert_v2v_to_torch`.
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
    if mode not in ('bilinear', 'nearest'):
        raise ValueError(f"mode must be 'bilinear' or 'nearest', got '{mode}'.")
    if padding_mode not in ('zeros', 'border', 'reflection'):
        raise ValueError(
            f"padding_mode must be 'zeros', 'border', or 'reflection', got '{padding_mode}'."
        )
    if not is_torch_mat:
        torch_transform = trans.convert_v2v_to_torch(transform, image.shape, target_shape)
    else:
        torch_transform = transform[:3, :]
    out_shape = target_shape if target_shape is not None else image.shape
    grid_size = (1, 1) + tuple(out_shape)
    grid = nn.functional.affine_grid(
        torch_transform.unsqueeze(0).float(),
        grid_size,
        align_corners=False
    )
    return nn.functional.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False
    ).squeeze()


def map_r2r(
        image: torch.Tensor,
        r2r: torch.Tensor,
        source_affine: torch.Tensor,
        target_affine: torch.Tensor,
        target_shape: tuple[int, int, int] | None = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
) -> torch.Tensor:
    """Map an image using a RAS-to-RAS transform without a v2v intermediate.

    Wrapper around :func:`map` that calls
    :func:`~nireg.transforms.matrices.convert_r2r_to_torch` to build the
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
    torch_mat = trans.convert_r2r_to_torch(
        r2r, image.shape, source_affine, target_shape, target_affine
    )
    return map(image, torch_mat, is_torch_mat=True,
               target_shape=target_shape, mode=mode, padding_mode=padding_mode)

