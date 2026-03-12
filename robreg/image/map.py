"""Utilities for mapping (resampling) 3-D images via affine transforms."""

import torch
import torch.nn as nn

import robreg.transforms.matrices as trans


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
        matrix and converted via :func:`~robreg.transforms.matrices.convert_v2v_to_torch`.
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
        Resampled image with the same shape as *image*.

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
    grid = nn.functional.affine_grid(
        torch_transform.unsqueeze(0).float(),
        image.unsqueeze(0).unsqueeze(0).size(),
        align_corners=False
    )
    return nn.functional.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False
    ).squeeze()
