import torch
import torch.nn as nn

import robreg.transforms.matrices as trans


def map(
        image: torch.Tensor,
        transform: torch.Tensor,
        is_torch_mat: bool = True,
        target_shape: tuple[int, int, int] | None = None,
        mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Map an input image to another space using the inverse transformation matrix.

    Parameters
    ----------
    image : torch.Tensor
        The input image tensor to be transformed. Shape is typically (depth, height, width).
    transform : torch.Tensor
        The 4x4 transformation matrix used for mapping the image.
    is_torch_mat: bool, optional
        Specify whether the transformation matrix is in torch format or not. Default is True.
    mode : str, optional
        The interpolation mode to use for sampling. Options include 'bilinear' (default) and 'nearest'.

    Returns
    -------
    torch.Tensor
        The transformed image tensor with the same dimensionality as the input tensor.

    Notes
    -----
    - The `mode` parameter controls the interpolation strategy:
        - 'bilinear' for smooth sampling.
        - 'nearest' for nearest-neighbor sampling.
    - The input image tensor is assumed to be 3-dimensional (depth, height, width).
    - Padding outside the valid image region is set to zero.

    """
    if not is_torch_mat:
        torch_transform = trans.convert_v2v_to_torch(transform, image.shape, target_shape)
    else:
        torch_transform = transform[:3, :]  # Convert to 3x4 for affine_grid
    grid = nn.functional.affine_grid(
        torch_transform.unsqueeze(0).float(),
        image.unsqueeze(0).unsqueeze(0).size(),
        align_corners=False
    )
    return nn.functional.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=False
    ).squeeze()
