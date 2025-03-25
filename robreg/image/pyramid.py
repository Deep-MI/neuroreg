from typing import Optional

import torch
from torch import Tensor

from .smooth import smooth_image


def get_pyramid_limits(
        shape1: torch.Size,
        shape2: Optional[torch.Size] = None,
        minsize: int = 32,
        maxsize: Optional[int] = None
) -> tuple[Tensor, Tensor]:
    """
    Compute the minimum and maximum levels for a pyramid representation of shapes.

    Parameters
    ----------
    shape1 : torch.Size
        The primary shape of an input, typically a tensor size.
    shape2 : torch.Size, optional
        A secondary shape to compare with `shape1` for determining the limits.
        If `shape2` is not provided, `shape1` will be used for calculations.
    minsize : int, optional
        The minimum allowed size for the smallest dimension in the pyramid. Default is 32.
    maxsize : int, optional
        The maximum allowed size for the largest dimension in the pyramid. If not provided,
        this is not used in the calculations. Default is None.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple containing:
          - `minsteps` (Tensor): The minimum number of steps the pyramid can take before
            exceeding the `maxsize` (if given).
          - `maxsteps` (Tensor): The maximum number of steps the pyramid can take before
            hitting the `minsize`.

    Example
    -------
    >>> shape = torch.tensor((256, 256, 256))
    >>> limits = get_pyramid_limits(shape, shape2=shape, minsize=16, maxsize=200)
    >>> print(limits)
    (tensor(1), tensor(4))
    """
    if shape2 is None:
        smallest = torch.as_tensor(shape1)
    else:
        smallest = torch.minimum(torch.as_tensor(shape1), torch.as_tensor(shape2))
    smin = torch.min(smallest)
    maxsteps = torch.floor(torch.log2(smin / minsize)).int()
    if maxsize is None:
        return (torch.tensor(0.0), maxsteps)
    smax = torch.max(smallest)
    minsteps = torch.ceil(torch.log2(smax / maxsize)).int()
    return (minsteps, maxsteps)

def build_gaussian_pyramid(
        image: Tensor,
        affine: Tensor,
        limits: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Build a Gaussian pyramid for a 3D image, including its downsampled versions.

    Parameters
    ----------
    image : Tensor
        A 3D or higher-dimensional tensor representing the input image to build the pyramid from.
    affine : Tensor
        The affine (vox2ras) transformation matrix (4x4) corresponding to the input image.
        It is updated for each downsampled image in the pyramid.
    limits : tuple[Tensor, Tensor], optional
        A tuple containing the minimum and maximum level indices for the pyramid.
        If not provided, they are computed using the `get_pyramid_limits` function based on the image size.

    Returns
    -------
    tuple[list[Tensor], list[Tensor]]
        A tuple containing:
          - `imgs` (list[Tensor]): A list of downsampled and smoothed image tensors, starting
            from the input image (if applicable).
          - `affines` (list[Tensor]): A list of affine matrices corresponding to each
            downsampled image.

    Example
    -------
    >>> image = torch.rand(256, 256, 256)  # Example 3D image
    >>> affine = torch.eye(4)  # Initial affine matrix
    >>> gaussian_pyramid, affine_pyramid = build_gaussian_pyramid(image, affine)
    >>> print(len(gaussian_pyramid))
    4  # Number of pyramid levels (example)
    """
    if limits is None:
        limits = get_pyramid_limits(image.shape)
    last_affine = torch.tensor(affine)
    downM = 2 * torch.eye(4, dtype=last_affine.dtype)
    downM[3, 3] = 1
    downM[0:3, 3:4] = 0.5 * torch.ones((3, 1), dtype=last_affine.dtype)
    imgs = []
    affines = []
    # Check if original image (smoothed) needs to be stored
    if limits[0] == 0:
        smoothed = smooth_image(image)
        imgs.append(smoothed.squeeze())
        affines.append(last_affine)
    smoothed = image
    for i in range(limits[1]):
        smoothed = smooth_image(smoothed)
        smoothed = torch.nn.functional.avg_pool3d(smoothed, 2)
        last_affine = last_affine @ downM
        if i >= limits[0]:
            imgs.append(smoothed.squeeze())
            affines.append(last_affine)
    return imgs, affines
