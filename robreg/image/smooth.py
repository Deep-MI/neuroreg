import torch
import torch.nn as nn
from torch import Tensor


def get_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.08, dim: int = 3) -> Tensor:
    """
    Generate a Gaussian kernel for smoothing in 1D, 2D, or 3D.

    Parameters
    ----------
    kernel_size : int, optional
        The size of the kernel along each dimension (must be odd). Default is 5.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. Default is 1.08,
        to be similar to FreeSurfer's robust registration.
    dim : int, optional
        The dimensionality of the Gaussian kernel (1, 2, or 3). Default is 3.

    Returns
    -------
    Tensor
        A normalized Gaussian kernel tensor of shape:
          - (kernel_size,) for `dim=1`
          - (kernel_size, kernel_size) for `dim=2`
          - (kernel_size, kernel_size, kernel_size) for `dim=3`

    Raises
    ------
    ValueError
        If `dim` is not 1, 2, or 3.

    Example
    -------
    >>> kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0, dim=2)
    >>> print(kernel.shape)
    torch.Size([3, 3])
    """
    if dim not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension {dim}. Expected 1, 2, or 3.")
    x_cord = torch.arange(kernel_size, dtype=torch.float32)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2
    # 1D Gaussian
    gaussian_kernel_x = (1. / (2. * torch.pi * variance)) * torch.exp(-((x_cord - mean) ** 2) / (2 * variance))
    gaussian_kernel_x = gaussian_kernel_x / torch.sum(gaussian_kernel_x)
    # Extend kernel to requested dimensions
    gaussian_kernel = gaussian_kernel_x
    if dim > 1:
        gaussian_kernel_xy = torch.outer(gaussian_kernel_x, gaussian_kernel_x)
        gaussian_kernel = gaussian_kernel_xy
    if dim > 2:
        gaussian_kernel = gaussian_kernel[..., None] * gaussian_kernel_x
    return gaussian_kernel


def smooth_image(image: Tensor, kernel_size: int = 5, sigma: float = 1.08) -> Tensor:
    """
    Smooth a 3D image using a Gaussian kernel with separable convolution.

    Parameters
    ----------
    image : Tensor
        A 3D or higher-dimensional tensor representing the input image to be smoothed.
        It is expected to have at least 3 dimensions (depth, height, width).
    kernel_size : int, optional
        The size of the Gaussian kernel along each dimension (must be odd). Default is 5.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. Default is 1.08.

    Returns
    -------
    Tensor
        A smoothed image with the same shape as the input image.

    Raises
    ------
    Exception
        If the input `image` does not have at least 3 dimensions.

    Example
    -------
    >>> input_image = torch.rand(64, 64, 64)  # Example 3D image
    >>> smoothed_image = smooth_image(input_image, kernel_size=3, sigma=1.0)
    >>> print(smoothed_image.shape)
    torch.Size([64, 64, 64])
    """
    # Generate the Gaussian kernel
    g = get_gaussian_kernel(kernel_size, sigma, dim=1).to(image.device)
    # Ensure the image tensor has at least 3 valid dimensions
    if image.dim() < 3:
        raise Exception("ERROR: smooth_image: there need to be at least 3 image dimensions!")
    # Add batch and channel dimensions if necessary
    if image.dim() < 4:
        image = image.unsqueeze(0)  # Add batch dimension
    if image.dim() < 5:
        image = image.unsqueeze(0)  # Add channel dimension
    # Apply separable 3D convolution
    gx = g.view(1, 1, 1, 1, kernel_size)  # Kernel for the x-dimension
    smoothed = nn.functional.conv3d(image, gx, padding="same")
    gy = g.view(1, 1, 1, kernel_size, 1)  # Kernel for the y-dimension
    smoothed = nn.functional.conv3d(smoothed, gy, padding="same")
    gz = g.view(1, 1, kernel_size, 1, 1)  # Kernel for the z-dimension
    smoothed = nn.functional.conv3d(smoothed, gz, padding="same")
    return smoothed
