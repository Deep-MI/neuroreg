import torch
from torch import Tensor


def compute_centroid(image: Tensor) -> Tensor:
    """
    Compute the weighted centroid of a 3D image based on its intensities.

    Parameters
    ----------
    image : Tensor
        A 3D tensor representing the image with shape (D, H, W),
        where D is the depth, H is the height, and W is the width. The values in the tensor
        are interpreted as intensity values.

    Returns
    -------
    Tensor
        A 1D tensor containing the coordinates of the centroid in the format (z, y, x).

    Example
    -------
    >>> image = torch.rand(64, 64, 64)  # An example 3D image
    >>> centroid = compute_centroid(image)
    >>> print(centroid)
    tensor([32.0456, 31.7854, 33.2187])  # Example result
    """
    # Get the dimensions of the image
    d, h, w = image.shape[-3:]
    # Create coordinate grids for each axis (z, y, x) on the same device as the image.
    z = torch.arange(d, dtype=image.dtype, device=image.device).view(d, 1, 1).expand(d, h, w)
    y = torch.arange(h, dtype=image.dtype, device=image.device).view(1, h, 1).expand(d, h, w)
    x = torch.arange(w, dtype=image.dtype, device=image.device).view(1, 1, w).expand(d, h, w)
    # Compute the moments (weighted sums)
    weighted_x = x * image
    weighted_y = y * image
    weighted_z = z * image
    # Compute the sum of intensities
    total_intensity = image.sum()
    # Compute the weighted centroid (moment) for each axis
    centroid_x = weighted_x.sum() / total_intensity
    centroid_y = weighted_y.sum() / total_intensity
    centroid_z = weighted_z.sum() / total_intensity
    # Return the centroid as a tensor (z, y, x) on the same device and dtype.
    return torch.stack((centroid_z, centroid_y, centroid_x))
