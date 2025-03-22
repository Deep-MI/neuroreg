import torch
from torch import Tensor
from ..image.centroid import compute_centroid

def get_ixform_centroids(simg: Tensor, timg: Tensor) -> Tensor:
    """
    Computes an initial voxel-to-voxel transformation between a source image
    and a target image by aligning their centroids.

    Note
    ----
    The header affines of the input images must agree for centroid alignment to be valid.

    Parameters
    ----------
    simg : Tensor
        A 3D tensor representing the source image with shape (D, H, W),
        where D is the depth, H is the height, and W is the width.
    timg : Tensor
        A 3D tensor representing the target image with shape (D, H, W).

    Returns
    -------
    Tensor
        A 4x4 tensor representing the initial voxel-to-voxel transformation matrix.
        This defines a translation that aligns the centroids of the source and target images.

    Example
    -------
    >>> simg = torch.rand(64, 64, 64)  # Source image
    >>> timg = torch.rand(64, 64, 64)  # Target image
    >>> transform = get_ixform_centroids(simg, timg)
    >>> print(transform)
    tensor([[1., 0., 0., 1.5234],
            [0., 1., 0., -0.2342],
            [0., 0., 1., 0.1945],
            [0., 0., 0., 1.]])
    """
    # Compute centroids for the source and target images
    cs = compute_centroid(simg)
    ct = compute_centroid(timg)
    # Create the voxel-to-voxel transformation matrix
    v2v = torch.eye(4)
    v2v[0:3, 3] = ct - cs  # Translate to align centroids
    return v2v



def get_vox2vox_from_header(saffine: Tensor, taffine: Tensor) -> Tensor:
    """
    Computes the initial transformation between source and target images based on their affine matrices.

    Parameters
    ----------
    saffine : Tensor
        A 4x4 tensor representing the affine transformation matrix of the source image (vox2ras).
    taffine : Tensor
        A 4x4 tensor representing the affine transformation matrix of the target image (vox2ras).

    Returns
    -------
    Tensor
        A 4x4 tensor representing the initial vox2vox transformation matrix, which maps
        voxel indices from the target to the source image.

    Example
    -------
    >>> saffine = torch.eye(4)  # Example source affine matrix (identity)
    >>> taffine = torch.tensor([[ 0.5, 0,   0,   0],
    ...                         [ 0,   0.5, 0,   0],
    ...                         [ 0,   0,   0.5, 0],
    ...                         [ 0,   0,   0,   1]])  # Example target affine
    >>> vox2vox = get_vox2vox_from_header(saffine, taffine)
    >>> print(vox2vox)
    tensor([[2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
            [0., 0., 0., 1.]])
    """
    return torch.inverse(taffine) @ saffine

