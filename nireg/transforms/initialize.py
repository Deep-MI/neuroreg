import torch
from torch import Tensor

from ..image.centroid import compute_centroid


def get_ixform_centroids(simg: Tensor, timg: Tensor) -> Tensor:
    """
    Compute an initial voxel-to-voxel transformation by aligning image centroids.

    Computes the intensity-weighted center of mass (moment) for each image
    and returns a translation that aligns them. Useful when images share
    the same voxel grid but anatomy is displaced (e.g., patient motion,
    synthetic displacement).

    Note
    ----
    Can fail when images are too different (tumor growth, missing structures).
    Users can disable with centroid_init=False in that case.

    Parameters
    ----------
    simg : Tensor
        Source image (D, H, W).
    timg : Tensor
        Target image (D, H, W).

    Returns
    -------
    Tensor
        4x4 voxel-to-voxel transformation matrix (translation only).
    """
    # Compute intensity-weighted centroids
    cs = compute_centroid(simg)
    ct = compute_centroid(timg)
    # Create the voxel-to-voxel transformation matrix
    v2v = torch.eye(4, dtype=simg.dtype)
    v2v[0:3, 3] = ct - cs  # Translate to align centroids
    return v2v



def get_vox2vox_from_header(saffine: Tensor, taffine: Tensor) -> Tensor:
    """
    Compute the initial transformation based on image affine matrices (from headers).

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

