"""
Volume sampling at surface vertices.

Handles sampling image intensities at transformed vertex locations,
managing coordinate system transformations (tkRAS -> voxel).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Literal


def sample_volume_at_vertices(
    volume: torch.Tensor,
    vertices_tkras: torch.Tensor,
    vox2ras_tkr: torch.Tensor,
    reg_matrix: Optional[torch.Tensor] = None,
    interpolation: Literal['nearest', 'bilinear', 'trilinear'] = 'trilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True
) -> torch.Tensor:
    """
    Sample volume intensities at surface vertex locations.

    This function handles the complete transformation pipeline:
    1. Apply registration transform to vertices (if provided)
    2. Transform from tkRAS to voxel coordinates
    3. Sample volume using grid_sample

    Parameters
    ----------
    volume : torch.Tensor, shape (H, W, D) or (1, 1, H, W, D)
        Volume to sample from
    vertices_tkras : torch.Tensor, shape (N, 3)
        Vertex coordinates in tkRAS space
    vox2ras_tkr : torch.Tensor, shape (4, 4)
        Voxel-to-tkRAS transformation matrix
    reg_matrix : torch.Tensor, shape (4, 4), optional
        Registration matrix (transforms vertices before sampling)
        If None, uses identity (no transformation)
    interpolation : str
        Interpolation mode: 'nearest', 'bilinear', or 'trilinear'
    padding_mode : str
        How to handle out-of-bounds: 'zeros', 'border', 'reflection'
    align_corners : bool
        Align corners for grid_sample (default: True)

    Returns
    -------
    values : torch.Tensor, shape (N,)
        Sampled intensity values at each vertex

    Notes
    -----
    The transformation pipeline is:
        vertices_tkras -> [reg_matrix] -> vertices_mov_tkras
                      -> [inv(vox2ras_tkr)] -> vertices_vox
                      -> normalize to [-1, 1] -> sample

    Out-of-bounds vertices (outside volume) will have value determined
    by padding_mode (typically 0).
    """
    device = volume.device
    dtype = volume.dtype

    # Ensure volume is 5D: (B, C, H, W, D)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)

    # Volume shape from nibabel is (i, j, k) stored as (B, C, i, j, k)
    _, _, Si, Sj, Sk = volume.shape

    # Convert vertices to homogeneous coordinates (N, 4)
    n_vertices = vertices_tkras.shape[0]
    vertices_hom = torch.cat([
        vertices_tkras,
        torch.ones((n_vertices, 1), device=device, dtype=dtype)
    ], dim=1)  # (N, 4)

    # Apply registration transform if provided
    if reg_matrix is not None:
        vertices_hom = torch.matmul(reg_matrix, vertices_hom.T).T  # (N, 4)

    # Transform from tkRAS to voxel coordinates
    # vox = inv(vox2ras_tkr) @ tkras  -> (i, j, k)
    ras2vox_tkr = torch.inverse(vox2ras_tkr.to(device).to(dtype))
    vertices_vox = torch.matmul(ras2vox_tkr, vertices_hom.T).T[:, :3]  # (N, 3): (i, j, k)

    # grid_sample with a 5D volume (B, C, i, j, k) expects grid coords (x, y, z)
    # where x indexes the LAST dim (k), y the middle dim (j), z the first spatial dim (i).
    # So: x = normalise(k, Sk), y = normalise(j, Sj), z = normalise(i, Si)
    grid = torch.zeros((n_vertices, 3), device=device, dtype=dtype)
    grid[:, 0] = 2.0 * vertices_vox[:, 2] / (Sk - 1) - 1.0  # x -> k (last dim)
    grid[:, 1] = 2.0 * vertices_vox[:, 1] / (Sj - 1) - 1.0  # y -> j
    grid[:, 2] = 2.0 * vertices_vox[:, 0] / (Si - 1) - 1.0  # z -> i (first spatial dim)

    # Reshape for grid_sample: (1, N, 1, 1, 3)
    grid = grid.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, N, 1, 1, 3)

    # Map interpolation mode
    mode_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',  # Actually trilinear for 3D
        'trilinear': 'bilinear'
    }
    mode = mode_map.get(interpolation, 'bilinear')

    # Sample the volume
    # grid_sample expects grid with shape (B, H_out, W_out, D_out, 3)
    sampled = F.grid_sample(
        volume,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )  # (1, 1, N, 1, 1)

    # Extract values: (N,)
    values = sampled.squeeze()

    return values


def compute_volume_gradient(
    volume: torch.Tensor,
    voxel_size: Optional[tuple[float, float, float]] = None
) -> torch.Tensor:
    """
    Compute 3D gradient of volume.

    Parameters
    ----------
    volume : torch.Tensor, shape (H, W, D) or (1, 1, H, W, D)
        Input volume
    voxel_size : tuple of float, optional
        Voxel dimensions (dx, dy, dz) in mm
        If None, assumes isotropic 1mm voxels

    Returns
    -------
    gradient : torch.Tensor, shape (3, H, W, D) or (1, 3, H, W, D)
        Gradient in each direction (x, y, z)
    """
    # Ensure 5D
    squeeze_output = False
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        squeeze_output = True
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)

    # Compute gradients using Sobel-like kernels
    # For simplicity, use central differences
    device = volume.device
    dtype = volume.dtype

    # Sobel-like 3D kernels for gradient estimation
    # Simple central difference: [-1, 0, 1] / 2
    kernel_x = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    kernel_x[0, 0, 1, 1, :] = torch.tensor([-0.5, 0, 0.5], device=device, dtype=dtype)

    kernel_y = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    kernel_y[0, 0, 1, :, 1] = torch.tensor([-0.5, 0, 0.5], device=device, dtype=dtype)

    kernel_z = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    kernel_z[0, 0, :, 1, 1] = torch.tensor([-0.5, 0, 0.5], device=device, dtype=dtype)

    # Apply convolution
    grad_x = F.conv3d(volume, kernel_x, padding=1)
    grad_y = F.conv3d(volume, kernel_y, padding=1)
    grad_z = F.conv3d(volume, kernel_z, padding=1)

    # Adjust for voxel size
    if voxel_size is not None:
        grad_x = grad_x / voxel_size[0]
        grad_y = grad_y / voxel_size[1]
        grad_z = grad_z / voxel_size[2]

    # Concatenate gradients
    gradient = torch.cat([grad_x, grad_y, grad_z], dim=1)  # (1, 3, H, W, D)

    if squeeze_output:
        gradient = gradient.squeeze(0)  # (3, H, W, D)

    return gradient


def sample_gradient_at_vertices(
    volume: torch.Tensor,
    vertices_tkras: torch.Tensor,
    vox2ras_tkr: torch.Tensor,
    reg_matrix: Optional[torch.Tensor] = None,
    voxel_size: Optional[tuple[float, float, float]] = None,
    interpolation: str = 'trilinear'
) -> torch.Tensor:
    """
    Sample volume gradient at surface vertex locations.

    Parameters
    ----------
    volume : torch.Tensor, shape (H, W, D)
        Volume to compute gradient from
    vertices_tkras : torch.Tensor, shape (N, 3)
        Vertex coordinates in tkRAS space
    vox2ras_tkr : torch.Tensor, shape (4, 4)
        Voxel-to-tkRAS transformation
    reg_matrix : torch.Tensor, shape (4, 4), optional
        Registration matrix
    voxel_size : tuple of float, optional
        Voxel dimensions for gradient computation
    interpolation : str
        Interpolation mode

    Returns
    -------
    gradients : torch.Tensor, shape (N, 3)
        Gradient vectors at each vertex location
    """
    # Compute volume gradient
    grad_volume = compute_volume_gradient(volume, voxel_size)  # (3, H, W, D)

    # Sample each gradient component
    grad_x = sample_volume_at_vertices(
        grad_volume[0], vertices_tkras, vox2ras_tkr, reg_matrix, interpolation
    )
    grad_y = sample_volume_at_vertices(
        grad_volume[1], vertices_tkras, vox2ras_tkr, reg_matrix, interpolation
    )
    grad_z = sample_volume_at_vertices(
        grad_volume[2], vertices_tkras, vox2ras_tkr, reg_matrix, interpolation
    )

    # Stack into (N, 3)
    gradients = torch.stack([grad_x, grad_y, grad_z], dim=1)

    return gradients

