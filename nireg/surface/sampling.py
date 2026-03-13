"""
Volume sampling at surface vertices.

Handles sampling image intensities at transformed vertex locations,
managing coordinate system transformations (tkRAS -> voxel).
"""

from typing import Literal

import torch
import torch.nn.functional as F


def _trilinear_manual(
    volume: torch.Tensor,
    vi: torch.Tensor,
    vj: torch.Tensor,
    vk: torch.Tensor,
    Si: int, Sj: int, Sk: int,
    dtype: torch.dtype,
    padding_mode: str,
) -> torch.Tensor:
    """MPS-compatible trilinear interpolation using only native index ops.

    Called by :func:`sample_volume_at_vertices` on MPS devices where
    ``F.grid_sample`` is unavailable.  Gradients flow through the fractional
    weights ``fi/fj/fk`` via ``vi - vi.detach().floor()``; the integer corner
    indices are detached so that ``.long()`` casting does not break autograd.

    Parameters
    ----------
    volume : torch.Tensor, shape (1, 1, Si, Sj, Sk)
        Volume to sample (already 5-D).
    vi, vj, vk : torch.Tensor, shape (N,)
        Voxel coordinates along i, j, k axes.
    Si, Sj, Sk : int
        Volume dimensions.
    dtype : torch.dtype
        Target dtype for fractional weights.
    padding_mode : str
        ``'zeros'`` or ``'border'``.
    """
    i0 = vi.detach().floor().long()
    i1 = i0 + 1
    j0 = vj.detach().floor().long()
    j1 = j0 + 1
    k0 = vk.detach().floor().long()
    k1 = k0 + 1

    # fractional parts — carry grad through vi/vj/vk
    fi = (vi - vi.detach().floor()).to(dtype)
    fj = (vj - vj.detach().floor()).to(dtype)
    fk = (vk - vk.detach().floor()).to(dtype)

    # clamp indices to valid range
    i0 = i0.clamp(0, Si - 1)
    i1 = i1.clamp(0, Si - 1)
    j0 = j0.clamp(0, Sj - 1)
    j1 = j1.clamp(0, Sj - 1)
    k0 = k0.clamp(0, Sk - 1)
    k1 = k1.clamp(0, Sk - 1)

    if padding_mode == 'zeros':
        oob = ((vi < 0) | (vi > Si - 1) |
               (vj < 0) | (vj > Sj - 1) |
               (vk < 0) | (vk > Sk - 1))

    v = volume[0, 0]
    c00 = torch.lerp(v[i0, j0, k0], v[i1, j0, k0], fi)
    c01 = torch.lerp(v[i0, j0, k1], v[i1, j0, k1], fi)
    c10 = torch.lerp(v[i0, j1, k0], v[i1, j1, k0], fi)
    c11 = torch.lerp(v[i0, j1, k1], v[i1, j1, k1], fi)
    c0  = torch.lerp(c00, c10, fj)
    c1  = torch.lerp(c01, c11, fj)
    values = torch.lerp(c0, c1, fk)

    if padding_mode == 'zeros':
        values = values.masked_fill(oob, 0.0)

    return values


def sample_volume_at_vertices(
    volume: torch.Tensor,
    vertices_tkras: torch.Tensor,
    vox2ras_tkr: torch.Tensor,
    reg_matrix: torch.Tensor | None = None,
    interpolation: Literal['nearest', 'trilinear'] = 'trilinear',
    padding_mode: str = 'zeros',
) -> torch.Tensor:
    """Sample volume intensities at surface vertex locations.

    Applies the full coordinate pipeline, then samples the volume using
    manual trilinear interpolation with only MPS/CUDA/CPU-native ops —
    avoiding ``F.grid_sample`` which is not implemented for MPS in
    PyTorch 2.x.

    Parameters
    ----------
    volume : torch.Tensor, shape (H, W, D) or (1, 1, H, W, D)
        Volume to sample from.
    vertices_tkras : torch.Tensor, shape (N, 3)
        Vertex coordinates in tkRAS space.
    vox2ras_tkr : torch.Tensor, shape (4, 4)
        Voxel-to-tkRAS transformation matrix.
    reg_matrix : torch.Tensor, shape (4, 4), optional
        Registration matrix applied to vertices before sampling.
        If ``None`` the identity is used (no transformation).
    interpolation : {'nearest', 'trilinear'}
        Interpolation mode.
    padding_mode : {'zeros', 'border'}
        Out-of-bounds handling: ``'zeros'`` returns 0 for vertices outside
        the volume; ``'border'`` clamps to the nearest edge voxel.

    Returns
    -------
    values : torch.Tensor, shape (N,)
        Sampled intensity values at each vertex.
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

    # voxel coordinates: i, j, k  (first/second/third spatial dim)
    vi = vertices_vox[:, 0]
    vj = vertices_vox[:, 1]
    vk = vertices_vox[:, 2]

    if interpolation == 'nearest':
        # Compute OOB mask *before* clamping so that padding_mode='zeros'
        # returns 0 for out-of-bounds vertices.  Without this the clamp
        # would silently act as padding_mode='border' regardless of what
        # the caller requested.
        if padding_mode == 'zeros':
            oob = ((vi < 0) | (vi > Si - 1) |
                   (vj < 0) | (vj > Sj - 1) |
                   (vk < 0) | (vk > Sk - 1))
        ii = vi.round().long().clamp(0, Si - 1)
        jj = vj.round().long().clamp(0, Sj - 1)
        kk = vk.round().long().clamp(0, Sk - 1)
        values = volume[0, 0][ii, jj, kk]
        if padding_mode == 'zeros':
            values = values.masked_fill(oob, 0.0)
        return values

    # ── choose backend based on device ─────────────────────────────────────
    # F.grid_sample is not implemented for MPS (PyTorch 2.x), so we use a
    # manual trilinear interpolation there.  On CPU and CUDA we keep
    # F.grid_sample because its internal fused kernel gives a different
    # (and empirically better-converging) float32 rounding path.
    if volume.device.type == 'mps':
        return _trilinear_manual(volume, vi, vj, vk, Si, Sj, Sk, dtype, padding_mode)

    # ── F.grid_sample path (CPU / CUDA) ────────────────────────────────────
    # grid_sample expects coords in [-1, 1] with x→last dim, y→middle, z→first.
    grid = torch.zeros((n_vertices, 3), device=device, dtype=dtype)
    grid[:, 0] = 2.0 * vk / (Sk - 1) - 1.0  # x → k (last dim)
    grid[:, 1] = 2.0 * vj / (Sj - 1) - 1.0  # y → j
    grid[:, 2] = 2.0 * vi / (Si - 1) - 1.0  # z → i (first spatial dim)
    grid = grid.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, N, 1, 1, 3)

    sampled = F.grid_sample(
        volume, grid, mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True
    )
    return sampled.squeeze()


def compute_volume_gradient(
    volume: torch.Tensor,
    voxel_size: tuple[float, float, float] | None = None
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
    reg_matrix: torch.Tensor | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    interpolation: str = 'trilinear',
    precomputed_grad: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sample volume gradient at surface vertex locations.

    Parameters
    ----------
    volume : torch.Tensor, shape (H, W, D)
        Volume to compute gradient from.  Ignored when *precomputed_grad*
        is supplied.
    vertices_tkras : torch.Tensor, shape (N, 3)
        Vertex coordinates in tkRAS space
    vox2ras_tkr : torch.Tensor, shape (4, 4)
        Voxel-to-tkRAS transformation
    reg_matrix : torch.Tensor, shape (4, 4), optional
        Registration matrix
    voxel_size : tuple of float, optional
        Voxel dimensions for gradient computation.  Ignored when
        *precomputed_grad* is supplied.
    interpolation : str
        Interpolation mode
    precomputed_grad : torch.Tensor, shape (3, H, W, D), optional
        Pre-computed gradient volume (e.g. cached in the model).  When
        provided the three ``conv3d`` passes are skipped entirely, which
        avoids redundant work during iterative optimisation where the
        moving volume — and therefore its gradient — never changes.

    Returns
    -------
    gradients : torch.Tensor, shape (N, 3)
        Gradient vectors at each vertex location
    """
    # Use caller-supplied gradient volume when available; otherwise compute.
    if precomputed_grad is not None:
        grad_volume = precomputed_grad          # (3, H, W, D)
    else:
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

    return torch.stack([grad_x, grad_y, grad_z], dim=1)

