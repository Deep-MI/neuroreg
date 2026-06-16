from __future__ import annotations

import torch
from torch import Tensor

_CENTERED_CUBIC_REDUCE_FILTER = torch.tensor(
    [
        0.708792,
        0.328616,
        -0.165157,
        -0.114448,
        0.0944036,
        0.0543881,
        -0.05193,
        -0.0284868,
        0.0281854,
        0.0152877,
        -0.0152508,
        -0.00825077,
        0.00824629,
        0.00445865,
        -0.0044582,
        -0.00241009,
        0.00241022,
        0.00130278,
        -0.00130313,
        -0.000704109,
        0.000704784,
    ],
    dtype=torch.float64,
)
"""Centered cubic B-spline reduce filter used by FreeSurfer's pyramid code."""


def _pseudo_mirror_indices(indices: Tensor, length: int) -> Tensor:
    """Map arbitrary indices to a pseudo-mirrored range ``[0, length)``."""
    if length < 1:
        raise ValueError("length must be positive")
    if length == 1:
        return torch.zeros_like(indices)

    period = 2 * length
    wrapped = torch.remainder(indices, period)
    return torch.where(wrapped < length, wrapped, period - wrapped - 1)


def _reduce_centered_lastdim(lines: Tensor, filt: Tensor) -> Tensor:
    """Reduce the last dimension by a factor of two on the centered half-grid."""
    n_in = lines.shape[-1]
    if n_in <= 1:
        return lines.clone()

    n_out = n_in // 2
    n_even = n_out * 2
    if n_out < 1:
        return lines[..., :1].clone()

    clipped = lines[..., :n_even]
    coeffs = filt.to(device=lines.device, dtype=lines.dtype)
    positions = torch.arange(n_even, device=lines.device, dtype=torch.int64)

    filtered = clipped * coeffs[0]
    for offset in range(1, int(coeffs.numel())):
        left = _pseudo_mirror_indices(positions - offset, n_even)
        right = _pseudo_mirror_indices(positions + offset, n_even)
        filtered = filtered + coeffs[offset] * (clipped.index_select(-1, left) + clipped.index_select(-1, right))

    return 0.5 * (filtered[..., 0::2] + filtered[..., 1::2])


def downsample2_bspline(volume: Tensor) -> Tensor:
    """Downsample a 3-D volume by 2x using a centered cubic B-spline reducer.

    This ports the reduction path FreeSurfer uses in ``MRIdownsample2BSpline``:
    a separable centered cubic B-spline filter followed by Haar averaging on the
    centered half-grid, with pseudo-mirror boundary handling.
    """
    if volume.ndim != 3:
        raise ValueError(f"downsample2_bspline expects a 3D tensor, got shape {tuple(volume.shape)}")

    reduced = volume
    for dim in range(3):
        if reduced.shape[dim] <= 1:
            continue

        perm = [axis for axis in range(3) if axis != dim] + [dim]
        inverse_perm = [0, 0, 0]
        for idx, axis in enumerate(perm):
            inverse_perm[axis] = idx

        lines = reduced.permute(*perm).contiguous()
        leading_shape = tuple(int(v) for v in lines.shape[:-1])
        flat = lines.reshape(-1, int(lines.shape[-1]))
        flat = _reduce_centered_lastdim(flat, _CENTERED_CUBIC_REDUCE_FILTER)
        reduced = flat.reshape(*leading_shape, int(flat.shape[-1])).permute(*inverse_perm).contiguous()

    return reduced


__all__ = ["downsample2_bspline"]
