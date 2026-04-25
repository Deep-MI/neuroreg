"""Differentiable image-similarity loss functions for cross-modal registration.

All public functions accept 3-D voxel tensors of shape (D, H, W) and return a
scalar tensor whose gradient flows through the warped source image *preds*.

Intensity-based losses (mse / huber / smooth_l1 / l1) live in
:mod:`torch.nn.functional` and are dispatched directly from the training loop.
The three functions here are for modality-independent similarity:

* :func:`ncc_loss`  — local normalised cross-correlation (same-modality)
* :func:`mi_loss`   — mutual information via Parzen-window histogram (cross-modal)
* :func:`nmi_loss`  — normalised MI (cross-modal, robust to masked/cropped volumes)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_EMPTY_MASK_PENALTY = 1e6


def _mask_to_bool(mask: Tensor | None, reference: Tensor) -> Tensor | None:
    """Validate and normalize a mask tensor to boolean form."""
    if mask is None:
        return None
    if tuple(mask.shape) != tuple(reference.shape):
        raise ValueError(
            "Mask shape must match image shape: "
            f"mask={tuple(mask.shape)} vs image={tuple(reference.shape)}."
        )
    return mask.to(device=reference.device) > 0.5


def masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Average a tensor over valid mask elements, or all elements if unmasked."""
    mask_bool = _mask_to_bool(mask, values)
    if mask_bool is None:
        return values.mean()
    if not torch.any(mask_bool):
        return values.sum() * 0 + values.new_tensor(_EMPTY_MASK_PENALTY)
    return values[mask_bool].mean()

# ---------------------------------------------------------------------------
# Local NCC
# ---------------------------------------------------------------------------

def ncc_loss(
    preds: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    win_size: int = 9,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
) -> Tensor:
    """Local normalised cross-correlation loss.

    Loss = 1 − mean(NCC²), so the loss is 0 for perfectly correlated images
    and 1 for orthogonal images.  Using NCC² makes the metric symmetric and
    well-defined for inverted contrasts.

    Suitable for same-modality registration where the intensity relationship is
    approximately linear (e.g. T1→T1, T2→T2).

    Parameters
    ----------
    preds : Tensor
        Warped source image, shape (D, H, W).
    target : Tensor
        Fixed target image, shape (D, H, W).
    win_size : int, default=9
        Sliding-window size in voxels.  Automatically clamped to the smallest
        spatial dimension and forced to be odd.
    smooth_nr : float, default=1e-5
        Small constant added to the NCC² numerator.  When equal to
        ``smooth_dr`` (the default), flat (zero-variance) windows produce
        NCC²=1 → loss=0 instead of a false penalty.
    smooth_dr : float, default=1e-5
        Small constant added to the NCC² denominator to prevent division by
        zero; also the minimum value each local variance is clamped to.

    Returns
    -------
    Tensor
        Scalar loss in [0, 1].
    """
    if preds.dim() > 3 or target.dim() > 3:
        raise ValueError(
            f"ncc_loss expects ≤3-D tensors; got shapes {tuple(preds.shape)} and {tuple(target.shape)}."
        )

    mask_bool = _mask_to_bool(mask, preds)

    # Add batch + channel dims for pooling: (1, 1, D, H, W)
    src = preds.unsqueeze(0).unsqueeze(0)
    trg = target.unsqueeze(0).unsqueeze(0)

    max_dim = min(src.shape[2], src.shape[3], src.shape[4])
    win_size = min(win_size, max_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 1)
    pad = win_size // 2

    pool_kw: dict = dict(kernel_size=win_size, stride=1, padding=pad)

    if mask_bool is None:
        src_mean = F.avg_pool3d(src, **pool_kw)
        trg_mean = F.avg_pool3d(trg, **pool_kw)
        ss_mean = F.avg_pool3d(src * src, **pool_kw)
        tt_mean = F.avg_pool3d(trg * trg, **pool_kw)
        st_mean = F.avg_pool3d(src * trg, **pool_kw)
        window_valid = torch.ones_like(src_mean, dtype=torch.bool)
    else:
        mask_5d = mask_bool.to(dtype=src.dtype).unsqueeze(0).unsqueeze(0)
        window_weight = F.avg_pool3d(mask_5d, **pool_kw)
        safe_weight = window_weight.clamp(min=1e-6)
        src_mean = F.avg_pool3d(src * mask_5d, **pool_kw) / safe_weight
        trg_mean = F.avg_pool3d(trg * mask_5d, **pool_kw) / safe_weight
        ss_mean = F.avg_pool3d(src * src * mask_5d, **pool_kw) / safe_weight
        tt_mean = F.avg_pool3d(trg * trg * mask_5d, **pool_kw) / safe_weight
        st_mean = F.avg_pool3d(src * trg * mask_5d, **pool_kw) / safe_weight
        window_valid = window_weight > 0

    # Clamp each variance to smooth_dr so the denominator product is always > 0.
    # With smooth_nr == smooth_dr, flat windows give ncc2 ≈ 1 → loss = 0.
    smooth_dr_t = src.new_tensor(smooth_dr)
    src_var = torch.max(ss_mean - src_mean * src_mean, smooth_dr_t)
    trg_var = torch.max(tt_mean - trg_mean * trg_mean, smooth_dr_t)
    cross   = st_mean - src_mean * trg_mean

    ncc2 = (cross.pow(2) + smooth_nr) / (src_var * trg_var + smooth_dr)
    # Cauchy-Schwarz guarantees ncc2 ≤ 1; clamp guards floating-point edge cases.
    ncc2 = ncc2.clamp(max=1.0)
    return 1.0 - masked_mean(ncc2.squeeze(), window_valid.squeeze())


# ---------------------------------------------------------------------------
# Parzen-window joint histogram (shared by MI and NMI)
# ---------------------------------------------------------------------------

def _parzen_joint_hist(
    a: Tensor,
    b: Tensor,
    num_bins: int = 32,
    sigma: float | None = None,
) -> Tensor:
    """Differentiable joint histogram via Gaussian Parzen windows.

    Parameters
    ----------
    a, b : Tensor
        Flat intensity tensors with values in [0, 1].
    num_bins : int, default=32
        Number of histogram bins along each intensity axis.
    sigma : float or None, default=None
        Standard deviation of the Gaussian kernel in normalised [0, 1]
        intensity units.  ``None`` auto-computes ``sigma = 0.5 / (num_bins-1)``,
        which places the kernel half-width at one bin spacing.

    Returns
    -------
    Tensor of shape (num_bins, num_bins)
        Normalised joint probability matrix summing to 1.
    """
    if sigma is None:
        sigma = 0.5 / (num_bins - 1)

    bins = torch.linspace(0.0, 1.0, num_bins, device=a.device, dtype=a.dtype)  # (B,)

    a_flat = a.reshape(-1, 1)     # (N, 1)
    b_flat = b.reshape(-1, 1)     # (N, 1)
    bins_row = bins.unsqueeze(0)  # (1, B)

    # Gaussian soft assignment: (N, B)
    a_w = torch.exp(-0.5 * ((a_flat - bins_row) / sigma).pow(2))
    b_w = torch.exp(-0.5 * ((b_flat - bins_row) / sigma).pow(2))

    # Row-normalise so each voxel's weights sum to 1
    a_w = a_w / (a_w.sum(dim=1, keepdim=True) + 1e-10)
    b_w = b_w / (b_w.sum(dim=1, keepdim=True) + 1e-10)

    # Joint PMF: (B, B), normalised by number of voxels
    joint = torch.einsum("ni,nj->ij", a_w, b_w) / float(a.numel())
    return joint


def _to_unit(x: Tensor) -> Tensor:
    """Scale tensor to [0, 1] using its own min/max (gradient-transparent)."""
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin + 1e-8)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def mi_loss(
    preds: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    num_bins: int = 32,
    sigma: float | None = None,
    smooth_nr: float = 1e-7,
    smooth_dr: float = 1e-7,
) -> Tensor:
    """Mutual-information loss (minimise −MI).

    Computed via the KL-divergence form::

        MI = Σ_ij p(a,b)_ij · log[ p(a,b)_ij / (p(a)_i · p(b)_j) ]

    Works across imaging modalities (T1↔T2, T1↔FLAIR, MRI↔CT).

    Parameters
    ----------
    preds : Tensor
        Warped source image (any value range; normalised to [0,1] internally).
    target : Tensor
        Fixed target image (any value range).
    num_bins : int, default=32
        Number of intensity bins for the joint histogram.
    sigma : float or None, default=None
        Parzen-window bandwidth in normalised [0, 1] intensity units.
        ``None`` auto-computes ``sigma = 0.5 / (num_bins-1)`` (half a bin
        spacing).  Pass an explicit float to override.
    smooth_nr : float, default=1e-7
        Small constant added to the joint PMF in the log numerator.
    smooth_dr : float, default=1e-7
        Small constant added to the marginal product in the log denominator,
        and a lower bound on the log argument to prevent log→−∞.

    Returns
    -------
    Tensor
        −MI; minimising this maximises mutual information.
    """
    mask_bool = _mask_to_bool(mask, preds)
    if mask_bool is not None:
        preds = preds[mask_bool]
        target = target[mask_bool]
        if preds.numel() == 0:
            return preds.sum() * 0 + preds.new_tensor(_EMPTY_MASK_PENALTY)

    joint = _parzen_joint_hist(_to_unit(preds), _to_unit(target), num_bins=num_bins, sigma=sigma)

    pa = joint.sum(dim=1, keepdim=True)   # (B, 1) marginal over preds
    pb = joint.sum(dim=0, keepdim=True)   # (1, B) marginal over target
    papb = pa * pb                         # (B, B) product-of-marginals

    mi = (joint * torch.log((joint + smooth_nr) / (papb + smooth_dr) + smooth_dr)).sum()
    return -mi


# ---------------------------------------------------------------------------
# Normalised mutual information
# ---------------------------------------------------------------------------

def nmi_loss(
    preds: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    num_bins: int = 32,
    sigma: float | None = None,
    smooth_nr: float = 1e-7,
    smooth_dr: float = 1e-7,
) -> Tensor:
    """Normalised mutual-information loss (minimise −NMI).

    NMI = (H(A) + H(B)) / H(A, B) lies in [1, 2]: 1 for independent images,
    2 for identical images.  Normalisation makes the metric robust to changes
    in marginal entropy (e.g. at coarse pyramid levels with masked regions).

    Works across imaging modalities (T1↔T2, T1↔FLAIR, MRI↔CT).

    Parameters
    ----------
    preds : Tensor
        Warped source image (any value range).
    target : Tensor
        Fixed target image (any value range).
    num_bins : int, default=32
        Number of intensity bins for the joint histogram.
    sigma : float or None, default=None
        Parzen-window bandwidth.  See :func:`mi_loss` for details.
    smooth_nr : float, default=1e-7
        Additive floor applied to every joint-histogram bin before entropy
        computation (then renormalised); prevents log(0).
    smooth_dr : float, default=1e-7
        Small constant added to H(A, B) in the denominator.

    Returns
    -------
    Tensor
        −NMI; minimising this maximises normalised mutual information.
    """
    mask_bool = _mask_to_bool(mask, preds)
    if mask_bool is not None:
        preds = preds[mask_bool]
        target = target[mask_bool]
        if preds.numel() == 0:
            return preds.sum() * 0 + preds.new_tensor(_EMPTY_MASK_PENALTY)

    joint = _parzen_joint_hist(_to_unit(preds), _to_unit(target), num_bins=num_bins, sigma=sigma)

    # Additive smoothing: floor every bin, then renormalise to a valid PMF.
    joint = joint + smooth_nr
    joint = joint / joint.sum()

    pa = joint.sum(dim=1)   # (B,) marginal over preds
    pb = joint.sum(dim=0)   # (B,) marginal over target

    H_ab = -(joint * joint.log()).sum()
    H_a  = -(pa * pa.log()).sum()
    H_b  = -(pb * pb.log()).sum()

    return -(H_a + H_b) / (H_ab + smooth_dr)


__all__ = ["ncc_loss", "mi_loss", "nmi_loss"]
