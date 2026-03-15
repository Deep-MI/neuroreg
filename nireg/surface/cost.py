"""Cost functions for surface-based registration."""

import logging

import torch

logger = logging.getLogger(__name__)


def bbr_contrast_cost(
    vwm: torch.Tensor,
    vctx: torch.Tensor,
    slope: float = 0.5,
    center: float = 0.0,
    contrast_sign: int = 1,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute BBR cost based on tissue contrast.

    The cost is minimized when tissues have correct contrast:
    - For T2/fMRI (contrast_sign=+1): cost is low when vctx > vwm (GM brighter)
    - For T1 (contrast_sign=-1): cost is low when vctx < vwm (GM darker)

    Parameters
    ----------
    vwm : torch.Tensor
        White matter intensities
    vctx : torch.Tensor
        Gray matter (cortex) intensities
    slope : float
        Slope parameter for tanh penalty
    center : float
        Center for percent contrast
    contrast_sign : int
        +1 for T2/fMRI (GM brighter), -1 for T1 (GM darker)
    """
    mean_intensity = (vctx + vwm) / 2.0 + eps
    percent_contrast = 100.0 * (vctx - vwm) / mean_intensity

    # Compute penalty - should be negative when contrast is correct
    if contrast_sign == 0:
        # Maximize absolute contrast
        penalty = -torch.abs(slope * (percent_contrast - center))
    elif contrast_sign == 1:
        # T2/fMRI: want vctx > vwm (positive percent_contrast)
        # penalty is negative when percent_contrast is positive
        penalty = -slope * (percent_contrast - center)
    elif contrast_sign == -1:
        # T1: want vctx < vwm (negative percent_contrast)
        # penalty is negative when percent_contrast is negative
        penalty = slope * (percent_contrast - center)
    else:
        raise ValueError(f"Invalid contrast_sign: {contrast_sign}")

    # Cost is minimized when penalty is negative (correct contrast)
    cost_per_vertex = 1.0 + torch.tanh(penalty)

    if mask is not None:
        valid = mask > 0.5
        if valid.sum() == 0:
            return torch.tensor(10.0, device=vwm.device, dtype=vwm.dtype)
        cost = cost_per_vertex[valid].mean()
    else:
        cost = cost_per_vertex.mean()
    return cost


def gradient_magnitude_cost(
    volume_gradient: torch.Tensor,
    normals: torch.Tensor,
    mask: torch.Tensor | None = None,
    use_normal_component: bool = True
) -> torch.Tensor:
    """Compute cost based on image gradient magnitude."""
    if use_normal_component:
        grad_normal = torch.abs(torch.sum(volume_gradient * normals, dim=1))
        if mask is not None:
            valid = mask > 0.5
            if valid.sum() == 0:
                return torch.tensor(0.0, device=volume_gradient.device)
            cost = -grad_normal[valid].mean()
        else:
            cost = -grad_normal.mean()
    else:
        grad_magnitude = torch.norm(volume_gradient, dim=1)
        if mask is not None:
            valid = mask > 0.5
            if valid.sum() == 0:
                return torch.tensor(0.0, device=volume_gradient.device)
            cost = -grad_magnitude[valid].mean()
        else:
            cost = -grad_magnitude.mean()
    return cost


def detect_contrast(
    vwm: torch.Tensor,
    vgm: torch.Tensor,
) -> str:
    """Auto-detect image contrast direction from sampled WM and GM intensities.

    Computes the percent contrast ``(vgm - vwm) / mean`` at each vertex and
    looks at the **median** across all vertices:

    * Positive median → GM is brighter than WM → **T2** contrast.
    * Negative median → WM is brighter than GM → **T1** contrast.

    Parameters
    ----------
    vwm : torch.Tensor, shape (N,)
        Sampled intensities at white-matter sample points.
    vgm : torch.Tensor, shape (N,)
        Sampled intensities at grey-matter sample points.

    Returns
    -------
    str
        ``'t2'`` or ``'t1'``.
    """
    eps = 1e-6
    mean_intensity = (vgm + vwm) / 2.0 + eps
    percent_contrast = (vgm - vwm) / mean_intensity
    median_contrast = torch.median(percent_contrast).item()
    detected = 't2' if median_contrast >= 0 else 't1'
    logger.info(
        "Auto-detected contrast: %s  (median percent contrast = %.4f)",
        detected, median_contrast
    )
    return detected

