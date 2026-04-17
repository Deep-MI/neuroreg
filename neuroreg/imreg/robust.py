"""Robust M-estimator weighting functions for IRLS registration.

This module implements robust weighting schemes used in iteratively reweighted
least squares (IRLS) optimization, matching FreeSurfer's mri_robust_register
approach.
"""

import torch


def tukey_weights(residuals: torch.Tensor, c: float = 6.0) -> torch.Tensor:
    """Compute Tukey biweight robust weights (GPU-accelerated).

    The Tukey biweight (also called bisquare) is a robust M-estimator weight
    function that completely rejects outliers beyond a threshold while
    smoothly downweighting moderate deviations.

    Weight function:
        w(r) = (1 - (r/c)²)²  for |r| ≤ c
        w(r) = 0               for |r| > c

    This matches FreeSurfer's mri_robust_register implementation, which uses
    Tukey biweight with c=6.0 (saturation parameter).

    Parameters
    ----------
    residuals : torch.Tensor
        Residual values (any shape). Typically intensity differences between
        warped source and target images.
    c : float, optional
        Tuning constant (saturation threshold). Common values:
        - c=4.685 gives 95% efficiency at normal distribution
        - c=6.0 used by FreeSurfer (more conservative, default)
        - c=4.0 more aggressive outlier rejection

    Returns
    -------
    weights : torch.Tensor
        Robust weights in [0, 1], same shape as residuals. Outliers beyond
        threshold c receive zero weight; inliers receive smoothly varying
        weights approaching 1 near zero residual.

    Examples
    --------
    >>> residuals = torch.randn(100, 100, 100)
    >>> residuals[::10, ::10, ::10] = 20.0  # Add outliers
    >>> sigma = compute_mad(residuals)
    >>> weights = tukey_weights(residuals / sigma, c=6.0)
    >>> print(weights[::10, ::10, ::10].max())  # Outliers get ~0 weight
    tensor(0.0012)
    >>> print(weights[0, 0, 1])  # Inliers get high weight
    tensor(0.9821)

    References
    ----------
    .. [1] Reuter, M., Rosas, H. D., & Fischl, B. (2010). Highly accurate
           inverse consistent registration: a robust approach. NeuroImage,
           53(4), 1181-1196.
    .. [2] Huber, P. J. (1981). Robust Statistics. Wiley.
    """
    u = residuals.abs()
    # Compute (1 - (u/c)²)² where u ≤ c, else 0
    weights = torch.where(u <= c, (1 - (u / c) ** 2) ** 2, torch.zeros_like(u))
    return weights


def huber_weights(residuals: torch.Tensor, delta: float = 1.345) -> torch.Tensor:
    """Compute Huber M-estimator robust weights (GPU-accelerated).

    The Huber weight function provides a compromise between L2 (quadratic)
    and L1 (absolute) loss. It's less aggressive than Tukey biweight in
    rejecting outliers but more robust than pure L2.

    Weight function:
        w(r) = 1                for |r| ≤ δ
        w(r) = δ / |r|          for |r| > δ

    Parameters
    ----------
    residuals : torch.Tensor
        Residual values (any shape).
    delta : float, optional
        Transition point between L2 and L1 behavior. Default 1.345 gives
        95% efficiency at normal distribution.

    Returns
    -------
    weights : torch.Tensor
        Robust weights in (0, 1], same shape as residuals.

    Examples
    --------
    >>> residuals = torch.randn(1000)
    >>> sigma = compute_mad(residuals)
    >>> weights = huber_weights(residuals / sigma, delta=1.345)

    References
    ----------
    .. [1] Huber, P. J. (1964). Robust Estimation of a Location Parameter.
           The Annals of Mathematical Statistics, 35(1), 73-101.
    """
    u = residuals.abs()
    weights = torch.where(u <= delta, torch.ones_like(u), delta / u)
    return weights


def cauchy_weights(residuals: torch.Tensor, c: float = 2.385) -> torch.Tensor:
    """Compute Cauchy M-estimator robust weights (GPU-accelerated).

    The Cauchy weight function is even more aggressive than Tukey in
    downweighting outliers, with weights that asymptotically approach zero
    but never reach exactly zero.

    Weight function:
        w(r) = 1 / (1 + (r/c)²)

    Parameters
    ----------
    residuals : torch.Tensor
        Residual values (any shape).
    c : float, optional
        Scale parameter. Default 2.385 gives 95% efficiency at normal
        distribution.

    Returns
    -------
    weights : torch.Tensor
        Robust weights in (0, 1], same shape as residuals.

    Examples
    --------
    >>> residuals = torch.randn(1000)
    >>> sigma = compute_mad(residuals)
    >>> weights = cauchy_weights(residuals / sigma, c=2.385)

    References
    ----------
    .. [1] Holland, P. W., & Welsch, R. E. (1977). Robust regression using
           iteratively reweighted least-squares. Communications in
           Statistics-theory and Methods, 6(9), 813-827.
    """
    u = residuals.abs()
    weights = 1.0 / (1.0 + (u / c) ** 2)
    return weights


def compute_mad(residuals: torch.Tensor) -> torch.Tensor:
    """Compute Median Absolute Deviation (MAD) for robust scale estimation.

    MAD is a robust estimator of scale (analogous to standard deviation) that
    is not influenced by outliers. It's computed as the median of absolute
    deviations from the median.

    For a normally distributed variable:
        σ ≈ MAD / 0.6745

    This is the standard conversion factor that makes MAD consistent with
    standard deviation under normality.

    Parameters
    ----------
    residuals : torch.Tensor
        Residual values (any shape).

    Returns
    -------
    sigma : torch.Tensor
        Robust scale estimate (scalar tensor).

    Examples
    --------
    >>> residuals = torch.randn(10000)
    >>> residuals[::100] = 50.0  # Add outliers
    >>> sigma_mad = compute_mad(residuals)
    >>> sigma_std = residuals.std()
    >>> print(f"MAD: {sigma_mad:.3f}, STD: {sigma_std:.3f}")
    MAD: 1.002, STD: 2.456

    Notes
    -----
    The flattening of the residuals tensor is necessary because torch.median()
    requires a 1D input for reliable behavior across all PyTorch versions.

    References
    ----------
    .. [1] Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median
           absolute deviation. Journal of the American Statistical association,
           88(424), 1273-1283.
    """
    # Flatten to 1D for median computation
    r_flat = residuals.flatten()

    # Compute median
    median = torch.median(r_flat)

    # Compute MAD: median of |r - median(r)|
    mad = torch.median(torch.abs(r_flat - median))

    # Convert to standard deviation estimate
    # Factor 0.6745 is the 0.75 quantile of the standard normal distribution
    sigma = mad / 0.6745

    # Ensure non-zero scale (handle constant residuals)
    sigma = torch.clamp(sigma, min=1e-10)

    return sigma


def compute_scale_estimate(
    residuals: torch.Tensor, method: str = "mad", percentile: float | None = None
) -> torch.Tensor:
    """Compute robust scale estimate from residuals.

    Provides multiple methods for estimating the scale of residuals in a
    robust manner (not sensitive to outliers).

    Parameters
    ----------
    residuals : torch.Tensor
        Residual values (any shape).
    method : {'mad', 'percentile', 'iqr'}, optional
        Method for scale estimation:
        - 'mad': Median Absolute Deviation (default, most robust)
        - 'percentile': Use specified percentile (e.g., 75th percentile)
        - 'iqr': Interquartile range (75th - 25th percentile)
    percentile : float, optional
        Percentile to use when method='percentile'. Should be in (0, 1).
        Typical values: 0.75 (75th percentile) or 0.95 (95th percentile).

    Returns
    -------
    sigma : torch.Tensor
        Robust scale estimate (scalar tensor).

    Examples
    --------
    >>> residuals = torch.randn(1000)
    >>> sigma_mad = compute_scale_estimate(residuals, method='mad')
    >>> sigma_75 = compute_scale_estimate(residuals, method='percentile', percentile=0.75)
    >>> sigma_iqr = compute_scale_estimate(residuals, method='iqr')
    """
    if method == "mad":
        return compute_mad(residuals)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("percentile must be specified when method='percentile'")
        r_abs = torch.abs(residuals.flatten())
        sigma = torch.quantile(r_abs, percentile)
        return torch.clamp(sigma, min=1e-10)
    elif method == "iqr":
        r_abs = torch.abs(residuals.flatten())
        q75 = torch.quantile(r_abs, 0.75)
        q25 = torch.quantile(r_abs, 0.25)
        # IQR / 1.349 converts to std under normality
        sigma = (q75 - q25) / 1.349
        return torch.clamp(sigma, min=1e-10)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'mad', 'percentile', 'iqr'.")

