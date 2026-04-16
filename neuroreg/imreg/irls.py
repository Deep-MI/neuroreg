"""
FreeSurfer-style closed-form IRLS rigid registration.

Faithfully implements mri_robust_register's algorithm:

  Outer loop (per pyramid level, max ``nmax`` steps):
      1. Warp source to current space (non-symmetric variant)
      2. Build linear system  A p = b
            A[i,:] = gradient row at voxel i  (6 DOF rigid Jacobian)
            b[i]   = blurred(S - T)[i]
      3. IRLS inner loop (max 20 steps, stop when weighted error *increases*):
            r  ← b           (initial residuals, p = 0)
            loop:
                σ  ← MAD(r) / 0.6745
                w  ← sqrt_tukey(r / σ, sat)   ← sqrt of Tukey weights
                p  ← QR-solve(√W · A, √W · b)
                r  ← b − A p
                ε  ← Σ w² r² / Σ w²
                if ε increased → revert to previous p, w, stop
      4. p → 4 × 4 delta transform via Rodrigues formula
      5. T ← T_delta @ T
      6. Convergence: AffineTransDist(T, T_prev, r=100) < ε_it

References
----------
Reuter et al., NeuroImage 2010.
FreeSurfer source: mri_robust_register/Regression.cpp,
                   RegistrationStep.h, MyMRI.cpp, Transformation.h
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from ..transforms.matrices import matrix_sqrt_schur, params_to_rigid_matrix
from ..transforms.metrics import affine_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FreeSurfer's exact separable 5-tap filters (from MyMRI.cpp)
# ---------------------------------------------------------------------------
_PREFILTER = torch.tensor([0.03504, 0.24878, 0.43234, 0.24878, 0.03504])
_DERFILTER = torch.tensor([-0.10689, -0.28461, 0.0, 0.28461, 0.10689])


def _conv1d_along(
    vol: torch.Tensor,
    kernel: torch.Tensor,
    dim: int,
    padding_mode: str = 'replicate',
) -> torch.Tensor:
    """Convolve a 3-D volume with a 1-D kernel along one spatial dimension.

    Parameters
    ----------
    vol    : [D, H, W]
    kernel : [K]        – 1-D filter
    dim    : 0 (D), 1 (H), or 2 (W)
    """
    K = kernel.shape[0]
    pad = K // 2
    k = kernel.to(dtype=vol.dtype, device=vol.device)

    # Match FreeSurfer MRIconvolve1d: if the convolved dimension has length 1,
    # skip the convolution and return a copy.
    if vol.shape[dim] == 1:
        return vol.clone()

    # Use explicit clamp/replicate padding to avoid artificial edges at the
    # image boundary. This matches FreeSurfer's repeated-edge handling.
    x = vol.unsqueeze(0).unsqueeze(0)

    # reshape kernel to [1, 1, k] and expand to the right conv3d weight shape
    if dim == 0:                         # along depth
        w = k.view(1, 1, K, 1, 1)
        pads = (0, 0, 0, 0, pad, pad)
    elif dim == 1:                       # along height
        w = k.view(1, 1, 1, K, 1)
        pads = (0, 0, pad, pad, 0, 0)
    else:                                # along width
        w = k.view(1, 1, 1, 1, K)
        pads = (pad, pad, 0, 0, 0, 0)

    if padding_mode == 'zeros':
        padded = F.pad(x, pads, mode='constant', value=0.0)
    else:
        padded = F.pad(x, pads, mode=padding_mode)
    return F.conv3d(padded, w).squeeze(0).squeeze(0)


def compute_partials(img: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Compute partial derivatives and blurred image using FreeSurfer's filters.

    Each output is the same shape as *img* [D, H, W].

    Separable scheme (matches MyMRI::getPartials exactly):
        fx   = der_x  ⊗ pre_y  ⊗ pre_z  applied to img
        fy   = pre_x  ⊗ der_y  ⊗ pre_z
        fz   = pre_x  ⊗ pre_y  ⊗ der_z
        blur = pre_x  ⊗ pre_y  ⊗ pre_z

    Returns
    -------
    fx, fy, fz, blur  – each [D, H, W]
    """
    pre = _PREFILTER
    der = _DERFILTER

    # fx: der along W(=x), pre along H(=y), pre along D(=z)
    fx = _conv1d_along(_conv1d_along(_conv1d_along(img, der, 2), pre, 1), pre, 0)

    # fy: pre along W, der along H, pre along D
    fy = _conv1d_along(_conv1d_along(_conv1d_along(img, pre, 2), der, 1), pre, 0)

    # fz: pre along W, pre along H, der along D
    fz = _conv1d_along(_conv1d_along(_conv1d_along(img, pre, 2), pre, 1), der, 0)

    # blur: pre along all three axes
    blur = _conv1d_along(_conv1d_along(_conv1d_along(img, pre, 2), pre, 1), pre, 0)

    return fx, fy, fz, blur


# ---------------------------------------------------------------------------
# Build the linear system  A p = b  for rigid 6-DOF registration
# ---------------------------------------------------------------------------

def construct_Ab(
    src_warped: torch.Tensor,
    trg: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build gradient matrix A and residual vector b.

    Mirrors FreeSurfer's ``RegistrationStep::constructAb``:

    * ``SpTh = (src_warped + trg) / 2``  – average image (at current space)
    * ``SmT  = blur(src_warped − trg)``  – blurred difference → **b**
    * Gradients ``fx, fy, fz`` computed from *SpTh*
    * Each valid voxel contributes one row to A::

        A[i] = [fx, fy, fz,
                fz·y − fy·z,
                fx·z − fz·x,
                fy·x − fx·y]

    Coordinates are centred at the image centre for numerical stability.

    Parameters
    ----------
    src_warped : [D, H, W]  source image warped into target space
    trg        : [D, H, W]  target image
    eps        : threshold for zero-gradient masking

    Returns
    -------
    A     : [N, 6]
    b     : [N]
    valid : [D*H*W] boolean mask indicating which voxels are included
    """
    D, H, W = src_warped.shape

    # Average image for gradients (matches FreeSurfer)
    SpTh = (src_warped + trg) * 0.5
    fx, fy, fz, _ = compute_partials(SpTh)
    
    # Difference for b: compute difference THEN blur (matches FreeSurfer)
    # FreeSurfer: SmT = src - trg; SmT = blur(SmT); b = SmT
    SmT = src_warped - trg
    _, _, _, b_vol = compute_partials(SmT)  # blur of difference

    # Coordinate grids (NO CENTERING - match FreeSurfer)
    # FreeSurfer uses coordinates directly: xp1, yp1, zp1
    z_idx = torch.arange(D, dtype=src_warped.dtype, device=src_warped.device)
    y_idx = torch.arange(H, dtype=src_warped.dtype, device=src_warped.device)
    x_idx = torch.arange(W, dtype=src_warped.dtype, device=src_warped.device)
    # meshgrid: z[D], y[H], x[W] → each [D, H, W]
    gz, gy, gx = torch.meshgrid(z_idx, y_idx, x_idx, indexing='ij')

    # Flatten everything
    fxf = fx.flatten()
    fyf = fy.flatten()
    fzf = fz.flatten()
    xf  = gx.flatten()
    yf  = gy.flatten()
    zf  = gz.flatten()
    bf  = b_vol.flatten()
    src_flat = src_warped.flatten()
    trg_flat = trg.flatten()

    # Valid mask: FreeSurfer checks for:
    # 1. Outside values (near zero or background)
    # 2. NaN gradients
    # 3. Near-zero gradients
    # 4. Finite values
    
    # Check for outside/background values (typically 0)
    # FreeSurfer uses: fabs(val - outside_val) > eps
    # For typical images, outside_val = 0, so this checks |val| > eps
    outside_eps = 1e-5
    valid = (src_flat.abs() > outside_eps) & (trg_flat.abs() > outside_eps)
    
    # Non-zero gradient
    valid &= (fxf.abs() + fyf.abs() + fzf.abs()) > eps
    
    # Finite values
    valid &= torch.isfinite(fxf) & torch.isfinite(fyf) & torch.isfinite(fzf)
    valid &= torch.isfinite(bf)

    fxv = fxf[valid]
    fyv = fyf[valid]
    fzv = fzf[valid]
    xv  = xf[valid]
    yv  = yf[valid]
    zv  = zf[valid]
    bv  = bf[valid]

    # Rigid Jacobian: [tx, ty, tz, rx, ry, rz]
    A = torch.stack([
        fxv,
        fyv,
        fzv,
        fzv * yv - fyv * zv,   # ∂/∂rx
        fxv * zv - fzv * xv,   # ∂/∂ry
        fyv * xv - fxv * yv,   # ∂/∂rz
    ], dim=1)                   # [N, 6]

    return A, bv, valid


# ---------------------------------------------------------------------------
# Weighted least squares solve (QR, matching FreeSurfer's getWeightedLSEst)
# ---------------------------------------------------------------------------

def solve_wls(
    A: torch.Tensor,
    b: torch.Tensor,
    w_sqrt: torch.Tensor,
) -> torch.Tensor:
    """Solve  (√W A) p = √W b  via QR / least-squares.

    Parameters
    ----------
    A      : [N, DOF]
    b      : [N]
    w_sqrt : [N]   sqrt of weights  (as in FreeSurfer's getSqrtTukeyDiaWeights)

    Returns
    -------
    p : [DOF]
    """
    Aw = A * w_sqrt.unsqueeze(1)   # [N, DOF]
    bw = b * w_sqrt                # [N]
    # torch.linalg.lstsq is QR-backed, equivalent to vnl_qr
    result = torch.linalg.lstsq(Aw, bw, driver='gelsd')
    return result.solution


# ---------------------------------------------------------------------------
# Tukey sqrt-weights  (FreeSurfer: getSqrtTukeyDiaWeights)
# w = 1 − (r/sat)²  for |r| < sat,  else 0
# These are the SQUARE-ROOT of the standard Tukey biweights.
# ---------------------------------------------------------------------------

def _sqrt_tukey(r: torch.Tensor, sat: float) -> torch.Tensor:
    t = r / sat
    w = torch.clamp(1.0 - t * t, min=0.0)
    return w


# ---------------------------------------------------------------------------
# IRLS inner loop  (matches Regression::getRobustEstWAB)
# ---------------------------------------------------------------------------

def irls_inner_loop(
    A: torch.Tensor,
    b: torch.Tensor,
    sat: float = 4.685,
    max_iterations: int = 20,
    eps: float = 2e-12,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """IRLS inner loop: iteratively re-weighted least squares.

    Exactly mirrors ``Regression<T>::getRobustEstWAB``:

    * Residuals start at ``r = b``  (p = 0)
    * Loop:
        1. ``σ = MAD(r)``
        2. ``w_sqrt = sqrt_tukey(r / σ, sat)``
        3. ``p = solve_wls(A, b, w_sqrt)``
        4. ``r = b − A p``
        5. ``ε = Σ w² r² / Σ w²``
        6. If ε increased → revert, stop

    Parameters
    ----------
    A   : [N, 6]
    b   : [N]
    sat : Tukey saturation threshold (default 4.685)

    Returns
    -------
    p       : [6]  parameter vector
    w_sqrt  : [N]  final sqrt-weights (for diagnostics / weight image)
    sigma   : float  final MAD-based scale estimate (normalised intensity units)
    err     : float  final weighted residual  Σw²r² / Σw²  (used for outer-loop cost)
    """
    N = b.shape[0]
    r = b.clone()                    # initial residuals: p = 0

    p       = b.new_zeros(A.shape[1])
    w_sqrt  = b.new_ones(N)
    err_prev = float('inf')
    err_cur  = float('inf')
    sigma    = torch.tensor(1.0)   # fallback

    p_last = p.clone()
    w_last = w_sqrt.clone()

    for iteration in range(max_iterations):
        # --- 1. Robust scale  (FreeSurfer MAD: median(|r - median(r)|) / 0.6745) ---
        r_median = torch.median(r)
        sigma = torch.median((r - r_median).abs()) / 0.6745
        if sigma < eps:
            err_cur = float((r * r).mean().item()) if N > 0 else 0.0
            if verbose:
                logger.debug("    IRLS: sigma too small (identical images?), stopping")
            break

        # --- 2. sqrt-Tukey weights on r/σ ---
        w_sqrt = _sqrt_tukey(r / sigma, sat)

        # --- 3. Weighted solve ---
        p = solve_wls(A, b, w_sqrt)

        # --- 4. New residuals ---
        r = b - A @ p

        # --- 5. Weighted error  ε = Σw²r² / Σw² ---
        w2  = w_sqrt * w_sqrt
        sw  = w2.sum()
        swr = (w2 * r * r).sum()
        if sw > 0:
            err_cur = (swr / sw).item()
        else:
            err_cur = float('inf')

        if verbose:
            zero_frac = (w_sqrt == 0).float().mean().item()
            logger.debug(
                "    IRLS iter %2d: err=%.6e  sigma=%.4f  "
                "mean_w=%.4f  outliers=%.1f%%",
                iteration + 1, err_cur, sigma.item(),
                w_sqrt.mean().item(), zero_frac * 100,
            )

        # --- 6. Stop if error increased ---
        if err_cur > err_prev:
            # revert to previous
            p      = p_last
            w_sqrt = w_last
            if verbose:
                logger.debug("    IRLS: error increased, reverting to iter %d", iteration)
            break

        if err_cur < eps:
            if verbose:
                logger.debug("    IRLS: converged at iter %d", iteration + 1)
            break

        # save for potential revert
        p_last     = p.clone()
        w_last     = w_sqrt.clone()
        err_prev   = err_cur

    return p, w_sqrt, float(sigma), float(err_cur)


# ---------------------------------------------------------------------------
# One full registration step (build Ab + IRLS)
# ---------------------------------------------------------------------------

def register_step(
    src_warped: torch.Tensor,
    trg: torch.Tensor,
    sat: float = 4.685,
    max_irls: int = 20,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Build A, b then run IRLS.  Returns (p, w_sqrt, valid_mask, sigma, err)."""
    A, b, valid = construct_Ab(src_warped, trg)
    p, w_sqrt, sigma, err = irls_inner_loop(A, b, sat=sat, max_iterations=max_irls,
                                            verbose=verbose)
    return p, w_sqrt, valid, sigma, err


# ---------------------------------------------------------------------------
# Full IRLS registration with image warping and convergence check
# ---------------------------------------------------------------------------

def register_irls(
    src: torch.Tensor,
    trg: torch.Tensor,
    initial_transform: torch.Tensor | None = None,
    nmax: int = 5,
    sat: float = 4.685,
    epsit: float = 0.01,
    max_irls: int = 20,
    symmetric: bool = True,
    adaptive_sat: bool = False,
    target_outlier_pct: float = 5.0,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """IRLS rigid registration at a single resolution level.

    Mirrors ``RegRobust::iterativeRegistrationHelper``.

    Parameters
    ----------
    src               : [D, H, W]  source image (float32)
    trg               : [D, H, W]  target image
    initial_transform : 4x4 vox-to-vox matrix (default: identity)
    nmax              : maximum outer-loop iterations (default 5)
    sat               : Tukey saturation threshold (default 4.685)
    epsit             : convergence threshold for affine distance (default 0.01)
    max_irls          : maximum IRLS inner iterations per step (default 20)
    symmetric         : if True, use symmetric (midspace) mode where both images
                        are warped half-way; if False, use directed mode where
                        only source is warped to target space (default True)
    adaptive_sat      : if True, increase sat when outliers exceed target (default False)
    target_outlier_pct: target outlier percentage (default 5.0%)
    verbose           : print per-iteration info

    Returns
    -------
    T    : [4, 4]  final vox-to-vox transform
    info : dict with keys 'iterations', 'converged', 'dists', 'weights', 'valid_mask', 'sigma_hist'
    """
    from ..image.map import map as map_image  # lazy import to avoid circular

    T = initial_transform.clone() if initial_transform is not None \
        else torch.eye(4, dtype=src.dtype)

    # Normalise both images to roughly [0, 1] so that residuals and σ are on a
    # consistent scale regardless of scanner intensity range.  Identical to
    # training_loop's normalise step: divide by the 99.5th percentile of the target.
    scale = torch.quantile(trg.reshape(-1).abs(), 0.995).clamp(min=1.0)
    src_n = src / scale
    trg_n = trg / scale

    src_shape: tuple[int, int, int] = (int(src_n.shape[0]), int(src_n.shape[1]), int(src_n.shape[2]))
    trg_shape: tuple[int, int, int] = (int(trg_n.shape[0]), int(trg_n.shape[1]), int(trg_n.shape[2]))

    info = dict(iterations=0, converged=False, dists=[], weights=None,
                valid_mask=None, image_shape=trg_shape, sigma_hist=[])

    # Track the T with the lowest inner-loop weighted cost across all outer
    # iterations.  At coarse levels the cost steadily falls; at the finest
    # level tiny oscillations can occur — keeping the best avoids storing a
    # slightly diverged final step.
    best_T      = T.clone()
    best_w_sqrt = None
    best_valid  = None
    best_err    = float('inf')
    
    # Adaptive sat: start with provided value, increase if outliers too high
    current_sat = sat

    for i in range(nmax):
        T_prev = T.clone()
        mh = None
        mhi = None

        # Warp images based on mode
        if symmetric:
            # Symmetric mode: compute midspace transforms and warp both images
            mh, mhi = matrix_sqrt_schur(T)
            
            src_warped = map_image(
                src_n, mh,
                is_torch_mat=False,
                target_shape=src_shape,
                mode='bilinear',
                padding_mode='zeros',
            ).float()
            
            trg_warped = map_image(
                trg_n, mhi,
                is_torch_mat=False,
                target_shape=src_shape,
                mode='bilinear',
                padding_mode='zeros',
            ).float()
            
            # Build system in midspace
            A, b, valid = construct_Ab(src_warped, trg_warped)
        else:
            # Directed mode: warp source to target space
            src_warped = map_image(
                src_n, T,
                is_torch_mat=False,
                target_shape=trg_shape,
                mode='bilinear',
                padding_mode='zeros',
            ).float()
            
            # Build system in target space
            A, b, valid = construct_Ab(src_warped, trg_n)

        # Solve IRLS on normalised images
        p, w_sqrt, sigma_val, err_val = irls_inner_loop(
            A, b, sat=current_sat, max_iterations=max_irls, verbose=verbose)
        info['sigma_hist'].append(sigma_val)
        
        # Check outlier percentage for adaptive sat adjustment
        zero_pct = (w_sqrt == 0).float().mean().item() * 100
        
        if adaptive_sat:
            # Bidirectional adaptive sat: adjust proportionally to error
            error = zero_pct - target_outlier_pct
            tolerance = 1.0  # Only adjust if error > 1%
            
            if abs(error) > tolerance:
                old_sat = current_sat
                
                # Proportional adjustment: bigger error = bigger change
                # Scale factor: 10% error → ~10% adjustment (clamped)
                # This is much gentler than the previous 50% jump
                adjustment_pct = error / 10.0  # 10% error → 10% adjustment
                adjustment_pct = max(-0.20, min(0.25, adjustment_pct))  # Limit: -20% to +25%
                current_sat = current_sat * (1.0 + adjustment_pct)
                
                # Keep sat within reasonable bounds
                current_sat = max(3.0, min(15.0, current_sat))
                
                if verbose:
                    direction = "increasing" if current_sat > old_sat else "decreasing"
                    logger.info(
                        "  Adaptive sat: outliers %.1f%% vs target %.1f%% (error %+.1f%%), "
                        "%s sat %.2f → %.2f",
                        zero_pct, target_outlier_pct, error, direction, old_sat, current_sat
                    )
            # Re-solve with new sat (A, b already built above)
            p, w_sqrt, sigma_val, err_new = irls_inner_loop(
                A, b, sat=current_sat, max_iterations=max_irls, 
                verbose=verbose
            )
            info['sigma_hist'][-1] = sigma_val
            err_val = err_new
            zero_pct = (w_sqrt == 0).float().mean().item() * 100

        # Compose transforms based on mode
        if symmetric:
            # Symmetric mode: M_new = inv(mhi) @ δ @ mh
            assert mh is not None and mhi is not None
            delta = params_to_rigid_matrix(p.float())
            mh2 = torch.inverse(mhi.double())  # = mh (by construction)
            T = mh2.float() @ delta @ mh
        else:
            # Directed mode: T_new = T_delta @ T_old
            T_delta = params_to_rigid_matrix(p.float())
            T = T_delta @ T_prev

        # Convergence metric (Jenkinson affine RMS distance on successive updates)
        dist = affine_dist(T, T_prev, radius=100.0)
        info['dists'].append(dist)
        info['iterations'] = i + 1

        if verbose:
            logger.info(
                "  Iter %2d: AffineTransDist=%.4f  sigma=%.4f  outliers=%.1f%%  sat=%.2f",
                i + 1, dist, sigma_val, zero_pct, current_sat,
            )

        # Keep best T by minimum inner-loop cost (no early stopping —
        # FreeSurfer's outer loop runs all nmax iterations and only stops
        # on AffineTransDist convergence).
        if err_val < best_err:
            best_err    = err_val
            best_T      = T.clone()
            best_w_sqrt = w_sqrt.clone()
            best_valid  = valid.clone()

        if dist <= epsit:
            info['converged'] = True
            if verbose:
                logger.info("  Converged after %d iterations (dist=%.4f)", i + 1, dist)
            break

    # Return the best T found (usually the same as final, guards against
    # minor oscillation at the finest level).
    T = best_T
    info['weights']    = best_w_sqrt
    info['valid_mask'] = best_valid
    return T, info

