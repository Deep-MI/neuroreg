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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FreeSurfer's exact separable 5-tap filters (from MyMRI.cpp)
# ---------------------------------------------------------------------------
_PREFILTER = torch.tensor([0.03504, 0.24878, 0.43234, 0.24878, 0.03504])
_DERFILTER  = torch.tensor([-0.10689, -0.28461, 0.0, 0.28461, 0.10689])


def _conv1d_along(vol: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
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

    # reshape kernel to [1, 1, k] and expand to the right conv3d weight shape
    if dim == 0:                         # along depth
        w = k.view(1, 1, K, 1, 1)
        p = (pad, 0, 0)
    elif dim == 1:                       # along height
        w = k.view(1, 1, 1, K, 1)
        p = (0, pad, 0)
    else:                                # along width
        w = k.view(1, 1, 1, 1, K)
        p = (0, 0, pad)

    return F.conv3d(vol.unsqueeze(0).unsqueeze(0), w, padding=p).squeeze(0).squeeze(0)


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
    outside_eps = 1e-4
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
) -> tuple[torch.Tensor, torch.Tensor, float]:
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
# Parameter vector → 4 × 4 matrix  (Rodrigues rotation vector, like
#   Transform3dRigid::getMatrix via Quaternion::importRotVec)
# ---------------------------------------------------------------------------

def rotation_vector_to_matrix(rv: torch.Tensor) -> torch.Tensor:
    """Convert a rotation vector to a 3 3 rotation matrix (Rodrigues).

    Parameters
    ----------
    rv : [3]  rotation vector (axis × angle)

    Returns
    -------
    R : [3, 3]
    """
    theta = torch.norm(rv)
    if theta < 1e-10:
        return torch.eye(3, dtype=rv.dtype, device=rv.device)
    axis = rv / theta
    K = torch.zeros(3, 3, dtype=rv.dtype, device=rv.device)
    K[0, 1] = -axis[2]
    K[0, 2] =  axis[1]
    K[1, 0] =  axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] =  axis[0]
    R = (torch.eye(3, dtype=rv.dtype, device=rv.device)
         + torch.sin(theta) * K
         + (1.0 - torch.cos(theta)) * (K @ K))
    return R


def params_to_rigid_matrix(p: torch.Tensor) -> torch.Tensor:
    """Convert 6-DOF rigid parameter vector to 4×4 homogeneous matrix
    in **DHW (nibabel) axis order** as expected by :func:`map_image`.

    The IRLS linear system associates:

    * ``p[0]`` ↔ W  (physical x, image dim 2)  –  ``fx``-column
    * ``p[1]`` ↔ H  (physical y, image dim 1)  –  ``fy``-column
    * ``p[2]`` ↔ D  (physical z, image dim 0)  –  ``fz``-column
    * ``p[3]`` ↔ rotation around W axis
    * ``p[4]`` ↔ rotation around H axis
    * ``p[5]`` ↔ rotation around D axis

    The rotation matrix is built in physical XYZ order first (using
    Rodrigues) and then permuted to DHW by swapping axes 0 ↔ 2.

    Parameters
    ----------
    p : [6]  [tx_W, ty_H, tz_D, rx_W, ry_H, rz_D]

    Returns
    -------
    T : [4, 4]  transform matrix in DHW nibabel axis order
    """
    T = torch.eye(4, dtype=p.dtype, device=p.device)

    # Rotation: build in physical XYZ, then permute rows/cols to DHW
    # XYZ → DHW: swap index 0 (x=W→D) and index 2 (z=D→W), keep 1 (y=H)
    R_xyz = rotation_vector_to_matrix(p[3:6])
    ii = [2, 1, 0]
    T[:3, :3] = R_xyz[ii][:, ii]

    # Translation: p[0]=tx lives in W (dim 2), p[2]=tz lives in D (dim 0)
    T[0, 3] = p[2]   # D row  ← tz (fz parameter)
    T[1, 3] = p[1]   # H row  ← ty (fy parameter)
    T[2, 3] = p[0]   # W row  ← tx (fx parameter)

    return T


# ---------------------------------------------------------------------------
# Convergence metric  (MyMatrix::AffineTransDistSq with r=100)
# ---------------------------------------------------------------------------

def affine_trans_dist(T_new: torch.Tensor, T_old: torch.Tensor,
                      r: float = 100.0) -> float:
    """Affine transformation distance used for convergence.

    ``D² = (1/5) r² · trace(ΔR^T ΔR) + ‖Δt‖²``
    ``D  = √D²``

    Matches ``MyMatrix::AffineTransDistSq(a, b, r=100)`` followed by sqrt.
    """
    dT = T_new.double() - T_old.double()
    tdq = (dT[:3, 3] ** 2).sum()
    dR  = dT[:3, :3]
    tr  = torch.trace(dR.T @ dR)
    return float(torch.sqrt(tr * r * r / 5.0 + tdq))


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
    sat: float = 6.0,
    epsit: float = 0.01,
    max_irls: int = 20,
    symmetric: bool = False,
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
    initial_transform : 4×4 vox-to-vox matrix (default: identity)
    nmax              : maximum outer-loop iterations (default 5)
    sat               : Tukey saturation threshold (default 4.685)
    epsit             : convergence threshold for affine distance (default 0.01)
    max_irls          : maximum IRLS inner iterations per step (default 20)
    symmetric         : if True, use symmetric (midspace) mode where both images
                        are warped half-way; if False, use directed mode where
                        only source is warped to target space (default False)
    adaptive_sat      : if True, increase sat when outliers exceed target (default True)
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

    info = dict(iterations=0, converged=False, dists=[], weights=None,
                valid_mask=None, image_shape=tuple(trg.shape), sigma_hist=[])

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

        # Warp images based on mode
        if symmetric:
            # Symmetric mode: compute midspace transforms and warp both images
            from ..transforms import matrix_sqrt_schur
            mh, mhi = matrix_sqrt_schur(T)
            
            src_warped = map_image(
                src_n, mh,
                is_torch_mat=False,
                target_shape=tuple(src_n.shape),
                mode='bilinear',
                padding_mode='zeros',
            ).float()
            
            trg_warped = map_image(
                trg_n, mhi,
                is_torch_mat=False,
                target_shape=tuple(src_n.shape),
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
                target_shape=tuple(trg_n.shape),
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
            p, w_sqrt, err_new = irls_inner_loop(
                A, b, sat=current_sat, max_iterations=max_irls, 
                verbose=verbose
            )[:3]  # Take only p, w_sqrt, sigma (skip err)
            err_val = err_new
            zero_pct = (w_sqrt == 0).float().mean().item() * 100

        # Compose transforms based on mode
        if symmetric:
            # Symmetric mode: M_new = inv(mhi) @ δ @ mh
            delta = params_to_rigid_matrix(p.float())
            mh2 = torch.inverse(mhi.double())  # = mh (by construction)
            T = mh2.float() @ delta @ mh
        else:
            # Directed mode: T_new = T_delta @ T_old
            T_delta = params_to_rigid_matrix(p.float())
            T = T_delta @ T_prev

        # Convergence metric (AffineTransDist)
        dist = affine_trans_dist(T, T_prev, r=100.0)
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


# ---------------------------------------------------------------------------
# Simple Gaussian pyramid builder (self-contained, no affine needed)
# ---------------------------------------------------------------------------

def _build_pyramid(img: torch.Tensor) -> list[torch.Tensor]:
    """Build a Gaussian pyramid from finest (index 0) to coarsest.

    Uses the same prefilter as FreeSurfer (getBlur) before each 2× downsample,
    mirroring ``MyMRI::subSample``.

    Returns
    -------
    levels : list[Tensor]   levels[0] = full-res (smoothed once), levels[k] = 2^k ×
             downsampled.  Each level has min dim ≥ 2.
    """
    levels = []
    cur = _conv1d_along(_conv1d_along(_conv1d_along(
        img.float(), _PREFILTER, 2), _PREFILTER, 1), _PREFILTER, 0)
    levels.append(cur)
    while min(cur.shape) >= 8:
        # smooth then 2× subsample
        blurred = _conv1d_along(_conv1d_along(_conv1d_along(
            cur, _PREFILTER, 2), _PREFILTER, 1), _PREFILTER, 0)
        cur = blurred[::2, ::2, ::2]
        levels.append(cur)
    return levels   # levels[0] finest … levels[-1] coarsest


# ---------------------------------------------------------------------------
# Multi-resolution (pyramid) IRLS registration
# ---------------------------------------------------------------------------

def register_irls_pyramid(
    src: torch.Tensor,
    trg: torch.Tensor,
    src_affine: torch.Tensor | None = None,
    trg_affine: torch.Tensor | None = None,
    initial_transform: torch.Tensor | None = None,
    min_voxels: int = 16,
    max_voxels: int = 64,
    nmax: int = 5,
    sat: float = 6.0,
    epsit: float = 0.01,
    max_irls: int = 20,
    isotropic: bool = True,
    symmetric: bool = False,
    adaptive_sat: bool = False,
    target_outlier_pct: float = 5.0,
    outliers_name: str | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, list]:
    """Pyramid IRLS registration (coarse-to-fine).

    Mirrors the multi-resolution loop in ``RegRobust::findSatMultiRes``:

    * Build Gaussian pyramid for both images (coarsest ~ max_voxels)
    * Register coarse → fine, propagating the transform at each level
    * Translation is scaled by 2 at each finer step; rotation is unchanged

    Parameters
    ----------
    src, trg          : [D, H, W]  full-resolution float32 images
    src_affine, trg_affine : 4×4 voxel-to-RAS affines (required if isotropic=True)
    initial_transform : 4×4 initial vox-to-vox (default: identity)
    min_voxels        : minimum side length to use a level (default 16)
    max_voxels        : start at the level whose max-dim first reaches this
                        (default 64, matching FreeSurfer)
    nmax              : max outer IRLS iterations per level (default 5)
    sat               : Tukey threshold (default 4.685)
    epsit             : AffineTransDist convergence threshold (default 0.01)
    max_irls          : max IRLS inner iterations (default 20)
    isotropic         : if True, resample to isotropic voxels before registration
                        (default True, matching FreeSurfer)
    symmetric         : if True, use symmetric (midspace) mode where both images
                        are warped half-way; if False, use directed mode where
                        only source is warped to target space (default False)
    adaptive_sat      : if True, increase sat when outliers exceed target (default True)
    target_outlier_pct: target outlier percentage (default 5.0%)
    outliers_name     : str, optional
                        If provided, save the outlier map (1 - Tukey weights) to this file.
                        High values indicate poorly registered voxels (outliers), low values
                        indicate well-registered voxels. Format auto-detected from extension
                        (.nii, .nii.gz, or .mgz).
    verbose           : print progress

    Returns
    -------
    T        : [4, 4]  final vox-to-vox transform in original voxel space
    all_info : list of info dicts (one per level, finest last)
    """
    # Isotropic resampling: resample both images to cubic voxels using the
    # finest voxel size, matching FreeSurfer's internal preprocessing.
    if isotropic:
        if src_affine is None or trg_affine is None:
            raise ValueError("src_affine and trg_affine required when isotropic=True")
        
        # Compute isotropic voxel size = max of finest resolutions (like register_pyramid)
        import numpy as np
        src_zooms = np.linalg.norm(src_affine.numpy()[:3, :3], axis=0)
        trg_zooms = np.linalg.norm(trg_affine.numpy()[:3, :3], axis=0)
        isosize = float(max(src_zooms.min(), trg_zooms.min()))
        
        if verbose:
            logger.info("Isotropic resampling: isosize=%.4f mm", isosize)
        
        # Resample using the same function as register_pyramid
        from ..image.map import resample_isotropic_tensor
        
        src_iso, src_iso_aff, Rsrc = resample_isotropic_tensor(
            src, src_affine.numpy(), isosize, mode='bilinear')
        trg_iso, trg_iso_aff, Rtrg = resample_isotropic_tensor(
            trg, trg_affine.numpy(), isosize, mode='bilinear')
        
        if verbose:
            logger.info("  Src resampled: %s → %s", src.shape, src_iso.shape)
            logger.info("  Trg resampled: %s → %s", trg.shape, trg_iso.shape)
        
        # Convert initial_transform to isotropic space if provided
        if initial_transform is not None:
            # T_iso_v2v = Rtrg @ T_orig_v2v @ inv(Rsrc)
            T_iso = Rtrg.double() @ initial_transform.double() @ torch.inverse(Rsrc.double())
        else:
            T_iso = None
        
        # Register in isotropic space
        pyramid_src = _build_pyramid(src_iso)
        pyramid_trg = _build_pyramid(trg_iso)
        iso_affine = trg_iso_aff   # saved for weight NIfTI headers
    else:
        # No resampling - work in original voxel space
        pyramid_src = _build_pyramid(src)
        pyramid_trg = _build_pyramid(trg)
        T_iso = initial_transform
        Rsrc = torch.eye(4, dtype=torch.float32)
        Rtrg = torch.eye(4, dtype=torch.float32)
        iso_affine = trg_affine.numpy() if trg_affine is not None else None

    # Pick levels: Start from the finest level where max-dim ≤ max_voxels,
    # then work down to level 0 (full resolution), respecting min_voxels.
    # This matches FreeSurfer's coarse-to-fine strategy.
    chosen: list[int] = []
    start_level = None
    
    # Find the finest (smallest index) level with max-dim ≤ max_voxels
    for lvl in range(len(pyramid_src)):
        s = pyramid_src[lvl]
        if max(s.shape) <= max_voxels:
            start_level = lvl
            break
    
    # If no level is <= max_voxels, start from the coarsest available
    if start_level is None:
        start_level = len(pyramid_src) - 1
    
    # Collect levels from start_level down to 0 (finest), respecting min_voxels
    for lvl in range(start_level, -1, -1):
        s = pyramid_src[lvl]
        if max(s.shape) < min_voxels:
            continue
        chosen.append(lvl)
    
    # chosen is already coarse-to-fine
    chosen_coarse_first = chosen

    # Initialize T (handle case where T_iso is None)
    T = T_iso if T_iso is not None else torch.eye(4, dtype=torch.float32)
    all_info: list[dict] = []

    for lvl in chosen_coarse_first:
        s = pyramid_src[lvl].float()
        t = pyramid_trg[lvl].float()
        scale = float(2 ** lvl)          # voxel size ratio vs full-res

        # Scale translation to this level's voxel space
        T_lvl = T.clone()
        T_lvl[:3, 3] = T[:3, 3] / scale

        if verbose:
            logger.info("Pyramid level %d  shape=%s  (scale ×1/%d)",
                        lvl, list(s.shape), int(scale))

        T_lvl, info = register_irls(
            s, t,
            initial_transform=T_lvl,
            nmax=nmax, sat=sat, epsit=epsit, max_irls=max_irls,
            symmetric=symmetric,
            adaptive_sat=adaptive_sat, target_outlier_pct=target_outlier_pct,
            verbose=verbose,
        )

        # Scale translation back to full-resolution voxels
        T_lvl[:3, 3] = T_lvl[:3, 3] * scale
        T = T_lvl
        info['iso_affine'] = iso_affine   # affine for weight NIfTI (isotropic space)
        all_info.append(info)

    # Convert back from isotropic voxel space to original voxel space
    if isotropic:
        # Transform chain: src_orig -> src_iso -> trg_iso -> trg_orig
        # T_orig_v2v = Rtrg @ T_iso_v2v @ inv(Rsrc)
        T = Rtrg.double() @ T.double() @ torch.inverse(Rsrc.double())
        T = T.float()

    # Save outlier map if requested
    if outliers_name is not None and all_info:
        import nibabel as nib

        final_info = all_info[-1]
        if 'weights' in final_info and 'valid_mask' in final_info:
            weights_sqrt = final_info['weights']  # sqrt(Tukey weights)
            valid_mask = final_info['valid_mask']
            reg_affine = final_info.get('iso_affine', None)

            if reg_affine is not None:
                # Square to get actual Tukey weights, then compute 1-w (outlier map)
                weights = weights_sqrt ** 2

                # Reconstruct shape is the shape from the last pyramid level
                # (which corresponds to the registration space)
                reg_shape = final_info['image_shape']

                # Reconstruct 3D volume in registration space
                weight_volume = torch.zeros(reg_shape, dtype=torch.float32)
                weight_volume.view(-1)[valid_mask] = weights

                # Create outlier map (1 - w): high values = outliers
                outlier_volume = 1.0 - weight_volume

                # Auto-detect format from extension
                if outliers_name.endswith('.nii') or outliers_name.endswith('.nii.gz'):
                    outlier_img = nib.Nifti1Image(outlier_volume.numpy(), reg_affine)
                else:  # Default to MGZ for .mgz or unknown extensions
                    outlier_img = nib.MGHImage(outlier_volume.numpy(), reg_affine)

                outlier_img.to_filename(outliers_name)

                if verbose:
                    outlier_pct = (outlier_volume > 0.5).sum().item() / outlier_volume.numel() * 100
                    logger.info("Saved outlier map: %s (%.1f%% high outliers)",
                                outliers_name, outlier_pct)
            else:
                logger.warning("Cannot save outlier map: no affine available")
        else:
            logger.warning("Cannot save outlier map: no weights in final level")

    return T, all_info

