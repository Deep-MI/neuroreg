"""Affine / rigid transformation matrix utilities."""

import numpy as np
import numpy.typing as npt
import torch


def _det_mps_compatible(R: torch.Tensor) -> torch.Tensor:
    """Compute determinant in an MPS-compatible way.
    
    MPS doesn't support torch.linalg.det, so we compute it manually.
    For 3x3 matrices, we use the standard formula.
    For 4x4 matrices, we use the same formula on the top-left 3x3 block
    since we only care about rotation determinants.
    
    Parameters
    ----------
    R : torch.Tensor
        3x3 or 4x4 matrix
    
    Returns
    -------
    det : torch.Tensor
        Scalar determinant value
    """
    if R.shape[0] == 4 and R.shape[1] == 4:
        # For 4x4 matrices, extract the 3x3 rotation block
        R = R[:3, :3]

    if R.shape != (3, 3):
        # Fallback to torch.linalg.det for other cases
        # (will work on CPU/CUDA, may fail on MPS for non-3x3)
        return torch.linalg.det(R)

    # Compute determinant using cofactor expansion along first row
    det = (
            R[0, 0] * (R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1])
            - R[0, 1] * (R[1, 0] * R[2, 2] - R[1, 2] * R[2, 0])
            + R[0, 2] * (R[1, 0] * R[2, 1] - R[1, 1] * R[2, 0])
    )
    return det


def matrix_sqrt_schur(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Principal square root of a 4×4 matrix via Schur decomposition.

    Uses :func:`scipy.linalg.sqrtm` which performs the Schur decomposition
    over ℂ and is valid for **any** matrix whose eigenvalues do not lie on
    the negative real axis — in particular for rigid and affine transforms
    whose rotation eigenvalues lie on the unit circle.

    Iterative square-root methods designed for positive-definite matrices can
    diverge for typical rotation matrices; **do not** use them for transforms.

    Parameters
    ----------
    M : torch.Tensor, shape (4, 4)
        Voxel-to-voxel (or any square) transform matrix.  May be float32 or
        float64; internally promoted to float64 for numerical stability.

    Returns
    -------
    mh : torch.Tensor, shape (4, 4)
        Principal square root such that ``mh @ mh ≈ M``.
    mhi : torch.Tensor, shape (4, 4)
        ``mh @ inv(M)`` — the "other half" mapping the target back to the
        midspace.  Using ``mh @ inv(M)`` instead of ``inv(mh)`` is more
        numerically stable when ``mh @ mh`` is not exactly ``M`` (following
        the FreeSurfer convention).

    Notes
    -----
    This function is intentionally **outside the autograd graph** (it calls
    ``.detach()`` before the numpy conversion).  Use it in the outer
    registration loop where symmetry is managed, not inside a differentiable
    model forward pass.
    """
    from scipy.linalg import sqrtm as scipy_sqrtm  # lazy import – not always needed

    M_np = M.detach().cpu().double().numpy()
    mh_np = scipy_sqrtm(M_np)
    # scipy may return a complex array with tiny imaginary residuals — take real part
    mh_np = np.real(mh_np)
    mh = torch.from_numpy(mh_np).to(dtype=M.dtype, device=M.device)
    work_dtype = torch.float32 if M.device.type == "mps" else torch.float64
    mi = torch.inverse(M.to(device=M.device, dtype=work_dtype))
    mhi = (mh.to(dtype=work_dtype) @ mi).to(dtype=M.dtype)
    return mh, mhi


def get_translation(translation: torch.Tensor) -> torch.Tensor:
    """Generate a 4 × 4 translation matrix (homogeneous coordinates).

    Parameters
    ----------
    translation : torch.Tensor, shape (3,)
        3-D translation vector.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Translation matrix.
    """
    trans = torch.eye(4, device=translation.device, dtype=translation.dtype)
    trans[:3, 3] = translation[:3]
    return trans


def get_rotation_rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    """Generate a 4 × 4 homogeneous rotation matrix from a Rodrigues vector.

    Parameters
    ----------
    rotvec : torch.Tensor, shape (3,)
        Rotation vector whose direction is the rotation axis and whose
        magnitude is the rotation angle in radians.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Homogeneous rotation matrix. The computation uses standard
        differentiable torch operations and remains compatible with autograd on
        CPU and GPU devices.
    """
    theta = torch.norm(rotvec)
    if theta < 1e-10:
        rmat = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    else:
        axis = rotvec / theta
        cross_mat = torch.zeros(3, 3, dtype=rotvec.dtype, device=rotvec.device)
        cross_mat[0, 1] = -axis[2]
        cross_mat[0, 2] = axis[1]
        cross_mat[1, 0] = axis[2]
        cross_mat[1, 2] = -axis[0]
        cross_mat[2, 0] = -axis[1]
        cross_mat[2, 1] = axis[0]
        rmat = (
                torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
                + torch.sin(theta) * cross_mat
                + (1.0 - torch.cos(theta)) * (cross_mat @ cross_mat)
        )

    r4x4 = torch.eye(4, dtype=rotvec.dtype, device=rotvec.device)
    r4x4[:3, :3] = rmat
    return r4x4


def get_rotation_euler(angles: torch.Tensor) -> torch.Tensor:
    """Generate a 4 × 4 rotation matrix from Euler angles (X → Y → Z).

    Parameters
    ----------
    angles : torch.Tensor, shape (3,)
        Euler angles [rx, ry, rz] in radians.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Rotation matrix.
    """
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles[0])
    zero = torch.zeros_like(angles[0])
    R_x = (one, zero, zero, zero, cos[0], -sin[0], zero, sin[0], cos[0])
    R_y = (cos[1], zero, sin[1], zero, one, zero, -sin[1], zero, cos[1])
    R_z = (cos[2], -sin[2], zero, sin[2], cos[2], zero, zero, zero, one)
    R_x = torch.stack(R_x, -1).view((3, 3))
    R_y = torch.stack(R_y, -1).view((3, 3))
    R_z = torch.stack(R_z, -1).view((3, 3))
    rmat = torch.matmul(torch.matmul(R_x, R_y), R_z)
    r4x4 = torch.eye(4, dtype=angles.dtype, device=angles.device)
    r4x4[:3, :3] = rmat
    return r4x4


def params_to_rigid_matrix(params: torch.Tensor) -> torch.Tensor:
    """Convert 6-DOF rigid parameters to a 4 × 4 matrix in DHW axis order.

    This helper is shared by the IRLS robreg implementation. The parameter
    vector follows the Jacobian convention used in the FreeSurfer-style linear
    system:

    * ``params[0]`` ↔ translation along W (image dim 2 / physical x)
    * ``params[1]`` ↔ translation along H (image dim 1 / physical y)
    * ``params[2]`` ↔ translation along D (image dim 0 / physical z)
    * ``params[3:6]`` ↔ Rodrigues rotation vector around W/H/D axes

    The rotation is first constructed in physical XYZ order and then permuted to
    nibabel's internal DHW indexing convention by swapping axes ``0 ↔ 2``.

    Parameters
    ----------
    params : torch.Tensor, shape (6,)
        Rigid parameter vector ``[tx_W, ty_H, tz_D, rx_W, ry_H, rz_D]``.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Homogeneous voxel-to-voxel update matrix in DHW axis order.
    """
    T = torch.eye(4, dtype=params.dtype, device=params.device)

    R_xyz = get_rotation_rodrigues(params[3:6])[:3, :3]
    ii = [2, 1, 0]
    T[:3, :3] = R_xyz[ii][:, ii]

    T[0, 3] = params[2]
    T[1, 3] = params[1]
    T[2, 3] = params[0]

    return T


def get_scaling(scales: torch.Tensor) -> torch.Tensor:
    """Generate a 4 × 4 diagonal scaling matrix.

    Parameters
    ----------
    scales : torch.Tensor, shape (3,)
        Scaling factors along x, y, z.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Scaling matrix.
    """
    S = torch.diag(torch.cat((scales, scales.new_tensor([1.0]))))
    return S


def rotation_error(R1: torch.Tensor, R2: torch.Tensor, check_valid: bool = True) -> float:
    """Compute rotation error between two rotation matrices in degrees.

    Uses the geodesic distance on SO(3): θ = arccos((trace(R1^T R2) - 1) / 2)

    Parameters
    ----------
    R1, R2 : torch.Tensor
        3×3 or 4×4 rotation matrices. If 4×4, only top-left 3×3 is used for
        the geodesic distance calculation, but full 4×4 is validated if
        check_valid=True (ensuring it's a pure rotation with no translation).
    check_valid : bool, optional
        If True (default), validates that inputs are proper rotation matrices.
        For 4×4 matrices, also validates no translation component.
        Set to False to skip validation for performance.

    Returns
    -------
    error : float
        Rotation error in degrees.

    Raises
    ------
    ValueError
        If check_valid=True and matrices are not proper rotations.
        For 4×4 matrices, also raises if translation is present.

    Examples
    --------
    >>> R1 = get_rotation_euler(torch.tensor([0.1, 0.2, 0.0]))[:3, :3]
    >>> R2 = get_rotation_euler(torch.tensor([0.15, 0.25, 0.05]))[:3, :3]
    >>> error = rotation_error(R1, R2)
    >>> print(f"Rotation error: {error:.2f}°")

    Notes
    -----
    A proper rotation matrix must satisfy:
    - R^T @ R = I (orthogonal)
    - det(R) = 1 (proper rotation, not reflection)

    For 4×4 homogeneous matrices, additionally:
    - Last row must be [0, 0, 0, 1]
    - Last column must be [0, 0, 0, 1]^T (no translation component)
    """
    if check_valid:
        # Validate with full matrices (handles both 3×3 and 4×4)
        if not _is_rotation_matrix(R1):
            raise ValueError(
                f"R1 is not a valid rotation matrix. "
                f"Orthogonality error: {torch.norm(R1.T @ R1 - torch.eye(R1.shape[0], device=R1.device)):.6f}, "
                f"Determinant: {_det_mps_compatible(R1):.6f} (should be 1.0)"
            )
        if not _is_rotation_matrix(R2):
            raise ValueError(
                f"R2 is not a valid rotation matrix. "
                f"Orthogonality error: {torch.norm(R2.T @ R2 - torch.eye(R2.shape[0], device=R2.device)):.6f}, "
                f"Determinant: {_det_mps_compatible(R2):.6f} (should be 1.0)"
            )

    # Extract 3×3 rotation part for geodesic distance calculation
    # (the formula uses trace of 3×3 rotation matrix)
    if R1.shape[0] == 4:
        R1 = R1[:3, :3]
    if R2.shape[0] == 4:
        R2 = R2[:3, :3]

    # Compute R1^T @ R2
    R_diff = R1.T @ R2

    # Geodesic distance: θ = arccos((tr(R) - 1) / 2)
    trace = torch.trace(R_diff)
    # Clamp to handle numerical errors
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta_rad = torch.arccos(cos_theta)
    theta_deg = torch.rad2deg(theta_rad)

    return theta_deg.item()


def _is_rotation_matrix(R: torch.Tensor, atol: float = 1e-4) -> bool:
    """Check if a 3×3 or 4×4 matrix is a proper rotation matrix.

    Parameters
    ----------
    R : torch.Tensor
        3×3 or 4×4 matrix to check. For 4×4 homogeneous coordinates,
        last row must be [0, 0, 0, 1] and last column must be [0, 0, 0, 1]
        (pure rotation, no translation).
    atol : float
        Absolute tolerance for checks.

    Returns
    -------
    bool
        True if R is a proper rotation matrix.
    """
    # Check shape
    if R.shape == (3, 3):
        # 3×3 rotation matrix
        n = 3
    elif R.shape == (4, 4):
        # 4×4 homogeneous rotation matrix (no translation)
        n = 4
        # Check last row is [0, 0, 0, 1]
        expected_last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device, dtype=R.dtype)
        if not torch.allclose(R[3, :], expected_last_row, atol=atol):
            return False
        # Check last column is [0, 0, 0, 1]^T (no translation)
        expected_last_col = torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device, dtype=R.dtype)
        if not torch.allclose(R[:, 3], expected_last_col, atol=atol):
            return False
    else:
        return False

    # Check orthogonality: R^T @ R = I
    # This works for both 3×3 and 4×4 matrices!
    identity_matrix = torch.eye(n, device=R.device, dtype=R.dtype)
    ortho_error = torch.norm(R.T @ R - identity_matrix)
    if ortho_error > atol:
        return False

    # Check determinant = 1 (not -1, which would be a reflection)
    det = _det_mps_compatible(R)
    if abs(det - 1.0) > atol:
        return False

    return True


def get_affine(
        translation: torch.Tensor, rotvec: torch.Tensor | None = None, scales: torch.Tensor | None = None
) -> torch.Tensor:
    """Generate a 4 × 4 affine matrix from translation, rotation, and scale.

    Parameters
    ----------
    translation : torch.Tensor, shape (3,)
        Translation vector.
    rotvec : torch.Tensor, shape (3,), optional
        Euler angles [rx, ry, rz] in radians.
    scales : torch.Tensor, shape (3,), optional
        Per-axis scaling factors.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Affine transformation matrix.
    """
    matrix = get_translation(translation)
    if rotvec is not None:
        matrix = torch.matmul(matrix, get_rotation_euler(rotvec))
    if scales is not None:
        matrix = torch.matmul(get_scaling(scales), matrix)
    return matrix


def convert_v2v_to_torch(v2v: torch.Tensor, source_shape, target_shape=None) -> torch.Tensor:
    """Convert a vox-to-vox affine matrix to PyTorch grid-sample format.

    Accounts for the coordinate-system differences between voxel space and
    PyTorch's normalised ``[-1, 1]`` grid space.

    Parameters
    ----------
    v2v : torch.Tensor, shape (4, 4)
        Vox-to-vox transformation matrix (source → target voxels),
        in nibabel axis order (D, H, W).
    source_shape : tuple[int, int, int]
        Shape of the source image (D, H, W).
    target_shape : tuple[int, int, int], optional
        Shape of the target image (D, H, W).  Defaults to *source_shape*.

    Returns
    -------
    torch.Tensor, shape (3, 4)
        PyTorch-compatible affine matrix for use with
        :func:`torch.nn.functional.affine_grid`.

    Raises
    ------
    ValueError
        If *v2v* is not shape (4, 4).

    Notes
    -----
    PyTorch's ``affine_grid`` / ``grid_sample`` use **(W, H, D) = (x, y, z)**
    axis order, while nibabel stores images as **(D, H, W)**.  For non-cubic
    images (D ≠ W) the scale factors along each axis differ, so the nibabel
    v2v must be **permuted** to (W, H, D) order **before** multiplying with
    the per-axis normalisation matrices.  Applying the permutation *after*
    the multiplication (as a row/column swap at the end) only works correctly
    when D = W (cubic images) and silently produces wrong results otherwise.
    """
    if target_shape is None:
        target_shape = source_shape
    if v2v.shape != torch.Size([4, 4]):
        raise ValueError(f"Expected v2v of shape (4, 4), but got {v2v.shape}.")

    dtype = v2v.dtype
    device = v2v.device
    work_dtype = torch.float32 if device.type == "mps" else torch.float64

    # Backward v2v in nibabel (D, H, W) order.
    inv_v2v_dhw = torch.inverse(v2v.to(dtype=work_dtype))

    # Reorder from nibabel (D, H, W) to PyTorch (W, H, D) = (x, y, z) order.
    # Swap axis 0 (D) ↔ axis 2 (W); leave axis 1 (H) and the homogeneous row/col.
    # This must be done BEFORE multiplying with the normalisation matrices (which
    # are built in (W, H, D) order via reversed(shape)), otherwise the per-axis
    # scale factors are applied to the wrong axes for non-cubic images.
    ii = [2, 1, 0, 3]
    inv_v2v_xyz = inv_v2v_dhw[ii][:, ii]

    # Normalisation scale factors in (W, H, D) = (x, y, z) order.
    # reversed((D, H, W)) = (W, H, D).
    sf_src = torch.tensor(list(reversed(source_shape)), dtype=work_dtype, device=device) / 2.0
    sf_trg = torch.tensor(list(reversed(target_shape)), dtype=work_dtype, device=device) / 2.0

    # denorm_trg : target normalised [-1, 1] → target voxel index [0, N-1]
    #   vox = (norm + 1) * (N/2) - 0.5    (align_corners=False)
    sf_trg4 = torch.cat((sf_trg, sf_trg.new_tensor([1.0])))
    denorm_trg = torch.diag(sf_trg4)
    denorm_trg[:-1, -1] += sf_trg4[:-1] - 0.5

    # norm_src : source voxel index [0, N-1] → source normalised [-1, 1]
    #   norm = (vox + 0.5) / (N/2) - 1    (align_corners=False)
    sf_src4 = torch.cat((sf_src, sf_src.new_tensor([1.0])))
    norm_src = torch.diag(1.0 / sf_src4)
    norm_src[:-1, -1] += -1.0 + 0.5 / sf_src4[:-1]

    # Full backward chain, all in (W, H, D) = (x, y, z) order:
    #   target_norm → target_vox → source_vox → source_norm
    torch_transform = norm_src @ inv_v2v_xyz @ denorm_trg

    # No axis reversal needed – everything is already in PyTorch (W, H, D) order.
    return torch_transform[:3, :4].to(dtype)


def convert_r2r_to_torch(
        r2r: torch.Tensor,
        source_shape,
        source_affine: torch.Tensor,
        target_shape=None,
        target_affine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a RAS-to-RAS affine matrix to PyTorch grid-sample format.

    Preferred over converting to vox-to-vox first then calling
    :func:`convert_v2v_to_torch`, because the intermediate v2v matrix
    ``inv(target_affine) @ r2r @ source_affine`` has non-unit singular values
    whenever the affines have anisotropic voxels and *r2r* contains a rotation.
    That apparent shear is a coordinate-representation artefact that cancels
    in the final PyTorch transform — but holding the sheared v2v externally
    invites misuse.

    This function builds the backward-sampling chain directly in physical
    (RAS) coordinates without forming the intermediate v2v:

    .. code-block:: none

        trg_norm  →  trg_vox  →  trg_RAS  →  src_RAS  →  src_vox  →  src_norm
                  denorm_trg    trg_affine   inv(r2r)   inv(src_aff)   norm_src

    which is identical to ``convert_v2v_to_torch`` when
    ``v2v = inv(target_affine) @ r2r @ source_affine``, but avoids the
    intermediate sheared matrix entirely.

    Parameters
    ----------
    r2r : torch.Tensor, shape (4, 4)
        RAS-to-RAS transformation matrix (source_RAS → target_RAS).
    source_shape : tuple[int, int, int]
        Shape of the source image (D, H, W).
    source_affine : torch.Tensor, shape (4, 4)
        Voxel-to-RAS affine of the source image.
    target_shape : tuple[int, int, int], optional
        Shape of the target image.  Defaults to *source_shape*.
    target_affine : torch.Tensor, shape (4, 4), optional
        Voxel-to-RAS affine of the target image.  Defaults to
        *source_affine* (same-space resampling).

    Returns
    -------
    torch.Tensor, shape (3, 4)
        PyTorch-compatible affine matrix for
        :func:`torch.nn.functional.affine_grid`.
    """
    if target_shape is None:
        target_shape = source_shape
    if target_affine is None:
        target_affine = source_affine

    device = r2r.device
    dtype = r2r.dtype

    # Compute the backward sampling chain in double precision to avoid
    # accumulated float32 rounding error, in nibabel (D, H, W) order.
    #   trg_vox → trg_RAS → src_RAS → src_vox
    r2r_d = r2r.double()
    src_affine_d = source_affine.double()
    trg_affine_d = target_affine.double()
    backward_phys_dhw = torch.inverse(src_affine_d) @ torch.inverse(r2r_d) @ trg_affine_d

    # Permute from nibabel (D, H, W) to PyTorch (W, H, D) order before
    # applying normalization (same fix as convert_v2v_to_torch).
    ii = [2, 1, 0, 3]
    backward_phys_xyz = backward_phys_dhw[ii][:, ii]

    # Normalization scale factors in (W, H, D) order (via reversed(shape)).
    sf_src = torch.tensor(list(reversed(source_shape)), dtype=torch.float64, device=device) / 2.0
    sf_trg = torch.tensor(list(reversed(target_shape)), dtype=torch.float64, device=device) / 2.0

    sf_trg4 = torch.cat((sf_trg, sf_trg.new_tensor([1.0])))
    denorm_trg = torch.diag(sf_trg4)
    denorm_trg[:-1, -1] += sf_trg4[:-1] - 0.5

    sf_src4 = torch.cat((sf_src, sf_src.new_tensor([1.0])))
    norm_src = torch.diag(1.0 / sf_src4)
    norm_src[:-1, -1] += -1.0 + 0.5 / sf_src4[:-1]

    # Full backward chain in (W, H, D) order.
    torch_transform = norm_src @ backward_phys_xyz @ denorm_trg

    # No axis reversal needed – already in PyTorch (W, H, D) order.
    return torch_transform[:3, :4].to(dtype)


def convert_torch_to_v2v(torch_transform: torch.Tensor, source_shape, target_shape=None) -> torch.Tensor:
    """Convert a PyTorch grid-sample affine matrix back to vox-to-vox format.

    Parameters
    ----------
    torch_transform : torch.Tensor, shape (4, 4)
        PyTorch-format affine matrix (must be passed as 4 × 4).
    source_shape : tuple[int, int, int]
        Shape of the source image (D, H, W).
    target_shape : tuple[int, int, int], optional
        Shape of the target image (D, H, W).  Defaults to *source_shape*.

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Vox-to-vox transformation matrix.

    Raises
    ------
    ValueError
        If *torch_transform* is not shape (4, 4).
    """
    if target_shape is None:
        target_shape = source_shape
    if torch_transform.size() != torch.Size([4, 4]):
        raise ValueError(f"Torch affine shape {torch_transform.size()} should be (4,4)!")

    dtype = torch_transform.dtype
    device = torch_transform.device
    ndim = 3

    # Prepare scale factors for source and target (match input dtype)
    scale_factor_source = torch.as_tensor(list(reversed(source_shape)), dtype=dtype, device=device) / 2.0
    scale_factor_target = torch.as_tensor(list(reversed(target_shape)), dtype=dtype, device=device) / 2.0
    # Create diagonal scaling and translation matrices
    scale_factor_source = torch.cat((scale_factor_source, scale_factor_source.new_tensor([1.0])), dim=0)
    scale_factor_target = torch.cat((scale_factor_target, scale_factor_target.new_tensor([1.0])), dim=0)
    # Rescale relative coordinates (-1, 1) --> (-0.5, (n-1)+0.5) and move center
    relative2target = torch.diag(scale_factor_source)
    relative2target[..., :-1, -1] += scale_factor_source[:-1] - 0.5
    # Rescale from relative coordinates (-0.5, (n-1)+0.5) --> (-1, 1) and move center
    source2relative = torch.diag(1.0 / scale_factor_target)
    source2relative[..., :-1, -1] += -1.0 + 0.5 / scale_factor_target[:-1]
    # Combine the scaling and translation matrices
    relative2target = relative2target.to(torch_transform.device)
    source2relative = source2relative.to(torch_transform.device)
    # Compute the inverse transformation
    v2v_inv = torch.matmul(relative2target, torch_transform)
    v2v_inv = torch.matmul(v2v_inv, source2relative)
    # Rearrange axes back to original order and add row/column for 4x4 transformation
    ii = list(reversed(range(ndim))) + [ndim]
    v2v_inv = v2v_inv[..., ii, :]
    v2v_inv = v2v_inv[..., ii]
    return torch.inverse(v2v_inv)


# LTA / FreeSurfer transform type constants (also re-exported from lta.py)
LINEAR_VOX_TO_VOX = 0
LINEAR_RAS_TO_RAS = 1


def convert_transform_type(
        matrix: npt.ArrayLike,
        src_affine: npt.ArrayLike,
        dst_affine: npt.ArrayLike,
        from_type: int,
        to_type: int,
) -> np.ndarray:
    """Convert a transformation matrix between vox-to-vox and RAS-to-RAS.

    Parameters
    ----------
    matrix : array-like, shape (4, 4)
        Input transformation matrix.
    src_affine : array-like, shape (4, 4)
        Source image voxel-to-RAS affine (nibabel ``img.affine``).
    dst_affine : array-like, shape (4, 4)
        Destination image voxel-to-RAS affine.
    from_type : int
        Type of the input matrix:
        ``LINEAR_VOX_TO_VOX`` (0) or ``LINEAR_RAS_TO_RAS`` (1).
    to_type : int
        Desired output type:
        ``LINEAR_VOX_TO_VOX`` (0) or ``LINEAR_RAS_TO_RAS`` (1).

    Returns
    -------
    np.ndarray, shape (4, 4)
        Converted transformation matrix.  Returns a copy when
        ``from_type == to_type``.

    Raises
    ------
    ValueError
        If *from_type* or *to_type* is not 0 or 1.

    Notes
    -----
    Conversion formulae (M = matrix, A_s = src_affine, A_d = dst_affine):

    * vox→vox to RAS→RAS:  ``A_d @ M @ inv(A_s)``
    * RAS→RAS to vox→vox:  ``inv(A_d) @ M @ A_s``
    """
    if from_type not in (LINEAR_VOX_TO_VOX, LINEAR_RAS_TO_RAS):
        raise ValueError(f"from_type must be 0 or 1, got {from_type}")
    if to_type not in (LINEAR_VOX_TO_VOX, LINEAR_RAS_TO_RAS):
        raise ValueError(f"to_type must be 0 or 1, got {to_type}")

    if from_type == to_type:
        return np.asarray(matrix, dtype=float).copy()

    M = np.asarray(matrix, dtype=float)
    As = np.asarray(src_affine, dtype=float)
    Ad = np.asarray(dst_affine, dtype=float)

    if from_type == LINEAR_VOX_TO_VOX:  # → RAS-to-RAS
        return Ad @ M @ np.linalg.inv(As)
    else:  # RAS-to-RAS → vox-to-vox
        return np.linalg.inv(Ad) @ M @ As
