"""Affine / rigid transformation matrix utilities."""

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def compute_sqrtm(matrix: Tensor, num_iters: int = 100) -> tuple[Tensor, Tensor]:
    r"""
    Compute the square root of a matrix using the Newton-Schulz iterative method.

    This method iteratively approximates the square root of a positive definite matrix.
    It is based on the source: https://github.com/msubhransu/matrix-sqrt.

    Parameters
    ----------
    matrix : Tensor
        A 2D square tensor (NxN) for which the square root is to be calculated.
    num_iters : int, optional
        The number of iterations to perform for the Newton-Schulz method.
        Defaults to 100. Must be greater than 0.

    Returns
    -------
    tuple[Tensor, Tensor]
        - The square root of the matrix (`s_matrix`) as a 2D tensor.
        - The approximation error (`error`) as a 1D tensor with a single value.

    Raises
    ------
    ValueError
        If the input tensor `matrix` is not 2D or square.
        If `num_iters` is less than or equal to 0.

    Example
    -------
    >>> matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    >>> sqrt_matrix, error = compute_sqrtm(matrix, num_iters=50)
    >>> print(sqrt_matrix)  # Approximated square root of the matrix
    tensor([[1.9799, 0.3033],
            [0.3033, 1.6593]])
    >>> print(error)  # Approximation error
    tensor([0.0000])

    Notes
    -----
    - This function is designed for positive definite matrices and may not work for
      non-positive definite matrices.
    - The algorithm stops early if the approximation error is below a threshold of 1e-5.
    """
    # Validate input dimensions
    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(f'Input dimension equals {matrix.dim()}, expected {expected_num_dims}')
    if num_iters <= 0:
        raise ValueError(f'Number of iterations equals {num_iters}, expected greater than 0')
    # Get matrix dimension
    dim = matrix.size(0)
    # Compute initial normalization
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)  # Initial normalization
    Id = torch.eye(dim, dim, requires_grad=False).to(matrix)  # Identity matrix
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)  # Initialize Z
    # Initialize placeholders
    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)
    # Iterate using Newton-Schulz method
    for _ in range(num_iters):
        T = 0.5 * (3.0 * Id - Z.mm(Y))  # Compute transformation matrix
        Y = Y.mm(T)  # Update Y
        Z = T.mm(Z)  # Update Z
        # Approximate the square root of the matrix
        s_matrix = Y * torch.sqrt(norm_of_matrix)
        # Compute the error
        error = torch.norm(matrix - torch.mm(s_matrix, s_matrix)) / norm_of_matrix
        # Check for early stopping if error is close to zero
        if torch.isclose(error, torch.tensor([0.]).to(error), atol=1e-5):
            break
    return s_matrix, error


def matrix_sqrt_schur(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Principal square root of a 4×4 matrix via Schur decomposition.

    Uses :func:`scipy.linalg.sqrtm` which performs the Schur decomposition
    over ℂ and is valid for **any** matrix whose eigenvalues do not lie on
    the negative real axis — in particular for rigid and affine transforms
    whose rotation eigenvalues lie on the unit circle.

    The Newton-Schulz method used in :func:`compute_sqrtm` requires positive
    definiteness and diverges for typical rotation matrices; **do not** use
    that function for transforms.

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

    M_np = M.detach().double().cpu().numpy()
    mh_np = scipy_sqrtm(M_np)
    # scipy may return a complex array with tiny imaginary residuals — take real part
    mh_np = mh_np.real
    mh = torch.from_numpy(mh_np).to(dtype=M.dtype, device=M.device)
    mi = torch.inverse(M.double().to(device=M.device))
    mhi = (mh.double() @ mi).to(dtype=M.dtype)
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
    """Generate a 4 × 4 rotation matrix from a Rodrigues vector.

    Parameters
    ----------
    rotvec : torch.Tensor, shape (3,)
        Rotation vector whose magnitude is the rotation angle (radians).

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Rotation matrix.
    """
    angle = torch.norm(rotvec)
    zero = torch.zeros(1, dtype=rotvec.dtype, device=rotvec.device).squeeze()
    cross_mat = torch.stack(
            [zero, -rotvec[2], rotvec[1],
             rotvec[2], zero, -rotvec[0],
             -rotvec[1], rotvec[0], zero], dim=-1
        ).view((3,3))
    angle2 = angle * angle
    if angle2 == 0:
        angle2 = 1
    rmat = ((torch.eye(3, dtype=rotvec.dtype, device=rotvec.device) +
            torch.sinc(angle / torch.pi) * cross_mat) +
            ((1 - torch.cos(angle)) / angle2) * (cross_mat @ cross_mat))
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


def get_affine(translation: torch.Tensor,
               rotvec: torch.Tensor | None = None,
               scales: torch.Tensor | None = None) -> torch.Tensor:
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

    dtype  = v2v.dtype
    device = v2v.device

    # Backward v2v in nibabel (D, H, W) order; float64 for numerical accuracy.
    inv_v2v_dhw = torch.inverse(v2v.double())

    # Reorder from nibabel (D, H, W) to PyTorch (W, H, D) = (x, y, z) order.
    # Swap axis 0 (D) ↔ axis 2 (W); leave axis 1 (H) and the homogeneous row/col.
    # This must be done BEFORE multiplying with the normalisation matrices (which
    # are built in (W, H, D) order via reversed(shape)), otherwise the per-axis
    # scale factors are applied to the wrong axes for non-cubic images.
    ii = [2, 1, 0, 3]
    inv_v2v_xyz = inv_v2v_dhw[ii][:, ii]

    # Normalisation scale factors in (W, H, D) = (x, y, z) order.
    # reversed((D, H, W)) = (W, H, D).
    sf_src = torch.tensor(list(reversed(source_shape)), dtype=torch.float64, device=device) / 2.0
    sf_trg = torch.tensor(list(reversed(target_shape)), dtype=torch.float64, device=device) / 2.0

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
    dtype  = r2r.dtype

    # Compute the backward sampling chain in double precision to avoid
    # accumulated float32 rounding error, in nibabel (D, H, W) order.
    #   trg_vox → trg_RAS → src_RAS → src_vox
    r2r_d        = r2r.double()
    src_affine_d = source_affine.double()
    trg_affine_d = target_affine.double()
    backward_phys_dhw = (
        torch.inverse(src_affine_d) @ torch.inverse(r2r_d) @ trg_affine_d
    )

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
    if torch_transform.size() != torch.Size([4, 4]) :
        raise ValueError(f"Torch affine shape {torch_transform.size()} should be (4,4)!")

    dtype  = torch_transform.dtype
    device = torch_transform.device
    ndim   = 3

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

    M  = np.asarray(matrix, dtype=float)
    As = np.asarray(src_affine, dtype=float)
    Ad = np.asarray(dst_affine, dtype=float)

    if from_type == LINEAR_VOX_TO_VOX:   # → RAS-to-RAS
        return Ad @ M @ np.linalg.inv(As)
    else:                                 # RAS-to-RAS → vox-to-vox
        return np.linalg.inv(Ad) @ M @ As

