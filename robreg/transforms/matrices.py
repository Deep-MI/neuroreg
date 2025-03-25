from typing import Optional

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


def matrix_decompose(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Decompose a 3x3 or 4x4 matrix into four components: rotation, shear, scale, and translation.

    This function performs singular value decomposition (SVD) to decompose the input matrix.
    If the matrix is 4x4, it extracts the translation vector from the last column and decomposes
    the upper-left 3x3 portion into rotation, symmetric scaling, and shear.

    Parameters
    ----------
    matrix : Tensor
        A 3x3 or 4x4 square tensor representing the affine transformation matrix.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        - Rotation matrix (`R`): A 3x3 tensor representing the pure rotation component.
        - Shear matrix (`S`): A 3x3 tensor representing the symmetric shear component, normalized by scales.
        - Scale factors (`D`): A 1D tensor of length 3 containing the diagonal scaling factors.
        - Translation vector (`T`): A 1D tensor of length 3 representing the translation component.

    Raises
    ------
    ValueError
        If the input matrix is not 2D, not square, or is neither 3x3 nor 4x4.

    Example
    -------
    >>> matrix = torch.tensor([[1.0, 0.0, 0.0, 5.0],
    ...                        [0.0, 1.0, 0.0, 6.0],
    ...                        [0.0, 0.0, 1.0, 7.0],
    ...                        [0.0, 0.0, 0.0, 1.0]])  # A 4x4 transformation matrix
    >>> R, S, D, T = matrix_decompose(matrix)
    >>> print("Rotation:", R)
    >>> print("Shear:", S)
    >>> print("Scale:", D)
    >>> print("Translation:", T)
    Translation: tensor([5., 6., 7.])

    Notes
    -----
    - The function is specifically designed for 3D transformation matrices of size 3x3 or 4x4.
    - The decomposition produces an exact reconstruction of the input matrix:
        For 4x4 -> [R S | T], For 3x3 -> R S, with S = shear * diag(scales).
    """
    # Check input validity
    if matrix.dim() != 2:
        raise ValueError(f'Input dimension equals {matrix.dim()}, expected 2')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Input matrix is not square')
    if matrix.shape[0] == 4:  # Handle 4x4 matrix
        T = matrix[0:3, 3]
        matrix = matrix[0:3, 0:3]
    else:  # Handle 3x3 matrix
        T = torch.zeros(3)
    if matrix.shape[0] != 3:
        raise ValueError('Input matrix should be 3x3 or 4x4')
    # Decompose the matrix using Singular Value Decomposition (SVD)
    U, W, Vh = torch.linalg.svd(matrix)
    # Compute the rotation matrix
    R = U @ Vh
    if torch.linalg.norm((R @ torch.t(R)) - torch.eye(3)) > 1e-5:
        raise ValueError("Rotation matrix does not preserve orthogonality (possible numerical issue)")
    # Compute the symmetric shear/scaling matrix
    W = torch.diag(W)
    S = torch.t(Vh) @ W @ Vh
    # Decompose S into scale and shear components
    D = torch.diagonal(S, 0)  # Extract scales (diagonal of the symmetric matrix)
    S = S / D  # Divide by scales to extract shear component
    return R, S, D, T


def get_translation(translation: torch.Tensor) -> torch.Tensor:
    """
    Generate a 4x4 translation matrix (homogeneous coordinates).

    Parameters
    ==========
        translation (torch.Tensor): 3D translation vector of shape (3,).

    Returns
    =======
        torch.Tensor: A 4x4 translation matrix.
    """
    trans = torch.eye(4, device=translation.device, dtype=translation.dtype)
    trans[:3, 3] = translation[:3]
    return trans


def get_rotation_rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Generate a 3x3 rotation matrix using a Rodrigues vector.

    Parameters
    ==========
        rotvec (torch.Tensor): 3D rotation vector of shape (3,).

    Returns
    =======
        torch.Tensor: A 3x3 rotation matrix.
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
    return rmat


def get_rotation_euler(angles: torch.Tensor) -> torch.Tensor:
    """
    Generate a 3x3 rotation matrix using Euler angles X,Y,Z.

    Parameters
    ==========
        angles (torch.Tensor): Euler angles X,Y,Z vector of shape (3,).

    Returns
    =======
        torch.Tensor: A 3x3 rotation matrix.
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
    return torch.matmul(torch.matmul(R_x, R_y), R_z)


def get_scaling(scales: torch.Tensor) -> torch.Tensor:
    """
    Generate a 4x4 scaling matrix.

    Parameters:
        scales (torch.Tensor): 3D scaling factors of shape (3,).

    Returns:
        torch.Tensor: A 4x4 scaling matrix.
    """
    S = torch.diag(torch.cat((scales, torch.tensor([1.0], device=scales.device))))
    return S


def get_affine(translation: torch.Tensor,
               rotvec: Optional[torch.Tensor] = None,
               scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generate a 4x4 affine transformation matrix from translation, rotation, and scaling.

    Parameters:
        translation (torch.Tensor): Translation vector of shape (3,).
        rotvec (Optional[torch.Tensor]): Euler X,Y,Z angles of shape (3,).
        scales (Optional[torch.Tensor]): Scaling factors of shape (3,).

    Returns:
        torch.Tensor: A 4x4 affine transformation matrix.
    """
    matrix = get_translation(translation)
    if rotvec is not None:
        matrix = torch.matmul(matrix, get_rotation_euler(rotvec))
    if scales is not None:
        matrix = torch.matmul(get_scaling(scales), matrix)
    return matrix


def convert_v2v_to_torch(v2v: torch.Tensor, source_shape, target_shape=None) -> torch.tensor:
    """
    Convert a vox2vox affine transformation matrix to a torch transformation matrix.

    This function accounts for scaling and translation based on the 3D shapes of the source and
    target volumes.

    Parameters
    ==========
        v2v (torch.Tensor): vox2vox transformation matrix (4x4) from source to target space.
        source_shape (tuple[int]): Shape of the source image (Depth, Height, Width).
        target_shape (Optional[tuple[int]]): Shape of the target image (Depth, Height, Width).
            If not provided, it defaults to `source_shape`.

    Returns
    =======
        torch.Tensor: A PyTorch-compatible affine transformation matrix (3x4) suitable for
            grid-sampling operations.
    """
    if target_shape is None:
        target_shape = source_shape
    # Validate the input transformation shape
    if v2v.shape != torch.Size([4, 4]):
        raise ValueError(f"Expected v2v of shape (4, 4), but got {v2v.shape}.")
    inv_transform = torch.inverse(v2v)
    ndim = 3
    device = inv_transform.device
    # Prepare scale factors for source and target grid spaces
    scale_factor_source = torch.tensor(list(reversed(source_shape)), dtype=v2v.dtype, device=device) / 2.0
    scale_factor_target = torch.tensor(list(reversed(target_shape)), dtype=v2v.dtype, device=device) / 2.0
    # Rescale from relative coordinates (-1, 1) --> image coordinates and move center
    scale_factor_target = torch.cat((scale_factor_target, torch.tensor([1.0], device=device)))
    source2relative = torch.diag(scale_factor_target)
    source2relative[:-1, -1] += scale_factor_target[:-1] - 0.5
    # Rescale to relative coordinates and move center to align with PyTorch's grid-space
    scale_factor_source = torch.cat((scale_factor_source, torch.tensor([1.0], device=device)))
    relative2target = torch.diag(1.0 / scale_factor_source)
    relative2target[:-1, -1] += -1 + 0.5 / scale_factor_source[:-1]
    # Combine transformations to get the final affine transformation
    relative2target = relative2target.to(inv_transform.device)
    source2relative = source2relative.to(inv_transform.device)
    torch_transform = torch.matmul(relative2target, inv_transform)
    torch_transform = torch.matmul(torch_transform, source2relative)
    # Rearrange axes back to original order and include homogeneous coordinates
    ii = list(reversed(range(ndim))) + [ndim]
    torch_transform = torch_transform[..., ii, :]
    torch_transform = torch_transform[..., ii]
    return torch_transform[:3,:4]


def convert_torch_to_v2v(torch_transform: torch.Tensor, source_shape, target_shape=None) -> torch.tensor:
    """
    Convert a torch transformation matrix (3x4) to a vox2vox transformation matrix (4x4).
    Parameters:
        torch_transform (torch.Tensor): A 3x4 transformation matrix (torch format).
        source_shape (tuple/list): Shape of the source image (D, H, W).
        target_shape (tuple/list, optional): Shape of the target image (D, H, W).
            If not provided, it defaults to source_shape.

    Returns:
        torch.Tensor: A 4x4 vox2vox transformation matrix.
    """
    if target_shape is None:
        target_shape = source_shape
    if torch_transform.size() != torch.Size([4, 4]) :
        raise ValueError(f"Torch affine shape {torch_transform.size()} should be (4,4)!")
    ndim = 3
    # Prepare scale factors for source and target
    scale_factor_source = torch.as_tensor(list(reversed(source_shape))) / 2.0
    scale_factor_target = torch.as_tensor(list(reversed(target_shape))) / 2.0
    # Create diagonal scaling and translation matrices
    scale_factor_source = torch.cat((scale_factor_source, torch.tensor([1.0])), dim=0)
    scale_factor_target = torch.cat((scale_factor_target, torch.tensor([1.0])), dim=0)
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

