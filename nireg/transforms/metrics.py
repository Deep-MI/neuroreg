"""Pure transform-metric utilities.

These functions operate on raw affine matrices (and, for corner-based metrics,
explicit source geometry) without any knowledge of the FreeSurfer ``.lta`` file
format. Geometry-aware LTA wrappers live in :mod:`nireg.transforms.lta`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def _rot_log_norm(R: np.ndarray) -> float:
    """Frobenius norm of the matrix logarithm of a 3×3 rotation matrix.

    Equal to ``sqrt(2) * theta`` where *theta* is the rotation angle in
    radians. Equivalent to the geodesic distance on SO(3).

    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    float
        Frobenius norm of ``log(R)``.
    """
    cos_theta = np.clip(0.5 * (np.trace(R[:3, :3]) - 1.0), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.sqrt(2.0) * theta)


def rigid_dist(
    M1: npt.ArrayLike,
    M2: npt.ArrayLike | None = None,
) -> float:
    r"""Rigid-transform distance between *M1* and *M2* (or *M1* vs identity).

    Both matrices must be **RAS-to-RAS**; the translation component is then
    in mm and the result has consistent physical units.

    .. math::

        D = \sqrt{\|T_d\|^2 + \|\log R_d\|_F^2}

    where :math:`d = M_1^{-1} M_2` when *M2* is given, else :math:`d = M_1`.
    :math:`T_d` is the translation part (mm) and :math:`\|\log R_d\|_F` is
    the Frobenius norm of the rotation-matrix logarithm (``sqrt(2)`` × rotation
    angle in radians).

    Corresponds to **dist type 1** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M1 : array-like, shape (4, 4)
        First (or only) RAS-to-RAS rigid transform.
    M2 : array-like, shape (4, 4), optional
        Second RAS-to-RAS rigid transform. When ``None``, the distance to the
        identity is returned.

    Returns
    -------
    float
        Rigid-transform distance (mm and radians added in quadrature).
    """
    M1a = np.asarray(M1, dtype=float)
    d = M1a if M2 is None else (np.linalg.inv(M1a) @ np.asarray(M2, dtype=float))
    tdq = float(np.dot(d[:3, 3], d[:3, 3]))
    rd = _rot_log_norm(d[:3, :3])
    return float(np.sqrt(rd * rd + tdq))


def affine_dist(
    M1: npt.ArrayLike | torch.Tensor,
    M2: npt.ArrayLike | torch.Tensor | None = None,
    radius: float = 100.0,
) -> float:
    r"""RMS affine-transform distance (Jenkinson 1999).

    Both matrices must be **RAS-to-RAS**; the translation component is then
    in mm and *radius* has a consistent mm interpretation.

    .. math::

        D = \sqrt{\frac{r^2}{5} \operatorname{Tr}(A^\top A) + \|T_d\|^2}

    where :math:`d = M_1 - M_2` (or :math:`M_1 - I` when *M2* is ``None``),
    *A* is the upper-left 3×3 linear part of *d*, and :math:`T_d` is the
    translation column (mm). *r* is the assumed brain radius in mm.

    Reference: Jenkinson (1999), *A method for motion correction of
    fMRI time-series*, FMRIB Technical Report TR99MJ1.

    Corresponds to **dist type 2** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M1 : array-like or torch.Tensor, shape (4, 4)
        First (or only) RAS-to-RAS transform.
    M2 : array-like or torch.Tensor, shape (4, 4), optional
        Second RAS-to-RAS transform. When ``None``, the distance to identity
        is returned.
    radius : float, default=100.0
        Radius of the brain sphere in mm.

    Returns
    -------
    float
        RMS displacement in mm.
    Notes
    -----
    The function accepts either NumPy-like arrays or torch tensors. When torch
    tensors are passed, the computation stays in torch until the final scalar
    conversion, which keeps it suitable for CPU/GPU registration loops.
    """
    if isinstance(M1, torch.Tensor) or isinstance(M2, torch.Tensor):
        M1_t = M1 if isinstance(M1, torch.Tensor) else torch.as_tensor(M1)
        if M2 is None:
            M2_t = torch.eye(4, dtype=M1_t.dtype, device=M1_t.device)
        elif isinstance(M2, torch.Tensor):
            M2_t = M2.to(device=M1_t.device)
        else:
            M2_t = torch.as_tensor(M2, dtype=M1_t.dtype, device=M1_t.device)
        dT = M1_t.double() - M2_t.double()
        tdq = (dT[:3, 3] ** 2).sum()
        dR = dT[:3, :3]
        tr = torch.trace(dR.T @ dR)
        return float(torch.sqrt(tr * radius * radius / 5.0 + tdq))

    dT = np.asarray(M1, dtype=float) - (np.eye(4) if M2 is None else np.asarray(M2, dtype=float))
    tdq = float(np.dot(dT[:3, 3], dT[:3, 3]))
    tr = float(np.trace(dT[:3, :3].T @ dT[:3, :3]))
    return float(np.sqrt((radius * radius / 5.0) * tr + tdq))


def corner_dist(
    M: npt.ArrayLike,
    src_shape: tuple[int, int, int],
    M2: npt.ArrayLike | None = None,
    src_affine: npt.ArrayLike | None = None,
) -> float:
    """Mean displacement at the 8 corners of the source volume.

    Places the 8 corners of the source volume in RAS space using *src_affine*,
    maps them through the RAS-to-RAS transform(s), and returns the mean
    Euclidean displacement.

    **Image-specific metric**: the result depends on the source image shape
    and affine, not just the transform. The corners are at the edges of the
    source FOV — for a large or padded FOV they may be far from brain tissue,
    making the result less representative than :func:`sphere_dist`. When
    comparing two transforms (*M* and *M2*), both must register the same
    source image (so *src_affine* and *src_shape* are unambiguous).

    Corresponds to **dist type 3** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M : array-like, shape (4, 4)
        First (or only) RAS-to-RAS transform.
    src_shape : tuple of int
        Voxel dimensions of the source volume.
    M2 : array-like, shape (4, 4), optional
        Second RAS-to-RAS transform. When ``None``, displacement from identity
        is measured.
    src_affine : array-like, shape (4, 4), optional
        Voxel-to-RAS affine of the source image. When given, corners are
        converted to RAS (mm) before applying the transforms; the result is in
        mm. When ``None``, corners remain in voxel units.

    Returns
    -------
    float
        Mean corner displacement in mm (or voxels if *src_affine* is ``None``).
    """
    Si, Sj, Sk = src_shape
    corners_vox = np.array(
        [[i * (Si - 1), j * (Sj - 1), k * (Sk - 1), 1.0] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=float,
    )
    corners = (np.asarray(src_affine, dtype=float) @ corners_vox.T).T if src_affine is not None else corners_vox
    p1 = (np.asarray(M, dtype=float) @ corners.T).T
    p2 = corners if M2 is None else (np.asarray(M2, dtype=float) @ corners.T).T
    return float(np.mean(np.linalg.norm(p1[:, :3] - p2[:, :3], axis=1)))


def sphere_dist(
    M1: npt.ArrayLike,
    M2: npt.ArrayLike | None = None,
    radius: float = 100.0,
) -> float:
    r"""Maximum displacement on a sphere of given radius (mm).

    Samples approximately 1,600 points on a sphere of *radius* mm centred at
    the RAS origin and returns the peak displacement caused by the transform
    difference. Both matrices must be **RAS-to-RAS** so that the sphere is in a
    meaningful physical space and displacements are in mm.

    .. math::

        M_d = M_1^{-1} M_2 \quad (\text{or } M_1 \text{ when } M_2 = \text{None})

        \mathrm{displacement}(p) = \|M_d \, p - p\|

    Unlike :func:`corner_dist` this metric is image-independent: the sphere is
    a canonical approximation of the head and the result is a pure property of
    the transform.

    Corresponds to **dist type 4** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M1 : array-like, shape (4, 4)
        First (or only) RAS-to-RAS transform.
    M2 : array-like, shape (4, 4), optional
        Second RAS-to-RAS transform. When ``None``, displacement from identity
        is returned.
    radius : float, default=100.0
        Sphere radius in mm (roughly the head radius).

    Returns
    -------
    float
        Maximum displacement in mm.
    """
    M1a = np.asarray(M1, dtype=float)
    Md = M1a if M2 is None else (np.linalg.inv(M1a) @ np.asarray(M2, dtype=float))

    pts: list[list[float]] = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    n = 10
    for i in range(-n + 1, n):
        angle1 = (i * np.pi * 0.5) / n
        r1 = np.cos(angle1)
        h = np.sin(angle1)
        n_long = int(4.0 * n * r1)
        for j in range(n_long):
            angle2 = (2.0 * np.pi * j) / n_long
            pts.append([radius * r1 * np.cos(angle2), radius * r1 * np.sin(angle2), radius * h])

    pts_arr = np.array(pts, dtype=float)
    hom = np.hstack([pts_arr, np.ones((len(pts_arr), 1))])
    mapped = (Md @ hom.T).T
    return float(np.max(np.linalg.norm(mapped[:, :3] - pts_arr, axis=1)))


def decompose_transform(M: npt.ArrayLike) -> dict:
    r"""Polar decomposition of a 4×4 affine matrix.

    Decomposes the upper-left 3×3 linear part as

    .. math::

        A = R \cdot S \cdot \operatorname{diag}(\text{scales})

    where *R* is a proper rotation matrix, *S* is a shear matrix (ones on the
    diagonal), and *diag(scales)* captures anisotropic scaling. The translation
    vector is in mm because the input is expected to be RAS-to-RAS.

    Corresponds to **dist type 7** in FreeSurfer's ``lta_diff`` (decompose).

    Parameters
    ----------
    M : array-like, shape (4, 4)
        Affine matrix to decompose. Pre-compose transforms before calling if a
        relative decomposition is needed (e.g. ``decompose_transform(M1 @ M2)``).

    Returns
    -------
    dict
        ``rotation`` : ndarray, shape (3, 3)
            Rotation matrix (det = +1).
        ``rot_vec`` : ndarray, shape (3,)
            Rotation vector (axis × angle, radians).
        ``rot_angle_deg`` : float
            Rotation angle in degrees.
        ``shear`` : ndarray, shape (3, 3)
            Shear matrix (unit diagonal).
        ``scales`` : ndarray, shape (3,)
            Per-axis scale factors.
        ``translation`` : ndarray, shape (3,)
            Translation vector in mm.
        ``abs_trans`` : float
            Euclidean norm of the translation vector (mm).
        ``determinant`` : float
            Determinant of the full 4×4 matrix.
    """
    from scipy.linalg import polar
    from scipy.spatial.transform import Rotation

    M_arr = np.asarray(M, dtype=float)
    A = M_arr[:3, :3]
    t = M_arr[:3, 3].copy()

    R, P = polar(A, side="right")
    if np.linalg.det(R) < 0:
        R, P = -R, -P

    scales = np.diag(P).copy()
    S = P / np.where(scales != 0, scales, 1.0)[np.newaxis, :]

    rot_obj = Rotation.from_matrix(R)
    rot_vec = rot_obj.as_rotvec()
    rot_angle_deg = float(np.degrees(np.linalg.norm(rot_vec)))

    return {
        "rotation": R,
        "rot_vec": rot_vec,
        "rot_angle_deg": rot_angle_deg,
        "shear": S,
        "scales": scales,
        "translation": t,
        "abs_trans": float(np.linalg.norm(t)),
        "determinant": float(np.linalg.det(M_arr)),
    }


__all__ = [
    "affine_dist",
    "corner_dist",
    "decompose_transform",
    "rigid_dist",
    "sphere_dist",
]

