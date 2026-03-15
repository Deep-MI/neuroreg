from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeAlias

import nibabel as nib
import numpy as np
import numpy.typing as npt

from .matrices import convert_transform_type

logger = logging.getLogger(__name__)

# ── internal helper ────────────────────────────────────────────────────────

_AnyHeader: TypeAlias = (
    str
    | Path
    | nib.nifti1.Nifti1Header
    | nib.freesurfer.mghformat.MGHHeader
    | nib.nifti1.Nifti1Image
    | nib.MGHImage
    | dict
)


def _header_info(src: _AnyHeader) -> dict:
    """Extract LTA volume-info fields from various input types.

    Accepted inputs
    ---------------
    * **str / Path** – file path: the header is loaded without reading image data.
    * **nibabel header** (``Nifti1Header``, ``MGHHeader``, …) – used directly;
      the affine is obtained via ``header.get_best_affine()``.
    * **nibabel image** (``Nifti1Image``, ``MGHImage``, …) – ``img.affine`` and
      ``img.header`` are used.  Because ``nib.load`` is lazy, no voxel data is
      read unless the caller has already called ``get_fdata()``.
    * **dict** with keys ``dims``, ``delta``, ``Mdc``, ``Pxyz_c`` – returned
      unchanged (legacy / internal path).

    Convention
    ----------
    ``Mdc`` columns are the unit direction cosines of the x/y/z voxel axes in
    scanner-RAS (``affine[:3,:3] / zooms``).  ``Pxyz_c`` uses FreeSurfer's
    ``shape/2`` centre-voxel convention.
    """
    if isinstance(src, dict):
        return src

    # path string → load header only (no data)
    if isinstance(src, (str, Path)):
        src = nib.load(src).header

    # nibabel image → extract header and affine
    if hasattr(src, 'affine'):
        affine = src.affine
        header = src.header
    else:
        # bare nibabel header
        affine = src.get_best_affine()
        header = src

    shape = [int(x) for x in header.get_data_shape()[:3]]
    zooms = np.array(header.get_zooms()[:3], dtype=float)

    return {
        'dims':   shape,
        'delta':  zooms.tolist(),
        'Mdc':    affine[:3, :3] / zooms,
        'Pxyz_c': affine[:3, :3] @ (np.array(shape) / 2.0) + affine[:3, 3],
    }


# ── public API ─────────────────────────────────────────────────────────────

def write_lta(
        filename: str,
        T: npt.ArrayLike,
        src_fname: str,
        src_img: _AnyHeader,
        dst_fname: str,
        dst_img: _AnyHeader,
        lta_type: int = 1
) -> None:
    """Write a FreeSurfer ``.lta`` file.

    Parameters
    ----------
    filename : str
        Output path.
    T : array-like, shape (4, 4)
        Transformation matrix to store.
    src_fname : str
        Filename of the source (moving) image, stored as metadata.
    src_img : path, nibabel header, nibabel image, or dict
        Source image geometry.  Accepted types:

        * ``str`` / ``Path`` – file is opened for the header only (no data loaded).
        * nibabel header (``Nifti1Header``, ``MGHHeader``, …).
        * nibabel image (``Nifti1Image``, ``MGHImage``, …) – data is not read.
        * ``dict`` with keys ``dims``, ``delta``, ``Mdc``, ``Pxyz_c`` (legacy).
    dst_fname : str
        Filename of the destination (target) image, stored as metadata.
    dst_img : path, nibabel header, nibabel image, or dict
        Destination image geometry (same accepted types as *src_img*).
    lta_type : {0, 1}
        Transform type: ``0`` = LINEAR_VOX_TO_VOX, ``1`` = LINEAR_RAS_TO_RAS
        (default).
    """
    import getpass
    from datetime import datetime

    if lta_type not in (0, 1):
        raise ValueError(
            f"write_lta: lta_type must be 0 (LINEAR_VOX_TO_VOX) or "
            f"1 (LINEAR_RAS_TO_RAS), got {lta_type!r}."
        )

    src = _header_info(src_img)
    dst = _header_info(dst_img)

    # validate required keys (only matters for the dict path)
    for role, info in (("src", src), ("dst", dst)):
        for field in ("dims", "delta", "Mdc", "Pxyz_c"):
            if field not in info:
                raise ValueError(
                    f"write_lta: {role} header dict is missing required field '{field}'"
                )

    src_dims  = " ".join(str(int(x)) for x in src["dims"])
    src_vsize = " ".join(f"{float(x):.15e}" for x in src["delta"])
    src_mdc   = src["Mdc"]
    src_c     = src["Pxyz_c"]

    dst_dims  = " ".join(str(int(x)) for x in dst["dims"])
    dst_vsize = " ".join(f"{float(x):.15e}" for x in dst["delta"])
    dst_mdc   = dst["Mdc"]
    dst_c     = dst["Pxyz_c"]

    type_name = "LINEAR_VOX_TO_VOX" if lta_type == 0 else "LINEAR_RAS_TO_RAS"

    with open(filename, "w") as f:
        f.write(f"# transform file {filename}\n")
        f.write(f"# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n")
        f.write(f"type      = {lta_type} # {type_name}\n")
        f.write("nxforms   = 1\n")
        f.write("mean      = 0.0 0.0 0.0\n")
        f.write("sigma     = 1.0\n")
        f.write("1 4 4\n")
        # Normalise T to a plain (4, 4) float64 ndarray regardless of whether
        # the caller passed a np.ndarray, torch.Tensor, or other array-like.
        if hasattr(T, 'detach'):          # torch.Tensor
            T_arr = T.detach().cpu().numpy()
        else:
            T_arr = np.asarray(T)
        T_arr = T_arr.reshape(4, 4).astype(float)
        for row in T_arr:
            f.write(" ".join(f"{v:.15e}" for v in row) + "\n")
        f.write("\n")
        f.write("src volume info\n")
        f.write("valid = 1  # volume info valid\n")
        f.write(f"filename = {src_fname}\n")
        f.write(f"volume = {src_dims}\n")
        f.write(f"voxelsize = {src_vsize}\n")
        f.write("xras   = " + " ".join(f"{x:.15e}" for x in src_mdc[:, 0]) + "\n")
        f.write("yras   = " + " ".join(f"{x:.15e}" for x in src_mdc[:, 1]) + "\n")
        f.write("zras   = " + " ".join(f"{x:.15e}" for x in src_mdc[:, 2]) + "\n")
        f.write("cras   = " + " ".join(f"{x:.15e}" for x in src_c) + "\n")
        f.write("dst volume info\n")
        f.write("valid = 1  # volume info valid\n")
        f.write(f"filename = {dst_fname}\n")
        f.write(f"volume = {dst_dims}\n")
        f.write(f"voxelsize = {dst_vsize}\n")
        f.write("xras   = " + " ".join(f"{x:.15e}" for x in dst_mdc[:, 0]) + "\n")
        f.write("yras   = " + " ".join(f"{x:.15e}" for x in dst_mdc[:, 1]) + "\n")
        f.write("zras   = " + " ".join(f"{x:.15e}" for x in dst_mdc[:, 2]) + "\n")
        f.write("cras   = " + " ".join(f"{x:.15e}" for x in dst_c) + "\n")

    logger.debug("Wrote LTA (%s): %s", type_name, filename)

def read_lta(filename: str, lta_type: int | None = None) -> dict:
    """Read a FreeSurfer LTA file and return its contents.

    Parameters
    ----------
    filename : str
        Path to the ``.lta`` file.
    lta_type : {0, 1, None}, optional
        If ``None`` (default) the matrix is returned exactly as stored.
        If set to ``LINEAR_VOX_TO_VOX`` (0) or ``LINEAR_RAS_TO_RAS`` (1)
        and the file contains a different type, the matrix is automatically
        converted using :func:`convert_lta_matrix` and ``"type"`` in the
        returned dict is updated to reflect the requested type.
        Conversion requires that the ``src`` / ``dst`` volume info blocks
        contain valid ``xras`` / ``yras`` / ``zras`` / ``cras`` /
        ``voxelsize`` fields so that the affine matrices can be
        reconstructed.

    Returns
    -------
    dict with keys:

    ``"matrix"`` : np.ndarray, shape (4, 4)
        The transformation matrix (converted if *lta_type* was specified).
    ``"type"`` : int
        Transform type of the returned matrix
        (0 = LINEAR_VOX_TO_VOX, 1 = LINEAR_RAS_TO_RAS).
    ``"src"`` : dict
        Source volume header fields (``filename``, ``volume``, ``voxelsize``,
        ``xras``, ``yras``, ``zras``, ``cras``).
    ``"dst"`` : dict
        Destination volume header fields (same keys as ``src``).

    Raises
    ------
    ValueError
        If the transformation matrix block cannot be found, or if a
        conversion is requested but the volume-info affine data are missing.
    """
    with open(filename) as f:
        lines = f.readlines()

    result: dict = {"src": {}, "dst": {}}

    # ── transform type ─────────────────────────────────────────────
    for line in lines:
        if line.startswith("type"):
            result["type"] = int(line.split("=")[1].split("#")[0].strip())
            break

    # ── 4×4 matrix ─────────────────────────────────────────────────
    mat = []
    for i, line in enumerate(lines):
        if "1 4 4" in line:
            for row in lines[i + 1: i + 5]:
                mat.append([float(v) for v in row.strip().split()])
            break
    if len(mat) != 4:
        raise ValueError(f"Could not parse 4×4 matrix from {filename}")
    result["matrix"] = np.array(mat)

    # ── volume info blocks ──────────────────────────────────────────
    def _parse_vol_block(start_idx: int) -> dict:
        info: dict = {}
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("filename"):
                info["filename"] = line.split("=", 1)[1].strip()
            elif line.startswith("volume"):
                info["volume"] = [int(v) for v in line.split("=", 1)[1].split()]
            elif line.startswith("voxelsize"):
                info["voxelsize"] = [float(v) for v in line.split("=", 1)[1].split()]
            elif line.startswith("xras"):
                info["xras"] = [float(v) for v in line.split("=", 1)[1].split()]
            elif line.startswith("yras"):
                info["yras"] = [float(v) for v in line.split("=", 1)[1].split()]
            elif line.startswith("zras"):
                info["zras"] = [float(v) for v in line.split("=", 1)[1].split()]
            elif line.startswith("cras"):
                info["cras"] = [float(v) for v in line.split("=", 1)[1].split()]
            elif line.startswith("dst volume info"):
                break
        return info

    for i, line in enumerate(lines):
        if line.strip().startswith("src volume info"):
            result["src"] = _parse_vol_block(i + 1)
        elif line.strip().startswith("dst volume info"):
            result["dst"] = _parse_vol_block(i + 1)

    # ── optional type conversion ────────────────────────────────────
    if lta_type is not None and lta_type != result["type"]:
        def _affine_from_info(info: dict, role: str) -> np.ndarray:
            required = ("xras", "yras", "zras", "cras", "voxelsize", "volume")
            missing = [k for k in required if k not in info]
            if missing:
                raise ValueError(
                    f"Cannot convert LTA type: {role} volume info is missing "
                    f"fields: {missing}"
                )
            vs = info["voxelsize"]
            A = np.eye(4)
            A[:3, 0] = np.array(info["xras"]) * vs[0]
            A[:3, 1] = np.array(info["yras"]) * vs[1]
            A[:3, 2] = np.array(info["zras"]) * vs[2]
            # cras is the RAS coordinate of the centre voxel (dims/2), not voxel 0.
            # Affine convention: ras = A[:3,:3] @ vox + A[:3,3]
            # => at vox = dims/2: cras = A[:3,:3] @ (dims/2) + A[:3,3]
            # => A[:3,3] = cras - A[:3,:3] @ (dims/2)
            dims = np.array(info["volume"], dtype=float)
            A[:3, 3] = np.array(info["cras"]) - A[:3, :3] @ (dims / 2.0)
            return A

        src_affine = _affine_from_info(result["src"], "src")
        dst_affine = _affine_from_info(result["dst"], "dst")
        result["matrix"] = convert_transform_type(
            result["matrix"], src_affine, dst_affine,
            from_type=result["type"], to_type=lta_type
        )
        result["type"] = lta_type

    return result


# ── transform comparison metrics ────────────────────────────────────────────

def _rot_log_norm(R: np.ndarray) -> float:
    """Frobenius norm of the matrix logarithm of a 3×3 rotation matrix.

    Equal to ``sqrt(2) * theta`` where *theta* is the rotation angle in
    radians.  Equivalent to the geodesic distance on SO(3).

    This is the helper used by :func:`rigid_dist`.
    """
    # tr R = 1 + 2 cos(theta)  →  cos(theta) = (tr - 1) / 2
    cos_theta = np.clip(0.5 * (np.trace(R[:3, :3]) - 1.0), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.sqrt(2.0) * theta)


def rigid_dist(
        M1: npt.ArrayLike,
        M2: npt.ArrayLike | None = None,
) -> float:
    """Rigid-transform distance between *M1* and *M2* (or *M1* vs identity).

    .. math::

        D = \\sqrt{\\|T_d\\|^2 + \\|\\log R_d\\|_F^2}

    where :math:`d = M_1^{-1} M_2` when *M2* is given, else :math:`d = M_1`.
    :math:`T_d` is the translation part and :math:`\\|\\log R_d\\|_F` is the
    Frobenius norm of the rotation-matrix logarithm (``sqrt(2)`` × rotation
    angle in radians).

    Corresponds to **dist type 1** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M1 : array-like, shape (4, 4)
        First (or only) rigid transform.
    M2 : array-like, shape (4, 4), optional
        Second rigid transform.  When ``None``, the distance to the
        identity is returned.

    Returns
    -------
    float
        Rigid-transform distance in mixed units (mm for translation,
        radians for rotation, added in quadrature).
    """
    M1a = np.asarray(M1, dtype=float)
    d = M1a if M2 is None else (np.linalg.inv(M1a) @ np.asarray(M2, dtype=float))
    tdq = float(np.dot(d[:3, 3], d[:3, 3]))
    rd = _rot_log_norm(d[:3, :3])
    return float(np.sqrt(rd * rd + tdq))


def affine_dist(
        M1: npt.ArrayLike,
        M2: npt.ArrayLike | None = None,
        radius: float = 100.0,
) -> float:
    """RMS affine-transform distance (Jenkinson 1999).

    .. math::

        D = \\sqrt{\\frac{r^2}{5} \\operatorname{Tr}(A^\\top A) + \\|T_d\\|^2}

    where :math:`d = M_1 - M_2` (or :math:`M_1 - I` when *M2* is ``None``),
    *A* is the upper-left 3×3 linear part of *d*, and :math:`T_d` is the
    translation column.  *r* is the assumed brain radius in mm.

    Reference: Jenkinson (1999), *A method for motion correction of
    fMRI time-series*, FMRIB Technical Report TR99MJ1.

    Corresponds to **dist type 2** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M1 : array-like, shape (4, 4)
    M2 : array-like, shape (4, 4), optional
        When ``None``, the distance to identity is returned.
    radius : float
        Radius of the brain sphere in mm (default 100).

    Returns
    -------
    float
        RMS displacement in mm.
    """
    M1a = np.asarray(M1, dtype=float)
    d = M1a - (np.eye(4) if M2 is None else np.asarray(M2, dtype=float))
    tdq = float(np.dot(d[:3, 3], d[:3, 3]))
    A = d[:3, :3]
    tr = float(np.trace(A.T @ A))
    return float(np.sqrt((radius ** 2 / 5.0) * tr + tdq))


def corner_diff(
        M: npt.ArrayLike,
        src_shape: tuple[int, int, int],
        M2: npt.ArrayLike | None = None,
        src_affine: npt.ArrayLike | None = None,
) -> float:
    """Mean displacement at the 8 corners of a volume.

    Each of the 8 corners of the source volume is mapped through *M* (and
    *M2* when given), and the mean Euclidean distance between the two mapped
    positions is returned.  When *M2* is ``None``, the distance from each
    mapped corner to its original position is used (i.e. displacement from
    identity).

    Corresponds to **dist type 3** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M : array-like, shape (4, 4)
        First (or only) transform matrix.
    src_shape : tuple of int (i_size, j_size, k_size)
        Voxel dimensions of the source volume, used to place the corners.
    M2 : array-like, shape (4, 4), optional
        Second transform.  When ``None``, displacement from identity is
        measured.
    src_affine : array-like, shape (4, 4), optional
        Voxel-to-RAS affine of the source image.  When given, corner voxel
        coordinates are first converted to RAS (mm) space before applying
        *M* / *M2*; the returned distance is in mm.  When ``None``, corners
        remain in voxel units and *M* is expected to be a vox-to-vox matrix.

    Returns
    -------
    float
        Mean displacement across the 8 corners (mm when *src_affine* is
        provided, voxels otherwise).
    """
    Si, Sj, Sk = src_shape
    # Build homogeneous voxel coordinates for all 8 corners
    corners_vox = np.array(
        [[i * (Si - 1), j * (Sj - 1), k * (Sk - 1), 1.0]
         for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=float,
    )  # (8, 4)

    if src_affine is not None:
        # map vox → RAS; M is then a RAS-to-RAS matrix
        corners = (np.asarray(src_affine, dtype=float) @ corners_vox.T).T
    else:
        corners = corners_vox  # M is a vox-to-vox matrix

    p1 = (np.asarray(M, dtype=float) @ corners.T).T
    p2 = corners if M2 is None else (np.asarray(M2, dtype=float) @ corners.T).T
    return float(np.mean(np.linalg.norm(p1[:, :3] - p2[:, :3], axis=1)))


def sphere_diff(
        M1: npt.ArrayLike,
        M2: npt.ArrayLike | None = None,
        radius: float = 100.0,
) -> float:
    """Maximum displacement on a sphere of given radius.

    Samples roughly 1 600 points uniformly on a sphere of *radius* mm
    centred at the coordinate origin and returns the maximum displacement
    caused by the transform difference.

    .. math::

        M_d = M_1^{-1} M_2 \\quad (\\text{or } M_1 \\text{ when } M_2 = \\text{None})

        \\text{displacement}(p) = \\|M_d \\, p - p\\|

    Corresponds to **dist type 4** in FreeSurfer's ``lta_diff``.

    Parameters
    ----------
    M1 : array-like, shape (4, 4)
    M2 : array-like, shape (4, 4), optional
        When ``None``, the maximum displacement of *M1* from identity is
        returned.
    radius : float
        Sphere radius in mm (default 100, roughly the head radius).

    Returns
    -------
    float
        Maximum displacement in mm.
    """
    M1a = np.asarray(M1, dtype=float)
    Md = M1a if M2 is None else (np.linalg.inv(M1a) @ np.asarray(M2, dtype=float))

    # Replicate the C++ sampling loop from MyMatrix::sphereDiff / lta_diff.cpp
    pts: list[list[float]] = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    n = 10  # latitude bands (same default as C++)
    for i in range(-n + 1, n):
        angle1 = (i * np.pi * 0.5) / n
        r1 = np.cos(angle1)
        h = np.sin(angle1)
        n_long = int(4.0 * n * r1)
        for j in range(n_long):
            angle2 = (2.0 * np.pi * j) / n_long
            pts.append([radius * r1 * np.cos(angle2),
                        radius * r1 * np.sin(angle2),
                        radius * h])

    pts_arr = np.array(pts, dtype=float)                      # (N, 3)
    hom = np.hstack([pts_arr, np.ones((len(pts_arr), 1))])    # (N, 4)
    mapped = (Md @ hom.T).T                                   # (N, 4)
    displacements = np.linalg.norm(mapped[:, :3] - pts_arr, axis=1)
    return float(np.max(displacements))


def decompose_transform(M: npt.ArrayLike) -> dict:
    """Polar decomposition of a 4×4 affine matrix.

    Decomposes the upper-left 3×3 linear part as

    .. math::

        A = R \\cdot S \\cdot \\operatorname{diag}(\\text{scales})

    where *R* is a proper rotation matrix, *S* is a shear matrix (ones on
    diagonal), and *diag(scales)* captures anisotropic scaling.

    Corresponds to **dist type 7** in FreeSurfer's ``lta_diff`` (decompose).

    Parameters
    ----------
    M : array-like, shape (4, 4)
        Affine matrix to decompose.  Pre-compose transforms before calling
        if a relative decomposition is needed (e.g. ``decompose_transform(M1 @ M2)``).

    Returns
    -------
    dict
        ``rotation`` : ndarray (3, 3)
            Rotation matrix (det = +1).
        ``rot_vec`` : ndarray (3,)
            Rotation vector (axis × angle, in radians).
        ``rot_angle_deg`` : float
            Rotation angle in degrees.
        ``shear`` : ndarray (3, 3)
            Shear matrix (unit diagonal).
        ``scales`` : ndarray (3,)
            Per-axis scale factors.
        ``translation`` : ndarray (3,)
            Translation vector (mm).
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

    # Polar decomposition: A = R @ P  (R orthogonal, P positive semi-definite)
    R, P = polar(A, side='right')

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        R, P = -R, -P

    # Decompose P further: P = S @ diag(scales)
    # scales = diagonal entries of P; S = P with each column divided by scales[col]
    scales = np.diag(P).copy()
    S = P / np.where(scales != 0, scales, 1.0)[np.newaxis, :]

    # Rotation vector and angle via scipy (handles edge cases at theta=0 and pi)
    rot_obj = Rotation.from_matrix(R)
    rot_vec = rot_obj.as_rotvec()              # axis × angle (radians)
    rot_angle_deg = float(np.degrees(np.linalg.norm(rot_vec)))

    return {
        'rotation':       R,
        'rot_vec':        rot_vec,
        'rot_angle_deg':  rot_angle_deg,
        'shear':          S,
        'scales':         scales,
        'translation':    t,
        'abs_trans':      float(np.linalg.norm(t)),
        'determinant':    float(np.linalg.det(M_arr)),
    }

