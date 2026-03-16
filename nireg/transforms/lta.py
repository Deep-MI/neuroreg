from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeAlias

import nibabel as nib
import numpy as np
import numpy.typing as npt

from .matrices import convert_transform_type

logger = logging.getLogger(__name__)

# ── type alias ─────────────────────────────────────────────────────────────

_AnyHeader: TypeAlias = (
    str
    | Path
    | nib.nifti1.Nifti1Header
    | nib.freesurfer.mghformat.MGHHeader
    | nib.nifti1.Nifti1Image
    | nib.MGHImage
    | dict
)

# ── private helpers ────────────────────────────────────────────────────────

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

    if isinstance(src, (str, Path)):
        src = nib.load(src).header

    if hasattr(src, 'affine'):
        affine = src.affine
        header = src.header
    else:
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


def _affine_from_info(info: dict) -> np.ndarray:
    """Reconstruct a 4×4 voxel-to-RAS affine from an LTA volume-info dict.

    Expects keys: ``xras``, ``yras``, ``zras``, ``cras``, ``voxelsize``,
    ``volume``.  These are the fields written by :meth:`LTA.write` and parsed
    by :meth:`LTA.read`.
    """
    required = ("xras", "yras", "zras", "cras", "voxelsize", "volume")
    missing = [k for k in required if k not in info]
    if missing:
        raise ValueError(f"LTA volume info is missing required fields: {missing}")
    vs   = info["voxelsize"]
    dims = np.array(info["volume"], dtype=float)
    A    = np.eye(4)
    A[:3, 0] = np.array(info["xras"]) * vs[0]
    A[:3, 1] = np.array(info["yras"]) * vs[1]
    A[:3, 2] = np.array(info["zras"]) * vs[2]
    A[:3, 3] = np.array(info["cras"]) - A[:3, :3] @ (dims / 2.0)
    return A


def _header_to_vol_info(hdr: dict, fname: str = '') -> dict:
    """Convert a :func:`_header_info` dict to LTA volume-info format.

    Maps the ``Mdc`` / ``Pxyz_c`` / ``dims`` / ``delta`` keys produced by
    :func:`_header_info` to the ``xras`` / ``yras`` / ``zras`` / ``cras`` /
    ``voxelsize`` / ``volume`` keys expected by :func:`_affine_from_info` and
    stored in ``.lta`` files.
    """
    Mdc    = np.asarray(hdr['Mdc'])
    Pxyz_c = np.asarray(hdr['Pxyz_c'])
    return {
        'filename': fname,
        'volume':   list(hdr['dims']),
        'voxelsize': list(hdr['delta']),
        'xras': Mdc[:, 0].tolist(),
        'yras': Mdc[:, 1].tolist(),
        'zras': Mdc[:, 2].tolist(),
        'cras': Pxyz_c.tolist(),
    }


# ── standalone distance functions ──────────────────────────────────────────

def _rot_log_norm(R: np.ndarray) -> float:
    """Frobenius norm of the matrix logarithm of a 3×3 rotation matrix.

    Equal to ``sqrt(2) * theta`` where *theta* is the rotation angle in
    radians.  Equivalent to the geodesic distance on SO(3).
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
    d   = M1a if M2 is None else (np.linalg.inv(M1a) @ np.asarray(M2, dtype=float))
    tdq = float(np.dot(d[:3, 3], d[:3, 3]))
    rd  = _rot_log_norm(d[:3, :3])
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
    d   = M1a - (np.eye(4) if M2 is None else np.asarray(M2, dtype=float))
    tdq = float(np.dot(d[:3, 3], d[:3, 3]))
    tr  = float(np.trace(d[:3, :3].T @ d[:3, :3]))
    return float(np.sqrt((radius ** 2 / 5.0) * tr + tdq))


def corner_dist(
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
    )
    corners = (
        (np.asarray(src_affine, dtype=float) @ corners_vox.T).T
        if src_affine is not None else corners_vox
    )
    p1 = (np.asarray(M, dtype=float) @ corners.T).T
    p2 = corners if M2 is None else (np.asarray(M2, dtype=float) @ corners.T).T
    return float(np.mean(np.linalg.norm(p1[:, :3] - p2[:, :3], axis=1)))


def sphere_dist(
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
    Md  = M1a if M2 is None else (np.linalg.inv(M1a) @ np.asarray(M2, dtype=float))

    pts: list[list[float]] = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    n = 10
    for i in range(-n + 1, n):
        angle1 = (i * np.pi * 0.5) / n
        r1 = np.cos(angle1)
        h  = np.sin(angle1)
        n_long = int(4.0 * n * r1)
        for j in range(n_long):
            angle2 = (2.0 * np.pi * j) / n_long
            pts.append([radius * r1 * np.cos(angle2),
                        radius * r1 * np.sin(angle2),
                        radius * h])

    pts_arr = np.array(pts, dtype=float)
    hom     = np.hstack([pts_arr, np.ones((len(pts_arr), 1))])
    mapped  = (Md @ hom.T).T
    return float(np.max(np.linalg.norm(mapped[:, :3] - pts_arr, axis=1)))


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

    rot_obj       = Rotation.from_matrix(R)
    rot_vec       = rot_obj.as_rotvec()
    rot_angle_deg = float(np.degrees(np.linalg.norm(rot_vec)))

    return {
        'rotation':      R,
        'rot_vec':       rot_vec,
        'rot_angle_deg': rot_angle_deg,
        'shear':         S,
        'scales':        scales,
        'translation':   t,
        'abs_trans':     float(np.linalg.norm(t)),
        'determinant':   float(np.linalg.det(M_arr)),
    }


# ── LTA class ──────────────────────────────────────────────────────────────

class LTA:
    """FreeSurfer Linear Transform Array.

    Wraps a 4×4 affine matrix together with source and destination volume
    geometry, mirroring the ``.lta`` file format used by FreeSurfer.

    Typical usage::

        lta = LTA.read("T2_to_T1.lta")
        print(lta.affine_dist())        # distance to identity
        print(lta.affine_dist(other))   # distance to another LTA
        lta.invert().write("T1_to_T2.lta")
    """

    def __init__(
        self,
        matrix: npt.ArrayLike,
        lta_type: int,
        src: dict,
        dst: dict,
    ) -> None:
        """
        Parameters
        ----------
        matrix : array-like, shape (4, 4)
        lta_type : {0, 1}
            0 = LINEAR_VOX_TO_VOX, 1 = LINEAR_RAS_TO_RAS.
        src : dict
            Source volume info (keys: volume, voxelsize, xras, yras, zras, cras).
        dst : dict
            Destination volume info (same keys).
        """
        self.matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        self.type   = lta_type
        self.src    = src
        self.dst    = dst

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def read(cls, filename: str | Path, lta_type: int | None = None) -> LTA:
        """Read a FreeSurfer ``.lta`` file.

        Parameters
        ----------
        filename : str or Path
        lta_type : {0, 1, None}, optional
            If given, convert the stored matrix to this type on load.

        Returns
        -------
        LTA
        """
        filename = str(filename)
        with open(filename) as f:
            lines = f.readlines()

        stored_type: int = 1
        for line in lines:
            if line.startswith('type'):
                stored_type = int(line.split('=')[1].split('#')[0].strip())
                break

        mat: list[list[float]] = []
        for i, line in enumerate(lines):
            if '1 4 4' in line:
                for row in lines[i + 1: i + 5]:
                    mat.append([float(v) for v in row.strip().split()])
                break
        if len(mat) != 4:
            raise ValueError(f'Could not parse 4×4 matrix from {filename}')

        def _parse_vol_block(start: int) -> dict:
            info: dict = {}
            for line in lines[start:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('filename'):
                    info['filename'] = line.split('=', 1)[1].strip()
                elif line.startswith('volume'):
                    info['volume'] = [int(v) for v in line.split('=', 1)[1].split()]
                elif line.startswith('voxelsize'):
                    info['voxelsize'] = [float(v) for v in line.split('=', 1)[1].split()]
                elif line.startswith('xras'):
                    info['xras'] = [float(v) for v in line.split('=', 1)[1].split()]
                elif line.startswith('yras'):
                    info['yras'] = [float(v) for v in line.split('=', 1)[1].split()]
                elif line.startswith('zras'):
                    info['zras'] = [float(v) for v in line.split('=', 1)[1].split()]
                elif line.startswith('cras'):
                    info['cras'] = [float(v) for v in line.split('=', 1)[1].split()]
                elif line.startswith('dst volume info'):
                    break
            return info

        src: dict = {}
        dst: dict = {}
        for i, line in enumerate(lines):
            if line.strip().startswith('src volume info'):
                src = _parse_vol_block(i + 1)
            elif line.strip().startswith('dst volume info'):
                dst = _parse_vol_block(i + 1)

        lta = cls(np.array(mat), stored_type, src, dst)

        if lta_type is not None and lta_type != stored_type:
            lta = cls(
                convert_transform_type(
                    lta.matrix, _affine_from_info(src), _affine_from_info(dst),
                    from_type=stored_type, to_type=lta_type,
                ),
                lta_type, src, dst,
            )

        return lta

    @classmethod
    def from_matrix(
        cls,
        matrix: npt.ArrayLike,
        src_fname: str,
        src_img: _AnyHeader,
        dst_fname: str,
        dst_img: _AnyHeader,
        lta_type: int = 1,
    ) -> LTA:
        """Create an LTA from a matrix and image geometry.

        Parameters
        ----------
        matrix : array-like, shape (4, 4)
            May be a ``torch.Tensor`` (detached automatically).
        src_fname : str
            Source filename stored as metadata in the ``.lta`` file.
        src_img : path, nibabel header/image, or dict
            Source image geometry.
        dst_fname : str
            Destination filename stored as metadata.
        dst_img : path, nibabel header/image, or dict
            Destination image geometry.
        lta_type : {0, 1}
            0 = LINEAR_VOX_TO_VOX, 1 = LINEAR_RAS_TO_RAS (default).
        """
        if hasattr(matrix, 'detach'):
            matrix = matrix.detach().cpu().numpy()
        M   = np.asarray(matrix, dtype=float).reshape(4, 4)
        src = _header_to_vol_info(_header_info(src_img), src_fname)
        dst = _header_to_vol_info(_header_info(dst_img), dst_fname)
        return cls(M, lta_type, src, dst)

    def write(self, filename: str | Path, lta_type: int | None = None) -> None:
        """Write to a FreeSurfer ``.lta`` file.

        Parameters
        ----------
        filename : str or Path
        lta_type : {0, 1, None}, optional
            Output transform type.  ``0`` = LINEAR_VOX_TO_VOX,
            ``1`` = LINEAR_RAS_TO_RAS.  When ``None`` (default) the matrix is
            written as stored.  When given and different from the stored type,
            the matrix is converted before writing; ``self`` is not mutated.
        """
        import getpass
        from datetime import datetime

        if lta_type is not None and lta_type not in (0, 1):
            raise ValueError(f'lta_type must be 0 or 1, got {lta_type!r}')

        out_type   = self.type if lta_type is None else lta_type
        out_matrix = (
            self.matrix if out_type == self.type
            else convert_transform_type(
                self.matrix, _affine_from_info(self.src), _affine_from_info(self.dst),
                from_type=self.type, to_type=out_type,
            )
        )

        filename  = str(filename)
        type_name = 'LINEAR_RAS_TO_RAS' if out_type == 1 else 'LINEAR_VOX_TO_VOX'

        def _fmt(vals: list) -> str:
            return ' '.join(f'{float(v):.15e}' for v in vals)

        with open(filename, 'w') as f:
            f.write(f'# transform file {filename}\n')
            f.write(f'# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n')
            f.write(f'type      = {out_type} # {type_name}\n')
            f.write('nxforms   = 1\n')
            f.write('mean      = 0.0 0.0 0.0\n')
            f.write('sigma     = 1.0\n')
            f.write('1 4 4\n')
            for row in out_matrix:
                f.write(_fmt(row) + '\n')
            f.write('\n')
            for role, info in (('src', self.src), ('dst', self.dst)):
                dims_str = ' '.join(str(int(x)) for x in info['volume'])
                f.write(f'{role} volume info\n')
                f.write('valid = 1  # volume info valid\n')
                f.write(f"filename = {info.get('filename', '')}\n")
                f.write(f'volume = {dims_str}\n')
                f.write(f"voxelsize = {_fmt(info['voxelsize'])}\n")
                f.write(f"xras   = {_fmt(info['xras'])}\n")
                f.write(f"yras   = {_fmt(info['yras'])}\n")
                f.write(f"zras   = {_fmt(info['zras'])}\n")
                f.write(f"cras   = {_fmt(info['cras'])}\n")

        logger.debug('Wrote LTA (%s): %s', type_name, filename)

    def __repr__(self) -> str:
        type_str = 'R2R' if self.type == 1 else 'V2V'
        src_fn   = self.src.get('filename', '?')
        dst_fn   = self.dst.get('filename', '?')
        return f'LTA({type_str}, {src_fn!r} → {dst_fn!r})'

    # ── matrix extraction ───────────────────────────────────────────────────

    def r2r(self) -> np.ndarray:
        """Return the 4×4 RAS-to-RAS matrix."""
        if self.type == 1:
            return self.matrix.copy()
        return convert_transform_type(
            self.matrix, _affine_from_info(self.src), _affine_from_info(self.dst),
            from_type=0, to_type=1,
        )

    def v2v(self) -> np.ndarray:
        """Return the 4×4 voxel-to-voxel matrix."""
        if self.type == 0:
            return self.matrix.copy()
        return convert_transform_type(
            self.matrix, _affine_from_info(self.src), _affine_from_info(self.dst),
            from_type=1, to_type=0,
        )

    def iso_vox(self) -> np.ndarray:
        """Return the iso-vox scaled matrix (FreeSurfer ``getIsoVOX``).

        ``diag(dst_vs) @ V2V @ inv(diag(src_vs))``
        """
        v2v    = self.v2v()
        src_vs = np.diag([*self.src['voxelsize'], 1.0])
        dst_vs = np.diag([*self.dst['voxelsize'], 1.0])
        return dst_vs @ v2v @ np.linalg.inv(src_vs)

    # ── operations ──────────────────────────────────────────────────────────

    def invert(self) -> LTA:
        """Return an inverted copy with src/dst swapped, stored as R2R."""
        return LTA(np.linalg.inv(self.r2r()), 1, self.dst, self.src)

    # ── single-transform analysis ────────────────────────────────────────────

    def decompose(self) -> dict:
        """Polar decomposition of the R2R matrix.

        See :func:`decompose_transform` for the return dict keys.
        """
        return decompose_transform(self.r2r())

    @property
    def det(self) -> float:
        """Determinant of the R2R matrix."""
        return float(np.linalg.det(self.r2r()))

    # ── distance methods ────────────────────────────────────────────────────

    def rigid_dist(self, other: LTA | None = None) -> float:
        """Rigid-transform distance to *other* (or identity).

        Delegates to :func:`rigid_dist`.
        """
        return rigid_dist(self.r2r(), other.r2r() if other is not None else None)

    def affine_dist(self, other: LTA | None = None, radius: float = 100.) -> float:
        """Affine RMS distance to *other* (Jenkinson 1999).

        Delegates to :func:`affine_dist`.
        """
        return affine_dist(self.r2r(), other.r2r() if other is not None else None,
                           radius=radius)

    def corner_dist(self, other: LTA | None = None, vox: bool = False) -> float:
        """Mean displacement at the 8 volume corners.

        Parameters
        ----------
        other : LTA, optional
            Second transform; ``None`` compares against identity.
        vox : bool
            If ``True``, work in iso-vox space (voxel-grid-aligned, mm units)
            instead of scanner RAS.  Corners are scaled by src voxel size;
            result is in mm.  Correct for images with different resolutions or
            orientations.  If ``False`` (default), use R2R with the full
            src affine; result is in RAS mm.

        Delegates to :func:`corner_dist`.
        """
        src_shape = tuple(self.src['volume'])
        if vox:
            vs         = self.src['voxelsize']
            src_affine = np.diag([vs[0], vs[1], vs[2], 1.0])
            M2         = other.iso_vox() if other is not None else None
            return corner_dist(self.iso_vox(), src_shape, M2=M2, src_affine=src_affine)
        M2 = other.r2r() if other is not None else None
        return corner_dist(self.r2r(), src_shape, M2=M2,
                           src_affine=_affine_from_info(self.src))

    def sphere_dist(self, other: LTA | None = None, radius: float = 100.) -> float:
        """Max displacement on a sphere of given radius.

        Delegates to :func:`sphere_dist`.
        """
        return sphere_dist(self.r2r(), other.r2r() if other is not None else None,
                           radius=radius)

