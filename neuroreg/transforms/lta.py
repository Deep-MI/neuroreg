from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeAlias

import nibabel as nib
import numpy as np
import numpy.typing as npt

from .matrices import convert_transform_type
from .metrics import (
    affine_dist,
    corner_dist,
    decompose_transform,
    rigid_dist,
    sphere_dist,
)

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

    if isinstance(src, str | Path):
        src = nib.load(src).header

    if hasattr(src, "affine"):
        affine = src.affine
        header = src.header
    else:
        affine = src.get_best_affine()
        header = src

    shape = [int(x) for x in header.get_data_shape()[:3]]
    zooms = np.array(header.get_zooms()[:3], dtype=float)

    return {
        "dims": shape,
        "delta": zooms.tolist(),
        "Mdc": affine[:3, :3] / zooms,
        "Pxyz_c": affine[:3, :3] @ (np.array(shape) / 2.0) + affine[:3, 3],
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
    vs = info["voxelsize"]
    dims = np.array(info["volume"], dtype=float)
    A = np.eye(4)
    A[:3, 0] = np.array(info["xras"]) * vs[0]
    A[:3, 1] = np.array(info["yras"]) * vs[1]
    A[:3, 2] = np.array(info["zras"]) * vs[2]
    A[:3, 3] = np.array(info["cras"]) - A[:3, :3] @ (dims / 2.0)
    return A


def _header_to_vol_info(hdr: dict, fname: str = "") -> dict:
    """Convert a :func:`_header_info` dict to LTA volume-info format.

    Maps the ``Mdc`` / ``Pxyz_c`` / ``dims`` / ``delta`` keys produced by
    :func:`_header_info` to the ``xras`` / ``yras`` / ``zras`` / ``cras`` /
    ``voxelsize`` / ``volume`` keys expected by :func:`_affine_from_info` and
    stored in ``.lta`` files.
    """
    Mdc = np.asarray(hdr["Mdc"])
    Pxyz_c = np.asarray(hdr["Pxyz_c"])
    return {
        "filename": fname,
        "volume": list(hdr["dims"]),
        "voxelsize": list(hdr["delta"]),
        "xras": Mdc[:, 0].tolist(),
        "yras": Mdc[:, 1].tolist(),
        "zras": Mdc[:, 2].tolist(),
        "cras": Pxyz_c.tolist(),
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
        if lta_type not in (0, 1):
            raise ValueError(f"lta_type must be 0 (LINEAR_VOX_TO_VOX) or 1 (LINEAR_RAS_TO_RAS), got {lta_type!r}")
        self.matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        self.type = lta_type
        self.src = src
        self.dst = dst

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

        if lta_type is not None and lta_type not in (0, 1):
            raise ValueError(f"lta_type must be 0 (LINEAR_VOX_TO_VOX) or 1 (LINEAR_RAS_TO_RAS), got {lta_type!r}")

        with open(filename) as f:
            lines = f.readlines()

        stored_type: int = 1
        nxforms: int = 1
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("type") and "=" in stripped:
                stored_type = int(stripped.split("=")[1].split("#")[0].strip())
            elif stripped.startswith("nxforms") and "=" in stripped:
                nxforms = int(stripped.split("=")[1].split("#")[0].strip())
            elif stripped.startswith("1 4 4"):
                break  # reached matrix; header fields are all above this

        if nxforms != 1:
            logger.warning(
                "%s: nxforms = %d; only the first transform will be read.",
                filename,
                nxforms,
            )

        if stored_type not in (0, 1):
            raise ValueError(
                f"{filename}: unsupported transform type {stored_type!r}; "
                f"expected 0 (LINEAR_VOX_TO_VOX) or 1 (LINEAR_RAS_TO_RAS)"
            )

        mat: list[list[float]] = []
        for i, line in enumerate(lines):
            if "1 4 4" in line:
                for row in lines[i + 1: i + 5]:
                    mat.append([float(v) for v in row.strip().split()])
                break
        if len(mat) != 4:
            raise ValueError(f"Could not parse 4×4 matrix from {filename}")

        def _parse_vol_block(start: int) -> dict:
            info: dict = {}
            for line in lines[start:]:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("valid"):
                    info["valid"] = int(line.split("=", 1)[1].split("#")[0].strip())
                elif line.startswith("filename"):
                    info["filename"] = line.split("=", 1)[1].strip()
                elif line.startswith("fname"):
                    # Old-style alias; only fill in if 'filename' not yet seen
                    info.setdefault("filename", line.split("=", 1)[1].strip())
                elif line.startswith("subject"):
                    # FreeSurfer writes "subject <name>" (no '=')
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        info["subject"] = parts[1].strip()
                    elif "=" in line:
                        info["subject"] = line.split("=", 1)[1].strip()
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

        src: dict = {}
        dst: dict = {}
        for i, line in enumerate(lines):
            if line.strip().startswith("src volume info"):
                src = _parse_vol_block(i + 1)
            elif line.strip().startswith("dst volume info"):
                dst = _parse_vol_block(i + 1)

        for role, info in (("src", src), ("dst", dst)):
            if info.get("valid", 1) == 0:
                logger.warning(
                    "%s: %s volume info has valid = 0; geometry may be unreliable.",
                    filename,
                    role,
                )

        lta = cls(np.array(mat), stored_type, src, dst)

        if lta_type is not None and lta_type != stored_type:
            lta = cls(
                convert_transform_type(
                    lta.matrix,
                    _affine_from_info(src),
                    _affine_from_info(dst),
                    from_type=stored_type,
                    to_type=lta_type,
                ),
                lta_type,
                src,
                dst,
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
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu().numpy()
        M = np.asarray(matrix, dtype=float).reshape(4, 4)
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
            raise ValueError(f"lta_type must be 0 or 1, got {lta_type!r}")

        out_type = self.type if lta_type is None else lta_type
        out_matrix = (
            self.matrix
            if out_type == self.type
            else convert_transform_type(
                self.matrix,
                _affine_from_info(self.src),
                _affine_from_info(self.dst),
                from_type=self.type,
                to_type=out_type,
            )
        )

        filename = str(filename)
        type_name = "LINEAR_RAS_TO_RAS" if out_type == 1 else "LINEAR_VOX_TO_VOX"

        def _fmt(vals: list) -> str:
            return " ".join(f"{float(v):.15e}" for v in vals)

        with open(filename, "w") as f:
            f.write(f"# transform file {filename}\n")
            f.write(f"# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n")
            f.write(f"type      = {out_type} # {type_name}\n")
            f.write("nxforms   = 1\n")
            f.write("mean      = 0.0 0.0 0.0\n")
            f.write("sigma     = 1.0\n")
            f.write("1 4 4\n")
            for row in out_matrix:
                f.write(_fmt(row) + "\n")
            f.write("\n")
            for role, info in (("src", self.src), ("dst", self.dst)):
                dims_str = " ".join(str(int(x)) for x in info["volume"])
                valid = info.get("valid", 1)
                f.write(f"{role} volume info\n")
                f.write(f"valid = {valid}  # volume info valid\n")
                f.write(f"filename = {info.get('filename', '')}\n")
                if "subject" in info:
                    f.write(f"subject {info['subject']}\n")
                f.write(f"volume = {dims_str}\n")
                f.write(f"voxelsize = {_fmt(info['voxelsize'])}\n")
                f.write(f"xras   = {_fmt(info['xras'])}\n")
                f.write(f"yras   = {_fmt(info['yras'])}\n")
                f.write(f"zras   = {_fmt(info['zras'])}\n")
                f.write(f"cras   = {_fmt(info['cras'])}\n")

        logger.debug("Wrote LTA (%s): %s", type_name, filename)

    def __repr__(self) -> str:
        type_str = "R2R" if self.type == 1 else "V2V"
        src_fn = self.src.get("filename", "?")
        dst_fn = self.dst.get("filename", "?")
        return f"LTA({type_str}, {src_fn!r} → {dst_fn!r})"

    # ── matrix extraction ───────────────────────────────────────────────────

    def r2r(self) -> np.ndarray:
        """Return the 4×4 RAS-to-RAS matrix."""
        if self.type == 1:
            return self.matrix.copy()
        return convert_transform_type(
            self.matrix,
            _affine_from_info(self.src),
            _affine_from_info(self.dst),
            from_type=0,
            to_type=1,
        )

    def v2v(self) -> np.ndarray:
        """Return the 4×4 voxel-to-voxel matrix."""
        if self.type == 0:
            return self.matrix.copy()
        return convert_transform_type(
            self.matrix,
            _affine_from_info(self.src),
            _affine_from_info(self.dst),
            from_type=1,
            to_type=0,
        )

    # ── operations ──────────────────────────────────────────────────────────

    def invert(self) -> LTA:
        """Return an inverted copy with src/dst swapped, stored as R2R."""
        return LTA(np.linalg.inv(self.r2r()), 1, self.dst, self.src)

    def concat(self, other: LTA) -> LTA:
        """Concatenate two transforms: ``self`` (A→B) followed by ``other`` (B→C).

        Returns a new LTA that maps directly from A to C, stored as R2R.
        The src geometry is taken from ``self`` and the dst geometry from
        ``other``; the caller is responsible for ensuring that the intermediate
        spaces (``self.dst`` / ``other.src``) are compatible.

        Equivalent to FreeSurfer's ``mri_concatenate_lta``.

        Parameters
        ----------
        other : LTA
            The second transform to apply (maps B → C).

        Returns
        -------
        LTA
            New LTA whose matrix is ``other.r2r() @ self.r2r()``,
            with ``src`` from ``self`` and ``dst`` from ``other``.
        """
        return LTA(other.r2r() @ self.r2r(), 1, self.src, other.dst)

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

        Operates on the RAS-to-RAS representation (converts automatically if
        stored as vox-to-vox).  Delegates to :func:`rigid_dist`.
        """
        return rigid_dist(self.r2r(), other.r2r() if other is not None else None)

    def affine_dist(self, other: LTA | None = None, radius: float = 100.0) -> float:
        """Affine RMS distance to *other* (Jenkinson 1999).

        Operates on the RAS-to-RAS representation (converts automatically if
        stored as vox-to-vox).  Delegates to :func:`affine_dist`.
        """
        return affine_dist(self.r2r(), other.r2r() if other is not None else None, radius=radius)

    def corner_dist(self, other: LTA | None = None) -> float:
        """Mean displacement at the 8 source-volume corners in RAS mm.

        Operates on the RAS-to-RAS representation (converts automatically if
        stored as vox-to-vox). **Image-specific**: depends on source volume
        shape and affine; see :func:`corner_dist` for the full description and
        limitations.

        * *other* is ``None`` - measures how far each corner moves from its
          original RAS position under this transform.
        * *other* is given - measures the separation between the two
          transforms' mappings of each corner; both LTAs must share the same
          source image.

        Parameters
        ----------
        other : LTA, optional
            Second transform. ``None`` compares this transform against identity.

        Returns
        -------
        float
            Mean corner displacement in mm. Delegates to :func:`corner_dist`.
        """
        src_volume = self.src["volume"]
        src_shape = (int(src_volume[0]), int(src_volume[1]), int(src_volume[2]))
        M2 = other.r2r() if other is not None else None
        return corner_dist(self.r2r(), src_shape, M2=M2, src_affine=_affine_from_info(self.src))

    def sphere_dist(self, other: LTA | None = None, radius: float = 100.0) -> float:
        """Max displacement on a sphere of given radius in RAS mm.

        Operates on the RAS-to-RAS representation (converts automatically if
        stored as vox-to-vox).  **Image-independent**: result depends only on
        the transform, not source/dst geometry.  Delegates to :func:`sphere_dist`.
        """
        return sphere_dist(self.r2r(), other.r2r() if other is not None else None, radius=radius)

