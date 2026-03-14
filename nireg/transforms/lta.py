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
        f.write(str(T).replace(" [", "").replace("[", "").replace("]", ""))
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

