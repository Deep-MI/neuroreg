import numpy as np
import numpy.typing as npt

# LTA transform type constants
LINEAR_VOX_TO_VOX = 0
LINEAR_RAS_TO_RAS = 1


def convert_lta_matrix(
    matrix: npt.ArrayLike,
    src_affine: npt.ArrayLike,
    dst_affine: npt.ArrayLike,
    from_type: int,
    to_type: int,
) -> np.ndarray:
    """Convert an LTA transformation matrix between vox-to-vox and RAS-to-RAS.

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
        Converted transformation matrix.

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
        return np.asarray(matrix, dtype=float)

    M  = np.asarray(matrix, dtype=float)
    As = np.asarray(src_affine, dtype=float)
    Ad = np.asarray(dst_affine, dtype=float)

    if from_type == LINEAR_VOX_TO_VOX:          # → RAS-to-RAS
        return Ad @ M @ np.linalg.inv(As)
    else:                                        # RAS-to-RAS → vox-to-vox
        return np.linalg.inv(Ad) @ M @ As


def write_lta(
        filename: str,
        T: npt.ArrayLike,
        src_fname: str,
        src_header: dict[str, list[float] | np.ndarray],
        dst_fname: str,
        dst_header: dict[str, list[float] | np.ndarray],
        lta_type: int = 1
) -> None:
    """
    Write linear transform array information to a `.lta` (linear transform array) file.

    This function saves a transformation matrix (T) to an LTA file along with
    metadata about the source and destination image volumes, including dimensions, voxel
    sizes, direction cosines, and centroids.

    Parameters
    ----------
    filename : str
        The name of the file to save the linear transformation.
    T : npt.ArrayLike
        The transformation matrix (4x4 linear transform array) to save.
    src_fname : str
        The filename of the source image.
    src_header : dict[str, Union[list[float], np.ndarray]]
        Header information of the source image, expected to contain:
          - "dims" (List[float]): Dimensions of the image (3D: x, y, z).
          - "delta" (List[float]): Voxel sizes in mm along each axis.
          - "Mdc" (np.ndarray): Voxel-to-RAS direction cosine matrix (3x3).
          - "Pxyz_c" (np.ndarray): The RAS coordinates of the voxel center (3D).
    dst_fname : str
        The filename of the destination image.
    dst_header : dict[str, Union[list[float], np.ndarray]]
        Header information of the destination image, expected to contain:
          - "dims" (list[float]): Dimensions of the image (3D: x, y, z).
          - "delta" (list[float]): Voxel sizes in mm along each axis.
          - "Mdc" (np.ndarray): Voxel-to-RAS direction cosine matrix (3x3).
          - "Pxyz_c" (np.ndarray): The RAS coordinates of the voxel center (3D).
    lta_type : int, optional
        Transform type: 0 = LINEAR_VOX_TO_VOX, 1 = LINEAR_RAS_TO_RAS (default).

    Raises
    ------
    ValueError
        If the `src_header` or `dst_header` is missing one or more of the required fields.

    Notes
    -----
    - The .lta format is specific to the FreeSurfer software suite and contains metadata
      as well as the transformation matrix.
    - The transformation matrix is assumed to be in RAS-to-RAS space (LINEAR_RAS_TO_RAS).
    - The "created by" line captures the current username and time of file creation.

    Examples
    --------
    >>> src_header = {
    ...     "dims": [256, 256, 256],
    ...     "delta": [1.0, 1.0, 1.0],
    ...     "Mdc": np.eye(3),
    ...     "Pxyz_c": np.array([0.0, 0.0, 0.0])
    ... }
    >>> dst_header = {
    ...     "dims": [128, 128, 128],
    ...     "delta": [2.0, 2.0, 2.0],
    ...     "Mdc": np.eye(3),
    ...     "Pxyz_c": np.array([0.0, 0.0, 0.0])
    ... }
    >>> T = np.eye(4)
    >>> writeLTA("transform.lta", T, "source.mgz", src_header, "target.mgz", dst_header)
    """
    import getpass
    from datetime import datetime

    fields = ("dims", "delta", "Mdc", "Pxyz_c")
    for field in fields:
        if field not in src_header:
            raise ValueError(
                f"writeLTA Error: src_header is missing required field: {field}"
            )
        if field not in dst_header:
            raise ValueError(
                f"writeLTA Error: dst_header is missing required field: {field}"
            )

    # Format dims and voxelsize as space-separated (no commas)
    src_dims = " ".join(str(int(x)) for x in src_header["dims"][0:3])
    src_vsize = " ".join(f"{float(x):.15e}" for x in src_header["delta"][0:3])
    src_v2r = src_header["Mdc"]
    src_c = src_header["Pxyz_c"]

    dst_dims = " ".join(str(int(x)) for x in dst_header["dims"][0:3])
    dst_vsize = " ".join(f"{float(x):.15e}" for x in dst_header["delta"][0:3])
    dst_v2r = dst_header["Mdc"]
    dst_c = dst_header["Pxyz_c"]

    with open(filename, "w") as f:
        f.write(f"# transform file {filename}\n")
        f.write(
            f"# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n"
        )
        # Write type based on parameter
        type_name = "LINEAR_VOX_TO_VOX" if lta_type == 0 else "LINEAR_RAS_TO_RAS"
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
        f.write("xras   = " + " ".join(f"{x:.15e}" for x in src_v2r[0, :]) + "\n")
        f.write("yras   = " + " ".join(f"{x:.15e}" for x in src_v2r[1, :]) + "\n")
        f.write("zras   = " + " ".join(f"{x:.15e}" for x in src_v2r[2, :]) + "\n")
        f.write("cras   = " + " ".join(f"{x:.15e}" for x in src_c) + "\n")
        f.write("dst volume info\n")
        f.write("valid = 1  # volume info valid\n")
        f.write(f"filename = {dst_fname}\n")
        f.write(f"volume = {dst_dims}\n")
        f.write(f"voxelsize = {dst_vsize}\n")
        f.write("xras   = " + " ".join(f"{x:.15e}" for x in dst_v2r[0, :]) + "\n")
        f.write("yras   = " + " ".join(f"{x:.15e}" for x in dst_v2r[1, :]) + "\n")
        f.write("zras   = " + " ".join(f"{x:.15e}" for x in dst_v2r[2, :]) + "\n")
        f.write("cras   = " + " ".join(f"{x:.15e}" for x in dst_c) + "\n")


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
            required = ("xras", "yras", "zras", "cras", "voxelsize")
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
            A[:3, 3] = np.array(info["cras"])
            return A

        src_affine = _affine_from_info(result["src"], "src")
        dst_affine = _affine_from_info(result["dst"], "dst")
        result["matrix"] = convert_lta_matrix(
            result["matrix"], src_affine, dst_affine,
            from_type=result["type"], to_type=lta_type
        )
        result["type"] = lta_type

    return result

