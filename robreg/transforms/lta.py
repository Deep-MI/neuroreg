from typing import Dict, Union

import numpy as np
import numpy.typing as npt


def write_lta(
        filename: str,
        T: npt.ArrayLike,
        src_fname: str,
        src_header: Dict[str, Union[list[float], np.ndarray]],
        dst_fname: str,
        dst_header: Dict[str, Union[list[float], np.ndarray]]
) -> None:
    """
    Write linear transform array information to a `.lta` (linear transform array) file.

    This function saves a voxel-to-voxel transformation matrix (T) to an LTA file along with
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
    src_header : Dict[str, Union[list[float], np.ndarray]]
        Header information of the source image, expected to contain:
          - "dims" (List[float]): Dimensions of the image (3D: x, y, z).
          - "delta" (List[float]): Voxel sizes in mm along each axis.
          - "Mdc" (np.ndarray): Voxel-to-RAS direction cosine matrix (3x3).
          - "Pxyz_c" (np.ndarray): The RAS coordinates of the voxel center (3D).
    dst_fname : str
        The filename of the destination image.
    dst_header : Dict[str, Union[list[float], np.ndarray]]
        Header information of the destination image, expected to contain:
          - "dims" (list[float]): Dimensions of the image (3D: x, y, z).
          - "delta" (list[float]): Voxel sizes in mm along each axis.
          - "Mdc" (np.ndarray): Voxel-to-RAS direction cosine matrix (3x3).
          - "Pxyz_c" (np.ndarray): The RAS coordinates of the voxel center (3D).

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

    src_dims = str(src_header["dims"][0:3]).replace("[", "").replace("]", "")
    src_vsize = str(src_header["delta"][0:3]).replace("[", "").replace("]", "")
    src_v2r = src_header["Mdc"]
    src_c = src_header["Pxyz_c"]

    dst_dims = str(dst_header["dims"][0:3]).replace("[", "").replace("]", "")
    dst_vsize = str(dst_header["delta"][0:3]).replace("[", "").replace("]", "")
    dst_v2r = dst_header["Mdc"]
    dst_c = dst_header["Pxyz_c"]

    with open(filename, "w") as f:
        f.write(f"# transform file {filename}\n")
        f.write(
            f"# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n"
        )
        f.write("type      = 1 # LINEAR_RAS_TO_RAS\n")
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
        f.write(f"xras   = {src_v2r[0, :]}\n".replace("[", "").replace("]", ""))
        f.write(f"yras   = {src_v2r[1, :]}\n".replace("[", "").replace("]", ""))
        f.write(f"zras   = {src_v2r[2, :]}\n".replace("[", "").replace("]", ""))
        f.write(f"cras   = {src_c}\n".replace("[", "").replace("]", ""))
        f.write("dst volume info\n")
        f.write("valid = 1  # volume info valid\n")
        f.write(f"filename = {dst_fname}\n")
        f.write(f"volume = {dst_dims}\n")
        f.write(f"voxelsize = {dst_vsize}\n")
        f.write(f"xras   = {dst_v2r[0, :]}\n".replace("[", "").replace("]", ""))
        f.write(f"yras   = {dst_v2r[1, :]}\n".replace("[", "").replace("]", ""))
        f.write(f"zras   = {dst_v2r[2, :]}\n".replace("[", "").replace("]", ""))
        f.write(f"cras   = {dst_c}\n".replace("[", "").replace("]", ""))