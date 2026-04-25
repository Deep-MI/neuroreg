from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..image.geometry import vox2tkras_from_volume_info
from .lta import LTA, _affine_from_info, _AnyHeader, _header_info, _header_to_vol_info
from .regdat import RegisterDat


def _is_nifti_like(path: str) -> bool:
    """Return whether a filename looks like a NIfTI/Analyze image path.

    Parameters
    ----------
    path : str
        Candidate image filename.

    Returns
    -------
    bool
        ``True`` when the suffix matches one of the NIfTI/Analyze extensions
        that trigger FSL's handedness convention adjustments.
    """
    lower = path.lower()
    return lower.endswith('.nii') or lower.endswith('.nii.gz') or lower.endswith('.img') or lower.endswith('.hdr')


def _diag_spacing(info: dict) -> np.ndarray:
    """Construct a diagonal voxel-size matrix from volume metadata.

    Parameters
    ----------
    info : dict
        FreeSurfer-style volume-info dictionary containing ``voxelsize``.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` diagonal spacing matrix.
    """
    vs = np.asarray(info['voxelsize'], dtype=float)
    mat = np.eye(4, dtype=float)
    mat[0, 0] = vs[0]
    mat[1, 1] = vs[1]
    mat[2, 2] = vs[2]
    return mat


def _apply_fsl_nifti_convention(
        ref: dict,
        mov: dict,
        d_ref: np.ndarray,
        d_mov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FSL's NIfTI handedness convention to spacing matrices.

    Parameters
    ----------
    ref, mov : dict
        Reference and moving FreeSurfer-style volume-info dictionaries.
    d_ref, d_mov : np.ndarray
        Reference and moving diagonal spacing matrices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Possibly adjusted ``(d_ref, d_mov)`` pair following FSL's NIfTI
        convention for positive-determinant affines.
    """
    ref_aff = _affine_from_info(ref)
    mov_aff = _affine_from_info(mov)

    if np.linalg.det(mov_aff[:3, :3]) > 0:
        d_mov = d_mov.copy()
        d_mov[0, 0] *= -1.0
        d_mov[0, 3] = float(mov['voxelsize'][0]) * (float(mov['volume'][0]) - 1.0)
    if np.linalg.det(ref_aff[:3, :3]) > 0:
        d_ref = d_ref.copy()
        d_ref[0, 0] *= -1.0
        d_ref[0, 3] = float(ref['voxelsize'][0]) * (float(ref['volume'][0]) - 1.0)
    return d_ref, d_mov


def _fsl_to_tkreg(ref: dict, mov: dict, fsl_matrix: np.ndarray) -> np.ndarray:
    """Convert an FSL voxel-to-voxel matrix to tkregister convention.

    Parameters
    ----------
    ref, mov : dict
        Reference and moving FreeSurfer-style volume-info dictionaries.
    fsl_matrix : np.ndarray
        ``(4, 4)`` affine matrix in FSL voxel-space convention.

    Returns
    -------
    np.ndarray
        Equivalent ``(4, 4)`` affine in tkregister voxel-space convention.
    """
    inv_d_mov = np.linalg.inv(_diag_spacing(mov))
    d_ref = _diag_spacing(ref)
    mov_path = mov.get('filename', '')
    ref_path = ref.get('filename', '')
    if _is_nifti_like(mov_path) or _is_nifti_like(ref_path):
        d_ref, d_mov = _apply_fsl_nifti_convention(ref, mov, d_ref, _diag_spacing(mov))
        inv_d_mov = np.linalg.inv(d_mov)
    t_mov = vox2tkras_from_volume_info(mov)
    t_ref = vox2tkras_from_volume_info(ref)
    return t_mov @ inv_d_mov @ np.linalg.inv(fsl_matrix) @ d_ref @ np.linalg.inv(t_ref)


def _tkreg_to_fsl(ref: dict, mov: dict, tkreg_matrix: np.ndarray) -> np.ndarray:
    """Convert a tkregister voxel-to-voxel matrix to FSL convention.

    Parameters
    ----------
    ref, mov : dict
        Reference and moving FreeSurfer-style volume-info dictionaries.
    tkreg_matrix : np.ndarray
        ``(4, 4)`` affine matrix in tkregister voxel-space convention.

    Returns
    -------
    np.ndarray
        Equivalent ``(4, 4)`` affine in FSL voxel-space convention.
    """
    d_mov = _diag_spacing(mov)
    d_ref = _diag_spacing(ref)
    mov_path = mov.get('filename', '')
    ref_path = ref.get('filename', '')
    if _is_nifti_like(mov_path) or _is_nifti_like(ref_path):
        d_ref, d_mov = _apply_fsl_nifti_convention(ref, mov, d_ref, d_mov)
    t_mov = vox2tkras_from_volume_info(mov)
    t_ref = vox2tkras_from_volume_info(ref)
    return np.linalg.inv(d_mov @ np.linalg.inv(t_mov) @ tkreg_matrix @ t_ref @ np.linalg.inv(d_ref))


@dataclass(slots=True)
class FSLMat:
    """FSL FLIRT affine matrix file.

    The stored matrix maps moving voxels to reference voxels in FSL conventions,
    so conversion to canonical scanner-RAS space requires explicit moving and
    reference image geometry.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)

    @classmethod
    def read(cls, filename: str | Path) -> FSLMat:
        """Read an FSL FLIRT matrix file.

        Parameters
        ----------
        filename : str or Path
            Path to a text ``.mat`` file containing a 4x4 FLIRT affine.

        Returns
        -------
        FSLMat
            Parsed FSL matrix wrapper.

        Raises
        ------
        ValueError
            If the file does not contain exactly four rows of four values.
        """
        path = Path(filename)
        rows = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            values = [float(v) for v in stripped.split()]
            if len(values) != 4:
                raise ValueError(f'{path}: expected 4 columns per row in FSL matrix')
            rows.append(values)
        if len(rows) != 4:
            raise ValueError(f'{path}: expected 4 rows in FSL matrix')
        return cls(np.asarray(rows, dtype=float))

    @classmethod
    def from_lta(cls, lta: LTA) -> FSLMat:
        """Create an FSL matrix wrapper from a canonical LTA.

        Parameters
        ----------
        lta : LTA
            Canonical scanner-RAS transform mapping moving to reference space.

        Returns
        -------
        FSLMat
            Wrapper containing the equivalent FLIRT voxel-space affine.
        """
        reg = RegisterDat.from_lta(lta)
        return cls(_tkreg_to_fsl(lta.dst, lta.src, reg.matrix))

    def to_lta(
            self,
            *,
            src_fname: str,
            src_img: _AnyHeader,
            dst_fname: str,
            dst_img: _AnyHeader,
    ) -> LTA:
        """Convert the FSL matrix to canonical scanner-RAS LTA form.

        Parameters
        ----------
        src_fname, dst_fname : str
            Source and destination filenames stored in the output LTA metadata.
        src_img, dst_img : header-like
            Source and destination image headers used to resolve the FSL voxel
            conventions.

        Returns
        -------
        LTA
            Canonical RAS-to-RAS transform wrapper.
        """
        src = _header_to_vol_info(_header_info(src_img), src_fname)
        dst = _header_to_vol_info(_header_info(dst_img), dst_fname)
        reg_matrix = _fsl_to_tkreg(dst, src, self.matrix)
        return RegisterDat(reg_matrix).to_lta(
            src_fname=src_fname,
            src_img=src_img,
            dst_fname=dst_fname,
            dst_img=dst_img,
        )

    def write(self, filename: str | Path) -> None:
        """Write the matrix in FSL text format.

        Parameters
        ----------
        filename : str or Path
            Output transform path.

        Returns
        -------
        None
            Writes the transform to ``filename``.
        """
        path = Path(filename)
        with path.open('w') as f:
            for row in self.matrix:
                f.write(' '.join(f'{float(v):.8f}' for v in row) + '\n')
