from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..image.geometry import vox2tkras_from_volume_info
from .lta import LTA, _affine_from_info, _AnyHeader, _header_info, _header_to_vol_info

_VALID_FLOAT2INT = {"tkregister", "round", "floor"}


@dataclass(slots=True)
class RegisterDat:
    """FreeSurfer tkregister-style volumetric registration file."""

    matrix: np.ndarray
    subject: str = "subject-unknown"
    inplane_resolution: float = 1.0
    between_plane_resolution: float = 1.0
    intensity: float = 0.1
    float2int: str = "round"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)
        if self.float2int not in _VALID_FLOAT2INT:
            raise ValueError(f"float2int must be one of {_VALID_FLOAT2INT}, got {self.float2int!r}")

    @classmethod
    def read(cls, filename: str | Path) -> RegisterDat:
        """Read a FreeSurfer ``register.dat`` / tkregister transform file.

        Parameters
        ----------
        filename : str or Path
            Path to a ``register.dat`` file.

        Returns
        -------
        RegisterDat
            Parsed register.dat wrapper.

        Raises
        ------
        ValueError
            If the file is shorter than the required header-plus-matrix layout.
        """
        path = Path(filename)
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 8:
            raise ValueError(f"{path}: register.dat file is too short")

        subject = lines[0]
        inplane = float(lines[1])
        between = float(lines[2])
        intensity = float(lines[3])
        matrix = np.array([[float(v) for v in row.split()] for row in lines[4:8]], dtype=float)
        float2int = lines[8] if len(lines) >= 9 else "tkregister"
        return cls(
            matrix=matrix,
            subject=subject,
            inplane_resolution=inplane,
            between_plane_resolution=between,
            intensity=intensity,
            float2int=float2int,
        )

    @classmethod
    def from_lta(
            cls,
            lta: LTA,
            *,
            subject: str | None = None,
            intensity: float | None = None,
            float2int: str = "round",
    ) -> RegisterDat:
        """Create a register.dat transform from a canonical LTA.

        Parameters
        ----------
        lta : LTA
            Canonical scanner-RAS transform mapping moving to reference space.
        subject : str or None, optional
            Subject name to store in the output file. When omitted, the subject
            metadata from ``lta`` is used when available.
        intensity : float or None, optional
            Intensity scale to store in the output file. When omitted, ``lta``
            metadata is used when available.
        float2int : {'tkregister', 'round', 'floor'}, default='round'
            Float-to-int conversion mode written to the output file.

        Returns
        -------
        RegisterDat
            Wrapper containing the equivalent tkregister transform.
        """
        src_affine = _affine_from_info(lta.src)
        dst_affine = _affine_from_info(lta.dst)
        src_vox2tkras = vox2tkras_from_volume_info(lta.src)
        dst_vox2tkras = vox2tkras_from_volume_info(lta.dst)
        src_ras2tkras = src_vox2tkras @ np.linalg.inv(src_affine)
        dst_tkras2ras = dst_affine @ np.linalg.inv(dst_vox2tkras)
        reg_matrix = src_ras2tkras @ np.linalg.inv(lta.r2r()) @ dst_tkras2ras
        subject_name = subject if subject is not None else (lta.subject or "subject-unknown")
        reg_intensity = intensity if intensity is not None else (0.1 if lta.fscale is None else float(lta.fscale))
        return cls(
            matrix=reg_matrix,
            subject=subject_name,
            inplane_resolution=float(lta.src["voxelsize"][0]),
            between_plane_resolution=float(lta.src["voxelsize"][2]),
            intensity=reg_intensity,
            float2int=float2int,
        )

    def to_lta(
            self,
            *,
            src_fname: str,
            src_img: _AnyHeader,
            dst_fname: str,
            dst_img: _AnyHeader,
    ) -> LTA:
        """Convert the tkregister transform to canonical RAS-to-RAS LTA.

        Parameters
        ----------
        src_fname, dst_fname : str
            Source and destination filenames stored in the output LTA metadata.
        src_img, dst_img : header-like
            Source and destination image headers used to recover tkregister
            geometry.

        Returns
        -------
        LTA
            Canonical scanner-RAS transform wrapper.

        Notes
        -----
        ``src`` is the moving/input volume and ``dst`` is the reference/target
        volume.
        """
        src = _header_to_vol_info(_header_info(src_img), src_fname)
        dst = _header_to_vol_info(_header_info(dst_img), dst_fname)
        src_affine = _affine_from_info(src)
        dst_affine = _affine_from_info(dst)
        src_vox2tkras = vox2tkras_from_volume_info(src)
        dst_vox2tkras = vox2tkras_from_volume_info(dst)
        src_ras2tkras = src_vox2tkras @ np.linalg.inv(src_affine)
        dst_tkras2ras = dst_affine @ np.linalg.inv(dst_vox2tkras)
        matrix = dst_tkras2ras @ np.linalg.inv(self.matrix) @ src_ras2tkras
        return LTA(matrix, 1, src, dst, subject=self.subject, fscale=self.intensity)

    def write(self, filename: str | Path) -> None:
        """Write the transform in ``register.dat`` format.

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
        with path.open("w") as f:
            f.write(f"{self.subject or 'subject-unknown'}\n")
            f.write(f"{float(self.inplane_resolution):.6f}\n")
            f.write(f"{float(self.between_plane_resolution):.6f}\n")
            f.write(f"{float(self.intensity):.6f}\n")
            for row_idx, row in enumerate(self.matrix, start=1):
                if row_idx < 4:
                    f.write(" ".join(f"{float(v):.15e}" for v in row) + " \n")
                else:
                    f.write(" ".join(f"{float(v):.15g}" for v in row) + "\n")
            f.write(f"{self.float2int}\n")
