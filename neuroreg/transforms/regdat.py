from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lta import LTA, _AnyHeader, _affine_from_info, _header_info, _header_to_vol_info

_VALID_FLOAT2INT = {"tkregister", "round", "floor"}


def _vox2tkr_from_info(info: dict) -> np.ndarray:
    dims = np.asarray(info["volume"], dtype=float)
    delta = np.asarray(info["voxelsize"], dtype=float)
    mat = np.array(
        [
            [-delta[0], 0.0, 0.0, delta[0] * dims[0] / 2.0],
            [0.0, 0.0, delta[2], -delta[2] * dims[2] / 2.0],
            [0.0, -delta[1], 0.0, delta[1] * dims[1] / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return mat


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
    def read(cls, filename: str | Path) -> "RegisterDat":
        """Read a register.dat/tkregister transform file."""
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
    ) -> "RegisterDat":
        """Create a register.dat transform from an LTA."""
        src_affine = _affine_from_info(lta.src)
        dst_affine = _affine_from_info(lta.dst)
        src_vox2tkr = _vox2tkr_from_info(lta.src)
        dst_vox2tkr = _vox2tkr_from_info(lta.dst)
        src_ras2tkr = src_vox2tkr @ np.linalg.inv(src_affine)
        dst_tkr2ras = dst_affine @ np.linalg.inv(dst_vox2tkr)
        reg_matrix = src_ras2tkr @ np.linalg.inv(lta.r2r()) @ dst_tkr2ras
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

        ``src`` is the moving/input volume. ``dst`` is the reference/target volume.
        """
        src = _header_to_vol_info(_header_info(src_img), src_fname)
        dst = _header_to_vol_info(_header_info(dst_img), dst_fname)
        src_affine = _affine_from_info(src)
        dst_affine = _affine_from_info(dst)
        src_vox2tkr = _vox2tkr_from_info(src)
        dst_vox2tkr = _vox2tkr_from_info(dst)
        src_ras2tkr = src_vox2tkr @ np.linalg.inv(src_affine)
        dst_tkr2ras = dst_affine @ np.linalg.inv(dst_vox2tkr)
        matrix = dst_tkr2ras @ np.linalg.inv(self.matrix) @ src_ras2tkr
        return LTA(matrix, 1, src, dst, subject=self.subject, fscale=self.intensity)

    def write(self, filename: str | Path) -> None:
        """Write the register.dat file."""
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
