from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .itk import _lps_to_ras
from .lta import LTA, _AnyHeader, _header_info, _header_to_vol_info, _invalid_vol_info


@dataclass(slots=True)
class AFNIAffine:
    """AFNI affine text transform.

    Supports the common ASCII 3D affine encodings used by AFNI tools such as
    ``3dAllineate`` and ``cat_matvec``: 3x4 text, augmented 4x4 text, or a
    single row of 12 values as in ``.aff12.1D`` files.

    This implementation interprets the stored matrix in AFNI's DICOM/LPS
    physical coordinate convention and converts it to canonical scanner-RAS
    for ``LTA`` interop.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)

    @classmethod
    def read(cls, filename: str | Path) -> AFNIAffine:
        path = Path(filename)
        rows: list[list[float]] = []
        flat_rows: list[list[float]] = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            values = [float(v) for v in line.split()]
            if len(values) == 12:
                flat_rows.append(values)
            elif len(values) in {4, 9, 16}:
                rows.append(values)
            else:
                raise ValueError(f"{path}: expected 4, 9, 12, or 16 values per AFNI affine line")

        matrix = np.eye(4, dtype=float)
        if flat_rows:
            if len(flat_rows) != 1 or rows:
                raise ValueError(f"{path}: multi-transform .aff12.1D files are not supported")
            values = flat_rows[0]
            matrix[:3, :4] = np.asarray(values, dtype=float).reshape(3, 4)
            return cls(matrix)
        if len(rows) == 3 and all(len(row) == 4 for row in rows):
            matrix[:3, :4] = np.asarray(rows, dtype=float)
            return cls(matrix)
        if len(rows) == 4 and all(len(row) == 4 for row in rows):
            matrix = np.asarray(rows, dtype=float)
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
                raise ValueError(f"{path}: AFNI 4x4 affine must end with '0 0 0 1'")
            return cls(matrix)
        if len(rows) == 1 and len(rows[0]) == 9:
            matrix[:3, :3] = np.asarray(rows[0], dtype=float).reshape(3, 3)
            return cls(matrix)
        raise ValueError(f"{path}: could not parse AFNI affine text transform")

    @classmethod
    def from_lta(cls, lta: LTA) -> AFNIAffine:
        return cls(_lps_to_ras(lta.r2r()))

    def to_lta(
        self,
        src_fname: str | None = None,
        src_img: _AnyHeader | None = None,
        dst_fname: str | None = None,
        dst_img: _AnyHeader | None = None,
    ) -> LTA:
        src_fname = "" if src_fname is None else src_fname
        dst_fname = "" if dst_fname is None else dst_fname
        src = _invalid_vol_info(src_fname) if src_img is None else _header_to_vol_info(_header_info(src_img), src_fname)
        dst = _invalid_vol_info(dst_fname) if dst_img is None else _header_to_vol_info(_header_info(dst_img), dst_fname)
        return LTA(_lps_to_ras(self.matrix), 1, src, dst)

    def write(self, filename: str | Path) -> None:
        path = Path(filename)
        matrix = self.matrix[:3, :4]
        if str(path).lower().endswith(".aff12.1d"):
            values = np.concatenate([matrix[0], matrix[1], matrix[2]])
            path.write_text(" ".join(f"{float(v):.9g}" for v in values) + "\n")
            return
        with path.open("w") as f:
            for row in matrix:
                f.write(" ".join(f"{float(v):.9g}" for v in row) + "\n")
