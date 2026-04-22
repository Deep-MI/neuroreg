from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lta import LTA, _AnyHeader, _header_info, _header_to_vol_info, _invalid_vol_info


@dataclass(slots=True)
class NiftyRegTransform:
    """NiftyReg 3D affine text matrix.

    FreeSurfer ``lta_convert`` treats the stored matrix as the inverse of the
    canonical scanner-RAS transform, i.e. target/reference RAS to
    source/moving RAS.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)

    @classmethod
    def read(cls, filename: str | Path) -> NiftyRegTransform:
        path = Path(filename)
        rows = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(v) for v in line.split()]
            if len(values) != 4:
                raise ValueError(f"{path}: expected 4 columns per row in NiftyReg affine matrix")
            rows.append(values)
        if len(rows) != 4:
            raise ValueError(f"{path}: expected 4 rows in NiftyReg affine matrix")
        return cls(np.asarray(rows, dtype=float))

    @classmethod
    def from_lta(cls, lta: LTA) -> NiftyRegTransform:
        return cls(np.linalg.inv(lta.r2r()))

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
        return LTA(np.linalg.inv(self.matrix), 1, src, dst)

    def write(self, filename: str | Path) -> None:
        path = Path(filename)
        with path.open("w") as f:
            for row in self.matrix:
                f.write(" ".join(f"{float(v):.7g}" for v in row) + "\n")
