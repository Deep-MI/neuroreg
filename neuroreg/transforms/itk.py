from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lta import LTA, _AnyHeader, _header_info, _header_to_vol_info, _invalid_vol_info

_ITK_TRANSFORM_RE = re.compile(r"^(AffineTransform|MatrixOffsetTransformBase)_(double|float)_3_3$")
_LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _lps_to_ras(matrix: np.ndarray) -> np.ndarray:
    return _LPS_RAS @ matrix @ _LPS_RAS


def _validate_transform_type(transform_type: str) -> str:
    if not _ITK_TRANSFORM_RE.match(transform_type):
        raise ValueError(
            "unsupported ITK transform type "
            f"{transform_type!r}; expected a 3D affine text transform such as "
            "'AffineTransform_double_3_3'"
        )
    return transform_type


@dataclass(slots=True)
class ITKTransform:
    """ITK/ANTs 3D affine text transform file.

    The stored matrix is the file-space LPS affine mapping fixed/reference points
    to moving/source points. Conversion to canonical scanner-RAS ``LTA`` therefore
    converts LPS to RAS and inverts the matrix.
    """

    matrix: np.ndarray
    transform_type: str = "AffineTransform_double_3_3"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)
        self.transform_type = _validate_transform_type(self.transform_type)

    @classmethod
    def read(cls, filename: str | Path) -> ITKTransform:
        path = Path(filename)
        transform_type: str | None = None
        parameters: list[float] | None = None
        fixed_parameters = np.zeros(3, dtype=float)

        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, rest = line.partition(":")
            if sep != ":":
                raise ValueError(f"{path}: malformed ITK transform line {raw_line!r}")
            key = key.strip()
            rest = rest.strip()
            if key == "Transform":
                transform_type = _validate_transform_type(rest)
            elif key == "Parameters":
                parameters = [float(v) for v in rest.split()]
                if len(parameters) != 12:
                    raise ValueError(f"{path}: expected 12 Parameters values in 3D ITK affine, got {len(parameters)}")
            elif key == "FixedParameters":
                values = [float(v) for v in rest.split()]
                if len(values) != 3:
                    raise ValueError(f"{path}: expected 3 FixedParameters values in 3D ITK affine, got {len(values)}")
                fixed_parameters = np.asarray(values, dtype=float)
            else:
                raise ValueError(f"{path}: unknown ITK transform field {key!r}")

        if transform_type is None:
            raise ValueError(f"{path}: missing Transform field")
        if parameters is None:
            raise ValueError(f"{path}: missing Parameters field")

        rotation = np.asarray(parameters[:9], dtype=float).reshape(3, 3)
        translation = np.asarray(parameters[9:], dtype=float)
        translation = translation + fixed_parameters - rotation @ fixed_parameters

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return cls(matrix=matrix, transform_type=transform_type)

    @classmethod
    def from_lta(cls, lta: LTA) -> ITKTransform:
        matrix = _lps_to_ras(np.linalg.inv(lta.r2r()))
        return cls(matrix=matrix)

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
        matrix = np.linalg.inv(_lps_to_ras(self.matrix))
        return LTA(matrix, 1, src, dst)

    def write(self, filename: str | Path) -> None:
        path = Path(filename)
        with path.open("w") as f:
            f.write("#Insight Transform File V1.0\n")
            f.write("#Transform 0\n")
            f.write(f"Transform: {self.transform_type}\n")
            params = [*self.matrix[:3, :3].reshape(-1), *self.matrix[:3, 3]]
            f.write("Parameters: " + " ".join(f"{float(v):.17g}" for v in params) + "\n")
            f.write("FixedParameters: 0 0 0\n")
