from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from .itk import _lps_to_ras, _validate_transform_type
from .lta import LTA, _AnyHeader, _header_info, _header_to_vol_info, _invalid_vol_info


@dataclass(slots=True)
class ANTsMatTransform:
    """ANTs / ITK Matlab-format affine transform.

    This corresponds to the common ``0GenericAffine.mat`` files written by
    ANTs. The stored transform uses the same physical-space ITK affine
    parameterization and LPS fixed-to-moving convention as text ``.tfm``
    files, but is serialized via ITK's Matlab transform IO backend instead of
    the legacy text format.
    """

    matrix: np.ndarray
    transform_type: str = "AffineTransform_double_3_3"
    fixed_parameters: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)
        self.transform_type = _validate_transform_type(self.transform_type)
        self.fixed_parameters = np.asarray(self.fixed_parameters, dtype=float).reshape(3)

    @classmethod
    def read(cls, filename: str | Path) -> ANTsMatTransform:
        path = Path(filename)
        data = loadmat(path)
        payload = {key: value for key, value in data.items() if not key.startswith("__")}
        transform_names = [key for key in payload if key != "fixed"]
        if len(transform_names) != 1:
            raise ValueError(
                f"{path}: expected exactly one transform entry plus optional 'fixed', got {sorted(payload)}"
            )
        transform_type = _validate_transform_type(transform_names[0])
        parameters = np.asarray(payload[transform_type], dtype=float).reshape(-1)
        if parameters.size != 12:
            raise ValueError(f"{path}: expected 12 affine parameters, got {parameters.size}")
        fixed_parameters = np.asarray(payload.get("fixed", np.zeros((3, 1), dtype=float)), dtype=float).reshape(-1)
        if fixed_parameters.size != 3:
            raise ValueError(f"{path}: expected 3 fixed parameters, got {fixed_parameters.size}")

        rotation = parameters[:9].reshape(3, 3)
        translation = parameters[9:] + fixed_parameters - rotation @ fixed_parameters

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return cls(matrix=matrix, transform_type=transform_type, fixed_parameters=fixed_parameters)

    @classmethod
    def from_lta(cls, lta: LTA) -> ANTsMatTransform:
        return cls(matrix=_lps_to_ras(np.linalg.inv(lta.r2r())))

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
        return LTA(np.linalg.inv(_lps_to_ras(self.matrix)), 1, src, dst)

    def write(self, filename: str | Path) -> None:
        path = Path(filename)
        rotation = self.matrix[:3, :3]
        center = self.fixed_parameters
        translation = self.matrix[:3, 3] - center + rotation @ center
        parameters = np.concatenate([rotation.reshape(-1), translation])
        savemat(
            path,
            {
                self.transform_type: parameters.reshape(-1, 1),
                "fixed": self.fixed_parameters.reshape(-1, 1),
            },
            format="4",
        )
