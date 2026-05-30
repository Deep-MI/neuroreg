"""Transform utilities: matrices, LTA file I/O, and transform metrics."""

from .afni import AFNIAffine
from .antsmat import ANTsMatTransform
from .fsl import FSLMat
from .io import TRANSFORM_FORMATS, infer_transform_format, read_transform_as_lta, write_lta_as_transform
from .itk import ITKTransform
from .lta import LTA, affine_from_volume_info
from .matrices import (
    LINEAR_RAS_TO_RAS,
    LINEAR_VOX_TO_VOX,
    convert_r2r_to_torch,
    convert_torch_to_v2v,
    convert_transform_type,
    convert_v2v_to_torch,
    get_affine,
    get_rotation_euler,
    get_rotation_rodrigues,
    get_scaling,
    get_translation,
    matrix_sqrt_schur,
    params_to_rigid_matrix,
    rotation_error,
)
from .metrics import (
    affine_dist,
    corner_dist,
    decompose_transform,
    rigid_dist,
    sphere_dist,
)
from .niftyreg import NiftyRegTransform
from .regdat import RegisterDat
from .xfm import XFM

__all__ = [
    # lta
    "AFNIAffine",
    "ANTsMatTransform",
    "affine_from_volume_info",
    "FSLMat",
    "ITKTransform",
    "LTA",
    "NiftyRegTransform",
    "RegisterDat",
    "TRANSFORM_FORMATS",
    "XFM",
    "affine_dist",
    "corner_dist",
    "decompose_transform",
    "infer_transform_format",
    "read_transform_as_lta",
    "rigid_dist",
    "sphere_dist",
    "write_lta_as_transform",
    # matrices
    "LINEAR_RAS_TO_RAS",
    "LINEAR_VOX_TO_VOX",
    "convert_r2r_to_torch",
    "convert_torch_to_v2v",
    "convert_transform_type",
    "convert_v2v_to_torch",
    "get_affine",
    "get_rotation_euler",
    "get_rotation_rodrigues",
    "get_scaling",
    "get_translation",
    "matrix_sqrt_schur",
    "params_to_rigid_matrix",
    "rotation_error",
]
