"""Transform utilities: matrices, LTA file I/O, initialisation."""

from .initialize import get_ixform_centroids, get_vox2vox_from_header
from .lta import (
    LTA,
    affine_dist,
    corner_dist,
    decompose_transform,
    rigid_dist,
    sphere_dist,
)
from .matrices import (
    LINEAR_RAS_TO_RAS,
    LINEAR_VOX_TO_VOX,
    compute_sqrtm,
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
)

__all__ = [
    # initialize
    "get_ixform_centroids",
    "get_vox2vox_from_header",
    # lta
    "LTA",
    "affine_dist",
    "corner_dist",
    "decompose_transform",
    "rigid_dist",
    "sphere_dist",
    # matrices
    "LINEAR_RAS_TO_RAS",
    "LINEAR_VOX_TO_VOX",
    "compute_sqrtm",
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
]
