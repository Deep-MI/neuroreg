"""Transform utilities: matrices, LTA file I/O, initialisation."""

from .initialize import get_ixform_centroids, get_vox2vox_from_header
from .lta import (
    affine_dist,
    corner_diff,
    decompose_transform,
    read_lta,
    rigid_dist,
    sphere_diff,
    write_lta,
)
from .matrices import (
    LINEAR_RAS_TO_RAS,
    LINEAR_VOX_TO_VOX,
    compute_sqrtm,
    convert_torch_to_v2v,
    convert_transform_type,
    convert_v2v_to_torch,
    get_affine,
    get_rotation_euler,
    get_rotation_rodrigues,
    get_scaling,
    get_translation,
    matrix_decompose,
)

__all__ = [
    # initialize
    "get_ixform_centroids",
    "get_vox2vox_from_header",
    # lta
    "affine_dist",
    "corner_diff",
    "decompose_transform",
    "read_lta",
    "rigid_dist",
    "sphere_diff",
    "write_lta",
    # matrices
    "LINEAR_RAS_TO_RAS",
    "LINEAR_VOX_TO_VOX",
    "compute_sqrtm",
    "convert_torch_to_v2v",
    "convert_transform_type",
    "convert_v2v_to_torch",
    "get_affine",
    "get_rotation_euler",
    "get_rotation_rodrigues",
    "get_scaling",
    "get_translation",
    "matrix_decompose",
]
