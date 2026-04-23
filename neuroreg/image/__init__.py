"""Image utilities: smoothing, pyramid, centroid, mapping, segmentation."""

from .centroid import compute_centroid
from .map import (
    create_image_like,
    header_map_image,
    infer_image_reslice_mode,
    map,
    map_r2r,
    resample_isotropic,
    reslice_r2r_image,
    save_header_mapped_image,
    save_resliced_r2r_image,
)
from .pyramid import build_gaussian_pyramid, get_pyramid_limits
from .segmentation import extract_wm_surface, simplify_segmentation, surfaces_from_segmentation
from .smooth import get_gaussian_kernel, smooth

__all__ = [
    "compute_centroid",
    "map",
    "map_r2r",
    "create_image_like",
    "header_map_image",
    "infer_image_reslice_mode",
    "resample_isotropic",
    "reslice_r2r_image",
    "save_resliced_r2r_image",
    "save_header_mapped_image",
    "build_gaussian_pyramid",
    "get_pyramid_limits",
    "get_gaussian_kernel",
    "smooth",
    "simplify_segmentation",
    "extract_wm_surface",
    "surfaces_from_segmentation",
]
