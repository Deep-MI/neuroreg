"""Image utilities: smoothing, pyramid, centroid, mapping, segmentation."""

from .binarize import binarize_image
from .bspline import downsample2_bspline
from .centroid import compute_centroid
from .compare import ImageDiff, compare_images
from .geometry import get_ras2tkras, get_tkras2ras, get_vox2tkras, vox2tkras_from_volume_info
from .info import describe_image, image_value_stats
from .io import load_image, save_image
from .map import (
    create_image_like,
    header_map_image,
    infer_image_reslice_mode,
    map,
    map_r2r,
    mask_geometry_differs,
    resample_isotropic,
    reslice_and_apply_mask,
    reslice_r2r_image,
    save_header_mapped_image,
    save_resliced_r2r_image,
)
from .pyramid import build_gaussian_pyramid, get_pyramid_limits
from .segmentation import extract_wm_surface, simplify_segmentation, surfaces_from_segmentation
from .smooth import get_gaussian_kernel, smooth

__all__ = [
    "compute_centroid",
    "get_ras2tkras",
    "get_tkras2ras",
    "get_vox2tkras",
    "vox2tkras_from_volume_info",
    "describe_image",
    "image_value_stats",
    "ImageDiff",
    "compare_images",
    "binarize_image",
    "load_image",
    "save_image",
    "map",
    "map_r2r",
    "create_image_like",
    "header_map_image",
    "infer_image_reslice_mode",
    "mask_geometry_differs",
    "resample_isotropic",
    "reslice_and_apply_mask",
    "reslice_r2r_image",
    "save_resliced_r2r_image",
    "save_header_mapped_image",
    "downsample2_bspline",
    "build_gaussian_pyramid",
    "get_pyramid_limits",
    "get_gaussian_kernel",
    "smooth",
    "simplify_segmentation",
    "extract_wm_surface",
    "surfaces_from_segmentation",
]
