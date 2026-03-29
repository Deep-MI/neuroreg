"""
Surface-based registration module for bbregister-style alignment.

This module provides tools for registering volumes to cortical surface meshes,
using boundary-based registration (BBR) similar to FreeSurfer's bbregister.
"""

from .cost import bbr_contrast_cost, detect_contrast, gradient_magnitude_cost
from .io import load_surface, load_surface_from_subject, load_surface_pair
from .optimize import BBRModel
from .projection import compute_vertex_normals, project_vertices
from .register import register_surface
from .sampling import sample_volume_at_vertices

__all__ = [
    "load_surface",
    "load_surface_pair",
    "load_surface_from_subject",
    "compute_vertex_normals",
    "project_vertices",
    "sample_volume_at_vertices",
    "bbr_contrast_cost",
    "detect_contrast",
    "gradient_magnitude_cost",
    "BBRModel",
    "register_surface",
]
