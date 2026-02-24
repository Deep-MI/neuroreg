"""
Surface-based registration module for bbregister-style alignment.

This module provides tools for registering volumes to cortical surface meshes,
using boundary-based registration (BBR) similar to FreeSurfer's bbregister.
"""

from .io import load_surface, load_surface_pair, load_surface_from_subject
from .projection import compute_vertex_normals, project_vertices
from .sampling import sample_volume_at_vertices
from .cost import bbr_contrast_cost, gradient_magnitude_cost
from .optimize import BBRModel

__all__ = [
    'load_surface',
    'load_surface_pair',
    'load_surface_from_subject',
    'compute_vertex_normals',
    'project_vertices',
    'sample_volume_at_vertices',
    'bbr_contrast_cost',
    'gradient_magnitude_cost',
    'BBRModel',
]


