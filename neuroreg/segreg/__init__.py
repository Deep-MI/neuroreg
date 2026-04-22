"""Public API for segmentation-based centroid registration.

The :func:`segreg` entry point fits RAS-space transforms from matched label
centroids and returns both the recovered transform and the target-geometry
metadata needed for writing LTAs or mapped outputs.
"""

from .register import RegistrationResult, segreg

__all__ = ["RegistrationResult", "segreg"]
