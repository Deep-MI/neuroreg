"""Segmentation-based registration via label centroids."""

from .points import RobustPointRegistrationInfo, register_points_robust
from .register import RegistrationResult, segreg

__all__ = ["RegistrationResult", "RobustPointRegistrationInfo", "register_points_robust", "segreg"]
