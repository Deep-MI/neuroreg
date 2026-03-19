"""Neural-network models and training loop for image registration."""

from .reg_model import RegModel
from .robust import cauchy_weights, compute_mad, compute_scale_estimate, huber_weights, tukey_weights
from .stopper import EarlyStopper
from .training import training_loop

__all__ = [
    "RegModel",
    "EarlyStopper",
    "training_loop",
    "tukey_weights",
    "huber_weights",
    "cauchy_weights",
    "compute_mad",
    "compute_scale_estimate",
]
