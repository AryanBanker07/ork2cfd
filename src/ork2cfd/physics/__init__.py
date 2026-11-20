"""Physics and boundary-layer sizing calculations for rocket external CFD."""

from .atmosphere import compute_atmosphere
from .boundary_layer import compute_boundary_layer
from .regime import classify_regime, recommend_solver

__all__ = [
    "classify_regime",
    "compute_atmosphere",
    "compute_boundary_layer",
    "recommend_solver",
]
