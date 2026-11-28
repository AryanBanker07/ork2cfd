"""Physics and boundary-layer sizing calculations for rocket external CFD."""

from .atmosphere import compute_atmosphere
from .boundary_layer import compute_boundary_layer
from .regime import classify_regime, recommend_solver
from .resources import (
    ResourceRecommendation,
    SystemHardware,
    detect_system_hardware,
    recommend_resources,
)

__all__ = [
    "ResourceRecommendation",
    "SystemHardware",
    "classify_regime",
    "compute_atmosphere",
    "compute_boundary_layer",
    "detect_system_hardware",
    "recommend_resources",
    "recommend_solver",
]
