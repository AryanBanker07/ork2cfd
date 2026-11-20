"""ork2cfd: Convert OpenRocket (.ork) designs to CFD-ready STEP, ANSYS scripts, and Workbench."""

__version__ = "0.1.0"

from .exceptions import (
    AnsysError,
    GeometryBuildError,
    GeometryValidationError,
    Ork2CfdError,
    OrkParseError,
)

__all__ = [
    "__version__",
    "Ork2CfdError",
    "OrkParseError",
    "GeometryBuildError",
    "GeometryValidationError",
    "AnsysError",
]
