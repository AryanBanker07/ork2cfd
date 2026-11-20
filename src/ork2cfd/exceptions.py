"""Domain exception hierarchy for ork2cfd."""


class Ork2CfdError(Exception):
    """Base exception for all ork2cfd errors."""


class OrkParseError(Ork2CfdError):
    """Raised when an OpenRocket (.ork) file cannot be parsed or is structurally invalid."""


class GeometryBuildError(Ork2CfdError):
    """Raised when 3D CAD / OCCT B-Rep construction fails."""


class GeometryValidationError(Ork2CfdError):
    """Raised when constructed geometry fails CFD watertightness or quality checks."""


class AnsysError(Ork2CfdError):
    """Raised when ANSYS script generation, Workbench execution, or Fluent launching fails."""
