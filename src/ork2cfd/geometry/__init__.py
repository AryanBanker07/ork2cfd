"""CFD-clean 3D B-Rep geometry construction, validation, and XCAF STEP export."""

from .domain_builder import DomainBuildOutput, build_cfd_domain
from .oml_builder import OmlBuildOutput, build_rocket_oml
from .profiles import ProfileSegmentMeta, build_merged_meridian_points_mm
from .step_exporter import export_step_with_names
from .validator import validate_brep_shape

__all__ = [
    "DomainBuildOutput",
    "OmlBuildOutput",
    "ProfileSegmentMeta",
    "build_cfd_domain",
    "build_merged_meridian_points_mm",
    "build_rocket_oml",
    "export_step_with_names",
    "validate_brep_shape",
]
