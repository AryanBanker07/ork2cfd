"""CFD-clean 3D B-Rep geometry construction, validation, and XCAF STEP export."""

from .domain_builder import DomainBuildOutput, build_cfd_domain, build_rocket_component_domain
from .oml_builder import OmlBuildOutput, build_rocket_oml
from .profiles import ProfileSegmentMeta, build_merged_meridian_points_mm
from .step_exporter import export_step_with_names, zone_color_rgb
from .stl_exporter import export_stl_with_regions
from .tessellation import ZoneMesh, tessellate_domain_zones
from .validator import validate_brep_shape
from .verifier import load_step_for_verification, verify_cad_model, write_verification_reports
from .viewer_exporter import export_cad_viewer_html, export_cad_views_png

__all__ = [
    "DomainBuildOutput",
    "OmlBuildOutput",
    "ProfileSegmentMeta",
    "ZoneMesh",
    "build_cfd_domain",
    "build_merged_meridian_points_mm",
    "build_rocket_component_domain",
    "build_rocket_oml",
    "export_cad_viewer_html",
    "export_cad_views_png",
    "export_step_with_names",
    "export_stl_with_regions",
    "load_step_for_verification",
    "tessellate_domain_zones",
    "validate_brep_shape",
    "verify_cad_model",
    "write_verification_reports",
    "zone_color_rgb",
]
