"""Immutable configuration DTOs for CAD generation, CFD domain, and ANSYS output."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

DEFAULT_SOLVER_ITERATIONS = 300
DEFAULT_MESH_CONV_TOL = 0.01
DEFAULT_MESH_CONV_MAX_LEVELS = 4
DEFAULT_MESH_CONV_FACTOR = 1.35


def default_fluent_cores() -> int:
    """Return a sensible default Fluent core count: min(8, logical_cpus // 2), at least 1."""
    logical = os.cpu_count() or 4
    return max(1, min(8, max(1, logical // 2)))


class DomainMode(str, Enum):
    """Geometry topology exported to STEP."""

    FLUID_DOMAIN = "fluid_domain"
    SYMMETRY_HALF = "symmetry_half"
    SYMMETRY_SECTOR = "symmetry_sector"
    SYMMETRY_HALF_SECTOR = "symmetry_half_sector"
    AXISYMMETRIC_2D = "axisymmetric_2d"
    OML_ONLY = "oml_only"


class EnclosureShape(str, Enum):
    """Farfield fluid domain outer shape."""

    BULLET = "bullet"
    CYLINDER = "cylinder"
    BOX = "box"


class FlowRegime(str, Enum):
    """Aerodynamic flow regime classification."""

    AUTO = "auto"
    SUBSONIC = "subsonic"
    TRANSONIC = "transonic"
    SUPERSONIC = "supersonic"


class MeshPreset(str, Enum):
    """Pre-tuned mesh density levels balancing accuracy and cell budget."""

    COARSE = "coarse"
    STUDENT_SAFE = "student_safe"
    STANDARD = "standard"
    FINE = "fine"


@dataclass(frozen=True)
class CadConfig:
    """Parameters governing CFD-clean B-Rep OML construction and STEP export."""

    tip_blunt_radius_m: float = 0.001
    fin_root_embed_m: float = 0.0015
    fin_te_thickness_m: float = 0.0005
    apply_fin_fillets: bool = False
    fillet_radius_override_m: float | None = None
    include_rail_buttons: bool = False
    include_launch_lugs: bool = False
    step_schema: str = "AP214"
    curve_samples: int = 80
    stl_linear_deflection_m: float = 0.0005
    stl_angular_deflection_deg: float = 12.0


@dataclass(frozen=True)
class BoiConfig:
    """Sizing parameters for Fluent Meshing Bodies of Influence (BOIs)."""

    enabled: bool = True
    nearfield_upstream_L: float = 0.35
    nearfield_downstream_L: float = 0.80
    nearfield_radius_factor: float = 1.60
    wake_length_L: float = 5.0
    wake_end_radius_factor: float = 3.0


@dataclass(frozen=True)
class DomainConfig:
    """Fluid domain enclosure and symmetry configuration."""

    mode: DomainMode = DomainMode.FLUID_DOMAIN
    shape: EnclosureShape = EnclosureShape.BULLET
    upstream_L: float | None = None
    downstream_L: float | None = None
    radial_L: float | None = None
    boi: BoiConfig = field(default_factory=BoiConfig)


@dataclass(frozen=True)
class FlightCondition:
    """Operating flight point and solver preferences."""

    mach: float = 0.8
    altitude_m: float = 500.0
    aoa_deg: float = 0.0
    target_yplus: float = 1.0
    regime_override: FlowRegime = FlowRegime.AUTO
    mesh_preset: MeshPreset = MeshPreset.STANDARD
    cores: int = field(default_factory=default_fluent_cores)
    iterations: int = DEFAULT_SOLVER_ITERATIONS
    ram_limit_gb: float | None = None


@dataclass(frozen=True)
class MeshConvergenceConfig:
    """Settings for progressive coarse-to-fine grid-independence CFD studies."""

    enabled: bool = False
    cd_rel_tol: float = DEFAULT_MESH_CONV_TOL
    max_levels: int = DEFAULT_MESH_CONV_MAX_LEVELS
    refinement_factor: float = DEFAULT_MESH_CONV_FACTOR
    max_cells: int | None = None


@dataclass(frozen=True)
class DeliverableConfig:
    """Output deliverables requested from the ork2cfd pipeline."""

    export_step: bool = True
    export_rocket_step: bool = True
    export_stl: bool = False
    export_verification: bool = True
    generate_scripts: bool = True
    build_wbpj: bool = True
    archive_wbpz: bool = False
    auto_run_cfd: bool = False
    ansys_version: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration object consumed by pipeline, CLI, and GUI."""

    cad: CadConfig = field(default_factory=CadConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    flight: FlightCondition = field(default_factory=FlightCondition)
    deliverables: DeliverableConfig = field(default_factory=DeliverableConfig)
    mesh_convergence: MeshConvergenceConfig = field(default_factory=MeshConvergenceConfig)
    stage_selection: str = "all"
