"""Immutable configuration DTOs for CAD generation, CFD domain, and ANSYS output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DomainMode(str, Enum):
    """Geometry topology exported to STEP."""

    FLUID_DOMAIN = "fluid_domain"
    SYMMETRY_HALF = "symmetry_half"
    SYMMETRY_SECTOR = "symmetry_sector"
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
    cores: int = 4
    iterations: int = 400


@dataclass(frozen=True)
class DeliverableConfig:
    """Output deliverables requested from the ork2cfd pipeline."""

    export_step: bool = True
    generate_scripts: bool = True
    build_wbpj: bool = True
    archive_wbpz: bool = False
    ansys_version: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration object consumed by pipeline, CLI, and GUI."""

    cad: CadConfig = field(default_factory=CadConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    flight: FlightCondition = field(default_factory=FlightCondition)
    deliverables: DeliverableConfig = field(default_factory=DeliverableConfig)
