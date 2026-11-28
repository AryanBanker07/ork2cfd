"""Result DTOs returned by physics, geometry validation, and pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .cfd import FlowRegime
from .rocket import RocketModel


@dataclass(frozen=True)
class AtmosphereState:
    """ISA atmospheric properties at flight altitude."""

    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    viscosity_pa_s: float
    speed_of_sound_m_s: float


@dataclass(frozen=True)
class BoundaryLayerReport:
    """Computed Reynolds number, y+ sizing, prism mesh parameters, and cell budget."""

    atmosphere: AtmosphereState
    velocity_m_s: float
    mach: float
    reynolds_length: float
    reynolds_diameter: float
    skin_friction_cf: float
    wall_shear_pa: float
    friction_velocity_m_s: float
    target_yplus: float
    first_cell_center_m: float
    first_layer_thickness_m: float
    bl_thickness_99_m: float
    prism_layers: int
    prism_growth_rate: float
    surface_min_size_m: float
    surface_max_size_m: float
    boi_nearfield_size_m: float
    boi_wake_size_m: float
    farfield_max_size_m: float
    estimated_cell_count: int
    exceeds_student_limit: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SolverRecommendation:
    """Flow-regime-aware Fluent solver and domain sizing recommendations."""

    regime: FlowRegime
    solver_type: str  # "pressure-based" | "density-based"
    energy_enabled: bool
    fluid_model: str  # "constant" | "ideal-gas"
    viscosity_model: str  # "constant" | "sutherland"
    turbulence_model: str  # "k-omega-sst"
    compressibility_effects: bool
    roughness_height_m: float
    inlet_bc_type: str
    outlet_bc_type: str
    farfield_bc_type: str
    recommended_upstream_L: float
    recommended_downstream_L: float
    recommended_radial_L: float
    courant_number: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeometryReport:
    """B-Rep validation diagnostics for constructed OML / fluid domain geometry."""

    is_valid: bool
    is_watertight: bool
    solid_count: int
    free_edge_count: int
    min_edge_length_m: float
    sliver_face_count: int
    oml_volume_m3: float
    oml_wetted_area_m2: float
    oml_frontal_area_m2: float
    domain_volume_m3: float
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    named_selections: dict[str, int] = field(default_factory=dict)
    boi_names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VerificationCheck:
    """Single quantitative CAD verification check result."""

    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    measured: str
    expected: str
    details: str = ""


@dataclass(frozen=True)
class VerificationReport:
    """Aggregate CAD quality, dimensional fidelity, and STEP round-trip audit."""

    overall_status: str  # "PASS" | "WARN" | "FAIL"
    checks: tuple[VerificationCheck, ...]
    zone_areas_m2: dict[str, float] = field(default_factory=dict)
    zone_face_counts: dict[str, int] = field(default_factory=dict)
    zone_triangle_counts: dict[str, int] = field(default_factory=dict)
    oml_bbox_min_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    oml_bbox_max_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    step_roundtrip_ok: bool | None = None
    summary_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CfdZoneResult:
    """Aerodynamic drag contribution for a single rocket wall boundary zone."""

    zone: str
    drag_force_n: float
    cd: float


@dataclass(frozen=True)
class CfdMachPoint:
    """Converged Fluent CFD drag result for one Mach operating point."""

    mach: float
    velocity_m_s: float
    total_cd: float
    total_drag_n: float
    wall_yplus_avg: float | None = None
    case_file: Path | None = None
    data_file: Path | None = None
    mach_contour_png: Path | None = None
    pressure_contour_png: Path | None = None
    zones: tuple[CfdZoneResult, ...] = ()


@dataclass(frozen=True)
class MeshRefinementLevel:
    """Single mesh-refinement level evaluated during a grid-independence study."""

    level: int
    mesh_scale: float
    surf_max_m: float
    mesh_cells: int | None
    min_orth_quality: float | None
    total_cd: float
    total_drag_n: float
    wall_yplus_avg: float | None = None
    rel_cd_change: float | None = None


@dataclass(frozen=True)
class MeshConvergenceReport:
    """Summary, stopping diagnostics, and GCI metrics for a mesh-convergence study."""

    converged: bool
    cd_rel_tol: float
    refinement_factor: float
    levels: tuple[MeshRefinementLevel, ...]
    optimal_level: int
    finest_level: int
    gci_fine_pct: float | None = None
    richardson_cd: float | None = None
    observed_order_p: float | None = None
    stop_reason: str = "converged"
    report_json: Path | None = None
    report_csv: Path | None = None
    report_md: Path | None = None
    plot_png: Path | None = None


@dataclass(frozen=True)
class CfdSolveResult:
    """Aggregated ANSYS Fluent CFD solve outputs parsed from cfd_summary.json / CSV."""

    rocket_name: str
    output_dir: Path
    altitude_m: float
    aoa_deg: float
    sym_factor: float
    s_ref_m2: float
    mesh_cells: int | None = None
    min_orth_quality: float | None = None
    runs: tuple[CfdMachPoint, ...] = ()
    results_csv: Path | None = None
    summary_json: Path | None = None
    mesh_convergence: MeshConvergenceReport | None = None

    @property
    def primary_run(self) -> CfdMachPoint | None:
        """Return the primary (first) solved Mach operating point, if any."""
        return self.runs[0] if self.runs else None


@dataclass(frozen=True)
class PackageResult:
    """Complete output package produced by the ork2cfd pipeline."""

    rocket: RocketModel
    bl_report: BoundaryLayerReport
    solver_rec: SolverRecommendation
    geometry_report: GeometryReport | None
    output_dir: Path
    step_path: Path | None = None
    stl_path: Path | None = None
    manifest_path: Path | None = None
    rocket_step_path: Path | None = None
    rocket_manifest_path: Path | None = None
    script_paths: dict[str, Path] = field(default_factory=dict)
    wbpj_path: Path | None = None
    wbpz_path: Path | None = None
    verification_report: VerificationReport | None = None
    viewer_html_path: Path | None = None
    views_png_path: Path | None = None
    verification_json_path: Path | None = None
    verification_md_path: Path | None = None
    cfd_result: CfdSolveResult | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
