"""Immutable configuration and result data models for fin shape optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..models.cfd import (
    DEFAULT_SOLVER_ITERATIONS,
    EnclosureShape,
    MeshPreset,
    default_fluent_cores,
)
from ..models.rocket import FinCrossSection

ACTIVE_CONTINUOUS_PARAMS: tuple[str, ...] = (
    "root_chord",
    "tip_chord",
    "span",
    "sweep_angle_deg",
)


@dataclass(frozen=True)
class FinDesignBounds:
    """Box bounds (SI meters / degrees) for active continuous trapezoidal fin parameters."""

    root_chord: tuple[float, float]
    tip_chord: tuple[float, float]
    span: tuple[float, float]
    sweep_angle_deg: tuple[float, float]

    def get_range(self, name: str) -> tuple[float, float]:
        """Return (low, high) tuple for an active parameter name."""
        if name not in ACTIVE_CONTINUOUS_PARAMS:
            raise ValueError(f"Unsupported continuous fin parameter '{name}'.")
        return getattr(self, name)

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize bounds to JSON-friendly dictionary."""
        return {
            "root_chord": list(self.root_chord),
            "tip_chord": list(self.tip_chord),
            "span": list(self.span),
            "sweep_angle_deg": list(self.sweep_angle_deg),
        }


@dataclass(frozen=True)
class FinCandidate:
    """Complete fin shape parameterization for one optimization evaluation."""

    cross_section: FinCrossSection
    root_chord: float
    tip_chord: float
    span: float
    sweep_angle_deg: float
    thickness: float

    @property
    def sweep_length(self) -> float:
        """Axial leading-edge sweep offset at the fin tip in meters."""
        return self.span * math.tan(math.radians(self.sweep_angle_deg))

    @property
    def taper_ratio(self) -> float:
        """Tip-to-root chord taper ratio lambda = tip_chord / root_chord."""
        return self.tip_chord / max(self.root_chord, 1e-9)

    @property
    def planform_area(self) -> float:
        """Single-fin planform area S = 0.5 * (c_r + c_t) * span in m^2."""
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    def to_vector(self, active_params: tuple[str, ...]) -> list[float]:
        """Extract ordered continuous vector matching active_params."""
        return [float(getattr(self, name)) for name in active_params]

    def with_vector(
        self,
        active_params: tuple[str, ...],
        values: list[float] | tuple[float, ...],
    ) -> FinCandidate:
        """Return a new FinCandidate updating only active_params while locking thickness."""
        updates = {
            name: float(val)
            for name, val in zip(active_params, values, strict=True)
        }
        return replace(self, **updates)

    def to_dict(self) -> dict[str, object]:
        """Serialize candidate parameters."""
        return {
            "cross_section": self.cross_section.value,
            "root_chord": round(self.root_chord, 6),
            "tip_chord": round(self.tip_chord, 6),
            "span": round(self.span, 6),
            "sweep_angle_deg": round(self.sweep_angle_deg, 4),
            "sweep_length": round(self.sweep_length, 6),
            "thickness": round(self.thickness, 6),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FinCandidate:
        """Deserialize a FinCandidate dictionary."""
        return cls(
            cross_section=FinCrossSection(str(payload["cross_section"])),
            root_chord=float(payload["root_chord"]),
            tip_chord=float(payload["tip_chord"]),
            span=float(payload["span"]),
            sweep_angle_deg=float(payload["sweep_angle_deg"]),
            thickness=float(payload["thickness"]),
        )


@dataclass(frozen=True)
class StabilityReport:
    """Analytical Barrowman CP / static margin and fin flutter feasibility check."""

    x_cp_m: float
    x_cg_m: float
    cg_source: str
    cna_total: float
    static_margin_cal: float
    v_flutter_m_s: float
    feasible: bool
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize report."""
        return {
            "x_cp_m": round(self.x_cp_m, 5),
            "x_cg_m": round(self.x_cg_m, 5),
            "cg_source": self.cg_source,
            "cna_total": round(self.cna_total, 4),
            "static_margin_cal": round(self.static_margin_cal, 4),
            "v_flutter_m_s": round(self.v_flutter_m_s, 2),
            "feasible": self.feasible,
            "violations": list(self.violations),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StabilityReport:
        """Deserialize StabilityReport."""
        return cls(
            x_cp_m=float(payload["x_cp_m"]),
            x_cg_m=float(payload["x_cg_m"]),
            cg_source=str(payload.get("cg_source", "unknown")),
            cna_total=float(payload["cna_total"]),
            static_margin_cal=float(payload["static_margin_cal"]),
            v_flutter_m_s=float(payload["v_flutter_m_s"]),
            feasible=bool(payload["feasible"]),
            violations=tuple(str(v) for v in (payload.get("violations") or ())),
        )


@dataclass(frozen=True)
class SimEvaluation:
    """Recorded evaluation from one optimization step (live Fluent or dry-run)."""

    sim_index: int
    phase: str
    candidate: FinCandidate
    stability: StabilityReport
    cd_total: float
    cd_fins: float | None = None
    total_drag_n: float | None = None
    wall_yplus_avg: float | None = None
    mesh_cells: int | None = None
    run_dir: str = ""
    is_dry_run: bool = False
    contour_png: str | None = None
    diverged: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize evaluation record for JSON checkpointing."""
        return {
            "sim_index": self.sim_index,
            "phase": self.phase,
            "candidate": self.candidate.to_dict(),
            "stability": self.stability.to_dict(),
            "cd_total": round(self.cd_total, 6),
            "cd_fins": round(self.cd_fins, 6) if self.cd_fins is not None else None,
            "total_drag_n": round(self.total_drag_n, 4) if self.total_drag_n is not None else None,
            "wall_yplus_avg": (
                round(self.wall_yplus_avg, 4) if self.wall_yplus_avg is not None else None
            ),
            "mesh_cells": self.mesh_cells,
            "run_dir": self.run_dir,
            "is_dry_run": self.is_dry_run,
            "contour_png": self.contour_png,
            "diverged": bool(self.diverged),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SimEvaluation:
        """Deserialize evaluation record from checkpoint JSON."""
        return cls(
            sim_index=int(payload["sim_index"]),
            phase=str(payload["phase"]),
            candidate=FinCandidate.from_dict(payload["candidate"]),  # type: ignore[arg-type]
            stability=StabilityReport.from_dict(payload["stability"]),  # type: ignore[arg-type]
            cd_total=float(payload["cd_total"]),
            cd_fins=float(payload["cd_fins"]) if payload.get("cd_fins") is not None else None,
            total_drag_n=(
                float(payload["total_drag_n"])
                if payload.get("total_drag_n") is not None
                else None
            ),
            wall_yplus_avg=(
                float(payload["wall_yplus_avg"])
                if payload.get("wall_yplus_avg") is not None
                else None
            ),
            mesh_cells=(
                int(payload["mesh_cells"]) if payload.get("mesh_cells") is not None else None
            ),
            run_dir=str(payload.get("run_dir", "")),
            is_dry_run=bool(payload.get("is_dry_run", False)),
            contour_png=(str(payload["contour_png"]) if payload.get("contour_png") else None),
            diverged=bool(payload.get("diverged", False)),
        )


@dataclass(frozen=True)
class LockedMeshCharacteristics:
    """Grid-independent mesh sizing locked from the initial mesh convergence study."""

    mesh_preset: MeshPreset
    mesh_scale: float
    target_yplus: float
    optimal_level: int
    finest_level: int
    converged: bool
    stop_reason: str
    surf_max_m: float | None = None
    mesh_cells: int | None = None
    gci_fine_pct: float | None = None
    source: str = "baseline_mesh_convergence"

    def to_dict(self) -> dict[str, object]:
        """Serialize locked mesh characteristics to JSON-safe dictionary."""
        return {
            "mesh_preset": self.mesh_preset.value,
            "mesh_scale": round(float(self.mesh_scale), 6),
            "target_yplus": round(float(self.target_yplus), 4),
            "optimal_level": int(self.optimal_level),
            "finest_level": int(self.finest_level),
            "converged": bool(self.converged),
            "stop_reason": str(self.stop_reason),
            "surf_max_m": round(float(self.surf_max_m), 6) if self.surf_max_m is not None else None,
            "mesh_cells": int(self.mesh_cells) if self.mesh_cells is not None else None,
            "gci_fine_pct": (
                round(float(self.gci_fine_pct), 4) if self.gci_fine_pct is not None else None
            ),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LockedMeshCharacteristics:
        """Deserialize LockedMeshCharacteristics dictionary."""
        surf_raw = payload.get("surf_max_m")
        cells_raw = payload.get("mesh_cells")
        gci_raw = payload.get("gci_fine_pct")
        return cls(
            mesh_preset=MeshPreset(str(payload["mesh_preset"])),
            mesh_scale=float(payload["mesh_scale"]),
            target_yplus=float(payload["target_yplus"]),
            optimal_level=int(payload["optimal_level"]),
            finest_level=int(payload["finest_level"]),
            converged=bool(payload.get("converged", False)),
            stop_reason=str(payload.get("stop_reason", "unknown")),
            surf_max_m=float(surf_raw) if surf_raw is not None else None,
            mesh_cells=int(cells_raw) if cells_raw is not None else None,
            gci_fine_pct=float(gci_raw) if gci_raw is not None else None,
            source=str(payload.get("source", "baseline_mesh_convergence")),
        )


@dataclass(frozen=True)
class FinOptimizationConfig:
    """Configuration for surrogate-based fin optimization under a strict simulation budget."""

    max_sims: int = 10
    stage_selection: str = "all"
    target_fin_stage: str = "booster"
    active_params: tuple[str, ...] = ACTIVE_CONTINUOUS_PARAMS
    screen_cross_sections: tuple[FinCrossSection, ...] = ()
    mach: float = 0.80
    altitude_m: float = 500.0
    aoa_deg: float = 0.0
    min_static_margin_cal: float = 2.0
    cg_x_m: float | None = None
    fin_shear_modulus_pa: float = 15.0e9
    min_flutter_sf: float = 0.0
    domain_mode: str = "auto"
    enclosure_shape: EnclosureShape = EnclosureShape.CYLINDER
    mesh_preset: MeshPreset = MeshPreset.COARSE
    target_yplus: float = 30.0
    verify_final: bool = True
    verify_yplus: float = 1.0
    verify_preset: MeshPreset = MeshPreset.STANDARD
    blunt_radius_m: float = 0.0004
    cores: int = field(default_factory=default_fluent_cores)
    iterations: int = DEFAULT_SOLVER_ITERATIONS
    ram_limit_gb: float | None = None
    max_cells: int | None = None
    seed: int = 42
    dry_run: bool = False
    export_best_cad: bool = True
    resume: bool = True
    mesh_convergence_baseline: bool = True
    mesh_conv_tol: float = 0.01
    mesh_conv_max_levels: int = 4
    mesh_conv_factor: float = 1.35
    mesh_convergence_source: Path | None = None


@dataclass(frozen=True)
class FinOptimizationResult:
    """Final deliverable summary from a fin shape optimization campaign."""

    output_dir: Path
    target_stage_name: str
    baseline_eval: SimEvaluation
    best_eval: SimEvaluation
    verification_eval: SimEvaluation | None
    evaluations: tuple[SimEvaluation, ...]
    optimized_ork_path: Path
    history_json_path: Path
    summary_csv_path: Path
    report_md_path: Path
    best_step_path: Path | None
    delta_cd: float
    delta_cd_pct: float
    cd_plot_path: Path | None = None
    cfd_images_dir: Path | None = None
    locked_mesh: LockedMeshCharacteristics | None = None
