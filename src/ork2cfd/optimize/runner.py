"""Crash-safe 10-simulation constrained fin optimization orchestrator."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from ..ansys.detector import select_best_installation
from ..ansys.fluent_launcher import find_solved_files, parse_cfd_results, run_fluent_solve_blocking
from ..ansys.mesh_convergence import is_valid_aerodynamic_cd, run_mesh_convergence_blocking
from ..exceptions import AnsysError, Ork2CfdError
from ..models.cfd import (
    BoiConfig,
    CadConfig,
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    FlightCondition,
    MeshConvergenceConfig,
    MeshPreset,
    PipelineConfig,
)
from ..models.rocket import RocketModel
from ..parser.ork_parser import parse_ork_file
from ..physics.atmosphere import compute_atmosphere
from ..physics.regime import recommend_solver
from ..physics.resources import recommend_resources
from ..pipeline import run_pipeline, sanitize_stem
from .constraints import estimate_cg_from_ork, evaluate_constraints
from .models import (
    ACTIVE_CONTINUOUS_PARAMS,
    FinCandidate,
    FinDesignBounds,
    FinOptimizationConfig,
    FinOptimizationResult,
    LockedMeshCharacteristics,
    SimEvaluation,
)
from .parameterizer import (
    apply_candidate_to_rocket,
    candidate_from_fin,
    default_bounds_for_fin,
    resolve_target_fin,
    write_modified_ork,
)
from .plots import render_fluent_hdf5_contours, render_optimization_cd_plot
from .surrogate import (
    GaussianProcessSurrogate,
    estimate_analytic_drag,
    propose_next_bo_candidate,
    sample_feasible_lhs,
)

ProgressCallback = Callable[[str, float], None]
CandidateEvaluator = Callable[
    [int, str, RocketModel, FinCandidate, Path, FinOptimizationConfig, bool],
    tuple[float, float | None, float | None, float | None, int | None],
]

_RUN_FLUENT_PRESET_RE = re.compile(r'MESH_PRESET\s*=\s*"([^"]+)"')


def load_locked_mesh_from_convergence(
    source: Path,
    *,
    fallback_preset: MeshPreset,
    fallback_yplus: float,
) -> LockedMeshCharacteristics | None:
    """Load optimal mesh preset, scale, and y+ from an existing mesh_convergence.json study."""
    src = Path(source)
    json_path = (src / "mesh_convergence.json") if src.is_dir() else src
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, list) or not levels:
        return None

    opt_id = int(data.get("optimal_level", levels[-1].get("level", len(levels))))
    opt_lv = next((lv for lv in levels if int(lv.get("level", -1)) == opt_id), levels[-1])

    preset = fallback_preset
    if data.get("base_preset"):
        with contextlib.suppress(ValueError):
            preset = MeshPreset(str(data["base_preset"]))
    else:
        run_py = json_path.parent / "run_fluent.py"
        if run_py.exists():
            m = _RUN_FLUENT_PRESET_RE.search(run_py.read_text(encoding="utf-8", errors="replace"))
            if m:
                with contextlib.suppress(ValueError):
                    preset = MeshPreset(m.group(1))

    yplus_raw = data.get("target_yplus")
    yplus = float(yplus_raw) if yplus_raw is not None else float(fallback_yplus)
    surf_raw = opt_lv.get("surf_max_m")
    cells_raw = opt_lv.get("mesh_cells")
    gci_raw = data.get("gci_fine_pct")
    return LockedMeshCharacteristics(
        mesh_preset=preset,
        mesh_scale=float(opt_lv.get("mesh_scale", 1.0)),
        target_yplus=yplus,
        optimal_level=int(opt_lv.get("level", opt_id)),
        finest_level=int(data.get("finest_level", len(levels))),
        converged=bool(data.get("converged", False)),
        stop_reason=str(data.get("stop_reason", "unknown")),
        surf_max_m=float(surf_raw) if surf_raw is not None else None,
        mesh_cells=int(cells_raw) if cells_raw is not None else None,
        gci_fine_pct=float(gci_raw) if gci_raw is not None else None,
        source=str(json_path),
    )


def resolve_domain_mode(rocket: RocketModel, config: FinOptimizationConfig) -> DomainMode:
    """Resolve DomainMode automatically for fast axial fin optimization runs."""
    mode_str = config.domain_mode.strip().lower()
    if mode_str not in {"", "auto"}:
        return DomainMode(mode_str)
    if abs(config.aoa_deg) < 1e-6:
        # Prefer full fin-centered sector (90 deg on 4-fin) to avoid sharp beveled wedge slivers
        if rocket.rotational_symmetry_order >= 2:
            return DomainMode.SYMMETRY_SECTOR
        if rocket.has_sector_mirror_symmetry:
            return DomainMode.SYMMETRY_HALF_SECTOR
    elif rocket.has_sector_mirror_symmetry or rocket.rotational_symmetry_order % 2 == 0:
        return DomainMode.SYMMETRY_HALF
    return DomainMode.FLUID_DOMAIN


def _write_outputs(
    output_dir: Path,
    stem: str,
    target_stage_name: str,
    cg_x_m: float,
    cg_source: str,
    config: FinOptimizationConfig,
    bounds: FinDesignBounds,
    evals: list[SimEvaluation],
    optimized_ork_path: Path | None = None,
    best_step_path: Path | None = None,
    locked_mesh: LockedMeshCharacteristics | None = None,
    config_signature: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    """Write checkpoint JSON, summary CSV, Cd vs fin shape plot, and Markdown report."""
    history_path = output_dir / "optimization_history.json"
    csv_path = output_dir / "optimization_summary.csv"
    report_path = output_dir / "optimization_report.md"
    cd_plot_path: Path | None = None

    baseline = evals[0] if evals else None
    valid_evals = [e for e in evals if not e.diverged]
    feasible_evals = [e for e in valid_evals if e.stability.feasible]
    best = (
        min(feasible_evals or valid_evals or evals, key=lambda e: e.cd_total)
        if evals
        else None
    )

    history_payload = {
        "stem": stem,
        "target_stage_name": target_stage_name,
        "mach": config.mach,
        "altitude_m": config.altitude_m,
        "aoa_deg": config.aoa_deg,
        "min_static_margin_cal": config.min_static_margin_cal,
        "cg_x_m": round(cg_x_m, 5),
        "cg_source": cg_source,
        "active_params": list(config.active_params),
        "config_signature": config_signature,
        "bounds": bounds.to_dict(),
        "locked_mesh": locked_mesh.to_dict() if locked_mesh is not None else None,
        "evaluations": [e.to_dict() for e in evals],
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sim_index",
            "phase",
            "cross_section",
            "root_chord_m",
            "tip_chord_m",
            "span_m",
            "sweep_angle_deg",
            "sweep_length_m",
            "thickness_m",
            "x_cp_m",
            "static_margin_cal",
            "v_flutter_m_s",
            "feasible",
            "cd_total",
            "cd_fins",
            "delta_cd_vs_base_pct",
            "wall_yplus_avg",
            "mesh_cells",
            "is_dry_run",
            "contour_png",
            "cd_airframe",
            "diverged",
        ])
        base_cd = baseline.cd_total if baseline and baseline.cd_total > 0 else None
        for ev in evals:
            delta_pct = (
                ((ev.cd_total - base_cd) / base_cd) * 100.0 if base_cd is not None else 0.0
            )
            c = ev.candidate
            s = ev.stability
            cd_air_val = (ev.cd_total - ev.cd_fins) if ev.cd_fins is not None else None
            writer.writerow([
                ev.sim_index,
                ev.phase,
                c.cross_section.value,
                f"{c.root_chord:.6f}",
                f"{c.tip_chord:.6f}",
                f"{c.span:.6f}",
                f"{c.sweep_angle_deg:.3f}",
                f"{c.sweep_length:.6f}",
                f"{c.thickness:.6f}",
                f"{s.x_cp_m:.5f}",
                f"{s.static_margin_cal:.3f}",
                f"{s.v_flutter_m_s:.1f}",
                s.feasible,
                f"{ev.cd_total:.6f}",
                f"{ev.cd_fins:.6f}" if ev.cd_fins is not None else "",
                f"{delta_pct:+.2f}",
                f"{ev.wall_yplus_avg:.2f}" if ev.wall_yplus_avg is not None else "",
                ev.mesh_cells if ev.mesh_cells is not None else "",
                ev.is_dry_run,
                ev.contour_png or "",
                f"{cd_air_val:.6f}" if cd_air_val is not None else "",
                ev.diverged,
            ])

    if evals:
        plot_candidate = output_dir / "optimization_cd_plot.png"
        with contextlib.suppress(Exception):
            cd_plot_path = render_optimization_cd_plot(
                evals,
                plot_candidate,
                rocket_name=stem,
                target_stage_name=target_stage_name,
                min_sm_cal=config.min_static_margin_cal,
            )

    if baseline is not None and best is not None:
        bc, oc = baseline.candidate, best.candidate
        bs, os_ = baseline.stability, best.stability
        delta_cd = best.cd_total - baseline.cd_total
        delta_pct = (delta_cd / max(baseline.cd_total, 1e-9)) * 100.0
        scope_lbl = (
            "Whole-Rocket Total $C_D$"
            if config.stage_selection == "all"
            else "Whole-Stage Total $C_D$"
        )
        base_air = (
            (baseline.cd_total - baseline.cd_fins) if baseline.cd_fins is not None else None
        )
        best_air = (best.cd_total - best.cd_fins) if best.cd_fins is not None else None
        extra_cd_rows: list[str] = []
        if base_air is not None and best_air is not None:
            extra_cd_rows.append(
                f"| Airframe (Nose+Body+Base) $C_D$ | `{base_air:.6f}` "
                f"| `{best_air:.6f}` | `{best_air - base_air:+.6f}` |"
            )
        if baseline.cd_fins is not None and best.cd_fins is not None:
            extra_cd_rows.append(
                f"| Fins $C_D$ | `{baseline.cd_fins:.6f}` "
                f"| `{best.cd_fins:.6f}` | `{best.cd_fins - baseline.cd_fins:+.6f}` |"
            )
        lines = [
            f"# Fin Shape Optimization Report — {stem}",
            "",
            f"- **Target Stage Optimized:** `{target_stage_name}` (all other stages kept fixed)",
            f"- **Fixed Fin Thickness:** `{oc.thickness * 1000:.2f} mm`",
            (
                f"- **Flight Point:** Mach `{config.mach:.2f}` @ "
                f"`{config.altitude_m:.1f} m`, AoA `{config.aoa_deg:.1f}°`"
            ),
            (
                f"- **Stability Constraint:** `SM >= {config.min_static_margin_cal:.2f} cal` "
                f"(`X_cg = {cg_x_m:.3f} m`, source: `{cg_source}`)"
            ),
            (
                f"- **Evaluations Completed:** `{len(evals)} / {config.max_sims}` "
                f"(`dry_run={best.is_dry_run}`)"
            ),
            "",
            "## Baseline vs Optimized Target Fin",
            "",
            "| Metric | Baseline | Optimized | Change |",
            "| :--- | ---: | ---: | ---: |",
            (
                f"| Cross Section | `{bc.cross_section.value}` "
                f"| `{oc.cross_section.value}` | — |"
            ),
            (
                f"| Root Chord (mm) | `{bc.root_chord*1000:.2f}` "
                f"| `{oc.root_chord*1000:.2f}` | `{(oc.root_chord-bc.root_chord)*1000:+.2f}` |"
            ),
            (
                f"| Tip Chord (mm) | `{bc.tip_chord*1000:.2f}` "
                f"| `{oc.tip_chord*1000:.2f}` | `{(oc.tip_chord-bc.tip_chord)*1000:+.2f}` |"
            ),
            (
                f"| Span / Height (mm) | `{bc.span*1000:.2f}` "
                f"| `{oc.span*1000:.2f}` | `{(oc.span-bc.span)*1000:+.2f}` |"
            ),
            (
                f"| LE Sweep Angle (deg) | `{bc.sweep_angle_deg:.2f}` "
                f"| `{oc.sweep_angle_deg:.2f}` | `{oc.sweep_angle_deg-bc.sweep_angle_deg:+.2f}` |"
            ),
            (
                f"| Thickness (mm) | `{bc.thickness*1000:.2f}` "
                f"| `{oc.thickness*1000:.2f}` | `0.00 (locked)` |"
            ),
            (
                f"| Barrowman $X_{{CP}}$ (m) | `{bs.x_cp_m:.4f}` "
                f"| `{os_.x_cp_m:.4f}` | `{os_.x_cp_m-bs.x_cp_m:+.4f}` |"
            ),
            (
                f"| Static Margin (cal) | `{bs.static_margin_cal:.2f}` "
                f"| `{os_.static_margin_cal:.2f}` "
                f"| `{os_.static_margin_cal-bs.static_margin_cal:+.2f}` |"
            ),
            (
                f"| **{scope_lbl}** | **`{baseline.cd_total:.6f}`** "
                f"| **`{best.cd_total:.6f}`** | **`{delta_cd:+.6f} ({delta_pct:+.2f}%)`** |"
            ),
            *extra_cd_rows,
            "",
        ]
        if cd_plot_path is not None:
            lines.append(f"- **Cd vs Fin Shape Plot:** `{cd_plot_path.name}`")
        cfd_imgs = [e.contour_png for e in evals if e.contour_png]
        if cfd_imgs:
            lines.append(
                f"- **Per-Simulation CFD Contours (`cfd_images/`):** `{len(cfd_imgs)}` PNG(s)"
            )
        if locked_mesh is not None:
            gci_s = f", GCI={locked_mesh.gci_fine_pct:.2f}%" if locked_mesh.gci_fine_pct else ""
            cells_s = f", cells={locked_mesh.mesh_cells:,}" if locked_mesh.mesh_cells else ""
            lines.append(
                f"- **Locked Converged Mesh:** `{locked_mesh.mesh_preset.value}` "
                f"(scale=`{locked_mesh.mesh_scale:.4f}`, y+=`{locked_mesh.target_yplus:.2f}`, "
                f"L{locked_mesh.optimal_level}/{locked_mesh.finest_level}{cells_s}{gci_s})"
            )
        if optimized_ork_path is not None:
            lines.append(f"- **Optimized `.ork`:** `{optimized_ork_path.name}`")
        if best_step_path is not None:
            lines.append(f"- **Optimized CAD STEP:** `{best_step_path}`")
        report_path.write_text("\n".join(lines), encoding="utf-8")

    return history_path, csv_path, report_path, cd_plot_path


def _default_fluent_evaluator(
    source_ork: Path,
    target_ids: tuple[int, int, int],
    domain_mode: DomainMode,
    fluent_line_callback: Callable[[str], None] | None = None,
    locked_mesh_box: list[LockedMeshCharacteristics | None] | None = None,
) -> CandidateEvaluator:
    """Create live Fluent evaluation closure with initial mesh-convergence locking & msmpi retry."""
    box = locked_mesh_box if locked_mesh_box is not None else [None]
    preferred_mpi: list[str | None] = [None]

    def _eval(
        sim_idx: int,
        phase: str,
        cand_rocket: RocketModel,
        candidate: FinCandidate,
        sim_dir: Path,
        config: FinOptimizationConfig,
        high_fidelity: bool,
    ) -> tuple[float, float | None, float | None, float | None, int | None]:
        sim_dir.mkdir(parents=True, exist_ok=True)
        cand_ork = sim_dir / f"candidate_{sim_idx:02d}.ork"
        write_modified_ork(source_ork, cand_ork, target_ids, candidate)

        solver_rec = recommend_solver(
            cand_rocket,
            FlightCondition(mach=config.mach, altitude_m=config.altitude_m, aoa_deg=config.aoa_deg),
        )
        res_rec = recommend_resources(
            rocket=cand_rocket,
            mach=config.mach,
            aoa_deg=config.aoa_deg,
            solver_type=solver_rec.solver_type,
            requested_cores=config.cores,
            ram_limit_gb=config.ram_limit_gb,
        )
        effective_max_cells = config.max_cells or res_rec.max_safe_cells

        locked = box[0]
        run_baseline_conv = (
            phase == "baseline" and locked is None and config.mesh_convergence_baseline
        )
        if run_baseline_conv:
            preset = config.mesh_preset
            yplus = config.target_yplus
            mesh_conv_cfg = MeshConvergenceConfig(
                enabled=True,
                cd_rel_tol=config.mesh_conv_tol,
                max_levels=config.mesh_conv_max_levels,
                refinement_factor=config.mesh_conv_factor,
                max_cells=effective_max_cells,
            )
            mesh_scale: float | None = None
        elif high_fidelity:
            mesh_conv_cfg = MeshConvergenceConfig(enabled=False)
            can_refine_verify = (
                locked is not None
                and locked.converged
                and (
                    locked.mesh_cells is None
                    or (locked.mesh_cells * 2.0 <= effective_max_cells)
                )
            )
            if locked is not None and can_refine_verify:
                preset = config.verify_preset
                yplus = config.verify_yplus
                if locked.mesh_preset == config.verify_preset:
                    mesh_scale = min(
                        1.0,
                        round(locked.mesh_scale / max(config.mesh_conv_factor, 1.05), 6),
                    )
                else:
                    mesh_scale = None
            elif locked is not None:
                preset = locked.mesh_preset
                yplus = locked.target_yplus
                mesh_scale = locked.mesh_scale
                if fluent_line_callback is not None:
                    with contextlib.suppress(Exception):
                        fluent_line_callback(
                            f"[ork2cfd] Verification Sim keeping locked RAM-safe mesh "
                            f"(scale={mesh_scale:.4f}, y+={yplus:.1f}, "
                            f"cells={locked.mesh_cells or 'auto'}) to prevent out-of-memory crash."
                        )
            else:
                preset = config.verify_preset
                yplus = config.verify_yplus
                mesh_scale = None
        else:
            preset = locked.mesh_preset if locked is not None else config.mesh_preset
            yplus = locked.target_yplus if locked is not None else config.target_yplus
            mesh_conv_cfg = MeshConvergenceConfig(enabled=False)
            mesh_scale = locked.mesh_scale if locked is not None else None

        cfg = PipelineConfig(
            cad=CadConfig(
                tip_blunt_radius_m=config.blunt_radius_m,
                apply_fin_fillets=False,
            ),
            domain=DomainConfig(
                mode=domain_mode,
                shape=config.enclosure_shape,
                boi=BoiConfig(enabled=True),
            ),
            flight=FlightCondition(
                mach=config.mach,
                altitude_m=config.altitude_m,
                aoa_deg=config.aoa_deg,
                target_yplus=yplus,
                mesh_preset=preset,
                cores=config.cores,
                iterations=config.iterations,
                ram_limit_gb=config.ram_limit_gb,
            ),
            deliverables=DeliverableConfig(
                export_step=True,
                export_rocket_step=False,
                export_stl=False,
                export_verification=False,
                generate_scripts=True,
                build_wbpj=False,
                archive_wbpz=False,
                auto_run_cfd=False,
            ),
            mesh_convergence=mesh_conv_cfg,
            stage_selection=config.stage_selection,
        )
        pkg_res = run_pipeline(
            ork_path=cand_ork,
            output_dir=sim_dir,
            config=cfg,
            rocket_override=cand_rocket,
        )
        ansys_install = select_best_installation(cfg.deliverables.ansys_version)

        if run_baseline_conv:
            conv_kw: dict[str, object] = {}
            if preferred_mpi[0] is not None:
                conv_kw["mpi"] = preferred_mpi[0]
            cfd_res = run_mesh_convergence_blocking(
                output_dir=sim_dir,
                ansys_install=ansys_install,
                conv_config=mesh_conv_cfg,
                base_preset=preset,
                cores=config.cores,
                surf_max_base_m=pkg_res.bl_report.surface_max_size_m,
                rocket_name=pkg_res.rocket.name,
                mach=config.mach,
                target_yplus=yplus,
                line_callback=fluent_line_callback,
                **conv_kw,
            )
            mc = cfd_res.mesh_convergence
            if mc is not None and mc.levels:
                opt_lv = next(
                    (lv for lv in mc.levels if lv.level == mc.optimal_level),
                    mc.levels[-1],
                )
                box[0] = LockedMeshCharacteristics(
                    mesh_preset=preset,
                    mesh_scale=opt_lv.mesh_scale,
                    target_yplus=yplus,
                    optimal_level=mc.optimal_level,
                    finest_level=mc.finest_level,
                    converged=mc.converged,
                    stop_reason=mc.stop_reason,
                    surf_max_m=opt_lv.surf_max_m,
                    mesh_cells=opt_lv.mesh_cells,
                    gci_fine_pct=mc.gci_fine_pct,
                    source=str(sim_dir / "mesh_convergence.json"),
                )
                opt_dir = sim_dir / "mesh_convergence" / f"level_{mc.optimal_level:02d}"
                opt_parsed = parse_cfd_results(opt_dir) if opt_dir.exists() else None
                if opt_parsed is not None and opt_parsed.runs:
                    cfd_res = opt_parsed
        else:
            solve_kw: dict[str, object] = {"mesh_scale": mesh_scale}
            if preferred_mpi[0] is not None:
                solve_kw["mpi"] = preferred_mpi[0]
            try:
                cfd_res = run_fluent_solve_blocking(
                    output_dir=sim_dir,
                    ansys_install=ansys_install,
                    cores=config.cores,
                    line_callback=fluent_line_callback,
                    **solve_kw,
                )
            except AnsysError as exc:
                if preferred_mpi[0] is not None:
                    raise
                print(
                    f"[ork2cfd] Sim {sim_idx} default MPI failed ({exc}); "
                    "retrying once with -mpi=msmpi..."
                )
                cfd_res = run_fluent_solve_blocking(
                    output_dir=sim_dir,
                    ansys_install=ansys_install,
                    cores=config.cores,
                    line_callback=fluent_line_callback,
                    mpi="msmpi",
                    mesh_scale=mesh_scale,
                )
                preferred_mpi[0] = "msmpi"
        if not cfd_res.runs:
            raise Ork2CfdError(
                f"Simulation {sim_idx} ({phase}) completed without Cd results in '{sim_dir}'."
            )
        run = cfd_res.runs[0]
        fin_cd = next((z.cd for z in run.zones if "fins" in z.zone.lower()), None)
        return (
            float(run.total_cd),
            float(fin_cd) if fin_cd is not None else None,
            float(run.total_drag_n),
            run.wall_yplus_avg,
            cfd_res.mesh_cells,
        )

    return _eval


def run_fin_optimization(
    ork_path: Path,
    output_dir: Path,
    config: FinOptimizationConfig | None = None,
    bounds: FinDesignBounds | None = None,
    evaluator: CandidateEvaluator | None = None,
    progress_callback: ProgressCallback | None = None,
    eval_callback: Callable[[list[SimEvaluation]], None] | None = None,
    fluent_line_callback: Callable[[str], None] | None = None,
) -> FinOptimizationResult:
    """
    Execute a constrained <=10-sim aerodynamic fin optimization campaign for any .ork rocket.

    Phases (cross-section screening is skipped by default to keep the .ork cross-section fixed):
      1. Baseline evaluation (plus optional cross-section screen if `screen_cross_sections` set)
      2. Maximin feasible Latin Hypercube DoE on the active cross-section
      3. Sequential Bayesian Optimization (ARD Matérn-5/2 GP + Expected Improvement)
      4. Optional high-fidelity verification run (when max_sims >= 8 and not dry-run)
    """
    cfg = config or FinOptimizationConfig()
    if cfg.max_sims < 3:
        raise Ork2CfdError("`max_sims` must be at least 3 for baseline + screen + improvement.")
    for pname in cfg.active_params:
        if pname not in ACTIVE_CONTINUOUS_PARAMS:
            raise Ork2CfdError(
                f"Unsupported fin parameter '{pname}'. Allowed: {ACTIVE_CONTINUOUS_PARAMS}."
            )

    src_ork = Path(ork_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rocket = parse_ork_file(src_ork)
    selected_indices = full_rocket.resolve_stage_indices(cfg.stage_selection)
    sim_rocket = full_rocket.select_stages(cfg.stage_selection)
    stem = sanitize_stem(sim_rocket.name)

    target_ids, target_stage, parent_comp, target_fin = resolve_target_fin(
        sim_rocket, cfg.target_fin_stage
    )
    # Map target stage index directly to full_rocket so identical stage names never collide
    full_target_ids = (selected_indices[target_ids[0]], target_ids[1], target_ids[2])

    stage_offset_m = (
        full_rocket.stages[selected_indices[0]].axial_start if selected_indices else 0.0
    )
    if cfg.cg_x_m is not None:
        raw_cg = float(cfg.cg_x_m)
        if stage_offset_m > 1e-6 and raw_cg >= stage_offset_m:
            cg_x_m = raw_cg - stage_offset_m
            cg_source = "user_specified_shifted"
        else:
            cg_x_m = raw_cg
            cg_source = "user_specified"
    else:
        cg_x_m, cg_source = estimate_cg_from_ork(src_ork, cfg.stage_selection)

    design_bounds = bounds or default_bounds_for_fin(target_fin, parent_comp)
    base_candidate = candidate_from_fin(target_fin)
    domain_mode = resolve_domain_mode(sim_rocket, cfg)
    config_signature: dict[str, object] = {
        "stem": stem,
        "stage_selection": cfg.stage_selection,
        "target_stage_index": int(full_target_ids[0]),
        "target_stage_name": target_stage.name,
        "mach": round(float(cfg.mach), 6),
        "altitude_m": round(float(cfg.altitude_m), 3),
        "aoa_deg": round(float(cfg.aoa_deg), 4),
        "min_static_margin_cal": round(float(cfg.min_static_margin_cal), 4),
        "cg_x_m": round(float(cg_x_m), 5),
        "active_params": list(cfg.active_params),
        "dry_run": bool(cfg.dry_run),
    }

    def _check_feasibility(cand: FinCandidate) -> tuple[RocketModel, object]:
        mutated = apply_candidate_to_rocket(sim_rocket, target_ids, cand)
        mutated_fin = mutated.stages[target_ids[0]].components[target_ids[1]].fins[target_ids[2]]
        stab = evaluate_constraints(
            mutated, mutated_fin, parent_comp, cand, cg_x_m, cg_source, cfg
        )
        return mutated, stab

    def _resume_matches(saved: dict[str, object]) -> bool:
        saved_sig = saved.get("config_signature")
        if isinstance(saved_sig, dict):
            return saved_sig == config_signature
        return (
            saved.get("stem") == stem
            and saved.get("target_stage_name") == target_stage.name
            and round(float(saved.get("mach", -1.0)), 6) == config_signature["mach"]
            and round(float(saved.get("altitude_m", -1e9)), 3) == config_signature["altitude_m"]
            and round(float(saved.get("aoa_deg", -1e9)), 4) == config_signature["aoa_deg"]
            and round(float(saved.get("min_static_margin_cal", -1.0)), 4)
            == config_signature["min_static_margin_cal"]
            and list(saved.get("active_params") or []) == config_signature["active_params"]
        )

    history_file = out_dir / "optimization_history.json"
    evals: list[SimEvaluation] = []
    initial_locked_mesh: LockedMeshCharacteristics | None = None
    if cfg.resume and history_file.exists():
        try:
            saved = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and _resume_matches(saved):
                evals = [SimEvaluation.from_dict(item) for item in (saved.get("evaluations") or [])]
                if isinstance(saved.get("locked_mesh"), dict):
                    initial_locked_mesh = LockedMeshCharacteristics.from_dict(
                        saved["locked_mesh"]
                    )
        except Exception:
            evals = []
            initial_locked_mesh = None
    if initial_locked_mesh is None and cfg.mesh_convergence_source is not None:
        initial_locked_mesh = load_locked_mesh_from_convergence(
            cfg.mesh_convergence_source,
            fallback_preset=cfg.mesh_preset,
            fallback_yplus=cfg.target_yplus,
        )
    if cfg.resume and initial_locked_mesh is None:
        baseline_conv = out_dir / "sim_01_baseline" / "mesh_convergence.json"
        if baseline_conv.exists():
            initial_locked_mesh = load_locked_mesh_from_convergence(
                baseline_conv,
                fallback_preset=cfg.mesh_preset,
                fallback_yplus=cfg.target_yplus,
            )
    locked_mesh_box: list[LockedMeshCharacteristics | None] = [initial_locked_mesh]
    if evals and eval_callback is not None:
        eval_callback(list(evals))

    live_eval_fn = (
        evaluator
        if evaluator is not None
        else _default_fluent_evaluator(
            src_ork,
            full_target_ids,
            domain_mode,
            fluent_line_callback=fluent_line_callback,
            locked_mesh_box=locked_mesh_box,
        )
    )

    def _run_one(phase: str, cand: FinCandidate, *, high_fidelity: bool = False) -> SimEvaluation:
        sim_idx = len(evals) + 1
        mutated, stab = _check_feasibility(cand)
        mutated_fin = mutated.stages[target_ids[0]].components[target_ids[1]].fins[target_ids[2]]
        sim_dir = out_dir / f"sim_{sim_idx:02d}_{phase}"
        if progress_callback is not None:
            progress_callback(
                f"Sim {sim_idx}/{cfg.max_sims} [{phase}]: {cand.cross_section.value} "
                f"cr={cand.root_chord*1000:.0f} ct={cand.tip_chord*1000:.0f} "
                f"span={cand.span*1000:.0f} sweep={cand.sweep_angle_deg:.1f}° "
                f"(SM={stab.static_margin_cal:.2f} cal)",
                min(0.92, sim_idx / max(cfg.max_sims, 1)),
            )

        if cfg.dry_run and evaluator is None:
            cd_tot, cd_fins = estimate_analytic_drag(mutated, mutated_fin, cfg)
            ev = SimEvaluation(
                sim_index=sim_idx,
                phase=phase,
                candidate=cand,
                stability=stab,
                cd_total=cd_tot,
                cd_fins=cd_fins,
                total_drag_n=None,
                wall_yplus_avg=None,
                mesh_cells=None,
                run_dir=str(sim_dir),
                is_dry_run=True,
            )
        else:
            cd_tot, cd_fins, drag_n, yp, cells = live_eval_fn(
                sim_idx, phase, mutated, cand, sim_dir, cfg, high_fidelity
            )
            contour_rel: str | None = None
            if evaluator is None:
                case_p, data_p, _pngs = find_solved_files(sim_dir)
                if case_p is not None and data_p is not None:
                    atmo = compute_atmosphere(cfg.altitude_m)
                    v_inf = max(0.01, cfg.mach) * atmo.speed_of_sound_m_s
                    local_png = sim_dir / f"sim_{sim_idx:02d}_cfd_contours.png"
                    rendered = render_fluent_hdf5_contours(
                        case_p,
                        data_p,
                        local_png,
                        title=(
                            f"Sim {sim_idx} [{phase}] | {cand.cross_section.value} "
                            f"cr={cand.root_chord*1000:.0f} ct={cand.tip_chord*1000:.0f} "
                            f"span={cand.span*1000:.0f} sweep={cand.sweep_angle_deg:.1f}° "
                            f"| Cd={cd_tot:.5f} | SM={stab.static_margin_cal:.2f} cal"
                        ),
                        rocket=mutated,
                        target_fin=mutated_fin,
                        mach_inf=cfg.mach,
                        p_inf_pa=atmo.pressure_pa,
                        rho_inf=atmo.density_kg_m3,
                        v_inf=v_inf,
                    )
                    if rendered is not None and rendered.exists():
                        images_dir = out_dir / "cfd_images"
                        images_dir.mkdir(parents=True, exist_ok=True)
                        dest_name = (
                            f"sim_{sim_idx:02d}_{phase}_{cand.cross_section.value}.png"
                        )
                        shutil.copy2(rendered, images_dir / dest_name)
                        contour_rel = f"cfd_images/{dest_name}"
            ev = SimEvaluation(
                sim_index=sim_idx,
                phase=phase,
                candidate=cand,
                stability=stab,
                cd_total=cd_tot,
                cd_fins=cd_fins,
                total_drag_n=drag_n,
                wall_yplus_avg=yp,
                mesh_cells=cells,
                run_dir=str(sim_dir),
                is_dry_run=False,
                contour_png=contour_rel,
            )
        ok_cd, bad_reason = is_valid_aerodynamic_cd(ev.cd_total, ev.wall_yplus_avg)
        if not ok_cd:
            if phase == "baseline" or sim_idx == 1:
                raise Ork2CfdError(
                    f"Baseline Simulation 1 diverged ({bad_reason}). Aborting optimization early "
                    "— check Mach number, Stage selection, or apply RAM-Safe Recommended Settings "
                    "(y+=30)."
                )
            ev = dataclasses.replace(ev, diverged=True)
            warn_msg = (
                f"[WARN] Sim {sim_idx} diverged ({bad_reason}); "
                "excluding from GP surrogate and best-fin selection."
            )
            if fluent_line_callback is not None:
                with contextlib.suppress(Exception):
                    fluent_line_callback(warn_msg)
        evals.append(ev)
        _write_outputs(
            out_dir,
            stem,
            target_stage.name,
            cg_x_m,
            cg_source,
            cfg,
            design_bounds,
            evals,
            locked_mesh=locked_mesh_box[0],
            config_signature=config_signature,
        )
        if eval_callback is not None:
            eval_callback(list(evals))
        return ev

    # Phase 1: Baseline + cross-section screening (fixed planform)
    if not evals:
        _run_one("baseline", base_candidate)

    tested_cs = {ev.candidate.cross_section for ev in evals}
    for cs in cfg.screen_cross_sections:
        if len(evals) >= cfg.max_sims:
            break
        if cs not in tested_cs:
            screen_cand = FinCandidate(
                cross_section=cs,
                root_chord=base_candidate.root_chord,
                tip_chord=base_candidate.tip_chord,
                span=base_candidate.span,
                sweep_angle_deg=base_candidate.sweep_angle_deg,
                thickness=base_candidate.thickness,
            )
            _run_one("cross_section_screen", screen_cand)
            tested_cs.add(cs)

    screen_pool = [
        e for e in evals if e.phase in {"baseline", "cross_section_screen"} and not e.diverged
    ] or [e for e in evals if e.phase in {"baseline", "cross_section_screen"}]
    feas_screen = [e for e in screen_pool if e.stability.feasible]
    best_cs = min(feas_screen or screen_pool, key=lambda e: e.cd_total).candidate.cross_section

    cs_template = FinCandidate(
        cross_section=best_cs,
        root_chord=base_candidate.root_chord,
        tip_chord=base_candidate.tip_chord,
        span=base_candidate.span,
        sweep_angle_deg=base_candidate.sweep_angle_deg,
        thickness=base_candidate.thickness,
    )

    reserve_verify = (
        1
        if (cfg.verify_final and cfg.max_sims >= 8 and (not cfg.dry_run or evaluator is not None))
        else 0
    )
    has_verified = any(e.phase == "verification" for e in evals)
    already_doe = sum(1 for e in evals if e.phase == "doe_lhs")
    total_optim_slots = cfg.max_sims - reserve_verify
    if len(evals) < total_optim_slots:
        remaining_after_screen = total_optim_slots - len(evals) + already_doe
        target_doe = min(
            max(len(cfg.active_params), 3),
            max(1, remaining_after_screen - 2),
        )
        need_doe = max(0, min(target_doe - already_doe, total_optim_slots - len(evals)))
        if need_doe > 0:
            existing_cs_vecs = [
                e.candidate.to_vector(cfg.active_params)
                for e in evals
                if e.candidate.cross_section == best_cs
            ]
            doe_cands = sample_feasible_lhs(
                bounds=design_bounds,
                active_params=cfg.active_params,
                template_candidate=cs_template,
                feasibility_fn=lambda c: _check_feasibility(c)[1],
                n_samples=need_doe,
                existing_vectors=existing_cs_vecs,
                seed=cfg.seed,
            )
            for cand in doe_cands:
                if len(evals) >= total_optim_slots:
                    break
                _run_one("doe_lhs", cand)

    # Phase 3: Sequential Bayesian Optimization (GP + Expected Improvement)
    while len(evals) < total_optim_slots:
        train_evals = [
            e
            for e in evals
            if e.candidate.cross_section == best_cs
            and e.phase != "verification"
            and not e.diverged
        ]
        if len(train_evals) < 2:
            train_evals = [e for e in evals if not e.diverged and e.phase != "verification"]
        if len(train_evals) < 2:
            train_evals = [e for e in evals if not e.diverged] or list(evals)
        x_train = [e.candidate.to_vector(cfg.active_params) for e in train_evals]
        y_train = [e.cd_total for e in train_evals]
        y_best_local = min(y_train)

        gp = GaussianProcessSurrogate(design_bounds, cfg.active_params)
        gp.fit(x_train, y_train)
        is_last_bo = (len(evals) + 1) == total_optim_slots and reserve_verify == 0
        next_cand = propose_next_bo_candidate(
            gp=gp,
            bounds=design_bounds,
            active_params=cfg.active_params,
            template_candidate=cs_template,
            feasibility_fn=lambda c: _check_feasibility(c)[1],
            evaluated_vectors=[e.candidate.to_vector(cfg.active_params) for e in evals],
            y_best=y_best_local,
            seed=cfg.seed + len(evals),
            exploit_only=is_last_bo,
        )
        _run_one("bayesian_ei", next_cand)

    feasible_all = [
        e
        for e in evals
        if e.stability.feasible and e.phase != "verification" and not e.diverged
    ]
    valid_pre_verify = [e for e in evals if e.phase != "verification" and not e.diverged]
    best_pre_verify = min(feasible_all or valid_pre_verify or evals, key=lambda e: e.cd_total)

    verification_eval: SimEvaluation | None = next(
        (e for e in evals if e.phase == "verification"), None
    )
    if reserve_verify == 1 and not has_verified and len(evals) < cfg.max_sims:
        verification_eval = _run_one(
            "verification",
            best_pre_verify.candidate,
            high_fidelity=True,
        )

    if (
        verification_eval is not None
        and verification_eval.stability.feasible
        and not verification_eval.diverged
    ):
        best_overall = verification_eval
    else:
        final_feasible = [e for e in evals if e.stability.feasible and not e.diverged]
        final_valid = [e for e in evals if not e.diverged]
        best_overall = min(final_feasible or final_valid or evals, key=lambda e: e.cd_total)
    baseline_eval = evals[0]

    optimized_ork = out_dir / f"{stem}_optimized.ork"
    write_modified_ork(src_ork, optimized_ork, full_target_ids, best_overall.candidate)

    best_step_path: Path | None = None
    if cfg.export_best_cad:
        best_cad_dir = out_dir / "best_cad"
        cad_res = run_pipeline(
            ork_path=optimized_ork,
            output_dir=best_cad_dir,
            config=PipelineConfig(
                cad=CadConfig(tip_blunt_radius_m=cfg.blunt_radius_m),
                domain=DomainConfig(
                    mode=domain_mode,
                    shape=cfg.enclosure_shape,
                    boi=BoiConfig(enabled=True),
                ),
                flight=FlightCondition(
                    mach=cfg.mach,
                    altitude_m=cfg.altitude_m,
                    aoa_deg=cfg.aoa_deg,
                    target_yplus=(
                        locked_mesh_box[0].target_yplus
                        if locked_mesh_box[0] is not None
                        else cfg.verify_yplus
                    ),
                    mesh_preset=(
                        locked_mesh_box[0].mesh_preset
                        if locked_mesh_box[0] is not None
                        else cfg.mesh_preset
                    ),
                    cores=cfg.cores,
                    iterations=cfg.iterations,
                ),
                deliverables=DeliverableConfig(
                    export_step=True,
                    export_rocket_step=True,
                    export_stl=False,
                    export_verification=True,
                    generate_scripts=True,
                    build_wbpj=False,
                    auto_run_cfd=False,
                ),
                stage_selection=cfg.stage_selection,
            ),
        )
        best_step_path = cad_res.step_path

    history_path, csv_path, report_path, cd_plot_path = _write_outputs(
        out_dir,
        stem,
        target_stage.name,
        cg_x_m,
        cg_source,
        cfg,
        design_bounds,
        evals,
        optimized_ork_path=optimized_ork,
        best_step_path=best_step_path,
        locked_mesh=locked_mesh_box[0],
        config_signature=config_signature,
    )
    if progress_callback is not None:
        progress_callback("Fin optimization complete.", 1.0)

    cfd_images_dir = out_dir / "cfd_images"
    delta_cd = best_overall.cd_total - baseline_eval.cd_total
    delta_cd_pct = (delta_cd / max(baseline_eval.cd_total, 1e-9)) * 100.0
    return FinOptimizationResult(
        output_dir=out_dir,
        target_stage_name=target_stage.name,
        baseline_eval=baseline_eval,
        best_eval=best_overall,
        verification_eval=verification_eval,
        evaluations=tuple(evals),
        optimized_ork_path=optimized_ork,
        history_json_path=history_path,
        summary_csv_path=csv_path,
        report_md_path=report_path,
        best_step_path=best_step_path,
        delta_cd=delta_cd,
        delta_cd_pct=delta_cd_pct,
        cd_plot_path=cd_plot_path,
        cfd_images_dir=cfd_images_dir if cfd_images_dir.exists() else None,
        locked_mesh=locked_mesh_box[0],
    )
