"""Thin Click/Rich CLI for ork2cfd (defaults to launching the desktop GUI)."""

from __future__ import annotations

import math
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ansys.next_steps import build_ansys_next_steps_summary
from .ansys.script_generator import _symmetry_multiplier
from .geometry import (
    export_cad_viewer_html,
    export_cad_views_png,
    load_step_for_verification,
    tessellate_domain_zones,
    verify_cad_model,
    write_verification_reports,
)
from .models.cfd import (
    DEFAULT_MESH_CONV_FACTOR,
    DEFAULT_MESH_CONV_MAX_LEVELS,
    DEFAULT_SOLVER_ITERATIONS,
    BoiConfig,
    CadConfig,
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    EnclosureShape,
    FlightCondition,
    FlowRegime,
    MeshConvergenceConfig,
    MeshPreset,
    PipelineConfig,
    default_fluent_cores,
)
from .models.rocket import FinCrossSection
from .optimize import FinOptimizationConfig, run_fin_optimization
from .parser.ork_parser import parse_ork_file
from .physics.boundary_layer import compute_boundary_layer
from .physics.regime import recommend_solver
from .pipeline import run_pipeline, sanitize_stem

console = Console()


def _require_finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise click.BadParameter(f"must be a finite number (got {value}).", param_hint=name)
    return val


def _require_positive(
    name: str,
    value: float,
    minimum: float = 0.0,
    inclusive: bool = False,
) -> float:
    val = _require_finite(name, value)
    if (val < minimum) if inclusive else (val <= minimum):
        cmp_lbl = f">= {minimum}" if inclusive else f"> {minimum}"
        raise click.BadParameter(f"must be {cmp_lbl} (got {value}).", param_hint=name)
    return val


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Convert OpenRocket (.ork) files into CFD-ready STEP, ANSYS scripts, and Workbench."""
    if ctx.invoked_subcommand is None:
        from .gui.app import launch_gui

        launch_gui()


@main.command()
@click.argument("ork_file", required=False, type=click.Path(exists=True, path_type=Path))
def gui(ork_file: Path | None = None) -> None:
    """Launch the ork2cfd desktop GUI (optionally pre-loading an .ork file)."""
    from .gui.app import launch_gui

    launch_gui(initial_ork=ork_file)


@main.command()
@click.argument("ork_file", type=click.Path(exists=True, path_type=Path))
def inspect(ork_file: Path) -> None:
    """Inspect an OpenRocket (.ork) file and print aerodynamic OML summary."""
    rocket = parse_ork_file(ork_file)
    table = Table(title=f"Rocket OML Summary: {rocket.name}")
    table.add_column("Stage", style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Type")
    table.add_column("Start (m)", justify="right")
    table.add_column("End (m)", justify="right")
    table.add_column("Fore/Aft R (mm)", justify="right")
    table.add_column("Fins")

    for st in rocket.stages:
        for comp in st.components:
            fins_str = ", ".join(
                f"{f.fin_count}x {f.fin_type} ({f.cross_section.value})" for f in comp.fins
            )
            table.add_row(
                st.name,
                comp.name,
                comp.component_type,
                f"{comp.axial_start:.3f}",
                f"{comp.axial_end:.3f}",
                f"{comp.fore_radius*1000:.1f} -> {comp.aft_radius*1000:.1f}",
                fins_str or "-",
            )

    console.print(table)
    n_sym = rocket.rotational_symmetry_order
    stage_kind = "Multistage" if rocket.is_multistage else "Single-stage"
    console.print(
        f"[bold]Stages:[/] {rocket.stage_count} ({stage_kind}) | "
        f"[bold]Rotational Symmetry:[/] {n_sym}-fold "
        f"({360.0 / n_sym:.1f}° sector / {180.0 / n_sym:.1f}° half-sector)"
    )
    console.print(
        f"[bold]Total Length:[/] {rocket.total_length:.3f} m | "
        f"[bold]Max Dia:[/] {rocket.max_diameter*1000:.1f} mm | "
        f"[bold]Ref Area:[/] {rocket.reference_area:.6f} m^2"
    )
    for w in rocket.warnings:
        console.print(f"[yellow]WARN:[/] {w}")


@main.command()
@click.argument("ork_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--mode",
    type=click.Choice([m.value for m in DomainMode]),
    default=DomainMode.FLUID_DOMAIN.value,
)
@click.option(
    "--shape",
    type=click.Choice([s.value for s in EnclosureShape]),
    default=EnclosureShape.BULLET.value,
)
@click.option(
    "--stage",
    default="all",
    help="Stage selection: all, stage_1, stage_2, up_to_2, or stage name.",
)
@click.option("--mach", type=float, default=0.8)
@click.option("--alt", "altitude_m", type=float, default=500.0)
@click.option("--aoa", "aoa_deg", type=float, default=0.0)
@click.option("--yplus", "target_yplus", type=float, default=1.0)
@click.option(
    "--regime",
    type=click.Choice([r.value for r in FlowRegime]),
    default=FlowRegime.AUTO.value,
)
@click.option(
    "--preset",
    type=click.Choice([p.value for p in MeshPreset]),
    default=MeshPreset.STANDARD.value,
)
@click.option("--blunt-mm", type=float, default=1.0)
@click.option("--boi/--no-boi", default=True)
@click.option("--fillets/--no-fillets", default=False)
@click.option("--rail-buttons/--no-rail-buttons", default=False)
@click.option("--launch-lugs/--no-launch-lugs", default=False)
@click.option("--step/--no-step", default=True, help="Export CFD fluid domain .step (m).")
@click.option(
    "--rocket-step/--no-rocket-step",
    default=True,
    help="Also export standalone rocket-only .step with per-component Named Selections.",
)
@click.option(
    "--rocket-only",
    is_flag=True,
    default=False,
    help="Export only the standalone rocket .step (no fluid enclosure, scripts, or Workbench).",
)
@click.option("--stl/--no-stl", default=False, help="Export multi-region ASCII STL mesh (m).")
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Generate 3D HTML CAD viewer, 4-view PNG, and verification report.",
)
@click.option("--wbpj/--no-wbpj", default=True)
@click.option("--wbpz/--no-wbpz", default=False)
@click.option(
    "--run-cfd/--no-run-cfd",
    default=False,
    help="Automatically run headless ANSYS Fluent to compute Cd and save solved .cas.h5/.dat.h5.",
)
@click.option(
    "--mesh-convergence/--no-mesh-convergence",
    default=False,
    show_default=True,
    help="Iteratively refine mesh from coarse to fine until |ΔCd|/Cd stabilizes.",
)
@click.option(
    "--mesh-conv-tol",
    type=float,
    default=1.0,
    show_default=True,
    help="Relative Cd convergence threshold in percent (e.g. 1.0 = <= 1.0%).",
)
@click.option(
    "--mesh-conv-levels",
    type=int,
    default=DEFAULT_MESH_CONV_MAX_LEVELS,
    show_default=True,
    help="Maximum refinement levels during mesh convergence study.",
)
@click.option(
    "--mesh-conv-factor",
    type=float,
    default=DEFAULT_MESH_CONV_FACTOR,
    show_default=True,
    help="Linear element refinement factor r per level (cells scale ~r^3).",
)
@click.option("--cores", type=int, default=default_fluent_cores, show_default=True)
@click.option(
    "--iters", "iterations", type=int, default=DEFAULT_SOLVER_ITERATIONS, show_default=True
)
def convert(
    ork_file: Path,
    out_dir: Path | None,
    mode: str,
    shape: str,
    stage: str,
    mach: float,
    altitude_m: float,
    aoa_deg: float,
    target_yplus: float,
    regime: str,
    preset: str,
    blunt_mm: float,
    boi: bool,
    fillets: bool,
    rail_buttons: bool,
    launch_lugs: bool,
    step: bool,
    rocket_step: bool,
    rocket_only: bool,
    stl: bool,
    verify: bool,
    wbpj: bool,
    wbpz: bool,
    run_cfd: bool,
    mesh_convergence: bool,
    mesh_conv_tol: float,
    mesh_conv_levels: int,
    mesh_conv_factor: float,
    cores: int,
    iterations: int,
) -> None:
    """Convert an .ork file into STEP geometry, ANSYS scripts, and Workbench project."""
    _require_positive("--mach", mach)
    _require_finite("--alt", altitude_m)
    _require_finite("--aoa", aoa_deg)
    _require_positive("--yplus", target_yplus)
    _require_finite("--blunt-mm", blunt_mm)
    _require_positive("--cores", float(cores), minimum=1.0, inclusive=True)
    _require_positive("--iters", float(iterations), minimum=1.0, inclusive=True)
    _require_positive("--mesh-conv-tol", mesh_conv_tol)
    _require_positive("--mesh-conv-levels", float(mesh_conv_levels), minimum=1.0, inclusive=True)
    _require_positive("--mesh-conv-factor", mesh_conv_factor, minimum=1.0, inclusive=False)

    rocket = parse_ork_file(ork_file).select_stages(stage)
    suffix = "rocket" if rocket_only else "cfd"
    target_dir = out_dir or (ork_file.parent / f"{sanitize_stem(rocket.name)}_{suffix}")

    use_cfd_step = False if rocket_only else step
    use_rocket_step = True if rocket_only else rocket_step

    cfg = PipelineConfig(
        cad=CadConfig(
            tip_blunt_radius_m=max(0.0, blunt_mm / 1000.0),
            apply_fin_fillets=fillets,
            include_rail_buttons=rail_buttons,
            include_launch_lugs=launch_lugs,
        ),
        domain=DomainConfig(
            mode=DomainMode(mode),
            shape=EnclosureShape(shape),
            boi=BoiConfig(enabled=boi),
        ),
        flight=FlightCondition(
            mach=mach,
            altitude_m=altitude_m,
            aoa_deg=aoa_deg,
            target_yplus=target_yplus,
            regime_override=FlowRegime(regime),
            mesh_preset=MeshPreset(preset),
            cores=cores,
            iterations=iterations,
        ),
        deliverables=DeliverableConfig(
            export_step=use_cfd_step,
            export_rocket_step=use_rocket_step,
            export_stl=stl,
            export_verification=verify,
            generate_scripts=use_cfd_step,
            build_wbpj=(wbpj and use_cfd_step),
            archive_wbpz=(wbpz and use_cfd_step),
            auto_run_cfd=(run_cfd and use_cfd_step),
        ),
        mesh_convergence=MeshConvergenceConfig(
            enabled=mesh_convergence,
            cd_rel_tol=max(0.01, mesh_conv_tol) / 100.0,
            max_levels=mesh_conv_levels,
            refinement_factor=mesh_conv_factor,
        ),
        stage_selection=stage,
    )

    res = run_pipeline(
        ork_path=ork_file,
        output_dir=target_dir,
        config=cfg,
        progress_callback=lambda msg, frac: console.print(f"[cyan][{int(frac*100):3d}%][/] {msg}"),
    )
    pkg_label = "Rocket-only CAD export" if rocket_only else "CFD package"
    console.print(f"\n[bold green]Done![/] {pkg_label} saved to: {res.output_dir}")
    if res.step_path:
        console.print(f"  STEP:      {res.step_path.name}")
    if res.rocket_step_path:
        console.print(f"  Rocket:    {res.rocket_step_path.name} (rocket-only, no enclosure)")
    if res.stl_path:
        console.print(f"  STL:       {res.stl_path.name}")
    if res.viewer_html_path:
        console.print(f"  Viewer:    {res.viewer_html_path.name}")
    if res.views_png_path:
        console.print(f"  Views:     {res.views_png_path.name}")
    if res.verification_report:
        vr = res.verification_report
        n_pass = sum(1 for c in vr.checks if c.status == "PASS")
        console.print(f"  Verify:    {vr.overall_status} ({n_pass}/{len(vr.checks)} checks PASS)")
    if res.wbpj_path:
        console.print(f"  Workbench: {res.wbpj_path.name}")
    if res.wbpz_path:
        console.print(f"  Archive:   {res.wbpz_path.name}")
    if res.cfd_result is not None and res.cfd_result.runs:
        cfd_table = Table(title="ANSYS Fluent CFD Drag Results")
        cfd_table.add_column("Mach", justify="right", style="cyan")
        cfd_table.add_column("Total Cd", justify="right", style="bold green")
        cfd_table.add_column("Total Drag (N)", justify="right")
        cfd_table.add_column("Wall y+", justify="right")
        cfd_table.add_column("Zone Cd Breakdown")
        for run in res.cfd_result.runs:
            yp_txt = f"{run.wall_yplus_avg:.2f}" if run.wall_yplus_avg is not None else "-"
            z_txt = ", ".join(
                f"{z.zone.removeprefix('rocket_')}: {z.cd:.4f}" for z in run.zones
            )
            cfd_table.add_row(
                f"{run.mach:.2f}",
                f"{run.total_cd:.4f}",
                f"{run.total_drag_n:.2f}",
                yp_txt,
                z_txt or "-",
            )
        console.print(cfd_table)
    if res.cfd_result is not None and res.cfd_result.mesh_convergence is not None:
        mc = res.cfd_result.mesh_convergence
        mc_table = Table(title="Mesh Convergence Study (Grid Independence)")
        mc_table.add_column("Level", justify="right", style="cyan")
        mc_table.add_column("Scale", justify="right")
        mc_table.add_column("Surf Max (mm)", justify="right")
        mc_table.add_column("Cells", justify="right")
        mc_table.add_column("Wall y+", justify="right")
        mc_table.add_column("Total Cd", justify="right", style="bold")
        mc_table.add_column("ΔCd %", justify="right")
        mc_table.add_column("Status")
        for lv in mc.levels:
            cells_s = f"{lv.mesh_cells:,}" if lv.mesh_cells is not None else "-"
            yp_s = f"{lv.wall_yplus_avg:.2f}" if lv.wall_yplus_avg is not None else "-"
            chg_s = f"{lv.rel_cd_change * 100.0:.2f}%" if lv.rel_cd_change is not None else "-"
            tags: list[str] = []
            if lv.level == mc.optimal_level:
                tags.append("Optimal")
            if lv.level == mc.finest_level:
                tags.append("Converged" if mc.converged else "Finest")
            status_s = " / ".join(tags) if tags else ("Baseline" if lv.level == 1 else "-")
            mc_table.add_row(
                str(lv.level),
                f"{lv.mesh_scale:.3f}",
                f"{lv.surf_max_m * 1000.0:.2f}",
                cells_s,
                yp_s,
                f"{lv.total_cd:.5f}",
                chg_s,
                status_s,
            )
        console.print(mc_table)
        gci_s = f" | GCI_fine = {mc.gci_fine_pct:.2f}%" if mc.gci_fine_pct is not None else ""
        rich_s = (
            f" | Richardson Cd = {mc.richardson_cd:.5f}" if mc.richardson_cd is not None else ""
        )
        status_txt = "CONVERGED" if mc.converged else mc.stop_reason
        console.print(
            f"  [bold green]Grid Independence:[/] {status_txt} "
            f"(Optimal L{mc.optimal_level}, Finest L{mc.finest_level}{gci_s}{rich_s})"
        )
        if mc.plot_png is not None:
            console.print(f"  Conv. Plot: {mc.plot_png.name}")
    elif mesh_convergence and use_cfd_step and res.cfd_result is None:
        console.print(
            "  [cyan]Mesh Study:[/] MESH_CONV_ENABLED = True written to run_fluent.py "
            "(pass --run-cfd to run automatically)."
        )
    for w in res.warnings:
        console.print(f"  [yellow]WARN:[/] {w}")

    if not res.script_paths and res.step_path is None:
        if res.geometry_report and res.geometry_report.named_selections:
            zones_str = ", ".join(sorted(res.geometry_report.named_selections))
            console.print(f"  Parts:     {zones_str}")
        return

    sym_mult = _symmetry_multiplier(rocket, DomainMode(mode))
    summary = build_ansys_next_steps_summary(
        stem=res.step_path.stem.removesuffix("_cfd")
        if res.step_path
        else sanitize_stem(rocket.name),
        domain_mode=DomainMode(mode),
        sym_multiplier=sym_mult,
        mach=mach,
        aoa_deg=aoa_deg,
        target_yplus=target_yplus,
        wbpj_filename=res.wbpj_path.name if res.wbpj_path else None,
    )
    console.print()
    console.print(
        Panel(
            f"{summary}\n\nFull step-by-step guide: README_RUN.md",
            title="Next Steps — View Results in ANSYS",
            border_style="cyan",
        )
    )


@main.command("calc-bl")
@click.argument("ork_file", type=click.Path(exists=True, path_type=Path))
@click.option("--stage", default="all", help="Stage selection: all, stage_1, stage_2, or name.")
@click.option("--mach", type=float, default=0.8)
@click.option("--alt", "altitude_m", type=float, default=500.0)
@click.option("--yplus", "target_yplus", type=float, default=1.0)
def calc_bl(
    ork_file: Path,
    stage: str,
    mach: float,
    altitude_m: float,
    target_yplus: float,
) -> None:
    """Calculate ISA conditions, Reynolds number, and y+ boundary-layer sizing."""
    rocket = parse_ork_file(ork_file).select_stages(stage)
    fc = FlightCondition(mach=mach, altitude_m=altitude_m, target_yplus=target_yplus)
    bl = compute_boundary_layer(rocket, fc)
    rec = recommend_solver(rocket, fc)

    console.print(f"[bold]{rocket.name}[/] | Mach {mach} @ {altitude_m} m ({rec.regime.value})")
    console.print(f"  V_inf = {bl.velocity_m_s:.2f} m/s | Re_L = {bl.reynolds_length:.3e}")
    console.print(
        f"  Target y+ = {bl.target_yplus} -> First layer Δy1 = {bl.first_layer_thickness_m:.3e} m "
        f"({bl.prism_layers} layers, growth {bl.prism_growth_rate})"
    )
    console.print(
        f"  Solver: {rec.solver_type} | {rec.fluid_model} | {rec.turbulence_model} | "
        f"Est. cells ~{bl.estimated_cell_count:,}"
    )


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--mode",
    type=click.Choice([m.value for m in DomainMode]),
    default=DomainMode.FLUID_DOMAIN.value,
)
@click.option(
    "--shape",
    type=click.Choice([s.value for s in EnclosureShape]),
    default=EnclosureShape.BULLET.value,
)
@click.option("--stage", default="all", help="Stage selection: all, stage_1, stage_2, or name.")
@click.option("--open/--no-open", "open_browser", default=False, help="Open 3D HTML viewer.")
def verify(
    input_file: Path,
    out_dir: Path | None,
    mode: str,
    shape: str,
    stage: str,
    open_browser: bool,
) -> None:
    """Verify an .ork or .step CAD model and generate 3D HTML viewer & 4-view PNG."""
    suffix = input_file.suffix.lower()
    if suffix in {".step", ".stp"}:
        stem = sanitize_stem(input_file.stem)
        target_dir = out_dir or (input_file.parent / f"{stem}_verify")
        target_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Loading STEP geometry:[/] {input_file}")
        domain_out = load_step_for_verification(input_file)
        zone_meshes = tessellate_domain_zones(domain_out, relative=True, include_bois=True)
        vr = verify_cad_model(
            domain_out=domain_out,
            zone_meshes=zone_meshes,
            step_path=input_file,
        )
        viewer_html = export_cad_viewer_html(
            zone_meshes=zone_meshes,
            verification_report=vr,
            html_path=target_dir / f"{stem}_cad_viewer.html",
            model_name=stem,
            domain_out=domain_out,
        )
        views_png = export_cad_views_png(
            zone_meshes=zone_meshes,
            verification_report=vr,
            png_path=target_dir / f"{stem}_cad_views.png",
            model_name=stem,
            domain_out=domain_out,
        )
        json_p, md_p = write_verification_reports(
            report=vr,
            json_path=target_dir / f"{stem}_verification.json",
            md_path=target_dir / f"{stem}_verification.md",
            model_name=stem,
            step_path=input_file,
            viewer_html_path=viewer_html,
            views_png_path=views_png,
        )
    else:
        rocket = parse_ork_file(input_file).select_stages(stage)
        target_dir = out_dir or (input_file.parent / f"{sanitize_stem(rocket.name)}_verify")
        cfg = PipelineConfig(
            domain=DomainConfig(mode=DomainMode(mode), shape=EnclosureShape(shape)),
            deliverables=DeliverableConfig(
                export_step=True,
                export_rocket_step=False,
                export_stl=False,
                export_verification=True,
                generate_scripts=False,
                build_wbpj=False,
            ),
            stage_selection=stage,
        )
        res = run_pipeline(
            ork_path=input_file,
            output_dir=target_dir,
            config=cfg,
            progress_callback=lambda m, f: console.print(f"[cyan][{int(f*100):3d}%][/] {m}"),
        )
        assert res.verification_report is not None
        vr = res.verification_report
        viewer_html = res.viewer_html_path
        views_png = res.views_png_path
        json_p = res.verification_json_path
        md_p = res.verification_md_path

    table = Table(title=f"CAD Verification Scorecard ({vr.overall_status})")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Measured")
    table.add_column("Expected")
    for chk in vr.checks:
        col = "green" if chk.status == "PASS" else ("yellow" if chk.status == "WARN" else "red")
        table.add_row(chk.name, f"[{col}]{chk.status}[/]", chk.measured, chk.expected)
    console.print(table)
    console.print(f"  Output Dir: {target_dir}")
    if viewer_html:
        console.print(f"  3D Viewer:  {viewer_html.name}")
    if views_png:
        console.print(f"  4-View PNG: {views_png.name}")
    if json_p and md_p:
        console.print(f"  Reports:    {json_p.name}, {md_p.name}")

    if open_browser and viewer_html is not None and viewer_html.exists():
        webbrowser.open(viewer_html.resolve().as_uri())


@main.command("optimize-fins")
@click.argument("ork_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--max-sims",
    type=int,
    default=10,
    show_default=True,
    help="Maximum ANSYS Fluent simulation budget.",
)
@click.option(
    "--target-fin-stage",
    default="booster",
    show_default=True,
    help="Stage whose fins are optimized ('booster', 'sustainer', stage_N, or name).",
)
@click.option(
    "--stage",
    default="all",
    show_default=True,
    help="Simulated flight vehicle stack ('all', 'stage_1', 'stage_2', etc.).",
)
@click.option("--mach", type=float, default=0.80, show_default=True)
@click.option("--alt", "altitude_m", type=float, default=500.0, show_default=True)
@click.option("--aoa", "aoa_deg", type=float, default=0.0, show_default=True)
@click.option(
    "--min-sm",
    type=float,
    default=2.0,
    show_default=True,
    help="Minimum Barrowman static stability margin (calibers).",
)
@click.option(
    "--cg-m",
    type=float,
    default=None,
    help="Optional fixed vehicle CG from nose tip (m); estimated from .ork XML if omitted.",
)
@click.option(
    "--params",
    default="root_chord,tip_chord,span,sweep_angle_deg",
    show_default=True,
    help="Comma-separated OpenRocket fin parameters to optimize.",
)
@click.option(
    "--cross-sections",
    default="",
    show_default=True,
    help=(
        "Optional comma-separated cross-sections to screen "
        "(empty keeps .ork cross-section fixed)."
    ),
)
@click.option(
    "--mode",
    default="auto",
    show_default=True,
    help="Domain mode ('auto' picks minimal symmetry sector when valid, or explicit DomainMode).",
)
@click.option(
    "--shape",
    type=click.Choice([s.value for s in EnclosureShape]),
    default=EnclosureShape.CYLINDER.value,
    show_default=True,
)
@click.option(
    "--preset",
    type=click.Choice([p.value for p in MeshPreset]),
    default=MeshPreset.COARSE.value,
    show_default=True,
)
@click.option("--yplus", "target_yplus", type=float, default=30.0, show_default=True)
@click.option(
    "--mesh-conv-baseline/--no-mesh-conv-baseline",
    default=True,
    show_default=True,
    help="Run initial mesh convergence on baseline and lock optimal mesh for subsequent sims.",
)
@click.option(
    "--mesh-conv-tol",
    type=float,
    default=1.0,
    show_default=True,
    help="Mesh convergence tolerance (% |ΔCd|/Cd) for baseline convergence study.",
)
@click.option(
    "--mesh-conv-from",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Existing mesh_convergence.json (or folder) whose optimal mesh is locked for all sims.",
)
@click.option(
    "--verify-final/--no-verify-final",
    default=True,
    show_default=True,
    help="Reserve last simulation for higher-fidelity verification of best candidate.",
)
@click.option("--blunt-mm", type=float, default=0.4, show_default=True)
@click.option("--cores", type=int, default=default_fluent_cores, show_default=True)
@click.option(
    "--iters", "iterations", type=int, default=DEFAULT_SOLVER_ITERATIONS, show_default=True
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Use fast analytical drag proxy instead of ANSYS Fluent (exports valid .ork & STEP).",
)
@click.option(
    "--no-cad",
    is_flag=True,
    default=False,
    help="Skip final OCCT CAD/STEP export of the winning geometry.",
)
@click.option("--resume/--no-resume", default=True, show_default=True)
def optimize_fins(
    ork_file: Path,
    out_dir: Path | None,
    max_sims: int,
    target_fin_stage: str,
    stage: str,
    mach: float,
    altitude_m: float,
    aoa_deg: float,
    min_sm: float,
    cg_m: float | None,
    params: str,
    cross_sections: str,
    mode: str,
    shape: str,
    preset: str,
    target_yplus: float,
    mesh_conv_baseline: bool,
    mesh_conv_tol: float,
    mesh_conv_from: Path | None,
    verify_final: bool,
    blunt_mm: float,
    cores: int,
    iterations: int,
    dry_run: bool,
    no_cad: bool,
    resume: bool,
) -> None:
    """Optimize OpenRocket fin shape to minimize Cd within <= max-sims ANSYS runs."""
    _require_positive("--max-sims", float(max_sims), minimum=3.0, inclusive=True)
    _require_positive("--mach", mach)
    _require_finite("--alt", altitude_m)
    _require_finite("--aoa", aoa_deg)
    _require_positive("--min-sm", min_sm)
    _require_positive("--yplus", target_yplus)
    _require_positive("--mesh-conv-tol", mesh_conv_tol)
    _require_finite("--blunt-mm", blunt_mm)
    _require_positive("--cores", float(cores), minimum=1.0, inclusive=True)
    _require_positive("--iters", float(iterations), minimum=1.0, inclusive=True)
    if cg_m is not None:
        _require_finite("--cg-m", cg_m)

    rocket = parse_ork_file(ork_file).select_stages(stage)
    target_dir = out_dir or (ork_file.parent / f"{sanitize_stem(rocket.name)}_fin_opt")
    active_params = tuple(p.strip() for p in params.split(",") if p.strip())
    cs_list = tuple(
        FinCrossSection.from_str(c.strip()) for c in cross_sections.split(",") if c.strip()
    )

    cfg = FinOptimizationConfig(
        max_sims=max_sims,
        stage_selection=stage,
        target_fin_stage=target_fin_stage,
        active_params=active_params,
        screen_cross_sections=cs_list,
        mach=mach,
        altitude_m=altitude_m,
        aoa_deg=aoa_deg,
        min_static_margin_cal=min_sm,
        cg_x_m=cg_m,
        domain_mode=mode,
        enclosure_shape=EnclosureShape(shape),
        mesh_preset=MeshPreset(preset),
        target_yplus=target_yplus,
        mesh_convergence_baseline=mesh_conv_baseline,
        mesh_conv_tol=max(0.05, mesh_conv_tol) / 100.0,
        mesh_convergence_source=mesh_conv_from,
        verify_final=verify_final,
        blunt_radius_m=max(0.0, blunt_mm / 1000.0),
        cores=cores,
        iterations=iterations,
        dry_run=dry_run,
        export_best_cad=not no_cad,
        resume=resume,
    )
    res = run_fin_optimization(
        ork_path=ork_file,
        output_dir=target_dir,
        config=cfg,
        progress_callback=lambda msg, frac: console.print(f"[cyan][{int(frac*100):3d}%][/] {msg}"),
    )

    table = Table(
        title=(
            f"Fin Optimization Results: {rocket.name} "
            f"(Target Stage: {res.target_stage_name}, Thickness Locked)"
        )
    )
    table.add_column("Sim", justify="right", style="cyan")
    table.add_column("Phase")
    table.add_column("Section")
    table.add_column("Root (mm)", justify="right")
    table.add_column("Tip (mm)", justify="right")
    table.add_column("Span (mm)", justify="right")
    table.add_column("Sweep (°)", justify="right")
    table.add_column("Thick (mm)", justify="right")
    table.add_column("SM (cal)", justify="right")
    table.add_column("Cd Total", justify="right", style="bold")
    table.add_column("ΔCd %", justify="right")

    base_cd = res.baseline_eval.cd_total
    for ev in res.evaluations:
        c = ev.candidate
        delta_pct = ((ev.cd_total - base_cd) / max(base_cd, 1e-9)) * 100.0
        is_best = ev.sim_index == res.best_eval.sim_index
        cd_str = f"[green]{ev.cd_total:.5f}[/]" if is_best else f"{ev.cd_total:.5f}"
        table.add_row(
            str(ev.sim_index),
            ev.phase,
            c.cross_section.value,
            f"{c.root_chord*1000:.1f}",
            f"{c.tip_chord*1000:.1f}",
            f"{c.span*1000:.1f}",
            f"{c.sweep_angle_deg:.1f}",
            f"{c.thickness*1000:.2f}",
            f"{ev.stability.static_margin_cal:.2f}",
            cd_str,
            f"{delta_pct:+.2f}%",
        )
    console.print(table)
    console.print(
        f"[bold green]Best Cd:[/] {res.best_eval.cd_total:.6f} "
        f"(Whole-Stage/Rocket, {res.delta_cd:+.6f}, {res.delta_cd_pct:+.2f}% vs baseline)"
    )
    if res.best_eval.cd_fins is not None:
        air_cd = res.best_eval.cd_total - res.best_eval.cd_fins
        console.print(
            f"  Component Split: Airframe Cd = {air_cd:.5f} | Fins Cd = {res.best_eval.cd_fins:.5f}"
        )
    if res.locked_mesh is not None:
        lm = res.locked_mesh
        console.print(
            f"  Locked Mesh:    preset={lm.mesh_preset.value}, scale={lm.mesh_scale:.4f}, "
            f"y+={lm.target_yplus:.2f}, L{lm.optimal_level}/{lm.finest_level}"
        )
    console.print(f"  Optimized .ork: {res.optimized_ork_path}")
    console.print(f"  Summary CSV:    {res.summary_csv_path}")
    if res.cd_plot_path is not None:
        console.print(f"  Cd Plot PNG:    {res.cd_plot_path}")
    if res.cfd_images_dir is not None and any(res.cfd_images_dir.glob("*.png")):
        console.print(f"  CFD Images:     {res.cfd_images_dir}")
    console.print(f"  Report MD:      {res.report_md_path}")
    if res.best_step_path is not None:
        console.print(f"  Best CFD STEP:  {res.best_step_path}")
