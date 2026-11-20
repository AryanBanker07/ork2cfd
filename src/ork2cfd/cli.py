"""Thin Click/Rich CLI for ork2cfd (defaults to launching the desktop GUI)."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .geometry import (
    export_cad_viewer_html,
    export_cad_views_png,
    load_step_for_verification,
    tessellate_domain_zones,
    verify_cad_model,
    write_verification_reports,
)
from .models.cfd import (
    BoiConfig,
    CadConfig,
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    EnclosureShape,
    FlightCondition,
    FlowRegime,
    MeshPreset,
    PipelineConfig,
)
from .parser.ork_parser import parse_ork_file
from .physics.boundary_layer import compute_boundary_layer
from .physics.regime import recommend_solver
from .pipeline import run_pipeline, sanitize_stem

console = Console()


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
@click.option("--stl/--no-stl", default=False, help="Export multi-region ASCII STL mesh (m).")
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Generate 3D HTML CAD viewer, 4-view PNG, and verification report.",
)
@click.option("--wbpj/--no-wbpj", default=True)
@click.option("--wbpz/--no-wbpz", default=False)
@click.option("--cores", type=int, default=4)
@click.option("--iters", "iterations", type=int, default=400)
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
    stl: bool,
    verify: bool,
    wbpj: bool,
    wbpz: bool,
    cores: int,
    iterations: int,
) -> None:
    """Convert an .ork file into STEP geometry, ANSYS scripts, and Workbench project."""
    rocket = parse_ork_file(ork_file).select_stages(stage)
    target_dir = out_dir or (ork_file.parent / f"{sanitize_stem(rocket.name)}_cfd")

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
            export_step=True,
            export_stl=stl,
            export_verification=verify,
            generate_scripts=True,
            build_wbpj=wbpj,
            archive_wbpz=wbpz,
        ),
        stage_selection=stage,
    )

    res = run_pipeline(
        ork_path=ork_file,
        output_dir=target_dir,
        config=cfg,
        progress_callback=lambda msg, frac: console.print(f"[cyan][{int(frac*100):3d}%][/] {msg}"),
    )
    console.print(f"\n[bold green]Done![/] CFD package saved to: {res.output_dir}")
    if res.step_path:
        console.print(f"  STEP:      {res.step_path.name}")
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
    for w in res.warnings:
        console.print(f"  [yellow]WARN:[/] {w}")


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
