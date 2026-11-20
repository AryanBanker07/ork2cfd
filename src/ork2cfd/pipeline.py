"""End-to-end ork2cfd orchestration pipeline shared by GUI and CLI."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .ansys.detector import select_best_installation
from .ansys.script_generator import generate_ansys_scripts
from .ansys.workbench_builder import build_workbench_project
from .geometry.domain_builder import build_cfd_domain
from .geometry.step_exporter import export_step_with_names
from .geometry.stl_exporter import export_stl_with_regions
from .geometry.tessellation import tessellate_domain_zones
from .geometry.verifier import verify_cad_model, write_verification_reports
from .geometry.viewer_exporter import export_cad_viewer_html, export_cad_views_png
from .models.cfd import PipelineConfig
from .models.results import GeometryReport, PackageResult, VerificationReport
from .parser.ork_parser import parse_ork_file
from .physics.boundary_layer import compute_boundary_layer
from .physics.regime import recommend_solver

ProgressCallback = Callable[[str, float], None]


def sanitize_stem(name: str) -> str:
    """Create a filesystem-safe stem for STEP, script, and Workbench filenames."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_.")
    return stem or "rocket"


def run_pipeline(
    ork_path: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PackageResult:
    """Execute the full .ork -> CFD STEP -> ANSYS scripts -> .wbpj/.wbpz pipeline."""
    cfg = config or PipelineConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _emit(msg: str, frac: float) -> None:
        if progress_callback is not None:
            progress_callback(msg, frac)

    _emit(f"Parsing OpenRocket file: {ork_path.name}...", 0.08)
    rocket = parse_ork_file(ork_path)
    stem = sanitize_stem(rocket.name)
    warnings: list[str] = list(rocket.warnings)

    _emit("Computing ISA atmosphere, y+ boundary layer, and solver settings...", 0.18)
    bl = compute_boundary_layer(rocket, cfg.flight, cfg.domain.mode)
    solver_rec = recommend_solver(rocket, cfg.flight)
    warnings.extend(bl.warnings)

    step_path: Path | None = None
    stl_path: Path | None = None
    manifest_path: Path | None = None
    geom_report: GeometryReport | None = None
    verification_report: VerificationReport | None = None
    viewer_html_path: Path | None = None
    views_png_path: Path | None = None
    verification_json_path: Path | None = None
    verification_md_path: Path | None = None

    need_step = (
        cfg.deliverables.export_step
        or cfg.deliverables.generate_scripts
        or cfg.deliverables.build_wbpj
    )
    need_domain = (
        need_step
        or cfg.deliverables.export_stl
        or cfg.deliverables.export_verification
    )
    if need_domain:
        _emit("Building watertight 3D OML, fluid domain, and BOIs...", 0.35)
        domain_out = build_cfd_domain(rocket, cfg.cad, cfg.domain, solver_rec)
        geom_report = domain_out.report
        warnings.extend(geom_report.warnings)

        if need_step:
            _emit("Exporting XCAF STEP with Named Selections...", 0.55)
            step_target = out_dir / f"{stem}_cfd.step"
            manifest_target = out_dir / f"{stem}_manifest.json"
            step_path, manifest_path = export_step_with_names(
                domain_out=domain_out,
                step_path=step_target,
                manifest_path=manifest_target,
                rocket=rocket,
                cad_config=cfg.cad,
                domain_config=cfg.domain,
            )

        if cfg.deliverables.export_stl:
            _emit("Exporting multi-region CFD STL mesh...", 0.62)
            stl_target = out_dir / f"{stem}_cfd.stl"
            stl_path, _tri_count = export_stl_with_regions(
                domain_out=domain_out,
                stl_path=stl_target,
                cad_config=cfg.cad,
            )

        if cfg.deliverables.export_verification:
            _emit("Verifying CAD model & generating 3D viewer and diagnostic views...", 0.68)
            zone_meshes = tessellate_domain_zones(domain_out, relative=True, include_bois=True)
            verification_report = verify_cad_model(
                domain_out=domain_out,
                zone_meshes=zone_meshes,
                rocket=rocket,
                cad_config=cfg.cad,
                domain_config=cfg.domain,
                step_path=step_path,
                stl_path=stl_path,
            )
            viewer_html_path = export_cad_viewer_html(
                zone_meshes=zone_meshes,
                verification_report=verification_report,
                html_path=out_dir / f"{stem}_cad_viewer.html",
                model_name=rocket.name,
                domain_out=domain_out,
                rocket=rocket,
            )
            views_png_path = export_cad_views_png(
                zone_meshes=zone_meshes,
                verification_report=verification_report,
                png_path=out_dir / f"{stem}_cad_views.png",
                model_name=rocket.name,
                domain_out=domain_out,
                rocket=rocket,
            )
            verification_json_path, verification_md_path = write_verification_reports(
                report=verification_report,
                json_path=out_dir / f"{stem}_verification.json",
                md_path=out_dir / f"{stem}_verification.md",
                model_name=rocket.name,
                step_path=step_path,
                viewer_html_path=viewer_html_path,
                views_png_path=views_png_path,
            )
            for chk in verification_report.checks:
                if chk.status == "FAIL":
                    warnings.append(
                        f"CAD Verification FAIL [{chk.name}]: {chk.details or chk.measured}"
                    )

    ansys_install = select_best_installation(cfg.deliverables.ansys_version)
    wbpj_target = out_dir / f"{stem}_cfd.wbpj"
    wbpz_target = (out_dir / f"{stem}_cfd.wbpz") if cfg.deliverables.archive_wbpz else None

    script_paths: dict[str, Path] = {}
    if cfg.deliverables.generate_scripts or cfg.deliverables.build_wbpj:
        assert step_path is not None and manifest_path is not None
        _emit(
            "Generating ANSYS PyFluent, Fluent journal, Discovery, and Workbench scripts...",
            0.72,
        )
        script_paths = generate_ansys_scripts(
            output_dir=out_dir,
            stem=stem,
            rocket=rocket,
            config=cfg,
            bl=bl,
            solver_rec=solver_rec,
            step_path=step_path,
            manifest_path=manifest_path,
            ansys_install=ansys_install,
            wbpj_path=wbpj_target,
            wbpz_path=wbpz_target,
        )

    wbpj_out: Path | None = None
    wbpz_out: Path | None = None
    if cfg.deliverables.build_wbpj:
        if ansys_install is None or ansys_install.runwb2_exe is None:
            warnings.append(
                "Local ANSYS Workbench (RunWB2.exe) was not detected; skipped .wbpj build "
                "(create_project.wbjn was still generated)."
            )
        else:
            _emit(
                f"Building ANSYS Workbench project via {ansys_install.label} (may take 15-40s)...",
                0.82,
            )
            wbjn_path = script_paths["create_project.wbjn"]
            wbpj_out, wbpz_out = build_workbench_project(
                wbjn_path=wbjn_path,
                expected_wbpj_path=wbpj_target,
                ansys_install=ansys_install,
                expected_wbpz_path=wbpz_target,
            )

    _emit("CFD package generation complete.", 1.0)
    return PackageResult(
        rocket=rocket,
        bl_report=bl,
        solver_rec=solver_rec,
        geometry_report=geom_report,
        output_dir=out_dir,
        step_path=step_path,
        stl_path=stl_path,
        manifest_path=manifest_path,
        script_paths=script_paths,
        wbpj_path=wbpj_out,
        wbpz_path=wbpz_out,
        verification_report=verification_report,
        viewer_html_path=viewer_html_path,
        views_png_path=views_png_path,
        verification_json_path=verification_json_path,
        verification_md_path=verification_md_path,
        warnings=tuple(dict.fromkeys(warnings)),
    )
