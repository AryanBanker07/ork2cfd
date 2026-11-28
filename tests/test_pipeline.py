"""End-to-end pipeline, CLI, and GUI widget/lifecycle smoke tests."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

from click.testing import CliRunner

from ork2cfd.cli import main as cli_main
from ork2cfd.gui.app import Ork2CfdApp
from ork2cfd.gui.help_texts import HELP_TEXTS
from ork2cfd.gui.preview_canvas import PreviewCanvas
from ork2cfd.models import (
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    FlightCondition,
    PipelineConfig,
)
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import recommend_solver
from ork2cfd.pipeline import run_pipeline


def test_full_pipeline_without_wbpj(synthetic_ork_file: Path, tmp_path: Path) -> None:
    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.SYMMETRY_HALF),
        flight=FlightCondition(mach=0.75, altitude_m=300.0),
        deliverables=DeliverableConfig(
            export_step=True,
            generate_scripts=True,
            build_wbpj=False,
        ),
    )

    events: list[tuple[str, float]] = []
    res = run_pipeline(
        ork_path=synthetic_ork_file,
        output_dir=tmp_path / "pkg",
        config=cfg,
        progress_callback=lambda m, f: events.append((m, f)),
    )

    assert res.step_path is not None and res.step_path.exists()
    assert res.manifest_path is not None and res.manifest_path.exists()
    assert res.rocket_step_path is not None and res.rocket_step_path.exists()
    assert res.rocket_manifest_path is not None and res.rocket_manifest_path.exists()
    assert res.viewer_html_path is not None and res.viewer_html_path.exists()
    assert res.views_png_path is not None and res.views_png_path.exists()
    assert res.verification_json_path is not None and res.verification_json_path.exists()
    assert res.verification_md_path is not None and res.verification_md_path.exists()
    assert res.verification_report is not None
    assert res.verification_report.overall_status in {"PASS", "WARN"}
    assert len(res.script_paths) == 6
    assert res.geometry_report is not None and res.geometry_report.is_watertight
    assert events and events[-1][1] == 1.0


def test_cli_inspect_calc_bl_and_convert(synthetic_ork_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()

    res_inspect = runner.invoke(cli_main, ["inspect", str(synthetic_ork_file)])
    assert res_inspect.exit_code == 0
    assert "TestRocket" in res_inspect.output

    res_bl = runner.invoke(
        cli_main, ["calc-bl", str(synthetic_ork_file), "--mach", "0.85", "--alt", "500"]
    )
    assert res_bl.exit_code == 0
    assert "Re_L" in res_bl.output

    out_dir = tmp_path / "cli_out"
    res_conv = runner.invoke(
        cli_main,
        [
            "convert",
            str(synthetic_ork_file),
            "-o",
            str(out_dir),
            "--mode",
            "fluid_domain",
            "--shape",
            "bullet",
            "--no-wbpj",
        ],
    )
    assert res_conv.exit_code == 0
    assert (out_dir / "TestRocket_cfd.step").exists()
    assert (out_dir / "TestRocket_rocket.step").exists()
    assert (out_dir / "TestRocket_rocket_manifest.json").exists()
    assert (out_dir / "run_fluent.py").exists()
    assert (out_dir / "create_project.wbjn").exists()
    from ork2cfd.models import DEFAULT_SOLVER_ITERATIONS, default_fluent_cores

    run_txt = (out_dir / "run_fluent.py").read_text(encoding="utf-8")
    assert f"CORES = {default_fluent_cores()}" in run_txt
    assert f"ITERATIONS = {DEFAULT_SOLVER_ITERATIONS}" in run_txt
    norm_out = " ".join(res_conv.output.split())
    assert "Next Steps" in norm_out
    assert "Wall Yplus" in norm_out


def test_preview_figure_render_offscreen(synthetic_ork_file: Path) -> None:
    """Verify Matplotlib rocket & domain figure plotting without requiring GUI mainloop."""
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        return

    try:
        canvas = PreviewCanvas(root, dark=True)
        rocket = parse_ork_file(synthetic_ork_file)
        rec = recommend_solver(rocket, FlightCondition(mach=0.8))
        canvas.update_preview(rocket, DomainConfig(mode=DomainMode.FLUID_DOMAIN), rec)
        assert len(canvas.ax_oml.patches) >= 4
        assert len(canvas.ax_dom.patches) >= 3
    finally:
        root.destroy()


def test_help_texts_complete() -> None:
    """Verify every GUI section and control option has a thorough help entry."""
    expected_keys = {
        "sec_input", "stage_selection", "sec_physics", "sec_cad", "sec_deliverables",
        "sec_next_steps",
        "mach", "altitude", "aoa", "yplus", "regime", "mesh_preset", "mesh_convergence",
        "mesh_conv_tol", "cores", "ram_limit", "iterations",
        "domain_mode", "enclosure_shape", "blunt_radius", "step_schema",
        "include_bois", "fin_fillets", "rail_buttons", "launch_lugs",
        "deliv_step",
        "export_preset",
        "deliv_rocket_step",
        "deliv_stl",
        "deliv_scripts",
        "deliv_wbpj",
        "deliv_wbpz",
        "deliv_verify",
        "deliv_auto_cfd",
        "sec_cfd_results",
        "sec_fin_opt",
        "opt_target_stage",
        "opt_max_sims",
        "opt_min_sm",
        "opt_verify_final",
        "opt_dry_run",
        "opt_params",
    }
    assert expected_keys.issubset(HELP_TEXTS.keys())
    for key in expected_keys:
        entry = HELP_TEXTS[key]
        assert entry.title.strip()
        assert len(entry.body.strip()) >= 60


def test_gui_app_lifecycle_smoke(synthetic_ork_file: Path, tmp_path: Path) -> None:
    """Verify Ork2CfdApp initializes, loads .ork, computes physics, and cleans up."""
    try:
        app = Ork2CfdApp(initial_ork=synthetic_ork_file)
        app.withdraw()
    except tk.TclError as exc:
        if "display" in str(exc).lower():
            return
        raise

    try:
        assert app._rocket is not None
        assert app._rocket.name == "TestRocket"
        assert len(app._info_icons) == 42
        assert app.var_auto_cfd.get() is False
        assert app.var_mesh_conv.get() is False
        assert app.btn_view_cad is not None
        assert app.btn_ansys_guide is not None
        assert app.btn_solve_cfd is not None
        assert app.btn_viz_fluent is not None
        assert app.btn_viz_cfdpost is not None
        assert app.btn_viz_contours is not None
        assert "Wall Yplus" in app.lbl_next_steps.cget("text")

        # 1-click Rocket .step Only toggle vs Full CFD Package
        app._on_export_preset_change("Rocket .step Only")
        assert app.var_rocket_step.get() is True
        assert app.var_step.get() is False
        assert app.var_scripts.get() is False
        assert app.var_wbpj.get() is False
        assert app.btn_generate.cget("text") == "Export Rocket .step"

        app._on_export_preset_change("Full CFD Package")
        assert app.var_step.get() is True
        assert app.var_rocket_step.get() is True
        assert app.btn_generate.cget("text") == "Generate CFD Package"
        app.var_out_dir.set(str(tmp_path / "gui_out"))
        cfg = app._collect_config()
        from ork2cfd.models import DEFAULT_SOLVER_ITERATIONS
        from ork2cfd.physics import recommend_resources

        assert cfg.flight.mach == 0.8
        assert cfg.flight.cores == recommend_resources(hardware=app._hw).recommended_cores
        assert cfg.flight.ram_limit_gb is not None and cfg.flight.ram_limit_gb >= 1.5
        assert cfg.flight.iterations == DEFAULT_SOLVER_ITERATIONS == 300
        assert "HW:" in app.lbl_hw_summary.cget("text")
        app._on_apply_recommended_settings()
        assert app.var_yplus.get() in {"1", "30"}
        assert cfg.mesh_convergence.enabled is False

        # Auto Mesh Convergence toggle updates config, summary text, and generate button label
        app.var_mesh_conv.set(True)
        app.var_mesh_conv_tol.set("0.5")
        app._on_mesh_conv_toggle()
        conv_cfg = app._collect_config()
        assert conv_cfg.mesh_convergence.enabled is True
        assert conv_cfg.mesh_convergence.cd_rel_tol == 0.005
        assert "Mesh Study: ON" in app.lbl_physics_summary.cget("text")
        app.var_auto_cfd.set(True)
        app._sync_generate_button_label()
        assert app.btn_generate.cget("text") == "Generate & Converge Mesh"
        app.var_auto_cfd.set(False)
        app.var_mesh_conv.set(False)
        app._on_mesh_conv_toggle()
        app._refresh_calculations()
        app.var_dark.set(False)
        app._toggle_theme()
        assert "Re_L=" in app.lbl_physics_summary.cget("text")

        # Smoke-test hover tooltip, click-to-pin popup, and ANSYS Results Guide window
        icon = app._info_icons[0]
        icon.show_tooltip()
        assert icon._tooltip_win is not None
        icon.hide_tooltip()
        assert icon._tooltip_win is None
        dlg = icon.open_dialog()
        if dlg is not None:
            dlg.destroy()
        guide_dlg = app._on_open_ansys_guide()
        if guide_dlg is not None:
            guide_dlg.destroy()

        # Verify GUI CFD Results Card populates Cd and enables visualization buttons
        from ork2cfd.ansys.detector import AnsysInstallation
        from ork2cfd.models import (
            CfdMachPoint,
            CfdSolveResult,
            CfdZoneResult,
            MeshConvergenceReport,
            MeshRefinementLevel,
        )

        gui_out = tmp_path / "gui_out"
        gui_out.mkdir(parents=True, exist_ok=True)
        (gui_out / "run_fluent.py").write_text("# runner", encoding="utf-8")
        c_file = gui_out / "TestRocket.cas.h5"
        d_file = gui_out / "TestRocket.dat.h5"
        p_file = gui_out / "TestRocket_M0.80_mach.png"
        c_file.write_text("cas", encoding="utf-8")
        d_file.write_text("dat", encoding="utf-8")
        p_file.write_text("png", encoding="utf-8")
        fake_exe = tmp_path / "stub.exe"
        fake_exe.write_text("stub", encoding="utf-8")
        app._ansys_install = AnsysInstallation(
            version_code="25.2",
            release_tag="v252",
            wb_script_version="25.2.170",
            root_dir=tmp_path,
            runwb2_exe=fake_exe,
            fluent_exe=fake_exe,
            discovery_exe=fake_exe,
            cfdpost_exe=fake_exe,
        )
        sample_cfd = CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=gui_out,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01131,
            mesh_cells=280000,
            min_orth_quality=0.25,
            mesh_convergence=MeshConvergenceReport(
                converged=True,
                cd_rel_tol=0.01,
                refinement_factor=1.35,
                levels=(
                    MeshRefinementLevel(1, 1.647, 0.016, 120000, 0.22, 0.4480, 202.0, 1.15),
                    MeshRefinementLevel(2, 1.220, 0.012, 280000, 0.25, 0.4321, 195.4, 1.08, 0.0368),
                ),
                optimal_level=2,
                finest_level=2,
                gci_fine_pct=1.42,
                richardson_cd=0.4295,
                observed_order_p=2.0,
            ),
            runs=(
                CfdMachPoint(
                    mach=0.80,
                    velocity_m_s=270.6,
                    total_cd=0.4321,
                    total_drag_n=195.4,
                    wall_yplus_avg=1.08,
                    case_file=c_file,
                    data_file=d_file,
                    mach_contour_png=p_file,
                    zones=(
                        CfdZoneResult("rocket_nosecone", 75.0, 0.1650),
                        CfdZoneResult("rocket_fins", 54.4, 0.1200),
                    ),
                ),
            ),
        )
        app._display_cfd_result(sample_cfd)
        assert "Whole-Rocket Cd = 0.4321" in app.lbl_cd_headline.cget("text")
        assert "fins: 0.1200" in app.lbl_cd_zones.cget("text")
        assert "Mesh Study (Converged)" in app.lbl_cd_zones.cget("text")
        assert app.btn_solve_cfd.cget("state") == "normal"
        assert app.btn_viz_fluent.cget("state") == "normal"
        assert app.btn_viz_cfdpost.cget("state") == "normal"
        assert app.btn_viz_contours.cget("state") == "normal"
    finally:
        app.destroy()


def test_pipeline_stl_only(synthetic_ork_file: Path, tmp_path: Path) -> None:
    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.OML_ONLY),
        flight=FlightCondition(mach=0.8),
        deliverables=DeliverableConfig(
            export_step=False,
            export_rocket_step=False,
            export_stl=True,
            export_verification=False,
            generate_scripts=False,
            build_wbpj=False,
        ),
    )

    res = run_pipeline(
        ork_path=synthetic_ork_file,
        output_dir=tmp_path / "stl_only",
        config=cfg,
    )

    assert res.step_path is None
    assert res.rocket_step_path is None
    assert res.manifest_path is None
    assert res.stl_path is not None and res.stl_path.exists()
    assert res.geometry_report is not None and res.geometry_report.is_valid
    assert "solid rocket_nosecone\n" in res.stl_path.read_text(encoding="utf-8")


def test_pipeline_rocket_step_only_with_verification(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig(
        deliverables=DeliverableConfig(
            export_step=False,
            export_rocket_step=True,
            export_stl=False,
            export_verification=True,
            generate_scripts=False,
            build_wbpj=False,
        ),
    )

    res = run_pipeline(
        ork_path=synthetic_ork_file,
        output_dir=tmp_path / "rocket_only",
        config=cfg,
    )

    assert res.step_path is None
    assert res.manifest_path is None
    assert res.rocket_step_path is not None and res.rocket_step_path.exists()
    assert res.rocket_manifest_path is not None and res.rocket_manifest_path.exists()
    assert res.viewer_html_path is not None and res.viewer_html_path.exists()
    assert res.views_png_path is not None and res.views_png_path.exists()
    assert res.verification_report is not None
    assert res.verification_report.overall_status == "PASS"
    assert res.verification_report.step_roundtrip_ok is True


def test_cli_convert_with_stl(synthetic_ork_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "cli_stl_out"
    res = runner.invoke(
        cli_main,
        [
            "convert",
            str(synthetic_ork_file),
            "-o",
            str(out_dir),
            "--mode",
            "oml_only",
            "--stl",
            "--no-wbpj",
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / "TestRocket_cfd.step").exists()
    assert (out_dir / "TestRocket_cfd.stl").exists()
    assert "STL:" in res.output


def test_cli_verify_multistage_stage_selection(tmp_path: Path) -> None:
    sample_ork = Path("sample orks/banana3.ork")
    if not sample_ork.exists():
        return
    runner = CliRunner()
    out_dir = tmp_path / "banana3_s1"
    res = runner.invoke(
        cli_main,
        [
            "verify",
            str(sample_ork),
            "-o",
            str(out_dir),
            "--stage",
            "stage_1",
            "--mode",
            "symmetry_sector",
            "--shape",
            "bullet",
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / "Banana3_stage_1_cfd.step").exists()
    assert (out_dir / "Banana3_stage_1_cad_viewer.html").exists()
    assert (out_dir / "Banana3_stage_1_verification.json").exists()


def test_cli_convert_rocket_only_flag(synthetic_ork_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "rocket_only_cli"
    res = runner.invoke(
        cli_main,
        [
            "convert",
            str(synthetic_ork_file),
            "-o",
            str(out_dir),
            "--rocket-only",
            "--stl",
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / "TestRocket_rocket.step").exists()
    assert (out_dir / "TestRocket_rocket_manifest.json").exists()
    assert (out_dir / "TestRocket_rocket.stl").exists()
    assert (out_dir / "TestRocket_cad_viewer.html").exists()
    assert not (out_dir / "TestRocket_cfd.step").exists()
    assert not (out_dir / "run_fluent.py").exists()
    assert not (out_dir / "TestRocket_cfd.wbpj").exists()
    stl_text = (out_dir / "TestRocket_rocket.stl").read_text(encoding="utf-8")
    assert "solid farfield\n" not in stl_text
    assert "rocket-only, no enclosure" in res.output


def test_pipeline_auto_run_cfd_and_cli_flag(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.models import CfdMachPoint, CfdSolveResult, CfdZoneResult

    fake_exe = tmp_path / "fluent.exe"
    fake_exe.write_text("stub", encoding="utf-8")
    fake_install = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=tmp_path,
        runwb2_exe=None,
        fluent_exe=fake_exe,
        discovery_exe=None,
        cfdpost_exe=fake_exe,
    )

    def _fake_solve(output_dir: Path, ansys_install, cores: int = 4, **_kwargs) -> CfdSolveResult:
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01131,
            mesh_cells=250000,
            runs=(
                CfdMachPoint(
                    mach=0.80,
                    velocity_m_s=270.6,
                    total_cd=0.4567,
                    total_drag_n=210.5,
                    wall_yplus_avg=1.05,
                    zones=(CfdZoneResult("rocket_fins", 60.0, 0.1302),),
                ),
            ),
        )

    monkeypatch.setattr("ork2cfd.pipeline.select_best_installation", lambda _v=None: fake_install)
    monkeypatch.setattr("ork2cfd.pipeline.run_fluent_solve_blocking", _fake_solve)

    runner = CliRunner()
    out_dir = tmp_path / "auto_cfd_cli"
    cli_res = runner.invoke(
        cli_main,
        [
            "convert",
            str(synthetic_ork_file),
            "-o",
            str(out_dir),
            "--mode",
            "symmetry_sector",
            "--no-wbpj",
            "--run-cfd",
        ],
    )
    assert cli_res.exit_code == 0
    assert "ANSYS Fluent CFD Drag Results" in cli_res.output
    assert "0.4567" in cli_res.output


def test_cli_convert_mesh_convergence_flag(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.models import (
        CfdMachPoint,
        CfdSolveResult,
        CfdZoneResult,
        MeshConvergenceReport,
        MeshRefinementLevel,
    )

    fake_exe = tmp_path / "fluent.exe"
    fake_exe.write_text("stub", encoding="utf-8")
    fake_install = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=tmp_path,
        runwb2_exe=None,
        fluent_exe=fake_exe,
        discovery_exe=None,
        cfdpost_exe=fake_exe,
    )
    captured: dict[str, object] = {}

    def _fake_conv(output_dir: Path, ansys_install, conv_config=None, **_kwargs) -> CfdSolveResult:
        captured["conv_config"] = conv_config
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01131,
            mesh_cells=540000,
            runs=(
                CfdMachPoint(
                    mach=0.80,
                    velocity_m_s=270.6,
                    total_cd=0.4510,
                    total_drag_n=208.0,
                    wall_yplus_avg=1.02,
                    zones=(CfdZoneResult("rocket_fins", 58.0, 0.1257),),
                ),
            ),
            mesh_convergence=MeshConvergenceReport(
                converged=True,
                cd_rel_tol=conv_config.cd_rel_tol,
                refinement_factor=conv_config.refinement_factor,
                levels=(
                    MeshRefinementLevel(1, 1.647, 0.016, 110000, 0.22, 0.4700, 216.0, 1.10),
                    MeshRefinementLevel(
                        2, 1.220, 0.012, 250000, 0.24, 0.4530, 209.0, 1.05, 0.0375
                    ),
                    MeshRefinementLevel(
                        3, 0.904, 0.009, 540000, 0.26, 0.4510, 208.0, 1.02, 0.0044
                    ),
                ),
                optimal_level=2,
                finest_level=3,
                gci_fine_pct=0.45,
                richardson_cd=0.4502,
                observed_order_p=2.1,
            ),
        )

    monkeypatch.setattr("ork2cfd.pipeline.select_best_installation", lambda _v=None: fake_install)
    monkeypatch.setattr("ork2cfd.pipeline.run_mesh_convergence_blocking", _fake_conv)

    runner = CliRunner()
    out_dir = tmp_path / "conv_cli"
    cli_res = runner.invoke(
        cli_main,
        [
            "convert",
            str(synthetic_ork_file),
            "-o",
            str(out_dir),
            "--mode",
            "symmetry_sector",
            "--no-wbpj",
            "--run-cfd",
            "--mesh-convergence",
            "--mesh-conv-tol",
            "1.0",
        ],
    )
    assert cli_res.exit_code == 0
    assert captured["conv_config"].enabled is True
    assert captured["conv_config"].cd_rel_tol == 0.01
    assert "Mesh Convergence Study" in cli_res.output
    assert "Grid Independence:" in cli_res.output
    assert "Optimal L2" in cli_res.output
    assert "MESH_CONV_ENABLED = True" in (out_dir / "run_fluent.py").read_text(encoding="utf-8")
