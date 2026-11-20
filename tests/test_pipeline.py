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
    assert (out_dir / "run_fluent.py").exists()
    assert (out_dir / "create_project.wbjn").exists()
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
        "mach", "altitude", "aoa", "yplus", "regime", "mesh_preset", "cores", "iterations",
        "domain_mode", "enclosure_shape", "blunt_radius", "step_schema",
        "include_bois", "fin_fillets", "rail_buttons", "launch_lugs",
        "deliv_step", "deliv_stl", "deliv_scripts", "deliv_wbpj", "deliv_wbpz", "deliv_verify",
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
    except tk.TclError:
        return

    try:
        assert app._rocket is not None
        assert app._rocket.name == "TestRocket"
        assert len(app._info_icons) == 28
        assert app.btn_view_cad is not None
        assert app.btn_ansys_guide is not None
        assert "Wall Yplus" in app.lbl_next_steps.cget("text")
        app.var_out_dir.set(str(tmp_path / "gui_out"))
        cfg = app._collect_config()
        assert cfg.flight.mach == 0.8
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
    finally:
        app.destroy()


def test_pipeline_stl_only(synthetic_ork_file: Path, tmp_path: Path) -> None:
    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.OML_ONLY),
        flight=FlightCondition(mach=0.8),
        deliverables=DeliverableConfig(
            export_step=False,
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
    assert res.manifest_path is None
    assert res.stl_path is not None and res.stl_path.exists()
    assert res.geometry_report is not None and res.geometry_report.is_valid
    assert "solid rocket_nosecone\n" in res.stl_path.read_text(encoding="utf-8")


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
