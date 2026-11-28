"""Tests for live GUI workflow progress tracking, Fluent stdout parsing, and widgets."""

from __future__ import annotations

import io
import json
import tkinter as tk
from pathlib import Path

import pytest

from ork2cfd.ansys.detector import AnsysInstallation
from ork2cfd.ansys.fluent_launcher import run_fluent_solve_blocking
from ork2cfd.gui.app import Ork2CfdApp
from ork2cfd.gui.progress_tracker import WorkflowProgressTracker, format_elapsed
from ork2cfd.models import FinCrossSection
from ork2cfd.optimize import FinCandidate, SimEvaluation, StabilityReport


def test_format_elapsed() -> None:
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(65.9) == "01:05"
    assert format_elapsed(3661) == "1:01:01"


def test_tracker_package_and_rocket_only_modes() -> None:
    trk = WorkflowProgressTracker(mode="package", include_verify=True, include_wbpj=True)
    assert trk.stage_labels == ("Parse", "3D CAD", "Verify", "Scripts", "Workbench")
    idle = trk.idle_snapshot()
    assert idle.fraction == 0.0 and not idle.is_busy and idle.badge_text == "READY"

    s1 = trk.on_pipeline_progress("Parsing OpenRocket file: 947.ork...", 0.08)
    s2 = trk.on_pipeline_progress("Building watertight 3D OML, fluid domain, and BOIs...", 0.35)
    s3 = trk.on_pipeline_progress("Verifying CAD model & generating 3D viewer...", 0.64)
    s4 = trk.on_pipeline_progress("Generating ANSYS PyFluent, Fluent journal...", 0.75)
    s5 = trk.on_pipeline_progress("Building ANSYS Workbench project via v252...", 0.84)
    done = trk.complete_snapshot("Done", "Saved")

    assert [
        s1.active_stage_index,
        s2.active_stage_index,
        s3.active_stage_index,
        s4.active_stage_index,
        s5.active_stage_index,
    ] == [0, 1, 2, 3, 4]
    assert (
        s1.fraction <= s2.fraction <= s3.fraction <= s4.fraction <= s5.fraction <= done.fraction
    )
    assert done.fraction == 1.0
    assert done.badge_text == "COMPLETE" and not done.is_busy

    rkt_trk = WorkflowProgressTracker(mode="package", rocket_only=True, include_verify=False)
    assert rkt_trk.stage_labels == ("Parse", "Rocket CAD")
    err = rkt_trk.error_snapshot("Boom failure\ntraceback")
    assert err.badge_text == "FAILED" and err.detail == "Boom failure"


def test_tracker_cfd_only_live_fluent_stdout() -> None:
    trk = WorkflowProgressTracker(mode="cfd_only", max_iters=300)
    expected_stages = ("Launch", "Surface Mesh", "Volume Mesh", "Setup", "Solve", "Results")
    assert trk.stage_labels == expected_stages
    trk.start_snapshot("Launching Fluent...", "dir")

    # Python `-py -i` echoed source lines must be ignored
    assert trk.on_fluent_line('>>> print("Starting")') is None
    assert (
        trk.on_fluent_line('...     print(f"[ork2cfd] Volume mesh total cell count: {n_cells}")')
        is None
    )

    s_launch = trk.on_fluent_line("[ork2cfd] Launching ANSYS Fluent 25.2 Meshing (4 cores)...")
    assert s_launch is not None
    assert s_launch.badge_text == "FLUENT LAUNCH"
    assert s_launch.active_stage_index == 0

    s_sm = trk.on_fluent_line("---- Surface Meshing")
    assert s_sm is not None and s_sm.badge_text == "SURFACE MESH" and s_sm.active_stage_index == 1

    s_cells = trk.on_fluent_line("---------------- 430328 cells were created in :  1.14 minutes")
    assert s_cells is not None and s_cells.active_stage_index == 2
    assert "430,328" in s_cells.headline and trk.mesh_cells == 430328

    s_orth = trk.on_fluent_line(
        "---------------- The mesh has a minimum Orthogonal Quality of:  0.08"
    )
    assert s_orth is not None and trk.min_orth_quality == pytest.approx(0.08)

    # Trailing-period orthogonal quality warnings (scientific & decimal) must not raise ValueError
    for raw_val, exp_val in (("0.002.", 0.002), ("1.56e-04.", 1.56e-4), ("9.58e-04.", 9.58e-4)):
        s_low = trk.on_fluent_line(
            f"---- Warning--- The mesh has a minimum Orthogonal Quality of: {raw_val}"
        )
        assert s_low is not None
        assert trk.min_orth_quality == pytest.approx(exp_val)
        assert s_low.badge_text == "LOW QUALITY"
        assert s_low.log_line is not None and "[WARN]" in s_low.log_line

    s_hyb = trk.on_fluent_line("Hybrid initialization is done.")
    assert s_hyb is not None and s_hyb.active_stage_index == 3

    hdr = (
        "  iter  continuity  x-velocity  y-velocity  z-velocity      "
        "energy           k       omega  cd_total_c     time/iter"
    )
    assert trk.on_fluent_line(hdr) is None

    row1 = (
        "     1  1.2000e-01  5.0000e-04  3.0000e-04  2.0000e-04  "
        "1.0000e-04  4.0000e-02  5.0000e-02  1.5000e-01  0:25:00  299"
    )
    row2 = (
        "     2  9.5000e-02  4.0000e-04  2.5000e-04  1.8000e-04  "
        "8.0000e-05  3.5000e-02  4.5000e-02  1.2000e-01  0:24:50  298"
    )
    row150 = (
        "   150  1.2300e-04  2.1000e-11  1.1000e-11  9.0000e-12  "
        "1.5000e-11  3.1000e-05  2.2000e-06  8.5000e-05  0:08:10  150"
    )
    r1 = trk.on_fluent_line(row1)
    r2 = trk.on_fluent_line(row2)
    r150 = trk.on_fluent_line(row150)

    assert r1 is not None and r1.should_log is True and "1 / 300" in r1.headline
    assert r2 is not None and r2.should_log is False and r2.fraction > r1.fraction
    assert r150 is not None and r150.should_log is True and r150.active_stage_index == 4
    assert "cd_conv 8.50e-05" in r150.detail and "ETA 0:08:10" in r150.detail

    # Incompressible solver line (6 floats, no cd_total_c)
    inc_hdr = (
        "  iter  continuity  x-velocity  y-velocity  z-velocity           "
        "k       omega     time/iter"
    )
    assert trk.on_fluent_line(inc_hdr) is None
    inc_row = (
        "   200  4.5000e-05  1.2000e-09  8.0000e-10  6.0000e-10  "
        "2.0000e-05  1.0000e-06  0:03:00  100"
    )
    r_inc = trk.on_fluent_line(inc_row)
    assert r_inc is not None
    assert "cd_conv" not in r_inc.detail and "Continuity 4.50e-05" in r_inc.detail

    s_yp = trk.on_fluent_line("[ork2cfd] Mach 0.80 wall_yplus_avg = 29.2500")
    assert s_yp is not None and s_yp.active_stage_index == 5 and "29.25" in s_yp.headline


def test_tracker_package_and_cfd_mesh_convergence_scaling() -> None:
    trk = WorkflowProgressTracker(mode="package_and_cfd", max_iters=200, mesh_conv_levels=3)
    p1 = trk.on_pipeline_progress("Building watertight 3D OML...", 0.35)
    p2 = trk.on_pipeline_progress("Running progressive mesh convergence study...", 0.90)
    assert 0.0 < p1.fraction < p2.fraction <= 0.21

    l1 = trk.on_fluent_line("[ork2cfd] Mesh Convergence Level 1/3 (mesh_scale=1.6470)...")
    l1_iter = trk.on_fluent_line(
        "   100  1.0000e-03  1e-8 1e-8 1e-8 1e-8 1e-4 1e-5 5e-4  0:05:00  100"
    )
    l2 = trk.on_fluent_line("[ork2cfd] Mesh Convergence Level 2/3 (mesh_scale=1.2200)...")
    l2_cells = trk.on_fluent_line("---------------- 260000 cells were created in :  0.80 minutes")
    assert l1 is not None and l1_iter is not None and l2 is not None and l2_cells is not None
    assert p2.fraction <= l1.fraction < l1_iter.fraction < l2.fraction < l2_cells.fraction < 0.80
    assert "Level 2/3" in l2.headline
    assert "L2/3" in l2_cells.headline


def test_run_fluent_solve_blocking_line_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "run_fluent.py").write_text("# runner", encoding="utf-8")
    fluent_exe = tmp_path / "fluent.exe"
    fluent_exe.write_text("stub", encoding="utf-8")
    install = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=tmp_path,
        runwb2_exe=None,
        fluent_exe=fluent_exe,
        discovery_exe=None,
    )

    lines_out = (
        "Launching Fluent...\n"
        "  iter  continuity  x-velocity  y-velocity  z-velocity           "
        "k       omega     time/iter\n"
        "    10  1.0000e-03  1e-8 1e-8 1e-8 1e-4 1e-5  0:01:00  90\n"
        "[ork2cfd] Wrote aero_results.csv\n"
    )

    class _MockProc:
        stdout = io.StringIO(lines_out)

        def wait(self) -> int:
            (tmp_path / "cfd_summary.json").write_text(
                json.dumps({
                    "rocket_name": "Rocket",
                    "altitude_m": 500.0,
                    "aoa_deg": 0.0,
                    "sym_factor": 4.0,
                    "s_ref_m2": 0.01,
                    "mesh_cells": 100000,
                    "runs": [{
                        "mach": 0.8,
                        "velocity_m_s": 270.0,
                        "total_cd": 0.5,
                        "total_drag_n": 200.0,
                    }],
                }),
                encoding="utf-8",
            )
            return 0

    monkeypatch.setattr(
        "ork2cfd.ansys.fluent_launcher.subprocess.Popen",
        lambda *_a, **_k: _MockProc(),
    )

    all_lines: list[str] = []
    milestone_lines: list[str] = []
    run_fluent_solve_blocking(
        tmp_path,
        install,
        log_callback=milestone_lines.append,
        line_callback=all_lines.append,
    )
    assert len(all_lines) == 4
    assert len(milestone_lines) == 1
    assert "[ork2cfd]" in milestone_lines[0]


def test_gui_live_progress_widgets_and_theme(
    synthetic_ork_file: Path, tmp_path: Path
) -> None:
    try:
        app = Ork2CfdApp(initial_ork=synthetic_ork_file)
        app.withdraw()
    except tk.TclError as exc:
        if "display" in str(exc).lower():
            return
        raise

    try:
        assert app.lbl_status_badge.cget("text") == "READY"
        assert app.lbl_progress_meta.cget("text") == "0%"
        assert len(app._stage_pills) == 5

        app._is_running = True
        app._tracker = WorkflowProgressTracker(mode="package_and_cfd", max_iters=100)
        app._on_progress("Building watertight 3D OML...", 0.45)
        assert "3D CAD" in app.lbl_status_badge.cget("text")
        assert len(app._stage_pills) == 6

        app._on_fluent_line(
            "  iter  continuity  x-velocity  y-velocity  z-velocity      "
            "energy           k       omega  cd_total_c     time/iter"
        )
        app._on_fluent_line(
            "    50  2.5000e-04  1e-10 1e-10 1e-10 1e-10 1e-5 1e-6 3.1000e-04  0:02:15  50"
        )
        assert "SOLVING" in app.lbl_status_badge.cget("text")
        assert "50 / 100" in app.lbl_status_headline.cget("text")
        assert "Continuity 2.50e-04" in app.lbl_status_detail.cget("text")
        assert "%" in app.lbl_progress_meta.cget("text")

        app.var_dark.set(False)
        app._toggle_theme()
        assert "SOLVING" in app.lbl_status_badge.cget("text")

        app._is_running = False
        app._apply_progress_snapshot(app._tracker.complete_snapshot("Done", "Verified"))
        assert app.lbl_progress_meta.cget("text").startswith("100%")

        app._clear_log()
        assert app.log_box.get("1.0", "end").strip() == ""

        # Fin Optimization live progress + convergence canvas check
        opt_trk = WorkflowProgressTracker(mode="fin_opt", total_sims=10, include_verify=True)
        assert opt_trk.stage_labels == ("Baseline", "LHS DoE", "GP-EI Opt", "Verify", "Export")
        s_bo = opt_trk.on_fin_opt_progress(
            "Sim 6/10 [bayesian_ei]: square cr=240 ct=50 span=120 sweep=45.0° (SM=2.45 cal)",
            0.60,
        )
        assert s_bo.active_stage_index == 2 and s_bo.badge_text == "GP-EI OPT"

        ev1 = SimEvaluation(
            sim_index=1,
            phase="baseline",
            candidate=FinCandidate(FinCrossSection.SQUARE, 0.30, 0.10, 0.14, 55.0, 0.003),
            stability=StabilityReport(3.1, 2.5, "xml", 12.0, 4.5, 400.0, True),
            cd_total=0.66,
            cd_fins=0.25,
        )
        ev2 = SimEvaluation(
            sim_index=2,
            phase="bayesian_ei",
            candidate=FinCandidate(FinCrossSection.SQUARE, 0.22, 0.05, 0.11, 38.0, 0.003),
            stability=StabilityReport(2.9, 2.5, "xml", 10.0, 2.6, 440.0, True),
            cd_total=0.58,
            cd_fins=0.18,
        )
        app._on_opt_eval_update([ev1, ev2], 2.0, "booster", "TestRocket")
        assert "Best #2" in app.lbl_opt_headline.cget("text")
        assert "Whole-Stage/Rocket Cd=0.58000" in app.lbl_opt_headline.cget("text")
        assert "Airframe=0.4000 + Fins=0.1800" in app.lbl_opt_headline.cget("text")
        assert app.btn_optimize_fins.cget("state") == "normal"

        # Deduplicated Target Fin Stage dropdown checks
        vals = list(app._opt_target_label_to_key.values())
        assert len(vals) == len(set(vals))
        sample_947 = Path("sample orks/finalfinal947.ork")
        if sample_947.exists():
            app._load_ork(sample_947)
            assert len(app._opt_target_label_to_key) == 2
            assert list(app._opt_target_label_to_key.values()) == ["stage_2", "stage_1"]

        # _on_opt_complete tolerates non-numeric var_opt_min_sm and honors min_sm_cal < 1.0
        from ork2cfd.models import CfdMachPoint, CfdSolveResult
        from ork2cfd.optimize import FinOptimizationResult

        app.var_opt_min_sm.set("invalid_sm")
        fake_opt_res = FinOptimizationResult(
            output_dir=tmp_path,
            target_stage_name="Booster",
            baseline_eval=ev1,
            best_eval=ev2,
            verification_eval=None,
            evaluations=(ev1, ev2),
            optimized_ork_path=tmp_path / "opt.ork",
            history_json_path=tmp_path / "hist.json",
            summary_csv_path=tmp_path / "sum.csv",
            report_md_path=tmp_path / "rep.md",
            best_step_path=None,
            delta_cd=-0.08,
            delta_cd_pct=-12.12,
        )
        app._on_opt_complete(fake_opt_res, min_sm_cal=0.6)
        assert "Best #2" in app.lbl_opt_headline.cget("text")

        # Diverged standalone CFD result is flagged in GUI headline
        div_cfd = CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=tmp_path,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            runs=(CfdMachPoint(0.8, 270.0, -5.0, -500.0, 2.0, diverged=True),),
        )
        app._display_cfd_result(div_cfd)
        assert "[DIVERGED]" in app.lbl_cd_headline.cget("text")
    finally:
        app.destroy()


def test_cli_rejects_invalid_numeric_flags(synthetic_ork_file: Path) -> None:
    from click.testing import CliRunner

    from ork2cfd.cli import main as cli_main

    runner = CliRunner()
    res_mach = runner.invoke(cli_main, ["convert", str(synthetic_ork_file), "--mach", "-0.5"])
    assert res_mach.exit_code != 0
    res_yp = runner.invoke(cli_main, ["convert", str(synthetic_ork_file), "--yplus", "0"])
    assert res_yp.exit_code != 0
    res_opt = runner.invoke(
        cli_main, ["optimize-fins", str(synthetic_ork_file), "--cores", "0", "--dry-run"]
    )
    assert res_opt.exit_code != 0
