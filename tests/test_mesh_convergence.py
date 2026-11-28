"""Tests for progressive coarse-to-fine mesh convergence (grid independence) study."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import pytest

from ork2cfd.ansys.detector import AnsysInstallation
from ork2cfd.ansys.fluent_launcher import parse_cfd_results, run_fluent_solve_blocking
from ork2cfd.ansys.mesh_convergence import (
    compute_gci_metrics,
    compute_mesh_convergence_scales,
    run_mesh_convergence_blocking,
)
from ork2cfd.ansys.script_generator import generate_ansys_scripts
from ork2cfd.exceptions import AnsysError, Ork2CfdError
from ork2cfd.models import (
    CfdMachPoint,
    CfdSolveResult,
    CfdZoneResult,
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    FlightCondition,
    MeshConvergenceConfig,
    MeshPreset,
    MeshRefinementLevel,
    PipelineConfig,
)
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import compute_boundary_layer, recommend_solver


def test_scaled_mesh_sizes_and_env_propagation(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.SYMMETRY_SECTOR),
        flight=FlightCondition(mach=0.80, altitude_m=500.0, mesh_preset=MeshPreset.STANDARD),
        deliverables=DeliverableConfig(build_wbpj=False),
        mesh_convergence=MeshConvergenceConfig(enabled=True, cd_rel_tol=0.01, max_levels=4),
    )
    bl = compute_boundary_layer(rocket, cfg.flight, cfg.domain.mode)
    rec = recommend_solver(rocket, cfg.flight)

    step_p = tmp_path / "Rocket_cfd.step"
    man_p = tmp_path / "Rocket_manifest.json"
    step_p.write_text("ISO-10303-21;", encoding="utf-8")
    man_p.write_text("{}", encoding="utf-8")

    scripts = generate_ansys_scripts(
        output_dir=tmp_path,
        stem="Rocket",
        rocket=rocket,
        config=cfg,
        bl=bl,
        solver_rec=rec,
        step_path=step_p,
        manifest_path=man_p,
    )
    run_src = scripts["run_fluent.py"].read_text(encoding="utf-8")
    ast.parse(run_src, filename="run_fluent.py")
    assert "MESH_CONV_ENABLED = True" in run_src
    assert "ORK2CFD_MESH_SCALE" in run_src
    assert 'MESH_PRESET = "standard"' in run_src

    ns: dict[str, object] = {
        "__name__": "test_run_fluent",
        "__file__": str(scripts["run_fluent.py"]),
    }
    exec(compile(run_src, str(scripts["run_fluent.py"]), "exec"), ns)
    scale_fn = ns["_scaled_mesh_sizes"]

    base = scale_fn(1.0)
    assert base["surf_max"] == pytest.approx(bl.surface_max_size_m, abs=1e-6)
    for mult in (1.647, 0.50, 0.25):
        s = scale_fn(mult)
        assert (
            1e-4
            <= s["surf_min"]
            < s["fin_face"]
            <= s["nose_face"]
            < s["surf_max"]
            <= s["boi_nf"]
            < s["boi_wake"]
            < s["farfield_max"]
        )

    # Verify ORK2CFD_MESH_SCALE environment variable propagation in run_fluent_solve_blocking
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
    captured_env: dict[str, str] = {}

    class _MockProc:
        stdout = io.StringIO("[ork2cfd] Done\n")

        def wait(self) -> int:
            (tmp_path / "cfd_summary.json").write_text(
                json.dumps({
                    "rocket_name": "Rocket",
                    "altitude_m": 500.0,
                    "aoa_deg": 0.0,
                    "sym_factor": 4.0,
                    "s_ref_m2": 0.01,
                    "mesh_cells": 220000,
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

    def _fake_popen(cmd, cwd=None, env=None, **_kwargs):
        captured_env.update(env or {})
        return _MockProc()

    monkeypatch.setattr("ork2cfd.ansys.fluent_launcher.subprocess.Popen", _fake_popen)
    run_fluent_solve_blocking(tmp_path, install, cores=4, mesh_scale=1.5)
    assert captured_env["ORK2CFD_MESH_SCALE"] == "1.500000"


def test_mesh_convergence_scales_and_gci_math() -> None:
    cfg = MeshConvergenceConfig(enabled=True, cd_rel_tol=0.01, max_levels=4, refinement_factor=1.35)
    scales_std = compute_mesh_convergence_scales(MeshPreset.STANDARD, cfg)
    assert len(scales_std) == 4
    assert scales_std[0] == pytest.approx(0.140 / 0.085, rel=1e-4)
    assert all(scales_std[i] > scales_std[i + 1] for i in range(3))

    scales_coarse = compute_mesh_convergence_scales(MeshPreset.COARSE, cfg)
    assert scales_coarse[0] == pytest.approx(1.0)

    with pytest.raises(Ork2CfdError):
        compute_mesh_convergence_scales(MeshPreset.STANDARD, MeshConvergenceConfig(max_levels=1))
    with pytest.raises(Ork2CfdError):
        compute_mesh_convergence_scales(
            MeshPreset.STANDARD, MeshConvergenceConfig(refinement_factor=1.0)
        )
    with pytest.raises(Ork2CfdError):
        compute_mesh_convergence_scales(MeshPreset.STANDARD, MeshConvergenceConfig(cd_rel_tol=0.0))

    # 2-grid conservative GCI (Fs=3.0, p=2.0)
    lv1 = MeshRefinementLevel(1, 1.647, 0.016, 100000, 0.22, 0.5200, 200.0, 1.1)
    lv2 = MeshRefinementLevel(2, 1.220, 0.012, 246038, 0.24, 0.5000, 192.0, 1.0, 0.04)
    gci2, rich2, p2 = compute_gci_metrics([lv1, lv2], 1.35)
    assert p2 == pytest.approx(2.0)
    assert rich2 < 0.5000
    assert gci2 == pytest.approx(3.0 * (0.04 / (1.35**2 - 1.0)) * 100.0, rel=1e-3)

    # 3-grid monotonic convergence: exact O(h^2) sequence f(h) = 0.5 + 0.04 * h^2
    h1, h2, h3 = 1.0, 1.35, 1.35**2
    f_exact = 0.5000
    l_c = MeshRefinementLevel(1, h3, 0.018, 100000, 0.22, f_exact + 0.04 * h3**2, 210.0, 1.1)
    l_m = MeshRefinementLevel(2, h2, 0.013, 246038, 0.24, f_exact + 0.04 * h2**2, 200.0, 1.0)
    l_f = MeshRefinementLevel(3, h1, 0.010, 605344, 0.26, f_exact + 0.04 * h1**2, 194.0, 0.98)
    gci3, rich3, p3 = compute_gci_metrics([l_c, l_m, l_f], 1.35)
    assert p3 == pytest.approx(2.0, abs=0.05)
    assert rich3 == pytest.approx(f_exact, abs=1e-4)
    assert gci3 == pytest.approx(1.25 * (0.04 / 0.54) * 100.0, rel=1e-3)


def test_run_mesh_convergence_early_stop_and_persistence(tmp_path: Path) -> None:
    out_dir = tmp_path / "conv_run"
    out_dir.mkdir()
    (out_dir / "run_fluent.py").write_text(
        'MESH_PRESET = "standard"\nSURF_MIN_M = 0.001200\nSURF_MAX_M = 0.010200\n',
        encoding="utf-8",
    )

    seq = [
        (110_000, 0.6500, 290.0, 1.12),
        (270_000, 0.6250, 279.0, 1.03),
        (650_000, 0.6225, 277.9, 0.99),  # |ΔCd|/Cd = 0.40% <= 1.0% -> stops here
        (1_500_000, 0.6224, 277.8, 0.98),
    ]
    seen_scales: list[float] = []

    def _mock_solve(
        output_dir: Path,
        ansys_install,
        mesh_scale: float = 1.0,
        **_kwargs,
    ) -> CfdSolveResult:
        idx = len(seen_scales)
        seen_scales.append(mesh_scale)
        cells, cd, drag, yp = seq[idx]
        cas = output_dir / "TestRocket.cas.h5"
        dat = output_dir / "TestRocket.dat.h5"
        cas.write_text(f"cas-level-{idx + 1}", encoding="utf-8")
        dat.write_text(f"dat-level-{idx + 1}", encoding="utf-8")
        summary_data = {
            "rocket_name": "TestRocket",
            "stem": "TestRocket",
            "altitude_m": 500.0,
            "aoa_deg": 0.0,
            "sym_factor": 4.0,
            "s_ref_m2": 0.01131,
            "mesh_scale": mesh_scale,
            "mesh_cells": cells,
            "min_orth_quality": 0.24,
            "runs": [
                {
                    "mach": 0.80,
                    "velocity_m_s": 270.6,
                    "total_cd": cd,
                    "total_drag_n": drag,
                    "wall_yplus_avg": yp,
                    "case_file": cas.name,
                    "data_file": dat.name,
                    "zones": [{"zone": "rocket_fins", "drag_force_n": drag * 0.4, "cd": cd * 0.4}],
                }
            ],
        }
        summary_p = output_dir / "cfd_summary.json"
        summary_p.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01131,
            mesh_cells=cells,
            min_orth_quality=0.24,
            runs=(
                CfdMachPoint(
                    mach=0.80,
                    velocity_m_s=270.6,
                    total_cd=cd,
                    total_drag_n=drag,
                    wall_yplus_avg=yp,
                    case_file=cas,
                    data_file=dat,
                    zones=(CfdZoneResult("rocket_fins", drag * 0.4, cd * 0.4),),
                ),
            ),
            summary_json=summary_p,
        )

    logs: list[str] = []
    res = run_mesh_convergence_blocking(
        output_dir=out_dir,
        ansys_install=None,
        conv_config=MeshConvergenceConfig(enabled=True, cd_rel_tol=0.01, max_levels=4),
        log_callback=logs.append,
        solve_fn=_mock_solve,
    )

    assert len(seen_scales) == 3
    assert res.mesh_convergence is not None
    mc = res.mesh_convergence
    assert mc.converged is True
    assert mc.stop_reason == "converged"
    assert mc.optimal_level == 2
    assert mc.finest_level == 3
    assert len(mc.levels) == 3
    assert mc.gci_fine_pct is not None
    assert mc.richardson_cd is not None
    assert mc.report_json is not None and mc.report_json.exists()
    assert mc.report_csv is not None and mc.report_csv.exists()
    assert mc.report_md is not None and mc.report_md.exists()
    assert mc.plot_png is not None and mc.plot_png.exists()
    assert any("Mesh converged at Level 3" in line for line in logs)

    # Verify reload from cfd_summary.json via parse_cfd_results
    reloaded = parse_cfd_results(out_dir)
    assert reloaded is not None and reloaded.mesh_convergence is not None
    assert reloaded.mesh_convergence.converged is True
    assert reloaded.mesh_convergence.optimal_level == 2
    assert len(reloaded.mesh_convergence.levels) == 3


def test_run_mesh_convergence_recovery_and_cell_cap(tmp_path: Path) -> None:
    out_dir = tmp_path / "recover_run"
    out_dir.mkdir()
    (out_dir / "run_fluent.py").write_text('MESH_PRESET = "coarse"\nSURF_MAX_M = 0.015\n')

    calls = 0

    def _failing_on_level3(output_dir: Path, ansys_install, mesh_scale: float = 1.0, **_k):
        nonlocal calls
        calls += 1
        if calls == 3:
            (output_dir / "fluent_run.log").write_text("Exceeded 512k student limit\n")
            raise AnsysError("Fluent exited with code 2 (512k limit)")
        cells = 140_000 if calls == 1 else 380_000
        cd = 0.6600 if calls == 1 else 0.6300
        cas = output_dir / "TestRocket.cas.h5"
        dat = output_dir / "TestRocket.dat.h5"
        cas.write_text(f"valid-level-{calls}", encoding="utf-8")
        dat.write_text(f"valid-dat-{calls}", encoding="utf-8")
        (output_dir / "cfd_summary.json").write_text(
            json.dumps({
                "rocket_name": "TestRocket",
                "mesh_cells": cells,
                "runs": [{"mach": 0.8, "total_cd": cd, "total_drag_n": 250.0}],
            }),
            encoding="utf-8",
        )
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=cells,
            runs=(CfdMachPoint(0.8, 270.0, cd, 250.0, 1.0, cas, dat),),
        )

    recovered = run_mesh_convergence_blocking(
        output_dir=out_dir,
        ansys_install=None,
        conv_config=MeshConvergenceConfig(enabled=True, cd_rel_tol=0.005, max_levels=4),
        solve_fn=_failing_on_level3,
    )
    assert recovered.mesh_convergence is not None
    assert recovered.mesh_convergence.converged is False
    assert recovered.mesh_convergence.stop_reason == "finer_level_failed_recovered"
    assert len(recovered.mesh_convergence.levels) == 2
    assert (out_dir / "TestRocket.cas.h5").read_text(encoding="utf-8") == "valid-level-2"
    assert (out_dir / "mesh_convergence" / "level_03_failed.log").exists()

    # Also test max_cells cap stops cleanly
    cap_dir = tmp_path / "cap_run"
    cap_dir.mkdir()
    cap_calls = 0

    def _cap_solve(output_dir: Path, ansys_install, mesh_scale: float = 1.0, **_k):
        nonlocal cap_calls
        cap_calls += 1
        cells = 150_000 * cap_calls
        cd = 0.65 - 0.03 * cap_calls
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=cells,
            runs=(CfdMachPoint(0.8, 270.0, cd, 240.0),),
        )

    capped = run_mesh_convergence_blocking(
        output_dir=cap_dir,
        ansys_install=None,
        conv_config=MeshConvergenceConfig(
            enabled=True, cd_rel_tol=0.001, max_levels=4, max_cells=300_000
        ),
        solve_fn=_cap_solve,
    )
    assert capped.mesh_convergence is not None
    assert capped.mesh_convergence.stop_reason == "max_cells_reached"
    assert len(capped.mesh_convergence.levels) == 2


def test_mesh_convergence_divergence_recovery_and_abort(tmp_path: Path) -> None:
    div_l1_dir = tmp_path / "div_l1"
    div_l1_dir.mkdir()
    (div_l1_dir / "run_fluent.py").write_text('MESH_PRESET = "coarse"\nSURF_MAX_M = 0.015\n')

    def _l1_bad(output_dir: Path, *_a, **_k):
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=120_000,
            runs=(CfdMachPoint(1.75, 568.0, 22147045.0, 1e9, 5600.0),),
        )

    with pytest.raises(AnsysError, match="diverged"):
        run_mesh_convergence_blocking(
            output_dir=div_l1_dir,
            ansys_install=None,
            conv_config=MeshConvergenceConfig(enabled=True, max_levels=3),
            solve_fn=_l1_bad,
        )

    div_l2_dir = tmp_path / "div_l2"
    div_l2_dir.mkdir()
    (div_l2_dir / "run_fluent.py").write_text('MESH_PRESET = "coarse"\nSURF_MAX_M = 0.015\n')
    l2_calls = 0

    def _l2_bad(output_dir: Path, *_a, **_k):
        nonlocal l2_calls
        l2_calls += 1
        cd = 0.55 if l2_calls == 1 else -22.15
        (output_dir / "TestRocket.cas.h5").write_text(f"cas-{l2_calls}", encoding="utf-8")
        (output_dir / "TestRocket.dat.h5").write_text(f"dat-{l2_calls}", encoding="utf-8")
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=120_000 * l2_calls,
            runs=(CfdMachPoint(0.8, 270.0, cd, cd * 400.0, 28.0),),
        )

    recovered = run_mesh_convergence_blocking(
        output_dir=div_l2_dir,
        ansys_install=None,
        conv_config=MeshConvergenceConfig(enabled=True, max_levels=3),
        solve_fn=_l2_bad,
    )
    assert recovered.mesh_convergence is not None
    assert recovered.mesh_convergence.stop_reason == "finer_level_diverged_recovered"
    assert len(recovered.mesh_convergence.levels) == 1
    assert (div_l2_dir / "TestRocket.cas.h5").read_text(encoding="utf-8") == "cas-1"
