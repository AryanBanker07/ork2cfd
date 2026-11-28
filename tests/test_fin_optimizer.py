"""Tests for the <=10-sim constrained Bayesian fin optimization workflow."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from ork2cfd.cli import main as cli_main
from ork2cfd.geometry.domain_builder import build_cfd_domain
from ork2cfd.models import DomainConfig, DomainMode, FinCrossSection, PipelineConfig
from ork2cfd.optimize import (
    ACTIVE_CONTINUOUS_PARAMS,
    FinCandidate,
    FinDesignBounds,
    FinOptimizationConfig,
    GaussianProcessSurrogate,
    apply_candidate_to_rocket,
    candidate_from_fin,
    compute_barrowman_cp,
    default_bounds_for_fin,
    estimate_cg_from_ork,
    propose_next_bo_candidate,
    resolve_domain_mode,
    resolve_target_fin,
    run_fin_optimization,
    sample_feasible_lhs,
    write_modified_ork,
)
from ork2cfd.optimize.constraints import evaluate_constraints
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import recommend_solver


def test_barrowman_cp_and_stability_margin_947(synthetic_ork_file: Path) -> None:
    sample_947 = Path("sample orks/finalfinal947.ork")
    ork_path = sample_947 if sample_947.exists() else synthetic_ork_file
    rocket = parse_ork_file(ork_path)
    x_cp, cna = compute_barrowman_cp(rocket)
    cg_x, cg_src = estimate_cg_from_ork(ork_path, "all")

    assert cna > 5.0
    assert 0.3 * rocket.total_length < x_cp < rocket.total_length
    assert 0.2 * rocket.total_length < cg_x < x_cp

    target_ids, _stage, comp, orig_fin = resolve_target_fin(rocket, "booster")
    base_cand = candidate_from_fin(orig_fin)
    cfg = FinOptimizationConfig(min_static_margin_cal=2.0)

    rep_ok = evaluate_constraints(rocket, orig_fin, comp, base_cand, cg_x, cg_src, cfg)
    assert rep_ok.feasible
    assert rep_ok.static_margin_cal >= 2.0

    # Mutating thickness must violate the locked-thickness constraint
    bad_thick = FinCandidate(
        cross_section=base_cand.cross_section,
        root_chord=base_cand.root_chord,
        tip_chord=base_cand.tip_chord,
        span=base_cand.span,
        sweep_angle_deg=base_cand.sweep_angle_deg,
        thickness=orig_fin.thickness + 0.001,
    )
    rep_t = evaluate_constraints(rocket, orig_fin, comp, bad_thick, cg_x, cg_src, cfg)
    assert not rep_t.feasible
    assert any("thickness_changed" in v for v in rep_t.violations)

    # Moving CG far aft must violate 2.0 cal static margin
    rep_sm = evaluate_constraints(
        rocket, orig_fin, comp, base_cand, x_cp - 0.5 * rocket.max_diameter, "test", cfg
    )
    assert not rep_sm.feasible
    assert any("static_margin" in v for v in rep_sm.violations)


def test_ork_parameterizer_roundtrip_booster_preserves_sustainer_and_thickness(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    # 1. Plain XML rocket roundtrip
    syn_rocket = parse_ork_file(synthetic_ork_file)
    s_ids, _st, _comp, s_fin = resolve_target_fin(syn_rocket, "booster")
    s_cand = FinCandidate(
        cross_section=FinCrossSection.AIRFOIL,
        root_chord=0.22,
        tip_chord=0.07,
        span=0.105,
        sweep_angle_deg=52.0,
        thickness=s_fin.thickness,
    )
    syn_out = tmp_path / "syn_opt.ork"
    write_modified_ork(synthetic_ork_file, syn_out, s_ids, s_cand)
    syn_re = parse_ork_file(syn_out)
    re_fin = syn_re.all_fins[0]
    assert re_fin.cross_section == FinCrossSection.AIRFOIL
    assert re_fin.root_chord == pytest.approx(0.22, rel=1e-5)
    assert re_fin.tip_chord == pytest.approx(0.07, rel=1e-5)
    assert re_fin.span == pytest.approx(0.105, rel=1e-5)
    assert re_fin.thickness == pytest.approx(s_fin.thickness, rel=1e-6)
    assert re_fin.sweep_length == pytest.approx(0.105 * math.tan(math.radians(52.0)), rel=1e-4)

    # 2. Multi-stage ZIP .ork: optimizing booster preserves sustainer option and thickness
    sample_947 = Path("sample orks/finalfinal947.ork")
    if not sample_947.exists():
        return
    r947 = parse_ork_file(sample_947)
    sust_ids, sust_stage, _sc, sust_fin = resolve_target_fin(r947, "sustainer")
    boost_ids, boost_stage, _bc, boost_fin = resolve_target_fin(r947, "booster")
    assert sust_ids[0] == 0 and sust_stage.name == "Sustainer"
    assert boost_ids[0] == 1 and boost_stage.name == "Stage"

    new_boost = FinCandidate(
        cross_section=FinCrossSection.DOUBLE_WEDGE,
        root_chord=0.25,
        tip_chord=0.08,
        span=0.115,
        sweep_angle_deg=60.0,
        thickness=boost_fin.thickness,
    )
    mutated_in_mem = apply_candidate_to_rocket(r947, boost_ids, new_boost)
    assert (
        mutated_in_mem.stages[1].components[boost_ids[1]].fins[boost_ids[2]].root_chord
        == pytest.approx(0.25)
    )
    zip_out = tmp_path / "947_opt.ork"
    write_modified_ork(sample_947, zip_out, boost_ids, new_boost)
    r947_after = parse_ork_file(zip_out)

    # Sustainer fins must be 100% untouched
    sust_after = r947_after.stages[0].components[sust_ids[1]].fins[sust_ids[2]]
    assert sust_after == sust_fin

    # Booster fins updated while thickness stays locked
    boost_after = r947_after.stages[1].components[boost_ids[1]].fins[boost_ids[2]]
    assert boost_after.cross_section == FinCrossSection.DOUBLE_WEDGE
    assert boost_after.root_chord == pytest.approx(0.25, rel=1e-5)
    assert boost_after.tip_chord == pytest.approx(0.08, rel=1e-5)
    assert boost_after.span == pytest.approx(0.115, rel=1e-5)
    assert boost_after.thickness == pytest.approx(boost_fin.thickness, rel=1e-6)
    assert resolve_domain_mode(r947, FinOptimizationConfig()) == DomainMode.SYMMETRY_SECTOR


def test_gp_surrogate_and_constrained_ei(synthetic_ork_file: Path) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    _ids, _st, comp, fin = resolve_target_fin(rocket, "booster")
    bounds: FinDesignBounds = default_bounds_for_fin(fin, comp)
    base = candidate_from_fin(fin)

    def _always_feasible(cand: FinCandidate):
        # Enforce tip_chord <= root_chord
        feasible = cand.tip_chord <= cand.root_chord
        return type("Rep", (), {"feasible": feasible, "static_margin_cal": 2.5})()

    samples = sample_feasible_lhs(
        bounds=bounds,
        active_params=ACTIVE_CONTINUOUS_PARAMS,
        template_candidate=base,
        feasibility_fn=_always_feasible,
        n_samples=5,
        existing_vectors=[base.to_vector(ACTIVE_CONTINUOUS_PARAMS)],
        seed=7,
    )
    assert len(samples) == 5
    for c in samples:
        assert c.tip_chord <= c.root_chord
        assert c.thickness == pytest.approx(base.thickness)

    all_cands = [base, *samples]
    x_train = [c.to_vector(ACTIVE_CONTINUOUS_PARAMS) for c in all_cands]
    y_train = [
        0.35 + (c.span - 0.08) ** 2 * 10.0 + (c.sweep_angle_deg - 55.0) ** 2 * 1e-4
        for c in all_cands
    ]

    gp = GaussianProcessSurrogate(bounds, ACTIVE_CONTINUOUS_PARAMS)
    gp.fit(x_train, y_train)
    mu, std = gp.predict(x_train)
    assert mu.shape == (6,)
    assert std.shape == (6,)
    assert max(abs(float(m) - float(y)) for m, y in zip(mu, y_train, strict=True)) < 0.02

    nxt = propose_next_bo_candidate(
        gp=gp,
        bounds=bounds,
        active_params=ACTIVE_CONTINUOUS_PARAMS,
        template_candidate=base,
        feasibility_fn=_always_feasible,
        evaluated_vectors=x_train,
        y_best=min(y_train),
        seed=11,
    )
    assert nxt.tip_chord <= nxt.root_chord
    assert nxt.thickness == pytest.approx(base.thickness)


def test_run_fin_optimization_947_booster_dry_run_and_resume(tmp_path: Path) -> None:
    sample_947 = Path("sample orks/finalfinal947.ork")
    if not sample_947.exists():
        pytest.skip("sample orks/finalfinal947.ork not present")

    out_dir = tmp_path / "opt_947"
    cfg = FinOptimizationConfig(
        max_sims=10,
        stage_selection="all",
        target_fin_stage="booster",
        mach=0.80,
        altitude_m=500.0,
        min_static_margin_cal=2.0,
        dry_run=True,
        export_best_cad=False,
        resume=True,
    )
    res = run_fin_optimization(sample_947, out_dir, config=cfg)

    assert len(res.evaluations) == 10
    assert res.target_stage_name == "Stage"
    phases = [e.phase for e in res.evaluations]
    assert phases[0] == "baseline"
    assert phases.count("cross_section_screen") == 0
    assert phases.count("doe_lhs") == 4
    assert phases.count("bayesian_ei") == 5

    orig = parse_ork_file(sample_947)
    _, _, _, orig_booster_fin = resolve_target_fin(orig, "booster")
    orig_booster_cs = orig_booster_fin.cross_section
    for ev in res.evaluations:
        assert ev.candidate.thickness == pytest.approx(0.003)
        assert ev.candidate.cross_section == orig_booster_cs
        if ev.sim_index > 1:
            assert ev.stability.feasible
            assert ev.stability.static_margin_cal >= 2.0

    assert res.best_eval.cd_total < res.baseline_eval.cd_total
    assert res.optimized_ork_path.exists()
    assert res.history_json_path.exists()
    assert res.summary_csv_path.exists()
    assert res.report_md_path.exists()
    assert res.cd_plot_path is not None and res.cd_plot_path.exists()

    # Verify sustainer untouched and optimized geometry builds a valid watertight OCCT fluid domain
    opt = parse_ork_file(res.optimized_ork_path)
    assert opt.stages[0].components[3].fins == orig.stages[0].components[3].fins
    dom = build_cfd_domain(
        opt,
        PipelineConfig().cad,
        DomainConfig(mode=DomainMode.SYMMETRY_HALF_SECTOR),
        recommend_solver(opt, PipelineConfig().flight),
    )
    assert dom.report.is_watertight

    # Resume on same directory must load all 10 existing evaluations cleanly
    res2 = run_fin_optimization(sample_947, out_dir, config=cfg)
    assert len(res2.evaluations) == 10
    assert res2.best_eval.sim_index == res.best_eval.sim_index


def test_run_fin_optimization_budget_with_verification_slot(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, str, bool]] = []

    def _mock_eval(sim_idx, phase, _rocket, cand, _sim_dir, _cfg, high_fidelity):
        calls.append((sim_idx, phase, high_fidelity))
        cs_pen = 0.04 if cand.cross_section == FinCrossSection.SQUARE else 0.0
        cd = 0.30 + cs_pen + 0.5 * cand.span + 0.2 * cand.tip_chord
        return cd, 0.12 + cs_pen, cd * 100.0, 1.0 if high_fidelity else 28.0, 250000

    cfg = FinOptimizationConfig(
        max_sims=10,
        stage_selection="all",
        target_fin_stage="booster",
        min_static_margin_cal=1.5,
        verify_final=True,
        dry_run=False,
        export_best_cad=False,
    )
    eval_counts: list[int] = []
    res = run_fin_optimization(
        synthetic_ork_file,
        tmp_path / "mock_opt",
        config=cfg,
        evaluator=_mock_eval,
        eval_callback=lambda evs: eval_counts.append(len(evs)),
    )
    assert eval_counts == list(range(1, 11))
    assert len(res.evaluations) == 10
    assert res.verification_eval is not None
    assert res.verification_eval.phase == "verification"
    assert res.verification_eval.sim_index == 10
    assert calls[-1] == (10, "verification", True)
    assert [c[1] for c in calls] == [
        "baseline",
        "doe_lhs",
        "doe_lhs",
        "doe_lhs",
        "doe_lhs",
        "bayesian_ei",
        "bayesian_ei",
        "bayesian_ei",
        "bayesian_ei",
        "verification",
    ]


def test_cli_optimize_fins_dry_run(synthetic_ork_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "cli_opt"
    res = runner.invoke(
        cli_main,
        [
            "optimize-fins",
            str(synthetic_ork_file),
            "-o",
            str(out_dir),
            "--max-sims",
            "8",
            "--min-sm",
            "1.5",
            "--dry-run",
            "--no-cad",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Best Cd:" in res.output
    assert (out_dir / "TestRocket_optimized.ork").exists()
    assert (out_dir / "optimization_summary.csv").exists()
    assert (out_dir / "optimization_cd_plot.png").exists()
    assert (out_dir / "optimization_report.md").exists()


def test_hdf5_contour_renderer_and_mpi_flag(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    import h5py
    import numpy as np

    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.ansys.fluent_launcher import run_fluent_solve_blocking
    from ork2cfd.optimize import render_fluent_hdf5_contours

    rocket = parse_ork_file(synthetic_ork_file)
    fin = rocket.all_fins[0]

    # Build a minimal valid Fluent HDF5 mesh + data pair with hexahedra on +Y plane
    cas_p = tmp_path / "mini.cas.h5"
    dat_p = tmp_path / "mini.dat.h5"
    pts: list[list[float]] = []
    face_nodes: list[int] = []
    face_nnodes: list[int] = []
    c0_list: list[int] = []
    c1_list: list[int] = []
    # Create 20 small quad faces / 20 cells along X=[1.0, 2.0], Y=[0.05, 0.25], Z=0
    for i in range(20):
        x0 = 1.0 + i * 0.05
        y0 = 0.06 + (i % 4) * 0.04
        base = len(pts) + 1
        pts.extend([
            [x0, y0, 0.0],
            [x0 + 0.05, y0, 0.0],
            [x0 + 0.05, y0 + 0.04, 0.0],
            [x0, y0 + 0.04, 0.0],
        ])
        face_nodes.extend([base, base + 1, base + 2, base + 3])
        face_nnodes.append(4)
        c0_list.append(i + 1)
        c1_list.append(0)

    with h5py.File(cas_p, "w") as fc:
        fc.create_dataset("meshes/1/nodes/coords/1", data=np.array(pts, dtype=np.float64))
        fc.create_dataset(
            "meshes/1/faces/nodes/1/nnodes", data=np.array(face_nnodes, dtype=np.int32)
        )
        fc.create_dataset(
            "meshes/1/faces/nodes/1/nodes", data=np.array(face_nodes, dtype=np.int32)
        )
        fc.create_dataset("meshes/1/faces/c0/1", data=np.array(c0_list, dtype=np.int32))
        fc.create_dataset("meshes/1/faces/c1/1", data=np.array(c1_list, dtype=np.int32))
        fc.create_dataset("meshes/1/cells/zoneTopology/maxId", data=np.array([20], dtype=np.int64))
        fc.create_dataset("meshes/1/faces/zoneTopology/name", data=np.array([b"rocket_fins.1"]))
        fc.create_dataset("meshes/1/faces/zoneTopology/minId", data=np.array([1], dtype=np.int64))
        fc.create_dataset("meshes/1/faces/zoneTopology/maxId", data=np.array([20], dtype=np.int64))

    with h5py.File(dat_p, "w") as fd:
        fd.create_dataset("results/1/phase-1/cells/SV_U/1", data=np.full(20, 260.0))
        fd.create_dataset("results/1/phase-1/cells/SV_V/1", data=np.full(20, 5.0))
        fd.create_dataset("results/1/phase-1/cells/SV_W/1", data=np.zeros(20))
        fd.create_dataset("results/1/phase-1/cells/SV_T/1", data=np.full(20, 285.0))
        fd.create_dataset("results/1/phase-1/cells/SV_P/1", data=np.full(20, 96000.0))

    png_out = tmp_path / "mini_contours.png"
    res_png = render_fluent_hdf5_contours(
        cas_p,
        dat_p,
        png_out,
        title="Test HDF5 Contour",
        rocket=rocket,
        target_fin=fin,
        mach_inf=0.80,
        p_inf_pa=95460.0,
        rho_inf=1.167,
        v_inf=270.7,
    )
    assert res_png == png_out and png_out.exists() and png_out.stat().st_size > 5000

    # Verify optional mpi flag passes through to fluent command cleanly
    pkg_dir = tmp_path / "mpi_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "run_fluent.py").write_text("# test", encoding="utf-8")
    fexe = tmp_path / "fluent.exe"
    fexe.write_text("stub", encoding="utf-8")
    inst = AnsysInstallation("25.2", "v252", "25.2.170", tmp_path, None, fexe, None)
    seen: list[list[str]] = []

    class _P:
        stdout = io.StringIO("ok\n")

        def wait(self) -> int:
            (pkg_dir / "aero_results.csv").write_text(
                "rocket,mach,altitude_m,aoa_deg,zone,drag_force_n,cd\n"
                "R,0.8,500,0,total,100,0.4\n",
                encoding="utf-8",
            )
            return 0

    monkeypatch.setattr(
        "ork2cfd.ansys.fluent_launcher.subprocess.Popen",
        lambda cmd, **_kw: seen.append(list(cmd)) or _P(),
    )
    run_fluent_solve_blocking(pkg_dir, inst, cores=4, mpi="msmpi")
    assert seen[0] == [
        str(fexe),
        "3ddp",
        "-meshing",
        "-g",
        "-t4",
        "-mpi=msmpi",
        "-py",
        "-i",
        "run_fluent.py",
    ]


def test_fin_opt_reuses_existing_mesh_convergence_characteristics(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.models import CfdMachPoint, CfdSolveResult, MeshPreset

    prior_dir = tmp_path / "prior_study"
    prior_dir.mkdir()
    conv_json = prior_dir / "mesh_convergence.json"
    conv_json.write_text(
        json.dumps({
            "converged": True,
            "cd_rel_tol": 0.01,
            "refinement_factor": 1.35,
            "optimal_level": 2,
            "finest_level": 3,
            "gci_fine_pct": 0.65,
            "base_preset": "standard",
            "target_yplus": 12.5,
            "levels": [
                {"level": 1, "mesh_scale": 1.647, "mesh_cells": 110000, "total_cd": 0.55},
                {"level": 2, "mesh_scale": 1.220, "mesh_cells": 240000, "total_cd": 0.52},
                {"level": 3, "mesh_scale": 0.904, "mesh_cells": 510000, "total_cd": 0.518},
            ],
        }),
        encoding="utf-8",
    )

    fake_exe = tmp_path / "fluent.exe"
    fake_exe.write_text("stub", encoding="utf-8")
    fake_inst = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=tmp_path,
        runwb2_exe=None,
        fluent_exe=fake_exe,
        discovery_exe=None,
    )
    seen_scales: list[float | None] = []
    seen_presets: list[str] = []

    def _fake_solve(
        output_dir: Path,
        ansys_install=None,
        mesh_scale=None,
        **_kw,
    ) -> CfdSolveResult:
        run_src = (output_dir / "run_fluent.py").read_text(encoding="utf-8")
        assert "standard" in run_src
        seen_presets.append("standard")
        seen_scales.append(mesh_scale)
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=240000,
            runs=(CfdMachPoint(0.8, 270.0, 0.45 - 0.01 * len(seen_scales), 190.0, 12.5),),
        )

    monkeypatch.setattr(
        "ork2cfd.optimize.runner.select_best_installation", lambda _v=None: fake_inst
    )
    monkeypatch.setattr("ork2cfd.optimize.runner.run_fluent_solve_blocking", _fake_solve)

    out_dir = tmp_path / "reused_opt"
    res = run_fin_optimization(
        synthetic_ork_file,
        out_dir,
        config=FinOptimizationConfig(
            max_sims=4,
            min_static_margin_cal=1.5,
            verify_final=False,
            export_best_cad=False,
            mesh_convergence_source=conv_json,
        ),
    )
    assert res.locked_mesh is not None
    assert res.locked_mesh.mesh_preset == MeshPreset.STANDARD
    assert res.locked_mesh.mesh_scale == pytest.approx(1.220)
    assert res.locked_mesh.target_yplus == pytest.approx(12.5)
    assert seen_scales == [pytest.approx(1.220)] * 4
    assert len(seen_presets) == 4


def test_fin_opt_baseline_mesh_convergence_locks_scale_for_following_sims_and_resume(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.models import (
        CfdMachPoint,
        CfdSolveResult,
        MeshConvergenceReport,
        MeshRefinementLevel,
    )

    fake_exe = tmp_path / "fluent.exe"
    fake_exe.write_text("stub", encoding="utf-8")
    fake_inst = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=tmp_path,
        runwb2_exe=None,
        fluent_exe=fake_exe,
        discovery_exe=None,
    )

    conv_calls = 0
    seen_solve_scales: list[float | None] = []

    def _fake_conv(output_dir: Path, *_a, **_kw) -> CfdSolveResult:
        nonlocal conv_calls
        conv_calls += 1
        opt_lv_dir = output_dir / "mesh_convergence" / "level_02"
        opt_lv_dir.mkdir(parents=True, exist_ok=True)
        (opt_lv_dir / "cfd_summary.json").write_text(
            json.dumps({
                "rocket_name": "TestRocket",
                "altitude_m": 500.0,
                "aoa_deg": 0.0,
                "sym_factor": 4.0,
                "s_ref_m2": 0.01,
                "mesh_cells": 245000,
                "runs": [{
                    "mach": 0.80,
                    "velocity_m_s": 270.0,
                    "total_cd": 0.5200,
                    "total_drag_n": 210.0,
                    "wall_yplus_avg": 29.1,
                }],
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
            mesh_cells=520000,
            runs=(CfdMachPoint(0.80, 270.0, 0.5190, 209.5, 29.0),),
            mesh_convergence=MeshConvergenceReport(
                converged=True,
                cd_rel_tol=0.01,
                refinement_factor=1.35,
                levels=(
                    MeshRefinementLevel(1, 1.647, 0.016, 110000, 0.22, 0.5500, 222.0, 29.5),
                    MeshRefinementLevel(
                        2, 1.220, 0.012, 245000, 0.24, 0.5200, 210.0, 29.1, 0.05
                    ),
                    MeshRefinementLevel(
                        3, 0.904, 0.009, 520000, 0.26, 0.5190, 209.5, 29.0, 0.0019
                    ),
                ),
                optimal_level=2,
                finest_level=3,
                gci_fine_pct=0.29,
                richardson_cd=0.5188,
                observed_order_p=2.0,
            ),
        )

    def _fake_solve(
        output_dir: Path,
        ansys_install=None,
        mesh_scale=None,
        **_kw,
    ) -> CfdSolveResult:
        seen_solve_scales.append(mesh_scale)
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=245000,
            runs=(CfdMachPoint(0.80, 270.0, 0.4900, 198.0, 29.1),),
        )

    monkeypatch.setattr(
        "ork2cfd.optimize.runner.select_best_installation", lambda _v=None: fake_inst
    )
    monkeypatch.setattr("ork2cfd.optimize.runner.run_mesh_convergence_blocking", _fake_conv)
    monkeypatch.setattr("ork2cfd.optimize.runner.run_fluent_solve_blocking", _fake_solve)

    out_dir = tmp_path / "conv_then_opt"
    cfg = FinOptimizationConfig(
        max_sims=4,
        min_static_margin_cal=1.5,
        verify_final=False,
        export_best_cad=False,
        mesh_convergence_baseline=True,
    )
    res = run_fin_optimization(synthetic_ork_file, out_dir, config=cfg)

    assert conv_calls == 1
    assert res.baseline_eval.cd_total == pytest.approx(0.5200)
    assert res.baseline_eval.mesh_cells == 245000
    assert seen_solve_scales == [pytest.approx(1.220), pytest.approx(1.220), pytest.approx(1.220)]
    assert res.locked_mesh is not None and res.locked_mesh.mesh_scale == pytest.approx(1.220)

    # Extend max_sims to 5 via resume: must reuse locked_mesh without re-running convergence
    res2 = run_fin_optimization(
        synthetic_ork_file,
        out_dir,
        config=FinOptimizationConfig(
            max_sims=5,
            min_static_margin_cal=1.5,
            verify_final=False,
            export_best_cad=False,
            mesh_convergence_baseline=True,
            resume=True,
        ),
    )
    assert conv_calls == 1
    assert len(seen_solve_scales) == 4
    assert seen_solve_scales[-1] == pytest.approx(1.220)
    assert res2.locked_mesh is not None and res2.locked_mesh.optimal_level == 2


def test_fin_opt_divergence_filtering_and_verification_ram_cap(
    synthetic_ork_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.exceptions import Ork2CfdError
    from ork2cfd.models import CfdMachPoint, CfdSolveResult

    # 1. Baseline divergence aborts early
    def _bad_base(_idx, _phase, _r, _c, _dir, _cfg, _hf):
        return -5.0, -1.0, -500.0, 10.0, 100_000

    with pytest.raises(Ork2CfdError, match="Baseline Simulation 1 diverged"):
        run_fin_optimization(
            synthetic_ork_file,
            tmp_path / "bad_base",
            config=FinOptimizationConfig(max_sims=4, dry_run=False, export_best_cad=False),
            evaluator=_bad_base,
        )

    # 2. Mid-run diverged sims are marked diverged and ignored in best selection
    def _mixed_eval(idx, _phase, _r, _c, _dir, _cfg, _hf):
        if idx == 3:
            return -22.15, -2.0, -33000.0, 3.2, 200_000
        if idx == 4:
            return 22_000_000.0, 22_000_000.0, 3e10, 5600.0, 200_000
        cd = 0.58 - 0.02 * idx
        return cd, cd * 0.35, cd * 400.0, 28.0, 200_000

    res_div = run_fin_optimization(
        synthetic_ork_file,
        tmp_path / "div_filter",
        config=FinOptimizationConfig(
            max_sims=6,
            min_static_margin_cal=1.5,
            verify_final=False,
            dry_run=False,
            export_best_cad=False,
        ),
        evaluator=_mixed_eval,
    )
    assert res_div.evaluations[2].diverged is True
    assert res_div.evaluations[3].diverged is True
    assert not res_div.best_eval.diverged
    assert 0.02 < res_div.best_eval.cd_total < 5.0

    # 3. Verification keeps locked mesh when baseline study did not converge or hits RAM cap
    prior = tmp_path / "unconv"
    prior.mkdir()
    conv_json = prior / "mesh_convergence.json"
    conv_json.write_text(
        json.dumps({
            "converged": False,
            "stop_reason": "max_levels_reached",
            "optimal_level": 4,
            "finest_level": 4,
            "base_preset": "standard",
            "target_yplus": 30.0,
            "levels": [
                {"level": 4, "mesh_scale": 0.669434, "mesh_cells": 950_000, "total_cd": 0.61}
            ],
        }),
        encoding="utf-8",
    )
    fake_exe = tmp_path / "fluent.exe"
    fake_exe.write_text("stub", encoding="utf-8")
    fake_inst = AnsysInstallation("25.2", "v252", "25.2.170", tmp_path, None, fake_exe, None)
    seen_verify: list[tuple[float | None, bool]] = []

    def _cap_solve(output_dir: Path, mesh_scale=None, **_kw):
        run_src = (output_dir / "run_fluent.py").read_text(encoding="utf-8")
        seen_verify.append((mesh_scale, "PRISM_LAYERS" in run_src and "30.0" or True))
        return CfdSolveResult(
            rocket_name="TestRocket",
            output_dir=output_dir,
            altitude_m=500.0,
            aoa_deg=0.0,
            sym_factor=4.0,
            s_ref_m2=0.01,
            mesh_cells=950_000,
            runs=(CfdMachPoint(0.8, 270.0, 0.55 - 0.01 * len(seen_verify), 210.0, 29.0),),
        )

    monkeypatch.setattr(
        "ork2cfd.optimize.runner.select_best_installation", lambda _v=None: fake_inst
    )
    monkeypatch.setattr("ork2cfd.optimize.runner.run_fluent_solve_blocking", _cap_solve)

    run_fin_optimization(
        synthetic_ork_file,
        tmp_path / "verify_cap",
        config=FinOptimizationConfig(
            max_sims=8,
            min_static_margin_cal=1.5,
            verify_final=True,
            export_best_cad=False,
            max_cells=1_000_000,
            mesh_convergence_source=conv_json,
        ),
    )
    assert len(seen_verify) == 8
    assert seen_verify[-1][0] == pytest.approx(0.669434)


def test_duplicate_stage_names_cg_shift_resume_signature_and_plot_cleanup(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    import json
    import xml.etree.ElementTree as ET

    import matplotlib.pyplot as plt

    from ork2cfd.optimize.plots import render_fluent_hdf5_contours
    from ork2cfd.parser import parse_ork_file

    # 1. Duplicate stage names ("Stage") with stage_selection="stage_2" & explicit CG shift
    dup_xml = """<?xml version="1.0" encoding="utf-8"?>
<openrocket version="1.9">
  <rocket>
    <name>DupStage</name>
    <subcomponents>
      <stage>
        <name>Stage</name>
        <subcomponents>
          <nosecone>
            <name>Nose</name>
            <length>0.30</length>
            <shape>ogive</shape>
            <aftradius>0.05</aftradius>
          </nosecone>
          <bodytube>
            <name>UpperBody</name>
            <length>0.70</length>
            <radius>0.05</radius>
            <subcomponents>
              <trapezoidfinset>
                <name>UpperFins</name>
                <fincount>4</fincount>
                <rootchord>0.12</rootchord>
                <tipchord>0.06</tipchord>
                <sweeplength>0.04</sweeplength>
                <height>0.07</height>
                <thickness>0.003</thickness>
                <crosssection>square</crosssection>
                <position type="bottom">-0.02</position>
              </trapezoidfinset>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
      <stage>
        <name>Stage</name>
        <subcomponents>
          <bodytube>
            <name>BoosterBody</name>
            <length>0.90</length>
            <radius>0.05</radius>
            <subcomponents>
              <trapezoidfinset>
                <name>BoosterFins</name>
                <fincount>4</fincount>
                <rootchord>0.22</rootchord>
                <tipchord>0.08</tipchord>
                <sweeplength>0.06</sweeplength>
                <height>0.11</height>
                <thickness>0.003</thickness>
                <crosssection>square</crosssection>
                <position type="bottom">-0.02</position>
              </trapezoidfinset>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
    dup_ork = tmp_path / "dup_stages.ork"
    dup_ork.write_text(dup_xml, encoding="utf-8")
    opt_dir = tmp_path / "dup_opt"
    res = run_fin_optimization(
        dup_ork,
        opt_dir,
        config=FinOptimizationConfig(
            max_sims=4,
            stage_selection="stage_2",
            target_fin_stage="booster",
            min_static_margin_cal=1.5,
            cg_x_m=1.35,  # full-vehicle CG -> shifted by stage_1 length (1.00 m) to 0.35 m
            dry_run=True,
            export_best_cad=False,
        ),
    )
    hist_data = json.loads(res.history_json_path.read_text(encoding="utf-8"))
    assert hist_data["cg_source"] == "user_specified_shifted"
    assert hist_data["cg_x_m"] == pytest.approx(0.35, abs=1e-4)

    # Upper stage fins must remain untouched; booster fins must hold the optimized chord
    out_root = ET.fromstring(res.optimized_ork_path.read_text(encoding="utf-8"))
    stages_el = out_root.findall("./rocket/subcomponents/stage")
    upper_rc = float(stages_el[0].findtext(".//trapezoidfinset/rootchord"))
    booster_rc = float(stages_el[1].findtext(".//trapezoidfinset/rootchord"))
    assert upper_rc == pytest.approx(0.12)
    assert booster_rc == pytest.approx(res.best_eval.candidate.root_chord, abs=1e-5)

    # 2. Config mismatch resets stale --resume history; resume=False ignores baseline conv
    res_changed = run_fin_optimization(
        dup_ork,
        opt_dir,
        config=FinOptimizationConfig(
            max_sims=4,
            stage_selection="stage_2",
            target_fin_stage="booster",
            mach=1.40,
            min_static_margin_cal=1.5,
            cg_x_m=1.35,
            dry_run=True,
            export_best_cad=False,
            resume=True,
        ),
    )
    assert len(res_changed.evaluations) == 4
    assert res_changed.evaluations[0].cd_total != pytest.approx(res.evaluations[0].cd_total)

    base_conv_dir = opt_dir / "sim_01_baseline"
    base_conv_dir.mkdir(parents=True, exist_ok=True)
    (base_conv_dir / "mesh_convergence.json").write_text(
        json.dumps({"optimal_level": 1, "levels": [{"level": 1, "mesh_scale": 1.6}]}),
        encoding="utf-8",
    )
    res_no_resume = run_fin_optimization(
        dup_ork,
        opt_dir,
        config=FinOptimizationConfig(
            max_sims=3,
            stage_selection="stage_2",
            target_fin_stage="booster",
            mach=1.40,
            min_static_margin_cal=1.5,
            cg_x_m=1.35,
            dry_run=True,
            export_best_cad=False,
            resume=False,
        ),
    )
    assert res_no_resume.locked_mesh is None

    # 3. render_fluent_hdf5_contours must not leak matplotlib figures on HDF5 read errors
    rkt = parse_ork_file(synthetic_ork_file)
    bad_h5 = tmp_path / "missing.cas.h5"
    before_figs = set(plt.get_fignums())
    assert (
        render_fluent_hdf5_contours(
            bad_h5,
            bad_h5,
            tmp_path / "out.png",
            title="t",
            rocket=rkt,
            target_fin=rkt.all_fins[0],
            mach_inf=0.8,
            p_inf_pa=101325.0,
            rho_inf=1.225,
            v_inf=270.0,
        )
        is None
    )
    assert set(plt.get_fignums()) == before_figs
