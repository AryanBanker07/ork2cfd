"""Tests verifying syntax and physics injection in generated ANSYS scripts & journals."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ork2cfd.ansys.detector import AnsysInstallation
from ork2cfd.ansys.script_generator import generate_ansys_scripts
from ork2cfd.models import (
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    FlightCondition,
    PipelineConfig,
)
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import compute_boundary_layer, recommend_solver


@pytest.mark.parametrize(
    "mach,expected_solver",
    [
        (0.2, "pressure-based"),
        (0.85, "pressure-based"),
        (2.0, "density-based"),
    ],
)
def test_generated_scripts_syntax_and_physics(
    synthetic_ork_file: Path,
    tmp_path: Path,
    mach: float,
    expected_solver: str,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.SYMMETRY_SECTOR),
        flight=FlightCondition(mach=mach, altitude_m=1000.0, aoa_deg=2.0),
        deliverables=DeliverableConfig(archive_wbpz=True),
    )
    bl = compute_boundary_layer(rocket, cfg.flight, cfg.domain.mode)
    rec = recommend_solver(rocket, cfg.flight)

    fake_install = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=Path(r"C:\Program Files\ANSYS Inc\v252"),
        runwb2_exe=None,
        fluent_exe=None,
        discovery_exe=None,
    )

    step_p = tmp_path / "TestRocket_cfd.step"
    man_p = tmp_path / "TestRocket_manifest.json"
    step_p.write_text("ISO-10303-21;", encoding="utf-8")
    man_p.write_text("{}", encoding="utf-8")

    scripts = generate_ansys_scripts(
        output_dir=tmp_path,
        stem="TestRocket",
        rocket=rocket,
        config=cfg,
        bl=bl,
        solver_rec=rec,
        step_path=step_p,
        manifest_path=man_p,
        ansys_install=fake_install,
    )

    assert set(scripts.keys()) == {
        "run_fluent.py",
        "fluent_meshing.jou",
        "fluent_solver.jou",
        "discovery_prep.py",
        "create_project.wbjn",
        "README_RUN.md",
    }

    # All generated Python/IronPython scripts must parse cleanly with ast.parse
    for py_name in ("run_fluent.py", "discovery_prep.py", "create_project.wbjn"):
        src = scripts[py_name].read_text(encoding="utf-8")
        ast.parse(src, filename=py_name)

    run_src = scripts["run_fluent.py"].read_text(encoding="utf-8")
    assert f'SOLVER_TYPE = "{expected_solver}"' in run_src
    assert "IN_FLUENT_PY =" in run_src
    assert "ORK2CFD_MACH_LIST" in run_src
    assert '"OneZonePer": "face"' in run_src
    assert '"OneObjectPer": "Part"' in run_src
    assert "_ensure_zone_labels_from_manifest" in run_src
    assert "_update_boundaries_and_regions" in run_src
    assert '"selected-labels"' in run_src
    assert '"Cartesian (X, Y, Z)"' in run_src
    assert '"y-plus"' in run_src
    assert '"ImproveVolumeMesh"' in run_src
    assert ".pressure_far_field[zname].settings" not in run_src
    assert "SYM_FACTOR = 4.0" in run_src
    assert "FIN_FACE_SIZE_M =" in run_src
    assert "NOSE_FACE_SIZE_M =" in run_src
    assert "MACH_LIST =" in run_src
    assert "poly-hexcore" in run_src
    assert "FirstHeight" in run_src
    assert "cfd_summary.json" in run_src
    assert "sep_face_zone_by_angle" in run_src

    mod_ast = ast.parse(run_src, filename="run_fluent.py")
    ext_fn_node = next(
        n
        for n in mod_ast.body
        if isinstance(n, ast.FunctionDef) and n.name == "_extract_computed_scalar"
    )
    ns: dict[str, object] = {}
    exec(compile(ast.Module(body=[ext_fn_node], type_ignores=[]), "<ext>", "exec"), ns)
    extract_fn = ns["_extract_computed_scalar"]
    sample_rep = [{"cd_total": [0.62998, ""]}, {"wall_yplus_avg": [0.9415, ""]}]
    assert extract_fn(sample_rep, "cd_total") == pytest.approx(0.62998)
    assert extract_fn(sample_rep, "wall_yplus_avg") == pytest.approx(0.9415)

    cfg_half_sec = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.SYMMETRY_HALF_SECTOR),
        flight=cfg.flight,
        deliverables=cfg.deliverables,
    )
    scripts_hs = generate_ansys_scripts(
        output_dir=tmp_path / "half_sector",
        stem="TestRocket",
        rocket=rocket,
        config=cfg_half_sec,
        bl=compute_boundary_layer(rocket, cfg_half_sec.flight, cfg_half_sec.domain.mode),
        solver_rec=rec,
        step_path=step_p,
        manifest_path=man_p,
        ansys_install=fake_install,
    )
    assert "SYM_FACTOR = 8.0" in scripts_hs["run_fluent.py"].read_text(encoding="utf-8")

    wbjn_src = scripts["create_project.wbjn"].read_text(encoding="utf-8")
    assert 'SetScriptVersion(Version="25.2.170")' in wbjn_src
    assert 'GetTemplate(TemplateName="Fluid Flow with Fluent Meshing")' in wbjn_src
    assert 'Archive(FilePath=' in wbjn_src

    mesh_jou = scripts["fluent_meshing.jou"].read_text(encoding="utf-8")
    assert "FirstHeight" in mesh_jou
    assert '"uniform"' in mesh_jou
    assert "poly-hexcore" in mesh_jou

    solver_jou = scripts["fluent_solver.jou"].read_text(encoding="utf-8")
    assert expected_solver in solver_jou
    assert "/define/boundary-conditions/" in solver_jou
    assert "/report/forces/wall-forces" in solver_jou

    readme_run = scripts["README_RUN.md"].read_text(encoding="utf-8")
    assert "## 4. Next Steps" in readme_run
    assert "Wall Yplus" in readme_run
    assert "Results → Reports → Forces" in readme_run
    assert "CFD-Post" in readme_run
    assert "Instancing Transform" in readme_run
    assert "4×" in readme_run

    hs_readme = scripts_hs["README_RUN.md"].read_text(encoding="utf-8")
    assert "8×" in hs_readme


def test_ansys_detector_and_workbench_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from ork2cfd.ansys.detector import (
        AnsysInstallation,
        detect_ansys_installations,
        select_best_installation,
    )
    from ork2cfd.ansys.workbench_builder import build_workbench_project
    from ork2cfd.exceptions import AnsysError

    # Fake AWP_ROOT299 installation tree
    fake_root = tmp_path / "v299"
    runwb2 = fake_root / "Framework" / "bin" / "Win64" / "RunWB2.exe"
    runwb2.parent.mkdir(parents=True)
    runwb2.write_text("stub", encoding="utf-8")
    cfdpost = fake_root / "CFD-Post" / "bin" / "cfdpost.exe"
    cfdpost.parent.mkdir(parents=True)
    cfdpost.write_text("stub", encoding="utf-8")
    (fake_root / "Framework" / "bin" / "Win64" / "Ans.Core.Version.txt").write_text(
        "29.9.123", encoding="utf-8"
    )
    monkeypatch.setenv("AWP_ROOT299", str(fake_root))

    installs = detect_ansys_installations()
    assert any(inst.release_tag == "v299" for inst in installs)
    chosen = select_best_installation("29.9")
    assert chosen is not None and chosen.wb_script_version == "29.9.123"
    assert chosen.cfdpost_exe == cfdpost

    # Missing exe error
    no_exe = AnsysInstallation(
        version_code="29.9",
        release_tag="v299",
        wb_script_version="29.9.123",
        root_dir=fake_root,
        runwb2_exe=None,
        fluent_exe=None,
        discovery_exe=None,
    )
    wbjn = tmp_path / "proj.wbjn"
    wbpj = tmp_path / "proj.wbpj"
    wbpz = tmp_path / "proj.wbpz"
    with pytest.raises(AnsysError):
        build_workbench_project(wbjn, wbpj, no_exe)

    # Missing wbjn error
    with pytest.raises(AnsysError):
        build_workbench_project(wbjn, wbpj, chosen)

    wbjn.write_text("# wbjn", encoding="utf-8")

    # Mocked non-zero returncode
    monkeypatch.setattr(
        "ork2cfd.ansys.workbench_builder.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(AnsysError):
        build_workbench_project(wbjn, wbpj, chosen)

    # Mocked success creating .wbpj and .wbpz
    def fake_run(*args, **kwargs):
        wbpj.write_text("<Storage/>", encoding="utf-8")
        wbpz.write_bytes(b"PK\x03\x04wbpz")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ork2cfd.ansys.workbench_builder.subprocess.run", fake_run)
    out_pj, out_pz = build_workbench_project(wbjn, wbpj, chosen, expected_wbpz_path=wbpz)
    assert out_pj == wbpj
    assert out_pz == wbpz


def test_fluent_launcher_handles_missing_pyfluent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.ansys.fluent_launcher import launch_fluent_run

    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    meshing_jou = out_dir / "fluent_meshing.jou"
    meshing_jou.write_text("; mesh journal", encoding="utf-8")

    def _raise_missing(_name: str):
        raise ModuleNotFoundError("No module named 'ansys'")

    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 1234

    def _fake_popen(cmd, cwd=None, **_kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakeProc()

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

    monkeypatch.setattr("ork2cfd.ansys.fluent_launcher.importlib.util.find_spec", _raise_missing)
    monkeypatch.setattr("ork2cfd.ansys.fluent_launcher.subprocess.Popen", _fake_popen)

    # Falls back to fluent_meshing.jou when run_fluent.py is absent
    proc = launch_fluent_run(out_dir, ansys_install=install, cores=4, show_gui=False)
    assert proc.pid == 1234
    assert captured["cmd"] == [
        str(fluent_exe),
        "3ddp",
        "-meshing",
        "-t4",
        "-g",
        "-i",
        str(meshing_jou),
    ]

    # Prefers native Fluent -py -i run_fluent.py when run_fluent.py exists
    run_py = out_dir / "run_fluent.py"
    run_py.write_text("# native -py script", encoding="utf-8")
    proc2 = launch_fluent_run(out_dir, ansys_install=install, cores=4, show_gui=False)
    assert proc2.pid == 1234
    assert captured["cmd"] == [
        str(fluent_exe),
        "3ddp",
        "-meshing",
        "-t4",
        "-g",
        "-py",
        "-i",
        str(run_py),
    ]


def test_parse_cfd_results_and_visualizers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import json

    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.ansys.fluent_launcher import (
        find_solved_files,
        launch_cfdpost_visualization,
        launch_fluent_visualization,
        parse_cfd_results,
        run_fluent_solve_blocking,
    )
    from ork2cfd.exceptions import AnsysError

    out_dir = tmp_path / "cfd_pkg"
    out_dir.mkdir()
    (out_dir / "run_fluent.py").write_text("# runner", encoding="utf-8")
    cas = out_dir / "Rocket.cas.h5"
    dat = out_dir / "Rocket.dat.h5"
    cas_m15 = out_dir / "Rocket_M1.50.cas.h5"
    dat_m15 = out_dir / "Rocket_M1.50.dat.h5"
    mpng = out_dir / "Rocket_M0.80_mach.png"
    cas.write_text("cas", encoding="utf-8")
    dat.write_text("dat", encoding="utf-8")
    cas_m15.write_text("cas15", encoding="utf-8")
    dat_m15.write_text("dat15", encoding="utf-8")
    mpng.write_text("png", encoding="utf-8")
    found_cas, found_dat, found_pngs = find_solved_files(out_dir)
    assert found_cas == cas
    assert found_dat == dat
    assert found_pngs == [mpng]

    # 1. CSV fallback before cfd_summary.json exists
    (out_dir / "aero_results.csv").write_text(
        "rocket,mach,altitude_m,aoa_deg,zone,drag_force_n,cd\n"
        "Rocket,0.8,500.0,0.0,rocket_nosecone,100.0,0.18\n"
        "Rocket,0.8,500.0,0.0,rocket_fins,150.0,0.27\n"
        "Rocket,0.8,500.0,0.0,total,250.0,0.45\n",
        encoding="utf-8",
    )
    csv_parsed = parse_cfd_results(out_dir)
    assert csv_parsed is not None
    assert csv_parsed.primary_run is not None
    assert csv_parsed.primary_run.total_cd == pytest.approx(0.45)
    assert len(csv_parsed.primary_run.zones) == 2

    # 2. Blocking solve via native -py + cfd_summary.json parsing
    fluent_exe = tmp_path / "fluent.exe"
    cfdpost_exe = tmp_path / "cfdpost.exe"
    fluent_exe.write_text("stub", encoding="utf-8")
    cfdpost_exe.write_text("stub", encoding="utf-8")
    install = AnsysInstallation(
        version_code="25.2",
        release_tag="v252",
        wb_script_version="25.2.170",
        root_dir=tmp_path,
        runwb2_exe=None,
        fluent_exe=fluent_exe,
        discovery_exe=None,
        cfdpost_exe=cfdpost_exe,
    )

    summary_payload = {
        "rocket_name": "Rocket",
        "stem": "Rocket",
        "altitude_m": 500.0,
        "aoa_deg": 0.0,
        "sym_factor": 4.0,
        "s_ref_m2": 0.01131,
        "mesh_cells": 312000,
        "min_orth_quality": 0.24,
        "runs": [
            {
                "mach": 0.8,
                "velocity_m_s": 270.6,
                "total_cd": 0.4412,
                "total_drag_n": 244.5,
                "wall_yplus_avg": 1.15,
                "case_file": "Rocket.cas.h5",
                "data_file": "Rocket.dat.h5",
                "mach_contour_png": "Rocket_M0.80_mach.png",
                "pressure_contour_png": None,
                "zones": [
                    {"zone": "rocket_nosecone", "drag_force_n": 95.0, "cd": 0.1714},
                    {"zone": "rocket_fins", "drag_force_n": 149.5, "cd": 0.2698},
                ],
            }
        ],
    }

    calls: list[dict[str, object]] = []

    class _MockSolveProc:
        pid = 999
        stdout = io.StringIO(
            "[ork2cfd] Volume mesh total cell count: 312000\n"
            "[ork2cfd] Wrote cfd_summary.json\n"
        )

        def wait(self) -> int:
            (out_dir / "cfd_summary.json").write_text(
                json.dumps(summary_payload), encoding="utf-8"
            )
            return 0

    def _mock_popen(cmd, cwd=None, env=None, **_kwargs):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        return _MockSolveProc()

    monkeypatch.setattr("ork2cfd.ansys.fluent_launcher.subprocess.Popen", _mock_popen)
    milestones: list[str] = []
    solved = run_fluent_solve_blocking(
        out_dir, install, cores=6, mach_list=[0.8, 1.5], log_callback=milestones.append
    )
    assert solved.mesh_cells == 312000
    assert solved.primary_run is not None
    assert solved.primary_run.total_cd == pytest.approx(0.4412)
    assert solved.primary_run.wall_yplus_avg == pytest.approx(1.15)
    assert len(milestones) == 2
    assert calls[0]["cmd"] == [
        str(fluent_exe), "3ddp", "-meshing", "-g", "-t6", "-py", "-i", "run_fluent.py",
    ]
    assert calls[0]["env"]["ORK2CFD_MACH_LIST"] == "0.8,1.5"

    # 3. Direct Fluent Solution GUI & CFD-Post launchers
    launch_fluent_visualization(out_dir, install, cores=6)
    assert "-g" not in calls[1]["cmd"]
    assert "-meshing" not in calls[1]["cmd"]
    assert calls[1]["cmd"] == [str(fluent_exe), "3ddp", "-t6", "-i", "_open_viz.jou"]
    jou_text = (out_dir / "_open_viz.jou").read_text(encoding="utf-8")
    assert '/file/read-case-data "Rocket.cas.h5"' in jou_text

    launch_cfdpost_visualization(out_dir, install)
    assert calls[2]["cmd"] == [str(cfdpost_exe), "-results", str(dat.resolve())]

    # 4. Non-zero Fluent exit raises AnsysError
    class _FailProc:
        stdout = io.StringIO("Error: boom\n")

        def wait(self) -> int:
            return 2

    monkeypatch.setattr(
        "ork2cfd.ansys.fluent_launcher.subprocess.Popen", lambda *a, **k: _FailProc()
    )
    with pytest.raises(AnsysError):
        run_fluent_solve_blocking(out_dir, install, cores=4)


def test_default_cores_and_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from ork2cfd.models import DEFAULT_SOLVER_ITERATIONS, FlightCondition, default_fluent_cores

    expected_map = {None: 2, 1: 1, 2: 1, 4: 2, 8: 4, 16: 8, 32: 8}
    for cpu_val, expected in expected_map.items():
        monkeypatch.setattr(os, "cpu_count", lambda v=cpu_val: v)
        assert default_fluent_cores() == expected
        fc = FlightCondition()
        assert fc.cores == expected
        assert fc.iterations == DEFAULT_SOLVER_ITERATIONS == 300


def test_fluid_domain_360_fixes_in_generated_run_fluent(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from ork2cfd.models import (
        BoiConfig,
        CadConfig,
        DomainConfig,
        DomainMode,
        EnclosureShape,
        MeshPreset,
    )

    sample_ork = Path("sample orks/finalfinal947.ork")
    ork_input = sample_ork if sample_ork.exists() else synthetic_ork_file
    rocket = parse_ork_file(ork_input).select_stages("all")
    cfg = PipelineConfig(
        cad=CadConfig(tip_blunt_radius_m=0.0004, step_schema="AP214"),
        domain=DomainConfig(
            mode=DomainMode.FLUID_DOMAIN,
            shape=EnclosureShape.BULLET,
            boi=BoiConfig(enabled=True),
        ),
        flight=FlightCondition(
            mach=0.80,
            altitude_m=500.0,
            aoa_deg=0.0,
            target_yplus=1.0,
            mesh_preset=MeshPreset.COARSE,
        ),
        deliverables=DeliverableConfig(build_wbpj=False),
    )
    bl = compute_boundary_layer(rocket, cfg.flight, cfg.domain.mode)
    rec = recommend_solver(rocket, cfg.flight)

    step_p = tmp_path / "947_cfd.step"
    man_p = tmp_path / "947_manifest.json"
    step_p.write_text("ISO-10303-21;", encoding="utf-8")
    man_p.write_text("{}", encoding="utf-8")

    scripts = generate_ansys_scripts(
        output_dir=tmp_path,
        stem="947",
        rocket=rocket,
        config=cfg,
        bl=bl,
        solver_rec=rec,
        step_path=step_p,
        manifest_path=man_p,
    )

    run_src = scripts["run_fluent.py"].read_text(encoding="utf-8")
    ast.parse(run_src, filename="run_fluent.py")
    assert 'DOMAIN_MODE = "fluid_domain"' in run_src
    assert "CD_STOP_CRITERION = 0.0001" in run_src or "CD_STOP_CRITERION = 1e-4" in run_src
    assert "CD_STOP_WINDOW = 25" in run_src
    assert "_separate_remeshed_zones" in run_src
    assert "_classify_region_types" in run_src
    assert "_configure_cd_convergence" in run_src

    mesh_jou = scripts["fluent_meshing.jou"].read_text(encoding="utf-8")
    assert "The geometry consists of both fluid and solid regions and/or voids" in mesh_jou

    # Execute generated script in isolated namespace to test runtime helpers
    ns: dict[str, object] = {
        "__name__": "test_run_fluent",
        "__file__": str(scripts["run_fluent.py"]),
    }
    exec(compile(run_src, str(scripts["run_fluent.py"]), "exec"), ns)

    # 1. Multi-shell sep_face_zone_by_angle separation for 2 disconnected shells [1, 92]
    sep_calls: list[tuple[list[int], float]] = []

    class _FakeMU:
        def __init__(self, zids, lbls, lbl_map=None) -> None:
            self._zids = list(zids)
            self._lbls = list(lbls)
            self._lbl_map = lbl_map or {}
            self.regions = ["fluid_domain-fluid_domain", "dead0"]
            self.vols = {"fluid_domain-fluid_domain": 531586.0, "dead0": 0.047}

        def get_face_zones_of_object(self, object_name="", labels=None):
            if labels:
                return self._lbl_map.get(labels[0], [])
            return list(self._zids)

        def get_labels(self, object_name=""):
            return list(self._lbls)

        def get_regions(self, object_name="", region_name_pattern="*"):
            return list(self.regions)

        def get_region_volume(self, object_name="", region_name=""):
            return self.vols.get(region_name, 0.0)

    def _sep_angle(zlist, ang):
        sep_calls.append((list(zlist), float(ang)))
        if zlist == [1]:
            mu._zids.extend([101, 102])
        elif zlist == [92]:
            mu._zids.extend([103, 104, 105, 106, 107])

    fake_session = SimpleNamespace(
        tui=SimpleNamespace(
            boundary=SimpleNamespace(
                separate=SimpleNamespace(sep_face_zone_by_angle=_sep_angle)
            ),
            objects=SimpleNamespace(
                volumetric_regions=SimpleNamespace(change_type=lambda *_a: None)
            ),
        )
    )
    manifest = {
        "named_selections_counts": {
            "farfield": 2,
            "outlet": 1,
            "rocket_nosecone": 1,
            "rocket_body": 1,
            "rocket_boattail": 1,
            "rocket_fins": 40,
            "rocket_base": 1,
        }
    }
    mu = _FakeMU([1, 92], ["fluid_domain-fluid_domain"])
    after = ns["_separate_remeshed_zones"](
        fake_session, mu, manifest, "fluid_domain-fluid_domain", 15.0
    )
    assert sep_calls == [([1], 15.0), ([92], 15.0)]
    assert len(after) > 2

    # Skip separation once all expected labels exist and zone count covers manifest
    sep_calls.clear()
    all_lbls = sorted(manifest["named_selections_counts"].keys())
    mu_done = _FakeMU(
        list(range(10, 17)),
        all_lbls,
        {lbl: [10 + idx] for idx, lbl in enumerate(all_lbls)},
    )
    skipped = ns["_separate_remeshed_zones"](
        fake_session, mu_done, manifest, "fluid_domain-fluid_domain", 15.0
    )
    assert skipped == list(range(10, 17))
    assert sep_calls == []

    # 2. _classify_region_types always marks voids/BOIs/interior cavities as 'dead'
    classify = ns["_classify_region_types"]
    assert classify(["fluid_domain-fluid_domain", "dead0"]) == ["fluid", "dead"]
    assert classify(
        ["fluid_domain-fluid_domain", "dead0-fluid_domain-fluid_domain:1", "boi_nearfield"]
    ) == ["fluid", "dead", "dead"]
    assert classify(
        ["fluid_domain-fluid_domain", "fluid"],
        {"fluid_domain-fluid_domain": 531586.0, "fluid": 0.047},
    ) == ["fluid", "dead"]
    assert classify(["fluid_domain-fluid_domain", "fluid"]) == ["fluid", "dead"]

    # 3. _update_boundaries_and_regions void / no-void handling
    class _FakeArgs:
        def __init__(self, state: dict) -> None:
            self._state = dict(state)
            self.set_calls: list[dict] = []

        def __call__(self):
            return dict(self._state)

        def set_state(self, payload: dict) -> None:
            self.set_calls.append(dict(payload))
            self._state.update(payload)

    class _FakeTask:
        def __init__(self, state: dict, status: str = "forced-up-to-date") -> None:
            self.Arguments = _FakeArgs(state)
            self._status = status
            self.executed = 0

        def State(self) -> str:
            return self._status

        def Execute(self) -> bool:
            self.executed += 1
            return True

    def _make_wf(cr_status: str):
        return SimpleNamespace(
            TaskObject={
                "Update Boundaries": _FakeTask(
                    {
                        "SelectionType": "label",
                        "BoundaryLabelList": ["farfield", "rocket_body"],
                    }
                ),
                "Create Regions": _FakeTask({"NumberOfFlowVolumes": 1}, status=cr_status),
                "Update Regions": _FakeTask(
                    {
                        "RegionNameList": [],
                        "RegionCurrentList": ["fluid_domain-fluid_domain", "dead0"],
                    }
                ),
            }
        )

    fake_session.meshing_utilities = mu
    wf_void = _make_wf("out-of-date")
    ns["_update_boundaries_and_regions"](
        wf_void,
        True,
        verified_labels=["farfield", "rocket_body"],
        meshing_session=fake_session,
        remesh_obj="fluid_domain-fluid_domain",
        has_voids=True,
    )
    cr_task = wf_void.TaskObject["Create Regions"]
    assert cr_task.executed == 1
    assert {"NumberOfFlowVolumes": 0} in cr_task.Arguments.set_calls
    ur_task = wf_void.TaskObject["Update Regions"]
    assert ur_task.executed == 1
    assert ur_task.Arguments.set_calls[-1]["RegionNameList"] == [
        "fluid_domain-fluid_domain",
        "dead0",
    ]
    assert ur_task.Arguments.set_calls[-1]["RegionTypeList"] == ["fluid", "dead"]

    wf_novoid = _make_wf("out-of-date")
    ns["_update_boundaries_and_regions"](
        wf_novoid,
        True,
        verified_labels=["farfield", "rocket_body"],
        meshing_session=fake_session,
        remesh_obj="fluid_domain-fluid_domain",
        has_voids=False,
    )
    assert wf_novoid.TaskObject["Create Regions"].executed == 0

    # 4. _configure_cd_convergence sets 1e-4 / 25-iter window and recreates cleanly
    class _FakeConvItem:
        report_defs = None
        stop_criterion = None
        previous_values_to_consider = None
        initial_values_to_ignore = None
        active = None
        print = None

    class _FakeConvReports:
        def __init__(self) -> None:
            self.items: dict[str, _FakeConvItem] = {}
            self.deleted: list[list[str]] = []

        def get_object_names(self):
            return list(self.items.keys())

        def create(self, name: str):
            obj = _FakeConvItem()
            self.items[name] = obj
            return obj

        def delete(self, names: list[str]):
            self.deleted.append(list(names))
            for n in names:
                self.items.pop(n, None)

    class _FakeEnumField:
        def __init__(self, allowed):
            self._allowed = list(allowed)

        def allowed_values(self):
            return list(self._allowed)

    conv_reports = _FakeConvReports()
    conv_cond = SimpleNamespace(
        convergence_reports=conv_reports,
        frequency=0,
        condition=_FakeEnumField(["all-conditions-are-met", "any-condition-is-met"]),
    )
    res_opts = SimpleNamespace(criterion_type=_FakeEnumField(["absolute", "relative", "none"]))
    fake_sroot = SimpleNamespace(
        solution=SimpleNamespace(
            monitor=SimpleNamespace(
                convergence_conditions=conv_cond,
                residual=SimpleNamespace(options=res_opts),
            )
        )
    )

    ns["_configure_cd_convergence"](fake_sroot, 300)
    assert "cd_total_conv" in conv_reports.items
    conv_item = conv_reports.items["cd_total_conv"]
    assert conv_item.report_defs == "cd_total"
    assert conv_item.stop_criterion == pytest.approx(1e-4)
    assert conv_item.previous_values_to_consider == 25
    assert conv_item.initial_values_to_ignore == 50
    assert conv_item.active is True
    assert conv_item.print is True
    assert conv_cond.frequency == 1
    assert conv_cond.condition == "any-condition-is-met"
    assert res_opts.criterion_type == "none"

    # Re-calling for a second Mach point recreates cd_total_conv cleanly
    ns["_configure_cd_convergence"](fake_sroot, 300)
    assert conv_reports.deleted == [["cd_total_conv"]]
    assert "cd_total_conv" in conv_reports.items


def test_script_escaping_subsonic_bc_zone_matching_and_launcher_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import json

    from ork2cfd.ansys.detector import AnsysInstallation
    from ork2cfd.ansys.fluent_launcher import (
        launch_fluent_visualization,
        parse_cfd_results,
        run_fluent_solve_blocking,
    )
    from ork2cfd.exceptions import AnsysError
    from ork2cfd.models import BodyComponent, DomainConfig, DomainMode, RocketModel, Stage

    tricky_rocket = RocketModel(
        name='3" Nike\nBooster',
        stages=(
            Stage(
                name="Stage 1",
                components=(
                    BodyComponent(
                        name="Nose",
                        component_type="nosecone",
                        length=0.3,
                        fore_radius=0.0,
                        aft_radius=0.04,
                        axial_start=0.0,
                        axial_end=0.3,
                    ),
                ),
                axial_start=0.0,
                axial_end=0.3,
            ),
        ),
    )
    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.FLUID_DOMAIN),
        flight=FlightCondition(mach=0.20, altitude_m=500.0, aoa_deg=0.0),
        deliverables=DeliverableConfig(archive_wbpz=True),
    )
    bl = compute_boundary_layer(tricky_rocket, cfg.flight, cfg.domain.mode)
    rec = recommend_solver(tricky_rocket, cfg.flight)
    assert rec.farfield_bc_type == "velocity-inlet"

    step_p = tmp_path / "nike_cfd.step"
    man_p = tmp_path / "nike_manifest.json"
    step_p.write_text("ISO-10303-21;", encoding="utf-8")
    man_p.write_text(
        json.dumps({"named_selections_counts": {"farfield": 2, "outlet": 1, "rocket_body": 1}}),
        encoding="utf-8",
    )

    scripts = generate_ansys_scripts(
        output_dir=tmp_path,
        stem="nike",
        rocket=tricky_rocket,
        config=cfg,
        bl=bl,
        solver_rec=rec,
        step_path=step_p,
        manifest_path=man_p,
    )
    for py_name in ("run_fluent.py", "create_project.wbjn", "discovery_prep.py"):
        ast.parse(scripts[py_name].read_text(encoding="utf-8"), filename=py_name)

    run_src = scripts["run_fluent.py"].read_text(encoding="utf-8")
    assert "sys.exit(1)" in run_src
    assert "_clear_cd_convergence" in run_src

    ns: dict[str, object] = {"__name__": "test_sub", "__file__": str(scripts["run_fluent.py"])}
    exec(compile(run_src, str(scripts["run_fluent.py"]), "exec"), ns)
    match_fn = ns["_match_zones"]
    assert match_fn(["my_inlet_wall", "inlet:1", "inlet.2", "inlet"], "inlet") == [
        "inlet",
        "inlet.2",
        "inlet:1",
    ]

    mesh_jou = scripts["fluent_meshing.jou"].read_text(encoding="utf-8")
    assert "rocket_boattail" not in mesh_jou
    assert "rocket_fins" not in mesh_jou
    assert "Update Regions" in mesh_jou and "dead0" in mesh_jou

    # Stale cfd_summary.json must be removed before run so failed Fluent cannot replay stale Cd
    pkg = tmp_path / "stale_pkg"
    pkg.mkdir()
    (pkg / "run_fluent.py").write_text("# runner", encoding="utf-8")
    stale_sum = pkg / "cfd_summary.json"
    stale_sum.write_text(
        json.dumps({
            "rocket_name": "R",
            "runs": [{"mach": 0.8, "velocity_m_s": 270.0, "total_cd": 0.5, "total_drag_n": 100.0}],
        }),
        encoding="utf-8",
    )
    fluent_exe = tmp_path / "fluent.exe"
    fluent_exe.write_text("stub", encoding="utf-8")
    install = AnsysInstallation("25.2", "v252", "25.2.170", tmp_path, None, fluent_exe, None)

    class _CrashZeroProc:
        stdout = io.StringIO("RuntimeError: boom\n")

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(
        "ork2cfd.ansys.fluent_launcher.subprocess.Popen", lambda *a, **k: _CrashZeroProc()
    )
    with pytest.raises(AnsysError):
        run_fluent_solve_blocking(pkg, install, cores=2)
    assert not stale_sum.exists()

    # Nested case path in launch_fluent_visualization + diverged flag in parse_cfd_results
    nested_dir = pkg / "sim_03_bayesian_ei"
    nested_dir.mkdir()
    nested_cas = nested_dir / "Rocket.cas.h5"
    nested_dat = nested_dir / "Rocket.dat.h5"
    nested_cas.write_text("cas", encoding="utf-8")
    nested_dat.write_text("dat", encoding="utf-8")
    monkeypatch.setattr(
        "ork2cfd.ansys.fluent_launcher.subprocess.Popen",
        lambda *_a, **_k: io.StringIO(""),
    )
    launch_fluent_visualization(pkg, install, case_path=nested_cas, cores=2)
    assert (pkg / "_open_viz.jou").read_text(encoding="utf-8").strip() == (
        '/file/read-case-data "sim_03_bayesian_ei/Rocket.cas.h5"'
    )

    stale_sum.write_text(
        json.dumps({
            "rocket_name": "R",
            "runs": [
                {
                    "mach": 1.2,
                    "velocity_m_s": 400.0,
                    "total_cd": -5.0,
                    "total_drag_n": -900.0,
                    "diverged": True,
                }
            ],
        }),
        encoding="utf-8",
    )
    parsed = parse_cfd_results(pkg)
    assert parsed is not None and parsed.primary_run is not None
    assert parsed.primary_run.diverged is True
