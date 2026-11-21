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
    assert "SYM_FACTOR = 4.0" in run_src
    assert "FIN_FACE_SIZE_M =" in run_src
    assert "NOSE_FACE_SIZE_M =" in run_src
    assert "MACH_LIST =" in run_src
    assert "poly-hexcore" in run_src
    assert "FirstHeight" in run_src

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
    (fake_root / "Framework" / "bin" / "Win64" / "Ans.Core.Version.txt").write_text(
        "29.9.123", encoding="utf-8"
    )
    monkeypatch.setenv("AWP_ROOT299", str(fake_root))

    installs = detect_ansys_installations()
    assert any(inst.release_tag == "v299" for inst in installs)
    chosen = select_best_installation("29.9")
    assert chosen is not None and chosen.wb_script_version == "29.9.123"

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
