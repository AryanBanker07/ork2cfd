"""Tests for CAD tessellation, quantitative verification, offline WebGL viewer, and 4-view PNG."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from ork2cfd.cli import main as cli_main
from ork2cfd.geometry import (
    build_cfd_domain,
    export_cad_viewer_html,
    export_cad_views_png,
    export_step_with_names,
    load_step_for_verification,
    tessellate_domain_zones,
    verify_cad_model,
    write_verification_reports,
)
from ork2cfd.models import (
    BoiConfig,
    CadConfig,
    DomainConfig,
    DomainMode,
    EnclosureShape,
    FlightCondition,
)
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import recommend_solver


@pytest.mark.parametrize(
    "mode",
    [
        DomainMode.OML_ONLY,
        DomainMode.FLUID_DOMAIN,
        DomainMode.SYMMETRY_HALF,
        DomainMode.AXISYMMETRIC_2D,
    ],
)
def test_tessellation_and_verification_modes(
    synthetic_ork_file: Path,
    tmp_path: Path,
    mode: DomainMode,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    rec = recommend_solver(rocket, FlightCondition(mach=0.8))
    cad = CadConfig()
    dom = DomainConfig(mode=mode, shape=EnclosureShape.BULLET, boi=BoiConfig(enabled=True))

    out = build_cfd_domain(rocket, cad, dom, rec)
    step_path = tmp_path / f"{mode.value}.step"
    man_path = tmp_path / f"{mode.value}_manifest.json"
    export_step_with_names(out, step_path, man_path, rocket, cad, dom)

    zones = tessellate_domain_zones(out, relative=True, include_bois=True)
    assert zones

    report = verify_cad_model(
        domain_out=out,
        zone_meshes=zones,
        rocket=rocket,
        cad_config=cad,
        domain_config=dom,
        step_path=step_path,
    )
    assert report.overall_status in {"PASS", "WARN"}
    assert report.step_roundtrip_ok is True

    by_name = {c.name: c.status for c in report.checks}
    assert by_name["occt_brep_validity"] == "PASS"
    assert by_name["watertight_manifold"] == "PASS"
    assert by_name["named_selections_complete"] == "PASS"
    assert by_name["step_file_roundtrip"] == "PASS"


def test_viewer_html_png_and_reports_generation(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    rec = recommend_solver(rocket, FlightCondition(mach=0.8))
    cad = CadConfig()
    dom = DomainConfig(mode=DomainMode.FLUID_DOMAIN, shape=EnclosureShape.BULLET)

    out = build_cfd_domain(rocket, cad, dom, rec)
    step_path = tmp_path / "rocket_cfd.step"
    man_path = tmp_path / "rocket_manifest.json"
    export_step_with_names(out, step_path, man_path, rocket, cad, dom)

    zones = tessellate_domain_zones(out, relative=True, include_bois=True)
    report = verify_cad_model(
        domain_out=out,
        zone_meshes=zones,
        rocket=rocket,
        cad_config=cad,
        domain_config=dom,
        step_path=step_path,
    )

    html_path = export_cad_viewer_html(
        zones, report, tmp_path / "rocket_cad_viewer.html", rocket.name, out, rocket=rocket
    )
    png_path = export_cad_views_png(
        zones, report, tmp_path / "rocket_cad_views.png", rocket.name, out, rocket=rocket
    )
    json_path, md_path = write_verification_reports(
        report,
        tmp_path / "rocket_verification.json",
        tmp_path / "rocket_verification.md",
        rocket.name,
        step_path=step_path,
        viewer_html_path=html_path,
        views_png_path=png_path,
    )

    html_text = html_path.read_text(encoding="utf-8")
    assert "webgl" in html_text
    assert "uClipAxis" in html_text
    assert "rocket_nosecone" in html_text
    assert "rocket_fins" in html_text
    assert "farfield" in html_text
    assert '<script src="http' not in html_text.lower()

    png_bytes = png_path.read_bytes()
    assert len(png_bytes) > 10_000
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["overall_status"] == report.overall_status
    assert len(parsed["checks"]) == len(report.checks)
    assert "PASS" in md_path.read_text(encoding="utf-8")


def test_standalone_step_load_and_verify(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    rec = recommend_solver(rocket, FlightCondition(mach=0.8))
    cad = CadConfig()
    dom = DomainConfig(mode=DomainMode.FLUID_DOMAIN, shape=EnclosureShape.BULLET)

    out = build_cfd_domain(rocket, cad, dom, rec)
    with_man_dir = tmp_path / "with_manifest"
    with_man_dir.mkdir()
    step_path = with_man_dir / "TestRocket_cfd.step"
    man_path = with_man_dir / "TestRocket_manifest.json"
    export_step_with_names(out, step_path, man_path, rocket, cad, dom)

    loaded = load_step_for_verification(step_path)
    assert loaded.report.is_watertight
    assert loaded.boi_solids
    assert "rocket_nosecone" in loaded.report.named_selections
    zones = tessellate_domain_zones(loaded)
    rep = verify_cad_model(loaded, zones, step_path=step_path)
    assert rep.overall_status in {"PASS", "WARN"}

    # Without sibling manifest
    no_man_dir = tmp_path / "no_manifest"
    no_man_dir.mkdir()
    standalone_step = no_man_dir / "standalone.step"
    shutil.copy2(step_path, standalone_step)
    loaded_bare = load_step_for_verification(standalone_step)
    assert loaded_bare.report.is_watertight
    zones_bare = tessellate_domain_zones(loaded_bare)
    rep_bare = verify_cad_model(loaded_bare, zones_bare, step_path=standalone_step)
    assert rep_bare.overall_status in {"PASS", "WARN"}


def test_cli_verify_ork_and_step(
    synthetic_ork_file: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    ork_out = tmp_path / "verify_ork"
    res_ork = runner.invoke(
        cli_main,
        ["verify", str(synthetic_ork_file), "-o", str(ork_out), "--mode", "oml_only"],
    )
    assert res_ork.exit_code == 0
    assert (ork_out / "TestRocket_cad_viewer.html").exists()
    assert (ork_out / "TestRocket_cad_views.png").exists()
    assert (ork_out / "TestRocket_verification.json").exists()

    step_file = ork_out / "TestRocket_cfd.step"
    step_out = tmp_path / "verify_step"
    res_step = runner.invoke(
        cli_main,
        ["verify", str(step_file), "-o", str(step_out)],
    )
    assert res_step.exit_code == 0
    assert (step_out / "TestRocket_cfd_cad_viewer.html").exists()
    assert (step_out / "TestRocket_cfd_cad_views.png").exists()
    assert (step_out / "TestRocket_cfd_verification.json").exists()
