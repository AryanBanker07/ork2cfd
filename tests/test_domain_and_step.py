"""Tests for fluid enclosure subtraction, symmetry modes, BOIs, and XCAF STEP export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ork2cfd.geometry import build_cfd_domain, export_step_with_names, export_stl_with_regions
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
        DomainMode.SYMMETRY_SECTOR,
        DomainMode.AXISYMMETRIC_2D,
    ],
)
def test_all_domain_modes_and_step_export(
    synthetic_ork_file: Path,
    tmp_path: Path,
    mode: DomainMode,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    rec = recommend_solver(rocket, FlightCondition(mach=0.8))
    cad = CadConfig(step_schema="AP214")
    dom = DomainConfig(mode=mode, shape=EnclosureShape.BULLET, boi=BoiConfig(enabled=True))

    out = build_cfd_domain(rocket, cad, dom, rec)
    assert out.report.is_valid
    assert out.report.is_watertight
    assert out.report.free_edge_count == 0

    step_path = tmp_path / f"{mode.value}.step"
    man_path = tmp_path / f"{mode.value}_manifest.json"
    export_step_with_names(out, step_path, man_path, rocket, cad, dom)

    assert step_path.exists()
    assert man_path.exists()
    step_txt = step_path.read_text(encoding="utf-8", errors="replace")
    assert "SI_UNIT($,.METRE.)" in step_txt

    if mode != DomainMode.AXISYMMETRIC_2D:
        assert "rocket_nosecone" in step_txt
        assert "rocket_body" in step_txt
        assert "rocket_fins" in step_txt

    if mode in {DomainMode.FLUID_DOMAIN, DomainMode.SYMMETRY_HALF, DomainMode.SYMMETRY_SECTOR}:
        assert "farfield" in step_txt
        assert "outlet" in step_txt
        assert "boi_nearfield" in step_txt
        assert "boi_wake" in step_txt

    if mode in {DomainMode.SYMMETRY_HALF, DomainMode.SYMMETRY_SECTOR}:
        assert "symmetry" in step_txt

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    assert manifest["units"] == "m"
    assert manifest["validation"]["is_watertight"] is True


@pytest.mark.parametrize(
    "shape,expected_inlet",
    [
        (EnclosureShape.CYLINDER, True),
        (EnclosureShape.BOX, True),
    ],
)
def test_cylinder_and_box_enclosures(
    synthetic_ork_file: Path,
    shape: EnclosureShape,
    expected_inlet: bool,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    rec = recommend_solver(rocket, FlightCondition(mach=0.2))
    out = build_cfd_domain(
        rocket,
        CadConfig(),
        DomainConfig(mode=DomainMode.FLUID_DOMAIN, shape=shape, boi=BoiConfig(enabled=False)),
        rec,
    )
    assert out.report.is_watertight
    assert ("inlet" in out.report.named_selections) is expected_inlet
    assert "outlet" in out.report.named_selections
    assert "farfield" in out.report.named_selections


@pytest.mark.parametrize("mode", [DomainMode.OML_ONLY, DomainMode.FLUID_DOMAIN])
def test_stl_multi_region_export(
    synthetic_ork_file: Path,
    tmp_path: Path,
    mode: DomainMode,
) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    rec = recommend_solver(rocket, FlightCondition(mach=0.8))
    cad = CadConfig()
    dom = DomainConfig(mode=mode, shape=EnclosureShape.BULLET, boi=BoiConfig(enabled=False))

    out = build_cfd_domain(rocket, cad, dom, rec)
    stl_out, tri_count = export_stl_with_regions(out, tmp_path / f"{mode.value}.stl", cad)

    assert stl_out.exists()
    assert tri_count > 100
    text = stl_out.read_text(encoding="utf-8")
    assert text.count("facet normal ") == tri_count
    for region in ("rocket_nosecone", "rocket_body", "rocket_fins"):
        assert f"solid {region}\n" in text
        assert f"endsolid {region}\n" in text

    if mode == DomainMode.FLUID_DOMAIN:
        for boundary in ("outlet", "farfield"):
            assert f"solid {boundary}\n" in text
            assert f"endsolid {boundary}\n" in text

    xs = [float(line.split()[1]) for line in text.splitlines() if "vertex " in line]
    assert xs
    assert min(xs) >= out.bbox_domain_min_m[0] - 0.01
    assert max(xs) <= out.bbox_domain_max_m[0] + 0.01
