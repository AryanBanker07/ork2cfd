"""Tests for fluid enclosure subtraction, symmetry modes, BOIs, and XCAF STEP export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ork2cfd.geometry import build_cfd_domain, export_step_with_names
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
