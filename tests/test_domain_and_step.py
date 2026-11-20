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
        DomainMode.SYMMETRY_HALF_SECTOR,
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

    if mode in {
        DomainMode.FLUID_DOMAIN,
        DomainMode.SYMMETRY_HALF,
        DomainMode.SYMMETRY_SECTOR,
        DomainMode.SYMMETRY_HALF_SECTOR,
    }:
        assert "farfield" in step_txt
        assert "outlet" in step_txt
        assert "boi_nearfield" in step_txt
        assert "boi_wake" in step_txt

    if mode in {
        DomainMode.SYMMETRY_HALF,
        DomainMode.SYMMETRY_SECTOR,
        DomainMode.SYMMETRY_HALF_SECTOR,
    }:
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


def _write_two_stage_ork(
    path: Path,
    fin_count_s1: int = 4,
    fin_count_s2: int = 4,
) -> Path:
    """Write a two-stage OpenRocket XML document with configurable stage fin counts."""
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<openrocket version="1.10" creator="OpenRocket 24.12">
  <rocket>
    <name>TwoStageTest</name>
    <subcomponents>
      <stage>
        <name>Sustainer</name>
        <subcomponents>
          <nosecone>
            <name>Nose</name>
            <length>0.40</length>
            <shape>ogive</shape>
            <shapeparameter>1.0</shapeparameter>
            <aftradius>0.05</aftradius>
          </nosecone>
          <bodytube>
            <name>Sustainer Tube</name>
            <length>0.80</length>
            <radius>0.05</radius>
            <subcomponents>
              <trapezoidfinset>
                <name>Sustainer Fins</name>
                <fincount>{fin_count_s1}</fincount>
                <position type="bottom">0.0</position>
                <thickness>0.004</thickness>
                <crosssection>square</crosssection>
                <rotation>0.0</rotation>
                <rootchord>0.20</rootchord>
                <tipchord>0.08</tipchord>
                <sweeplength>0.08</sweeplength>
                <height>0.10</height>
              </trapezoidfinset>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
      <stage>
        <name>Booster</name>
        <subcomponents>
          <bodytube>
            <name>Booster Tube</name>
            <length>0.70</length>
            <radius>0.05</radius>
            <subcomponents>
              <trapezoidfinset>
                <name>Booster Fins</name>
                <fincount>{fin_count_s2}</fincount>
                <position type="bottom">0.0</position>
                <thickness>0.004</thickness>
                <crosssection>square</crosssection>
                <rotation>0.0</rotation>
                <rootchord>0.24</rootchord>
                <tipchord>0.10</tipchord>
                <sweeplength>0.08</sweeplength>
                <height>0.12</height>
              </trapezoidfinset>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
    path.write_text(xml, encoding="utf-8")
    return path


def test_multistage_symmetry_quadrant_vs_half_sector_and_stage_selection(tmp_path: Path) -> None:
    from ork2cfd.ansys.script_generator import _symmetry_multiplier

    ork_4x4 = _write_two_stage_ork(tmp_path / "two_stage_4x4.ork", 4, 4)
    rocket = parse_ork_file(ork_4x4)
    assert rocket.is_multistage is True
    assert rocket.stage_count == 2
    assert rocket.rotational_symmetry_order == 4
    assert rocket.has_sector_mirror_symmetry is True

    rec = recommend_solver(rocket, FlightCondition(mach=0.8))
    cad = CadConfig()

    out_full = build_cfd_domain(
        rocket, cad, DomainConfig(mode=DomainMode.FLUID_DOMAIN, boi=BoiConfig(enabled=False)), rec
    )
    out_quad = build_cfd_domain(
        rocket,
        cad,
        DomainConfig(mode=DomainMode.SYMMETRY_SECTOR, boi=BoiConfig(enabled=False)),
        rec,
    )
    out_half_sec = build_cfd_domain(
        rocket,
        cad,
        DomainConfig(mode=DomainMode.SYMMETRY_HALF_SECTOR, boi=BoiConfig(enabled=False)),
        rec,
    )

    assert out_quad.report.is_watertight
    assert out_half_sec.report.is_watertight
    full_vol = out_full.report.domain_volume_m3
    assert out_quad.report.domain_volume_m3 == pytest.approx(full_vol / 4.0, rel=0.01)
    assert out_half_sec.report.domain_volume_m3 == pytest.approx(full_vol / 8.0, rel=0.01)
    assert _symmetry_multiplier(rocket, DomainMode.SYMMETRY_SECTOR) == 4.0
    assert _symmetry_multiplier(rocket, DomainMode.SYMMETRY_HALF_SECTOR) == 8.0

    # Per-stage selection & re-zeroed axial coordinates
    s1 = rocket.select_stages("stage_1")
    assert s1.stage_count == 1
    assert s1.total_length == pytest.approx(1.20)
    out_s1 = build_cfd_domain(
        s1, cad, DomainConfig(mode=DomainMode.OML_ONLY, boi=BoiConfig(enabled=False)), rec
    )
    assert out_s1.report.is_watertight

    s2 = rocket.select_stages("stage_2")
    assert s2.stage_count == 1
    assert s2.stages[0].axial_start == pytest.approx(0.0)
    assert s2.total_length == pytest.approx(0.70)
    out_s2 = build_cfd_domain(
        s2, cad, DomainConfig(mode=DomainMode.OML_ONLY, boi=BoiConfig(enabled=False)), rec
    )
    assert out_s2.report.is_watertight

    # Coprime 3-fin Sustainer + 4-fin Booster -> GCD = 1 -> safe 360° fallback + warning
    ork_3x4 = _write_two_stage_ork(tmp_path / "two_stage_3x4.ork", 3, 4)
    rkt_coprime = parse_ork_file(ork_3x4)
    assert rkt_coprime.rotational_symmetry_order == 1
    assert rkt_coprime.has_sector_mirror_symmetry is False
    out_coprime = build_cfd_domain(
        rkt_coprime,
        cad,
        DomainConfig(mode=DomainMode.SYMMETRY_SECTOR, boi=BoiConfig(enabled=False)),
        rec,
    )
    assert out_coprime.report.is_watertight
    assert any("rotational symmetry order = 1" in w for w in out_coprime.report.warnings)
    assert "symmetry" not in out_coprime.report.named_selections
