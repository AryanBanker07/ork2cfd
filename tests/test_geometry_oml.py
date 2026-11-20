"""Geometry kernel tests: watertight B-Rep OML across fin profiles, cant, and real ORK files."""

from __future__ import annotations

from pathlib import Path

import pytest

from ork2cfd.geometry import build_rocket_oml
from ork2cfd.models import CadConfig
from ork2cfd.parser import parse_ork_file
from tests.conftest import REAL_ORK_CANDIDATES, make_synthetic_ork_xml


@pytest.mark.parametrize("cross_section", ["square", "rounded", "airfoil", "hexagonal"])
def test_oml_watertight_across_fin_cross_sections(tmp_path: Path, cross_section: str) -> None:
    p = tmp_path / f"cs_{cross_section}.ork"
    p.write_text(
        make_synthetic_ork_xml(fin_tag="trapezoidfinset", cross_section=cross_section),
        encoding="utf-8",
    )
    rocket = parse_ork_file(p)
    out = build_rocket_oml(rocket, CadConfig())

    assert out.oml_stats.is_valid
    assert out.oml_stats.is_watertight
    assert out.oml_stats.free_edge_count == 0
    assert out.oml_stats.solid_count == 1
    assert out.oml_stats.volume_m3 > out.body_stats.volume_m3


@pytest.mark.parametrize("fin_tag", ["ellipticalfinset", "freeformfinset"])
def test_oml_elliptical_and_freeform_fins(tmp_path: Path, fin_tag: str) -> None:
    p = tmp_path / f"{fin_tag}.ork"
    p.write_text(make_synthetic_ork_xml(fin_tag=fin_tag), encoding="utf-8")
    rocket = parse_ork_file(p)
    out = build_rocket_oml(rocket, CadConfig())

    assert out.oml_stats.is_watertight
    assert out.oml_stats.free_edge_count == 0
    assert out.oml_stats.volume_m3 > out.body_stats.volume_m3


def test_oml_canted_fins_and_rail_buttons(tmp_path: Path) -> None:
    p = tmp_path / "canted_rail.ork"
    p.write_text(
        make_synthetic_ork_xml(cant_deg=1.5, include_rail_button=True),
        encoding="utf-8",
    )
    rocket = parse_ork_file(p)
    out = build_rocket_oml(rocket, CadConfig(include_rail_buttons=True))

    assert out.oml_stats.is_watertight
    assert out.oml_stats.free_edge_count == 0
    assert out.oml_stats.volume_m3 > out.body_stats.volume_m3


@pytest.mark.parametrize("ork_path", [p for p in REAL_ORK_CANDIDATES if p.exists()])
def test_oml_real_team_ork_files(ork_path: Path) -> None:
    rocket = parse_ork_file(ork_path)
    out = build_rocket_oml(rocket, CadConfig())
    assert out.oml_stats.is_valid
    assert out.oml_stats.is_watertight
    assert out.oml_stats.free_edge_count == 0
    assert out.oml_stats.volume_m3 > 0.0
