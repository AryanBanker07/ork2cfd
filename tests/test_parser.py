"""Unit tests for OpenRocket (.ork) parser across XML, ZIP, GZIP, and real fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ork2cfd.exceptions import OrkParseError
from ork2cfd.models import FinCrossSection, SurfaceFinish
from ork2cfd.parser import parse_ork_file
from tests.conftest import REAL_ORK_CANDIDATES, make_synthetic_ork_xml


def test_parse_plain_zip_and_gzip_formats(
    synthetic_ork_file: Path,
    synthetic_zip_ork_file: Path,
    synthetic_gzip_ork_file: Path,
) -> None:
    r_plain = parse_ork_file(synthetic_ork_file)
    r_zip = parse_ork_file(synthetic_zip_ork_file)
    r_gz = parse_ork_file(synthetic_gzip_ork_file)

    assert r_plain.name == "TestRocket"
    assert r_zip.name == "ZipRocket"
    assert r_gz.name == "GzRocket"

    # Motorconfiguration dummy <stage number="0"/> must NOT create empty/extra stages
    assert len(r_plain.stages) == 1
    assert len(r_plain.all_components) == 4
    assert pytest.approx(r_plain.total_length, rel=1e-6) == 1.95
    assert pytest.approx(r_plain.max_diameter, rel=1e-6) == 0.10

    # Auto radius inheritance on NoseCone and Transition
    nose = r_plain.all_components[0]
    boattail = r_plain.all_components[3]
    assert pytest.approx(nose.aft_radius, rel=1e-6) == 0.05
    assert pytest.approx(boattail.fore_radius, rel=1e-6) == 0.05
    assert pytest.approx(boattail.aft_radius, rel=1e-6) == 0.035

    # Rail buttons parsed
    upper = r_plain.all_components[1]
    assert len(upper.rail_buttons) == 1
    assert upper.rail_buttons[0].instance_count == 2


def test_parse_fin_types_cross_sections_and_offsets(tmp_path: Path) -> None:
    for ftag, cs, cant in [
        ("trapezoidfinset", "airfoil", 1.5),
        ("ellipticalfinset", "rounded", 0.0),
        ("freeformfinset", "hexagonal", 0.0),
    ]:
        p = tmp_path / f"{ftag}.ork"
        p.write_text(
            make_synthetic_ork_xml(fin_tag=ftag, cross_section=cs, cant_deg=cant),
            encoding="utf-8",
        )
        rocket = parse_ork_file(p)
        fin = rocket.all_fins[0]
        if cs == "airfoil":
            assert fin.cross_section == FinCrossSection.AIRFOIL
            assert pytest.approx(fin.cant_deg) == 1.5
        elif cs == "rounded":
            assert fin.cross_section == FinCrossSection.ROUNDED
            # Bottom offset -0.02 shifts TE upstream by 2 cm
            assert pytest.approx(fin.root_te_x, rel=1e-6) == 0.80 - 0.02
        else:
            assert fin.cross_section == FinCrossSection.DOUBLE_WEDGE
            assert len(fin.points) == 4


def test_parse_missing_or_corrupt_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OrkParseError):
        parse_ork_file(tmp_path / "does_not_exist.ork")

    bad = tmp_path / "bad.ork"
    bad.write_text("not xml at all <<<", encoding="utf-8")
    with pytest.raises(OrkParseError):
        parse_ork_file(bad)


def test_surface_finish_roughness() -> None:
    assert SurfaceFinish.POLISHED.roughness_m < SurfaceFinish.SMOOTH.roughness_m
    assert SurfaceFinish.SMOOTH.roughness_m < SurfaceFinish.NORMAL.roughness_m
    assert SurfaceFinish.NORMAL.roughness_m < SurfaceFinish.ROUGH.roughness_m


@pytest.mark.parametrize("ork_path", [p for p in REAL_ORK_CANDIDATES if p.exists()])
def test_parse_real_team_ork_files(ork_path: Path) -> None:
    rocket = parse_ork_file(ork_path)
    assert rocket.total_length > 1.0
    assert rocket.max_diameter > 0.05
    assert len(rocket.all_fins) >= 1
    assert rocket.reference_area > 0.0
