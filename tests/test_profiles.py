"""Unit tests for analytical OpenRocket curves, nose blunting, and meridian merging."""

from __future__ import annotations

from pathlib import Path

import pytest

from ork2cfd.geometry.profiles import (
    build_merged_meridian_points_mm,
    build_meridian_spans_mm,
    generate_component_profile_points,
)
from ork2cfd.models import BodyComponent, CadConfig
from ork2cfd.parser import parse_ork_file
from tests.conftest import make_synthetic_ork_xml


@pytest.mark.parametrize(
    "shape,param",
    [
        ("haack", 0.0),
        ("haack", 1.0 / 3.0),
        ("ogive", 0.0),
        ("ogive", 0.7),
        ("ellipsoid", 0.0),
        ("parabolic", 0.5),
        ("power", 0.5),
        ("conical", 0.0),
    ],
)
def test_nosecone_analytical_endpoints_and_monotonicity(shape: str, param: float) -> None:
    nose = BodyComponent(
        name="Nose",
        component_type="nosecone",
        length=0.5,
        fore_radius=0.0,
        aft_radius=0.06,
        shape=shape,
        shape_parameter=param,
        axial_start=0.0,
        axial_end=0.5,
    )
    pts = generate_component_profile_points(nose, num_points=60, global_coords=True)
    assert len(pts) == 61
    assert pytest.approx(pts[0][0], abs=1e-9) == 0.0
    assert pytest.approx(pts[0][1], abs=1e-6) == 0.0
    assert pytest.approx(pts[-1][0], abs=1e-9) == 0.5
    assert pytest.approx(pts[-1][1], abs=1e-5) == 0.06

    radii = [r for _x, r in pts]
    assert all(radii[i] <= radii[i + 1] + 1e-9 for i in range(len(radii) - 1))


def test_collinear_body_tubes_merged_into_single_span(tmp_path: Path) -> None:
    p = tmp_path / "multi_tube.ork"
    p.write_text(make_synthetic_ork_xml(), encoding="utf-8")
    rocket = parse_ork_file(p)

    spans, meta = build_meridian_spans_mm(rocket, CadConfig(tip_blunt_radius_m=0.001))
    # Nosecone (curved), merged Upper+Lower bodytube (straight), Boattail (straight) -> 3 spans!
    assert len(spans) == 3
    assert spans[0].is_curved is True
    assert spans[1].is_curved is False
    assert len(spans[1].points_mm) == 2
    assert pytest.approx(spans[1].points_mm[0][0], abs=1e-2) == 450.0
    assert pytest.approx(spans[1].points_mm[1][0], abs=1e-2) == 1850.0
    assert [m.zone_name for m in meta] == ["rocket_nosecone", "rocket_body", "rocket_boattail"]

    pts, _ = build_merged_meridian_points_mm(rocket, CadConfig(tip_blunt_radius_m=0.001))
    assert pts[0] == (0.0, 0.0)
    assert pytest.approx(pts[-1][0], abs=1e-2) == 1950.0
