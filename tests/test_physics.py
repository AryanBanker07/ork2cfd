"""Unit tests for ISA atmosphere, y+ boundary layer sizing, and flow regime solver logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from ork2cfd.models import DomainMode, FlightCondition, FlowRegime, MeshPreset
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import (
    classify_regime,
    compute_atmosphere,
    compute_boundary_layer,
    recommend_solver,
)


def test_isa_sea_level_and_stratosphere() -> None:
    sl = compute_atmosphere(0.0)
    assert pytest.approx(sl.temperature_k, rel=1e-4) == 288.15
    assert pytest.approx(sl.pressure_pa, rel=1e-4) == 101325.0
    assert pytest.approx(sl.density_kg_m3, rel=1e-3) == 1.225
    assert pytest.approx(sl.speed_of_sound_m_s, rel=1e-3) == 340.29

    high = compute_atmosphere(15000.0)
    assert pytest.approx(high.temperature_k, rel=1e-4) == 216.65
    assert high.pressure_pa < sl.pressure_pa
    assert high.density_kg_m3 < sl.density_kg_m3


def test_boundary_layer_yplus_scaling(synthetic_ork_file: Path) -> None:
    rocket = parse_ork_file(synthetic_ork_file)
    bl_y1 = compute_boundary_layer(
        rocket, FlightCondition(mach=0.8, altitude_m=500.0, target_yplus=1.0)
    )
    bl_y30 = compute_boundary_layer(
        rocket, FlightCondition(mach=0.8, altitude_m=500.0, target_yplus=30.0)
    )

    assert bl_y1.reynolds_length > 1e6
    ratio = bl_y30.first_layer_thickness_m / bl_y1.first_layer_thickness_m
    assert pytest.approx(ratio, rel=1e-5) == 30.0
    assert 6 <= bl_y1.prism_layers <= 26

    # Symmetry sector reduces estimated cell count compared to full domain
    bl_sec = compute_boundary_layer(
        rocket,
        FlightCondition(mach=0.8, altitude_m=500.0, mesh_preset=MeshPreset.STUDENT_SAFE),
        domain_mode=DomainMode.SYMMETRY_SECTOR,
    )
    assert bl_sec.estimated_cell_count < bl_y1.estimated_cell_count


def test_regime_classification_and_solver_selection(synthetic_ork_file: Path) -> None:
    rocket = parse_ork_file(synthetic_ork_file)

    assert classify_regime(FlightCondition(mach=0.2)) == FlowRegime.SUBSONIC
    assert classify_regime(FlightCondition(mach=0.8)) == FlowRegime.TRANSONIC
    assert classify_regime(FlightCondition(mach=2.0)) == FlowRegime.SUPERSONIC

    rec_sub = recommend_solver(rocket, FlightCondition(mach=0.2))
    assert rec_sub.solver_type == "pressure-based"
    assert rec_sub.energy_enabled is False

    rec_trans = recommend_solver(rocket, FlightCondition(mach=0.9))
    assert rec_trans.solver_type == "pressure-based"
    assert rec_trans.energy_enabled is True
    assert rec_trans.fluid_model == "ideal-gas"

    rec_sup = recommend_solver(rocket, FlightCondition(mach=2.5))
    assert rec_sup.solver_type == "density-based"
    assert rec_sup.energy_enabled is True
    assert rec_sup.farfield_bc_type == "pressure-far-field"
