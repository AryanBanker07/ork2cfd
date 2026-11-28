"""Unit tests for ISA atmosphere, y+ boundary layer sizing, and flow regime solver logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from ork2cfd.models import DomainMode, FlightCondition, FlowRegime, MeshPreset
from ork2cfd.parser import parse_ork_file
from ork2cfd.physics import (
    SystemHardware,
    classify_regime,
    compute_atmosphere,
    compute_boundary_layer,
    detect_system_hardware,
    recommend_resources,
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

    rec_sup_mod = recommend_solver(rocket, FlightCondition(mach=1.75))
    assert rec_sup_mod.regime == FlowRegime.SUPERSONIC
    assert rec_sup_mod.solver_type == "pressure-based"
    assert rec_sup_mod.energy_enabled is True
    assert rec_sup_mod.courant_number == pytest.approx(5.0)

    rec_sup = recommend_solver(rocket, FlightCondition(mach=2.5))
    assert rec_sup.solver_type == "density-based"
    assert rec_sup.energy_enabled is True
    assert rec_sup.farfield_bc_type == "pressure-far-field"
    assert rec_sup.courant_number == pytest.approx(2.0)


def test_hardware_detection_and_resource_recommendations(synthetic_ork_file: Path) -> None:
    hw_real = detect_system_hardware()
    assert hw_real.total_ram_gb >= 1.0
    assert 0.5 <= hw_real.available_ram_gb <= hw_real.total_ram_gb
    assert 1 <= hw_real.physical_cores <= hw_real.logical_threads

    rocket = parse_ork_file(synthetic_ork_file)
    hw_8 = SystemHardware(
        total_ram_gb=8.0, available_ram_gb=5.0, physical_cores=4, logical_threads=8
    )
    hw_15 = SystemHardware(
        total_ram_gb=15.4, available_ram_gb=11.0, physical_cores=8, logical_threads=16
    )
    hw_64 = SystemHardware(
        total_ram_gb=64.0, available_ram_gb=48.0, physical_cores=16, logical_threads=32
    )

    rec_8 = recommend_resources(
        hardware=hw_8,
        rocket=rocket,
        mach=0.8,
        solver_type="pressure-based",
        estimated_cells=600_000,
    )
    rec_15_pb = recommend_resources(
        hardware=hw_15,
        rocket=rocket,
        mach=1.75,
        solver_type="pressure-based",
        estimated_cells=600_000,
    )
    rec_15_db = recommend_resources(
        hardware=hw_15,
        rocket=rocket,
        mach=2.2,
        solver_type="density-based",
        estimated_cells=600_000,
    )
    rec_64 = recommend_resources(
        hardware=hw_64,
        rocket=rocket,
        mach=0.8,
        solver_type="pressure-based",
        estimated_cells=600_000,
    )

    assert rec_8.recommended_cores <= hw_8.physical_cores
    assert rec_15_pb.recommended_cores <= hw_15.physical_cores
    assert rec_64.recommended_cores <= hw_64.physical_cores
    assert rec_8.max_safe_cells < rec_15_pb.max_safe_cells < rec_64.max_safe_cells
    assert rec_15_db.max_safe_cells < rec_15_pb.max_safe_cells
    assert rec_15_pb.recommended_yplus == pytest.approx(30.0)
    assert rec_64.recommended_yplus == pytest.approx(1.0)
    assert rec_15_pb.recommended_domain_mode == DomainMode.SYMMETRY_SECTOR

    rec_ht = recommend_resources(
        hardware=hw_15,
        rocket=rocket,
        mach=0.8,
        solver_type="pressure-based",
        estimated_cells=2_000_000,
        requested_cores=16,
    )
    assert rec_ht.exceeds_ram_budget is True
    assert any("physical CPU cores" in w for w in rec_ht.warnings)
    assert any("exceeds allotted budget" in w for w in rec_ht.warnings)
