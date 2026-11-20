"""Flow-regime classification and Fluent solver configuration recommendations."""

from __future__ import annotations

import math

from ..models.cfd import FlightCondition, FlowRegime
from ..models.results import SolverRecommendation
from ..models.rocket import RocketModel


def classify_regime(flight: FlightCondition) -> FlowRegime:
    """Determine flow regime from override or free-stream Mach number."""
    if flight.regime_override != FlowRegime.AUTO:
        return flight.regime_override
    if flight.mach < 0.30:
        return FlowRegime.SUBSONIC
    if flight.mach < 1.20:
        return FlowRegime.TRANSONIC
    return FlowRegime.SUPERSONIC


def recommend_solver(rocket: RocketModel, flight: FlightCondition) -> SolverRecommendation:
    """Return regime-appropriate Fluent solver, turbulence, BC, and domain settings."""
    regime = classify_regime(flight)
    roughness_m = rocket.dominant_finish.roughness_m
    notes: list[str] = []

    if regime == FlowRegime.SUBSONIC:
        notes.append("Incompressible pressure-based coupled solver selected (Mach < 0.3).")
        return SolverRecommendation(
            regime=regime,
            solver_type="pressure-based",
            energy_enabled=False,
            fluid_model="constant",
            viscosity_model="constant",
            turbulence_model="k-omega-sst",
            compressibility_effects=False,
            roughness_height_m=roughness_m,
            inlet_bc_type="velocity-inlet",
            outlet_bc_type="pressure-outlet",
            farfield_bc_type="pressure-far-field",
            recommended_upstream_L=5.0,
            recommended_downstream_L=12.0,
            recommended_radial_L=8.0,
            courant_number=20.0,
            notes=tuple(notes),
        )

    if regime == FlowRegime.TRANSONIC:
        notes.append(
            "Compressible ideal-gas pressure-based coupled solver with energy equation "
            "selected for transonic shock/drag-rise fidelity."
        )
        return SolverRecommendation(
            regime=regime,
            solver_type="pressure-based",
            energy_enabled=True,
            fluid_model="ideal-gas",
            viscosity_model="sutherland",
            turbulence_model="k-omega-sst",
            compressibility_effects=True,
            roughness_height_m=roughness_m,
            inlet_bc_type="pressure-far-field",
            outlet_bc_type="pressure-far-field",
            farfield_bc_type="pressure-far-field",
            recommended_upstream_L=6.0,
            recommended_downstream_L=15.0,
            recommended_radial_L=10.0,
            courant_number=10.0,
            notes=tuple(notes),
        )

    # Supersonic
    mach = max(1.2, flight.mach)
    mu_angle = math.asin(1.0 / mach)
    # Ensure radial farfield clears Mach cone well downstream of the rocket
    mach_cone_radial = 1.25 * math.tan(mu_angle) * 6.0
    radial_l = max(6.0, min(14.0, round(mach_cone_radial, 1)))
    notes.append(
        f"Supersonic density-based implicit solver with ideal-gas + Sutherland viscosity "
        f"(Mach angle {math.degrees(mu_angle):.1f} deg)."
    )
    return SolverRecommendation(
        regime=regime,
        solver_type="density-based",
        energy_enabled=True,
        fluid_model="ideal-gas",
        viscosity_model="sutherland",
        turbulence_model="k-omega-sst",
        compressibility_effects=True,
        roughness_height_m=roughness_m,
        inlet_bc_type="pressure-far-field",
        outlet_bc_type="pressure-far-field",
        farfield_bc_type="pressure-far-field",
        recommended_upstream_L=3.0,
        recommended_downstream_L=10.0,
        recommended_radial_L=radial_l,
        courant_number=5.0,
        notes=tuple(notes),
    )
