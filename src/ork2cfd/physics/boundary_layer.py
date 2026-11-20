"""Boundary-layer y+ sizing, Reynolds estimation, and CFD mesh budget calculator."""

from __future__ import annotations

import math

from ..models.cfd import DomainMode, FlightCondition, MeshPreset
from ..models.results import BoundaryLayerReport
from ..models.rocket import RocketModel
from .atmosphere import compute_atmosphere

_PRESET_FACTORS: dict[MeshPreset, tuple[float, float, float, float, float]] = {
    # (surf_min/D, surf_max/D, nearfield_boi/D, wake_boi/D, farfield_max/L)
    MeshPreset.COARSE: (0.012, 0.120, 0.180, 0.450, 1.50),
    MeshPreset.STUDENT_SAFE: (0.009, 0.090, 0.140, 0.350, 1.25),
    MeshPreset.STANDARD: (0.005, 0.045, 0.065, 0.160, 0.85),
    MeshPreset.FINE: (0.0025, 0.022, 0.032, 0.080, 0.50),
}


def compute_boundary_layer(
    rocket: RocketModel,
    flight: FlightCondition,
    domain_mode: DomainMode = DomainMode.FLUID_DOMAIN,
) -> BoundaryLayerReport:
    """Calculate Reynolds numbers, first prism layer height for target y+, and cell budget."""
    atmo = compute_atmosphere(flight.altitude_m)
    mach = max(0.01, flight.mach)
    velocity = mach * atmo.speed_of_sound_m_s

    length_ref = max(0.05, rocket.total_length)
    diam_ref = max(0.01, rocket.max_diameter)

    re_len = max(100.0, (atmo.density_kg_m3 * velocity * length_ref) / atmo.viscosity_pa_s)
    re_diam = (atmo.density_kg_m3 * velocity * diam_ref) / atmo.viscosity_pa_s

    # Schlichting turbulent flat-plate skin friction + compressibility correction
    log_re = max(2.0, math.log10(re_len))
    cf_inc = (2.0 * log_re - 0.65) ** -2.3
    comp_factor = (1.0 + 0.144 * mach * mach) ** -0.65
    cf = cf_inc * comp_factor

    tau_w = 0.5 * atmo.density_kg_m3 * (velocity**2) * cf
    u_tau = math.sqrt(max(1e-12, tau_w / atmo.density_kg_m3))

    target_yplus = max(0.1, flight.target_yplus)
    y_center = (target_yplus * atmo.viscosity_pa_s) / (atmo.density_kg_m3 * u_tau)
    delta_y1 = 2.0 * y_center

    # Turbulent boundary-layer thickness estimate and prism stack calculation
    delta_99 = 0.37 * length_ref / (re_len**0.2)
    growth_rate = 1.20
    ratio_term = 1.0 + (delta_99 * (growth_rate - 1.0)) / max(delta_y1, 1e-9)
    n_raw = math.ceil(math.log(max(1.1, ratio_term)) / math.log(growth_rate))
    prism_layers = max(6, min(26, n_raw))

    f_min, f_max, f_nf, f_wake, f_far = _PRESET_FACTORS[flight.mesh_preset]
    surf_min = max(0.0002, f_min * diam_ref)
    surf_max = max(surf_min * 2.5, f_max * diam_ref)
    boi_nf = max(surf_max, f_nf * diam_ref)
    boi_wake = max(boi_nf * 1.5, f_wake * diam_ref)
    farfield_max = max(boi_wake * 2.0, f_far * length_ref)

    estimated_cells = _estimate_cell_count(
        rocket=rocket,
        domain_mode=domain_mode,
        surf_max=surf_max,
        prism_layers=prism_layers,
        boi_nf=boi_nf,
        boi_wake=boi_wake,
    )
    exceeds_student = estimated_cells > 512_000

    warnings: list[str] = []
    if 5.0 < target_yplus < 30.0:
        warnings.append(
            f"Target y+ = {target_yplus:.1f} falls in the buffer layer (5..30); "
            "prefer y+ <= 1 (wall-resolved) or y+ >= 30 (wall functions)."
        )
    if exceeds_student:
        warnings.append(
            f"Estimated mesh (~{estimated_cells:,} cells) may exceed the ANSYS Student "
            "512k CFD cell limit; consider 'Student Safe' preset or Symmetry Sector mode."
        )

    return BoundaryLayerReport(
        atmosphere=atmo,
        velocity_m_s=velocity,
        mach=mach,
        reynolds_length=re_len,
        reynolds_diameter=re_diam,
        skin_friction_cf=cf,
        wall_shear_pa=tau_w,
        friction_velocity_m_s=u_tau,
        target_yplus=target_yplus,
        first_cell_center_m=y_center,
        first_layer_thickness_m=delta_y1,
        bl_thickness_99_m=delta_99,
        prism_layers=prism_layers,
        prism_growth_rate=growth_rate,
        surface_min_size_m=surf_min,
        surface_max_size_m=surf_max,
        boi_nearfield_size_m=boi_nf,
        boi_wake_size_m=boi_wake,
        farfield_max_size_m=farfield_max,
        estimated_cell_count=estimated_cells,
        exceeds_student_limit=exceeds_student,
        warnings=tuple(warnings),
    )


def _estimate_cell_count(
    rocket: RocketModel,
    domain_mode: DomainMode,
    surf_max: float,
    prism_layers: int,
    boi_nf: float,
    boi_wake: float,
) -> int:
    """Heuristic pre-meshing volume cell estimate for user budget guidance."""
    if domain_mode == DomainMode.AXISYMMETRIC_2D:
        return int(max(15_000, (rocket.total_length / surf_max) * 180))

    sector_scale = 1.0
    if domain_mode == DomainMode.SYMMETRY_HALF:
        sector_scale = 0.5
    elif domain_mode == DomainMode.SYMMETRY_SECTOR:
        sector_scale = 1.0 / (2.0 * max(1, rocket.dominant_fin_count))

    d = max(0.02, rocket.max_diameter)
    l_tot = max(0.1, rocket.total_length)
    fin_area = sum(
        0.5 * (f.root_chord + f.tip_chord) * f.span * 2.0 * f.fin_count
        for f in rocket.all_fins
    )
    wetted_area = math.pi * d * l_tot + fin_area
    surface_faces = max(2_000, int((wetted_area / (surf_max * surf_max)) * 1.4))
    prism_cells = surface_faces * prism_layers

    # Volume inside nearfield & wake BOIs
    r_nf = max(d, 0.5 * rocket.max_fin_span_diameter) * 1.6
    v_nf = math.pi * (r_nf**2) * (1.8 * l_tot)
    v_wake = math.pi * ((3.0 * d) ** 2) * (4.0 * l_tot)
    boi_cells = int(v_nf / (boi_nf**3) * 0.4 + v_wake / (boi_wake**3) * 0.35)

    total = int((prism_cells + boi_cells) * sector_scale * 1.25)
    return max(25_000, total)
