"""Analytical aerodynamic stability (Barrowman CP) and structural fin constraints."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from ..models.rocket import BodyComponent, FinSet, RocketModel
from ..parser.ork_parser import _clean_float, _load_xml_root, _resolve_axial_span
from ..physics.atmosphere import compute_atmosphere
from .models import FinCandidate, FinOptimizationConfig, StabilityReport

_TYPICAL_PROPELLANT_DENSITY_KG_M3 = 1550.0
_SHELL_AREAL_DENSITY_KG_M2 = 0.0025 * 1600.0


def compute_barrowman_cp(rocket: RocketModel) -> tuple[float, float]:
    """Compute subsonic Barrowman center of pressure X_cp (m) and total C_Na (/rad)."""
    s_ref = max(rocket.reference_area, 1e-9)
    d_ref = max(rocket.max_diameter, 1e-6)
    cna_total = 0.0
    moment_total = 0.0

    for comp in rocket.all_components:
        if comp.component_type in {"nosecone", "transition"} and comp.length > 0.0:
            a1 = math.pi * comp.fore_radius**2
            a2 = math.pi * comp.aft_radius**2
            da = a2 - a1
            if abs(da) > 1e-12:
                n_steps = 200
                dx = comp.length / n_steps
                vol = sum(
                    math.pi * (comp.radius_at_local_x((i + 0.5) * dx) ** 2) * dx
                    for i in range(n_steps)
                )
                cna_comp = 2.0 * da / s_ref
                xcp_comp = comp.axial_start + (comp.length * a2 - vol) / da
                cna_total += cna_comp
                moment_total += cna_comp * xcp_comp

        for fin in comp.fins:
            if fin.fin_count <= 0 or fin.span <= 0.0 or fin.root_chord <= 0.0:
                continue
            x_mid = 0.5 * (max(0.0, fin.root_le_x) + min(comp.length, fin.root_te_x))
            r_body = comp.radius_at_local_x(x_mid)
            k_fb = 1.0 + r_body / max(fin.span + r_body, 1e-9)
            chord_sum = max(fin.root_chord + fin.tip_chord, 1e-9)
            mid_chord_offset = fin.sweep_length + 0.5 * (fin.tip_chord - fin.root_chord)
            l_fin = math.hypot(fin.span, mid_chord_offset)
            denom = 1.0 + math.sqrt(1.0 + (2.0 * l_fin / chord_sum) ** 2)
            cna_fin = k_fb * (4.0 * fin.fin_count * (fin.span / d_ref) ** 2) / denom
            xcp_fin = (
                fin.global_root_le_x
                + (fin.sweep_length / 3.0) * (fin.root_chord + 2.0 * fin.tip_chord) / chord_sum
                + (1.0 / 6.0)
                * (fin.root_chord + fin.tip_chord - (fin.root_chord * fin.tip_chord) / chord_sum)
            )
            cna_total += cna_fin
            moment_total += cna_fin * xcp_fin

    if abs(cna_total) < 1e-9:
        return 0.5 * rocket.total_length, 0.0
    return moment_total / cna_total, cna_total


def _local_component_centroid(
    elem: ET.Element,
    parent_length: float,
    parent_start_x: float,
    default_len: float,
) -> float:
    """Resolve global axial centroid of an internal OpenRocket subcomponent."""
    pos_el = elem.find("position")
    if pos_el is None:
        pos_el = elem.find("axialoffset")
    method = "top"
    offset = 0.0
    if pos_el is not None:
        method = (pos_el.attrib.get("type") or pos_el.attrib.get("method") or "top").strip().lower()
        offset = _clean_float(pos_el.text, 0.0)
    sub_len = max(0.0, _clean_float(elem.findtext("length"), default_len))
    le_x, te_x = _resolve_axial_span(method, offset, sub_len, parent_length, parent_start_x)
    return parent_start_x + 0.5 * (le_x + te_x)


def estimate_cg_from_ork(ork_path: Path, stage_selection: str = "all") -> tuple[float, str]:
    """
    Estimate loaded center of gravity X_cg (m from nose tip) from .ork mass & motor tags.

    Uses explicit `<overridemass>`, `<mass>`, and loaded motor cylinder volume approximations
    when exact OpenRocket simulation CG data is absent from the XML.
    """
    root = _load_xml_root(ork_path)
    rocket_el = root.find("rocket")
    if rocket_el is None:
        return 0.0, "heuristic_fallback"
    subs = rocket_el.find("subcomponents")
    if subs is None:
        return 0.0, "heuristic_fallback"

    stage_elems = subs.findall("stage")
    sel = stage_selection.strip().lower()
    if sel in {"", "all"} or len(stage_elems) <= 1:
        active_stages = stage_elems
    elif sel == "sustainer":
        active_stages = [stage_elems[0]]
    elif sel == "booster":
        active_stages = [stage_elems[-1]]
    elif sel.startswith("stage_") and sel[6:].isdigit():
        idx = int(sel[6:]) - 1
        active_stages = [stage_elems[idx]] if 0 <= idx < len(stage_elems) else stage_elems
    elif sel.isdigit():
        idx = int(sel) - 1
        active_stages = [stage_elems[idx]] if 0 <= idx < len(stage_elems) else stage_elems
    elif sel.startswith("up_to_") and sel[6:].isdigit():
        k = int(sel[6:])
        active_stages = stage_elems[:k] if 1 <= k <= len(stage_elems) else stage_elems
    else:
        matched = [s for s in stage_elems if (s.findtext("name") or "").strip().lower() == sel]
        active_stages = [matched[0]] if matched else stage_elems

    total_mass = 0.0
    moment_x = 0.0
    cursor_x = 0.0

    for st_el in active_stages:
        st_sub = st_el.find("subcomponents")
        if st_sub is None:
            continue
        for comp in st_sub:
            tag = comp.tag.lower()
            if tag not in {"nosecone", "bodytube", "transition"}:
                continue
            comp_len = max(0.0, _clean_float(comp.findtext("length"), 0.0))
            override_m = comp.findtext("overridemass")
            if override_m is not None:
                c_mass = _clean_float(override_m, 0.0)
            else:
                r = max(
                    _clean_float(comp.findtext("radius"), 0.0),
                    _clean_float(comp.findtext("aftradius"), 0.0),
                    _clean_float(comp.findtext("foreradius"), 0.0),
                )
                c_mass = 2.0 * math.pi * max(r, 0.02) * comp_len * _SHELL_AREAL_DENSITY_KG_M2
            if c_mass > 0.0:
                total_mass += c_mass
                moment_x += c_mass * (cursor_x + 0.5 * comp_len)

            mmount = comp.find("motormount")
            if mmount is not None:
                overhang = _clean_float(mmount.findtext("overhang"), 0.0)
                for mot in mmount.findall("motor"):
                    m_dia = _clean_float(mot.findtext("diameter"), 0.075)
                    m_len = _clean_float(mot.findtext("length"), 0.5)
                    m_mass = 0.25 * math.pi * (m_dia**2) * m_len * _TYPICAL_PROPELLANT_DENSITY_KG_M3
                    m_cg = cursor_x + comp_len + overhang - 0.5 * m_len
                    total_mass += m_mass
                    moment_x += m_mass * m_cg

            c_sub = comp.find("subcomponents")
            if c_sub is not None:
                for sc in c_sub.iter():
                    sc_tag = sc.tag.lower()
                    if sc_tag == "masscomponent":
                        sc_mass = _clean_float(sc.findtext("mass"), 0.0)
                        if sc_mass > 0.0:
                            cx = _local_component_centroid(sc, comp_len, cursor_x, 0.1)
                            total_mass += sc_mass
                            moment_x += sc_mass * cx
                    elif sc_tag in {"bulkhead", "centeringring"}:
                        sc_mass = _clean_float(sc.findtext("overridemass"), 0.0)
                        if sc_mass > 0.0:
                            cx = _local_component_centroid(sc, comp_len, cursor_x, 0.02)
                            total_mass += sc_mass
                            moment_x += sc_mass * cx

            cursor_x += comp_len

    if total_mass <= 1e-6:
        return 0.55 * max(cursor_x, 0.1), "heuristic_fallback"
    return moment_x / total_mass, "ork_xml_mass_estimate"


def compute_fin_flutter_speed(
    fin: FinSet,
    altitude_m: float,
    shear_modulus_pa: float = 15.0e9,
) -> float:
    """Compute NACA TN 4197 / Martin fin torsional flutter boundary velocity (m/s)."""
    atmo = compute_atmosphere(altitude_m)
    c_r = max(fin.root_chord, 1e-4)
    c_t = max(fin.tip_chord, 0.0)
    span = max(fin.span, 1e-4)
    thick = max(fin.thickness, 1e-5)

    area = 0.5 * (c_r + c_t) * span
    aspect_ratio = (span * span) / max(area, 1e-9)
    taper = c_t / c_r
    t_over_c = thick / c_r
    denom = (
        1.337
        * (aspect_ratio**3)
        * atmo.pressure_pa
        * (taper + 1.0)
    ) / max(2.0 * (aspect_ratio + 2.0) * (t_over_c**3), 1e-18)
    ratio = max(shear_modulus_pa, 1.0) / max(denom, 1e-9)
    return atmo.speed_of_sound_m_s * math.sqrt(ratio)


def evaluate_constraints(
    rocket: RocketModel,
    target_fin: FinSet,
    parent_comp: BodyComponent,
    candidate: FinCandidate,
    cg_x_m: float,
    cg_source: str,
    config: FinOptimizationConfig,
) -> StabilityReport:
    """Evaluate geometry validity, fixed thickness, Barrowman >= min SM, and optional flutter."""
    violations: list[str] = []

    if abs(candidate.thickness - target_fin.thickness) > 1e-9:
        violations.append(
            f"thickness_changed ({candidate.thickness*1000:.2f} != "
            f"{target_fin.thickness*1000:.2f} mm)"
        )
    if candidate.root_chord <= 0.02:
        violations.append(f"root_chord_too_small ({candidate.root_chord*1000:.1f} mm)")
    if candidate.tip_chord < 0.015:
        violations.append(f"tip_chord_too_small ({candidate.tip_chord*1000:.1f} mm < 15 mm)")
    if candidate.tip_chord > candidate.root_chord + 1e-6:
        violations.append("tip_chord_exceeds_root_chord")
    if candidate.span <= 0.02:
        violations.append(f"span_too_small ({candidate.span*1000:.1f} mm)")
    if not (-15.0 <= candidate.sweep_angle_deg <= 75.0):
        violations.append(f"sweep_angle_out_of_range ({candidate.sweep_angle_deg:.1f} deg)")

    new_local_le = target_fin.root_te_x - candidate.root_chord
    if new_local_le < -0.005:
        violations.append(
            f"root_chord_overhangs_fore_tube (local LE={new_local_le:.3f} m "
            f"on '{parent_comp.name}')"
        )

    x_cp_m, cna_total = compute_barrowman_cp(rocket)
    d_ref = max(rocket.max_diameter, 1e-6)
    static_margin_cal = (x_cp_m - cg_x_m) / d_ref
    if static_margin_cal < config.min_static_margin_cal - 1e-6:
        violations.append(
            f"static_margin {static_margin_cal:.2f} cal < {config.min_static_margin_cal:.2f} cal"
        )

    v_flutter = compute_fin_flutter_speed(
        target_fin,
        altitude_m=config.altitude_m,
        shear_modulus_pa=config.fin_shear_modulus_pa,
    )
    if config.min_flutter_sf > 0.0:
        atmo = compute_atmosphere(config.altitude_m)
        v_req = config.min_flutter_sf * config.mach * atmo.speed_of_sound_m_s
        if v_flutter < v_req:
            violations.append(f"flutter_speed {v_flutter:.1f} < {v_req:.1f} m/s")

    return StabilityReport(
        x_cp_m=x_cp_m,
        x_cg_m=cg_x_m,
        cg_source=cg_source,
        cna_total=cna_total,
        static_margin_cal=static_margin_cal,
        v_flutter_m_s=v_flutter,
        feasible=(len(violations) == 0),
        violations=tuple(violations),
    )
