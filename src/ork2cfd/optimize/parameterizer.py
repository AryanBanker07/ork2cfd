"""Fin parameter resolution, immutable rocket mutation, and .ork XML export."""

from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import replace
from pathlib import Path

from ..exceptions import Ork2CfdError
from ..models.rocket import BodyComponent, FinCrossSection, FinSet, RocketModel, Stage
from ..parser.ork_parser import _BODY_TAGS, _FIN_TAGS, _clean_float, _load_xml_root
from .models import FinCandidate, FinDesignBounds

_XML_CROSS_SECTION_NAMES: dict[FinCrossSection, str] = {
    FinCrossSection.SQUARE: "square",
    FinCrossSection.ROUNDED: "rounded",
    FinCrossSection.AIRFOIL: "airfoil",
    FinCrossSection.DOUBLE_WEDGE: "hexagonal",
}


def resolve_target_fin(
    rocket: RocketModel,
    target_fin_stage: str = "booster",
) -> tuple[tuple[int, int, int], Stage, BodyComponent, FinSet]:
    """
    Locate the target trapezoidal fin set in `rocket` for optimization.

    `target_fin_stage` supports:
    - `"booster"` (default): aft-most stage that contains fins (leaves sustainer fins fixed)
    - `"sustainer"`: first stage that contains fins
    - `"stage_1"`, `"stage_2"`, ... or exact stage name match
    """
    finned_stages = [
        (s_idx, st)
        for s_idx, st in enumerate(rocket.stages)
        if any(comp.fins for comp in st.components)
    ]
    if not finned_stages:
        raise Ork2CfdError(f"Rocket '{rocket.name}' has no aerodynamic fin sets to optimize.")

    sel = target_fin_stage.strip().lower()
    chosen_s_idx: int | None = None
    if sel in {"", "booster", "aft"}:
        chosen_s_idx = finned_stages[-1][0]
    elif sel in {"sustainer", "forward", "fore"}:
        chosen_s_idx = finned_stages[0][0]
    elif sel.startswith("stage_") and sel[6:].isdigit():
        k = int(sel[6:]) - 1
        if 0 <= k < len(rocket.stages):
            chosen_s_idx = k
    elif sel.isdigit():
        k = int(sel) - 1
        if 0 <= k < len(rocket.stages):
            chosen_s_idx = k
    else:
        for s_idx, st in enumerate(rocket.stages):
            if st.name.strip().lower() == sel:
                chosen_s_idx = s_idx
                break

    if chosen_s_idx is None:
        names = [st.name for st in rocket.stages]
        raise Ork2CfdError(
            f"Unknown target fin stage '{target_fin_stage}'. Available stages: {names}."
        )

    stage = rocket.stages[chosen_s_idx]
    for c_idx, comp in enumerate(stage.components):
        for f_idx, fin in enumerate(comp.fins):
            if fin.fin_type != "trapezoid":
                raise Ork2CfdError(
                    f"Target fin '{fin.name}' on stage '{stage.name}' is '{fin.fin_type}'; "
                    "only 'trapezoid' fin sets support parametric OpenRocket optimization."
                )
            return (chosen_s_idx, c_idx, f_idx), stage, comp, fin

    raise Ork2CfdError(f"Selected stage '{stage.name}' contains no fin sets.")


def candidate_from_fin(fin: FinSet) -> FinCandidate:
    """Convert an OpenRocket FinSet to a FinCandidate with thickness preserved."""
    span = max(fin.span, 1e-4)
    sweep_angle_deg = math.degrees(math.atan2(fin.sweep_length, span))
    return FinCandidate(
        cross_section=fin.cross_section,
        root_chord=fin.root_chord,
        tip_chord=fin.tip_chord,
        span=fin.span,
        sweep_angle_deg=sweep_angle_deg,
        thickness=fin.thickness,
    )


def default_bounds_for_fin(fin: FinSet, parent_comp: BodyComponent) -> FinDesignBounds:
    """Construct sensible model-agnostic design bounds around a baseline FinSet."""
    base = candidate_from_fin(fin)
    c_r = max(fin.root_chord, 0.03)
    c_t = max(fin.tip_chord, 0.02)
    span = max(fin.span, 0.03)

    # Keep root LE from protruding forward ahead of the parent tube mount
    max_root_on_tube = max(c_r, fin.root_te_x + 0.004)
    root_lo = max(0.04, 0.65 * c_r)
    root_hi = min(max_root_on_tube, 1.25 * c_r)
    if root_hi <= root_lo + 0.005:
        root_hi = root_lo + 0.02

    tip_lo = max(0.015, 0.25 * c_t)
    tip_hi = max(tip_lo + 0.02, min(1.25 * c_t, root_hi))

    span_lo = max(0.03, 0.65 * span)
    span_hi = max(span_lo + 0.02, 1.15 * span)

    sweep_lo = max(10.0, base.sweep_angle_deg - 25.0)
    sweep_hi = min(68.0, base.sweep_angle_deg + 12.0)
    if sweep_hi <= sweep_lo + 5.0:
        sweep_lo, sweep_hi = 15.0, 65.0

    return FinDesignBounds(
        root_chord=(root_lo, root_hi),
        tip_chord=(tip_lo, tip_hi),
        span=(span_lo, span_hi),
        sweep_angle_deg=(sweep_lo, sweep_hi),
    )


def apply_candidate_to_rocket(
    rocket: RocketModel,
    target_ids: tuple[int, int, int],
    candidate: FinCandidate,
) -> RocketModel:
    """Immutably replace only the targeted FinSet while keeping all other stages/thickness fixed."""
    t_s, t_c, t_f = target_ids
    stages = list(rocket.stages)
    target_stage = stages[t_s]
    comps = list(target_stage.components)
    target_comp = comps[t_c]
    fins = list(target_comp.fins)
    orig = fins[t_f]

    # Anchor trailing edge (orig.root_te_x / orig.global_root_te_x) so fin stays aft at stage base
    new_local_le = orig.root_te_x - candidate.root_chord
    new_global_le = orig.global_root_te_x - candidate.root_chord
    new_fin = replace(
        orig,
        root_chord=candidate.root_chord,
        tip_chord=candidate.tip_chord,
        span=candidate.span,
        sweep_length=candidate.sweep_length,
        thickness=orig.thickness,
        cross_section=candidate.cross_section,
        root_le_x=new_local_le,
        global_root_le_x=new_global_le,
    )
    fins[t_f] = new_fin
    comps[t_c] = replace(target_comp, fins=tuple(fins))
    stages[t_s] = replace(target_stage, components=tuple(comps))
    return replace(rocket, stages=tuple(stages))


def _set_or_create_text(parent: ET.Element, tag: str, value: str) -> None:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = value


def _shift_fore_anchored_position(fin_el: ET.Element, delta_chord: float) -> None:
    """Keep fin trailing edge fixed when an .ork file uses top/absolute positioning."""
    for tag, attr in (("position", "type"), ("axialoffset", "method")):
        el = fin_el.find(tag)
        if el is None:
            continue
        mode = (el.attrib.get(attr) or "bottom").strip().lower()
        if mode in {"top", "absolute"}:
            cur = _clean_float(el.text, 0.0)
            el.text = f"{cur + delta_chord:.8g}"
        elif mode == "middle":
            cur = _clean_float(el.text, 0.0)
            el.text = f"{cur + 0.5 * delta_chord:.8g}"


def write_modified_ork(
    source_ork: Path,
    dest_ork: Path,
    target_ids: tuple[int, int, int],
    candidate: FinCandidate,
) -> Path:
    """
    Write an updated OpenRocket (.ork) file preserving thickness, format, and all other stages.

    Supports both plain XML and ZIP-packaged `.ork` containers identically.
    """
    src = Path(source_ork)
    dst = Path(dest_ork)
    dst.parent.mkdir(parents=True, exist_ok=True)
    root = _load_xml_root(src)

    rocket_el = root.find("rocket")
    if rocket_el is None:
        raise Ork2CfdError(f"Missing <rocket> root element in '{src}'.")
    r_subs = rocket_el.find("subcomponents")
    if r_subs is None:
        raise Ork2CfdError(f"Missing <subcomponents> under <rocket> in '{src}'.")

    stages = r_subs.findall("stage")
    t_s, t_c, t_f = target_ids
    if not (0 <= t_s < len(stages)):
        raise Ork2CfdError(f"Target stage index {t_s} out of range for '{src}'.")

    st_subs = stages[t_s].find("subcomponents")
    if st_subs is None:
        raise Ork2CfdError(f"Stage {t_s} has no <subcomponents> in '{src}'.")

    body_elems = [
        el
        for el in st_subs
        if el.tag.lower() in _BODY_TAGS and _clean_float(el.findtext("length"), 0.0) > 0.0
    ]
    if not (0 <= t_c < len(body_elems)):
        raise Ork2CfdError(f"Target component index {t_c} out of range in stage {t_s}.")

    comp_subs = body_elems[t_c].find("subcomponents")
    if comp_subs is None:
        raise Ork2CfdError(f"Component {t_c} has no <subcomponents>.")
    fin_elems = [el for el in comp_subs if el.tag.lower() in _FIN_TAGS]
    if not (0 <= t_f < len(fin_elems)):
        raise Ork2CfdError(f"Target fin index {t_f} out of range in component {t_c}.")

    fin_el = fin_elems[t_f]
    orig_root = _clean_float(fin_el.findtext("rootchord"), candidate.root_chord)
    delta_chord = orig_root - candidate.root_chord
    if abs(delta_chord) > 1e-9:
        _shift_fore_anchored_position(fin_el, delta_chord)

    _set_or_create_text(fin_el, "rootchord", f"{candidate.root_chord:.8g}")
    _set_or_create_text(fin_el, "tipchord", f"{candidate.tip_chord:.8g}")
    _set_or_create_text(fin_el, "height", f"{candidate.span:.8g}")
    _set_or_create_text(fin_el, "sweeplength", f"{candidate.sweep_length:.8g}")
    _set_or_create_text(
        fin_el,
        "crosssection",
        _XML_CROSS_SECTION_NAMES[candidate.cross_section],
    )
    # Intentionally preserve <thickness> and every other stage's fin parameters verbatim

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    raw = src.read_bytes()
    if raw.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as z_in:
            names = z_in.namelist()
            target_name = (
                "rocket.ork"
                if "rocket.ork" in names
                else next(
                    (n for n in names if n.endswith(".ork") or n.endswith(".xml")),
                    names[0],
                )
            )
            with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
                for info in z_in.infolist():
                    data = xml_bytes if info.filename == target_name else z_in.read(info.filename)
                    z_out.writestr(info, data)
    else:
        dst.write_bytes(xml_bytes)

    return dst
