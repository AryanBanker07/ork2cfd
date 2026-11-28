"""Parser for OpenRocket (.ork) files supporting XML, ZIP, and GZIP containers."""

from __future__ import annotations

import gzip
import io
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path

from ..exceptions import OrkParseError
from ..models.rocket import (
    BodyComponent,
    FinCrossSection,
    FinSet,
    LaunchLug,
    RailButton,
    RocketModel,
    Stage,
    SurfaceFinish,
)
from .validation import validate_rocket_model

_BODY_TAGS = frozenset({"nosecone", "bodytube", "transition"})
_FIN_TAGS = frozenset({"trapezoidfinset", "ellipticalfinset", "freeformfinset", "tubefinset"})


def _clean_float(val_str: str | None, default: float = 0.0) -> float:
    """Parse a float value from OpenRocket XML, stripping any 'auto' prefix."""
    if val_str is None:
        return default
    cleaned = val_str.replace("auto", "").strip()
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _clean_int(val_str: str | None, default: int = 1) -> int:
    """Parse an integer value safely."""
    if val_str is None:
        return default
    try:
        return int(val_str.strip())
    except ValueError:
        return default


def _clean_bool(val_str: str | None, default: bool = False) -> bool:
    """Parse a boolean string from OpenRocket XML."""
    if val_str is None:
        return default
    return val_str.strip().lower() in {"true", "1", "yes"}


def _load_xml_root(file_path: Path) -> ET.Element:
    """Load XML root element from plain XML, ZIP, or GZIP .ork file."""
    if not file_path.exists():
        raise OrkParseError(f"OpenRocket file not found: {file_path}")

    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        raise OrkParseError(f"Unable to read file {file_path}: {exc}") from exc

    if raw_bytes.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                names = zf.namelist()
                target = "rocket.ork" if "rocket.ork" in names else None
                if target is None:
                    for name in names:
                        if name.endswith(".ork") or name.endswith(".xml"):
                            target = name
                            break
                if target is None and names:
                    target = names[0]
                if target is None:
                    raise OrkParseError(f"ZIP archive '{file_path.name}' is empty.")
                raw_bytes = zf.read(target)
        except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError) as exc:
            raise OrkParseError(f"Corrupt ZIP .ork archive '{file_path}': {exc}") from exc
    elif raw_bytes.startswith(b"\x1f\x8b"):
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except (OSError, EOFError, zlib.error) as exc:
            raise OrkParseError(f"Corrupt GZIP .ork file '{file_path}': {exc}") from exc

    try:
        return ET.fromstring(raw_bytes)
    except ET.ParseError:
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            return ET.fromstring(text)
        except ET.ParseError as exc:
            raise OrkParseError(f"Malformed XML in '{file_path}': {exc}") from exc


def _extract_position_offset(elem: ET.Element) -> tuple[str, float]:
    """Read alignment mode and offset from <position> or <axialoffset>."""
    pos_elem = elem.find("position")
    if pos_elem is not None:
        method = pos_elem.attrib.get("type", "bottom").strip().lower()
        return method, _clean_float(pos_elem.text, 0.0)

    ax_elem = elem.find("axialoffset")
    if ax_elem is not None:
        method = ax_elem.attrib.get("method", "bottom").strip().lower()
        return method, _clean_float(ax_elem.text, 0.0)

    return "bottom", 0.0


def _resolve_axial_span(
    method: str,
    offset: float,
    item_length: float,
    parent_length: float,
    parent_axial_start: float,
) -> tuple[float, float]:
    """Compute local (le_x, te_x) relative to parent front using OpenRocket sign convention."""
    if method == "bottom":
        te_x = parent_length + offset
        le_x = te_x - item_length
    elif method == "middle":
        le_x = 0.5 * (parent_length - item_length) + offset
        te_x = le_x + item_length
    elif method == "absolute":
        le_x = offset - parent_axial_start
        te_x = le_x + item_length
    else:  # "top"
        le_x = offset
        te_x = le_x + item_length
    return le_x, te_x


def _parse_fin_set(
    elem: ET.Element,
    parent_length: float,
    parent_axial_start: float,
) -> FinSet:
    """Parse trapezoid, elliptical, freeform, or tube fin sets."""
    name = elem.findtext("name", "Fin Set")
    fin_type = elem.tag.lower().replace("finset", "")
    fin_count = _clean_int(
        elem.findtext("fincount"),
        default=_clean_int(elem.findtext("instancecount"), 4),
    )

    thickness = _clean_float(elem.findtext("thickness"), 0.003)
    cross_section = FinCrossSection.from_str(elem.findtext("crosssection"))
    cant_deg = _clean_float(elem.findtext("cant"), 0.0)
    rotation_deg = _clean_float(
        elem.findtext("rotation") or elem.findtext("angleoffset"), 0.0
    )
    fillet_radius = _clean_float(elem.findtext("filletradius"), 0.0)
    outer_diameter = _clean_float(elem.findtext("outerdiameter"), 0.0)

    points: list[tuple[float, float]] = []
    if fin_type == "freeform":
        finpoints_elem = elem.find("finpoints")
        if finpoints_elem is not None:
            for pt_el in finpoints_elem.findall("point"):
                px = _clean_float(pt_el.attrib.get("x"), 0.0)
                py = _clean_float(pt_el.attrib.get("y"), 0.0)
                points.append((px, py))
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            root_chord = max(xs) - min(xs)
            # Root chord is distance between first and last root points when defined on y=0
            root_pts = [p[0] for p in points if abs(p[1]) < 1e-6]
            if len(root_pts) >= 2:
                root_chord = max(root_pts) - min(root_pts)
            span = max(ys)
            tip_chord = 0.0
            sweep_length = 0.0
        else:
            root_chord = 0.0
            tip_chord = 0.0
            span = 0.0
            sweep_length = 0.0
    elif fin_type == "elliptical":
        root_chord = _clean_float(elem.findtext("rootchord"), 0.0)
        span = _clean_float(elem.findtext("height"), 0.0)
        tip_chord = 0.0
        sweep_length = 0.0
    elif fin_type == "tube":
        root_chord = _clean_float(elem.findtext("length"), 0.0)
        tip_chord = root_chord
        span = outer_diameter
        sweep_length = 0.0
    else:  # trapezoid
        root_chord = _clean_float(elem.findtext("rootchord"), 0.0)
        tip_chord = _clean_float(elem.findtext("tipchord"), 0.0)
        span = _clean_float(elem.findtext("height"), 0.0)
        sweep_length = _clean_float(elem.findtext("sweeplength"), 0.0)

    method, offset = _extract_position_offset(elem)
    root_le_x, root_te_x = _resolve_axial_span(
        method, offset, root_chord, parent_length, parent_axial_start
    )

    return FinSet(
        name=name,
        fin_type=fin_type,
        fin_count=fin_count,
        root_chord=root_chord,
        tip_chord=tip_chord,
        span=span,
        sweep_length=sweep_length,
        thickness=thickness,
        cross_section=cross_section,
        cant_deg=cant_deg,
        rotation_deg=rotation_deg,
        fillet_radius=fillet_radius,
        points=tuple(points),
        root_le_x=root_le_x,
        root_te_x=root_te_x,
        global_root_le_x=parent_axial_start + root_le_x,
        global_root_te_x=parent_axial_start + root_te_x,
        outer_diameter=outer_diameter,
    )


def _parse_rail_button(
    elem: ET.Element,
    parent_length: float,
    parent_axial_start: float,
) -> RailButton:
    """Parse an external rail button element."""
    name = elem.findtext("name", "Rail Button")
    outer_diameter = _clean_float(
        elem.findtext("outerdiameter"),
        default=2.0 * _clean_float(elem.findtext("radius"), 0.005),
    )
    height = _clean_float(elem.findtext("height"), 0.008)
    instance_count = _clean_int(elem.findtext("instancecount"), 1)
    instance_sep = _clean_float(elem.findtext("instanceseparation"), 0.0)
    angle_deg = _clean_float(elem.findtext("angleoffset") or elem.findtext("rotation"), 0.0)

    method, offset = _extract_position_offset(elem)
    le_x, _ = _resolve_axial_span(method, offset, outer_diameter, parent_length, parent_axial_start)
    center_x = parent_axial_start + le_x + 0.5 * outer_diameter

    return RailButton(
        name=name,
        outer_diameter=outer_diameter,
        height=height,
        axial_x=center_x,
        angle_deg=angle_deg,
        instance_count=instance_count,
        instance_separation=instance_sep,
    )


def _parse_launch_lug(
    elem: ET.Element,
    parent_length: float,
    parent_axial_start: float,
) -> LaunchLug:
    """Parse an external launch lug element."""
    name = elem.findtext("name", "Launch Lug")
    outer_radius = _clean_float(elem.findtext("radius"), 0.006)
    length = _clean_float(elem.findtext("length"), 0.05)
    instance_count = _clean_int(elem.findtext("instancecount"), 1)
    instance_sep = _clean_float(elem.findtext("instanceseparation"), 0.0)
    angle_deg = _clean_float(elem.findtext("angleoffset") or elem.findtext("rotation"), 0.0)

    method, offset = _extract_position_offset(elem)
    le_x, _ = _resolve_axial_span(method, offset, length, parent_length, parent_axial_start)

    return LaunchLug(
        name=name,
        outer_radius=outer_radius,
        length=length,
        axial_x=parent_axial_start + le_x,
        angle_deg=angle_deg,
        instance_count=instance_count,
        instance_separation=instance_sep,
    )


def _parse_body_component(
    elem: ET.Element,
    axial_start: float,
    warnings: list[str],
) -> BodyComponent | None:
    """Parse one external aerodynamic body component and its aerodynamic subcomponents."""
    c_type = elem.tag.lower()
    if c_type not in _BODY_TAGS:
        return None

    name = elem.findtext("name", c_type.capitalize())
    length = _clean_float(elem.findtext("length"), 0.0)
    if length <= 0.0:
        warnings.append(f"Skipped '{name}' ({c_type}) with non-positive length {length:.4f} m.")
        return None

    shape = elem.findtext("shape", "")
    shape_param = _clean_float(elem.findtext("shapeparameter"), 0.0)
    shape_clipped = _clean_bool(elem.findtext("shapeclipped"), False)
    finish = SurfaceFinish.from_str(elem.findtext("finish"))

    if c_type == "nosecone":
        aft_r = _clean_float(elem.findtext("aftradius") or elem.findtext("radius"), 0.0)
        fore_r = 0.0
        if _clean_bool(elem.findtext("isflipped"), False):
            fore_r, aft_r = aft_r, fore_r
            c_type = "transition"
            warnings.append(f"NoseCone '{name}' is flipped; treating as tailcone profile.")
    elif c_type == "transition":
        fore_r = _clean_float(elem.findtext("foreradius"), 0.0)
        aft_r = _clean_float(elem.findtext("aftradius"), 0.0)
    else:  # bodytube
        radius = _clean_float(elem.findtext("radius"), 0.0)
        fore_r = radius
        aft_r = radius

    fins: list[FinSet] = []
    rail_buttons: list[RailButton] = []
    launch_lugs: list[LaunchLug] = []

    sub = elem.find("subcomponents")
    if sub is not None:
        for sc in sub:
            sc_tag = sc.tag.lower()
            if sc_tag in _FIN_TAGS:
                fins.append(
                    _parse_fin_set(
                        sc, parent_length=length, parent_axial_start=axial_start
                    )
                )
            elif sc_tag == "railbutton":
                rail_buttons.append(
                    _parse_rail_button(sc, parent_length=length, parent_axial_start=axial_start)
                )
            elif sc_tag == "launchlug":
                launch_lugs.append(
                    _parse_launch_lug(sc, parent_length=length, parent_axial_start=axial_start)
                )
            elif sc_tag in {"podset", "parallelstage"}:
                warnings.append(
                    f"Component '{sc.findtext('name', sc_tag)}' is a parallel pod/booster; "
                    "currently omitted from axisymmetric baseline OML."
                )

    return BodyComponent(
        name=name,
        component_type=c_type,
        length=length,
        fore_radius=fore_r,
        aft_radius=aft_r,
        shape=shape,
        shape_parameter=shape_param,
        shape_clipped=shape_clipped,
        finish=finish,
        axial_start=axial_start,
        axial_end=axial_start + length,
        fins=tuple(fins),
        rail_buttons=tuple(rail_buttons),
        launch_lugs=tuple(launch_lugs),
    )


def _find_stage_elements(rocket_elem: ET.Element) -> list[ET.Element]:
    """Locate real <stage> elements under <rocket><subcomponents>, avoiding motorconfiguration."""
    sub = rocket_elem.find("subcomponents")
    if sub is not None:
        stages = [el for el in sub if el.tag.lower() == "stage"]
        if stages:
            return stages
    direct_stages = [el for el in rocket_elem if el.tag.lower() == "stage"]
    if direct_stages:
        return direct_stages
    return [rocket_elem]


def _resolve_auto_radii(stages: list[Stage]) -> list[Stage]:
    """Propagate missing ('auto') radii forward and backward across sequential components."""
    flat: list[tuple[int, int, BodyComponent]] = []
    for s_idx, st in enumerate(stages):
        for c_idx, comp in enumerate(st.components):
            flat.append((s_idx, c_idx, comp))

    comps = [item[2] for item in flat]
    # Forward pass
    for idx, c in enumerate(comps):
        if c.component_type == "bodytube" and c.aft_radius == 0.0 and idx > 0:
            r = comps[idx - 1].aft_radius
            comps[idx] = _with_radii(c, r, r)
        elif c.component_type == "transition" and c.fore_radius == 0.0 and idx > 0:
            comps[idx] = _with_radii(c, comps[idx - 1].aft_radius, c.aft_radius)

    # Backward pass
    for idx in range(len(comps) - 1, -1, -1):
        c = comps[idx]
        if c.component_type == "nosecone" and c.aft_radius == 0.0 and idx + 1 < len(comps):
            comps[idx] = _with_radii(c, 0.0, comps[idx + 1].fore_radius)
        elif c.component_type == "bodytube" and c.aft_radius == 0.0 and idx + 1 < len(comps):
            r = comps[idx + 1].fore_radius
            comps[idx] = _with_radii(c, r, r)
        elif c.component_type == "transition" and c.aft_radius == 0.0 and idx + 1 < len(comps):
            comps[idx] = _with_radii(c, c.fore_radius, comps[idx + 1].fore_radius)

    # Reconstruct immutable Stage list
    rebuilt: list[list[BodyComponent]] = [[] for _ in stages]
    for (s_idx, _c_idx, _), updated in zip(flat, comps, strict=False):
        rebuilt[s_idx].append(updated)

    return [
        Stage(
            name=st.name,
            components=tuple(rebuilt[s_idx]),
            axial_start=st.axial_start,
            axial_end=st.axial_end,
        )
        for s_idx, st in enumerate(stages)
    ]


def _with_radii(comp: BodyComponent, fore_r: float, aft_r: float) -> BodyComponent:
    """Return a copy of BodyComponent with updated fore and aft radii."""
    return BodyComponent(
        name=comp.name,
        component_type=comp.component_type,
        length=comp.length,
        fore_radius=fore_r,
        aft_radius=aft_r,
        shape=comp.shape,
        shape_parameter=comp.shape_parameter,
        shape_clipped=comp.shape_clipped,
        finish=comp.finish,
        axial_start=comp.axial_start,
        axial_end=comp.axial_end,
        fins=comp.fins,
        rail_buttons=comp.rail_buttons,
        launch_lugs=comp.launch_lugs,
    )


def parse_ork_file(file_path: str | Path) -> RocketModel:
    """Parse an OpenRocket (.ork) file into a validated external-OML RocketModel."""
    path = Path(file_path)
    root = _load_xml_root(path)

    rocket_elem = root.find("rocket")
    if rocket_elem is None:
        rocket_elem = root

    rocket_name = rocket_elem.findtext("name") or path.stem
    stage_elems = _find_stage_elements(rocket_elem)

    stages: list[Stage] = []
    warnings: list[str] = []
    current_axial_x = 0.0

    for idx, s_elem in enumerate(stage_elems):
        s_name = s_elem.findtext("name") or f"Stage {idx + 1}"
        sub = s_elem.find("subcomponents")
        if sub is None and s_elem.tag.lower() == "rocket":
            sub = s_elem
        if sub is None:
            continue

        stage_start = current_axial_x
        comps: list[BodyComponent] = []
        for comp_el in sub:
            comp = _parse_body_component(comp_el, axial_start=current_axial_x, warnings=warnings)
            if comp is not None:
                current_axial_x = comp.axial_end
                comps.append(comp)

        if comps:
            stages.append(
                Stage(
                    name=s_name,
                    components=tuple(comps),
                    axial_start=stage_start,
                    axial_end=current_axial_x,
                )
            )

    resolved_stages = _resolve_auto_radii(stages)
    model = RocketModel(
        name=rocket_name,
        stages=tuple(resolved_stages),
        source_path=str(path),
        warnings=tuple(warnings),
    )

    all_warnings = validate_rocket_model(model)
    return RocketModel(
        name=model.name,
        stages=model.stages,
        source_path=model.source_path,
        warnings=tuple(all_warnings),
    )
