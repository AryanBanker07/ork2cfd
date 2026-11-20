"""CFD-ready analytical aerodynamic profiles, nose tip blunting, and meridian merging."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..models._curves import evaluate_profile_radius
from ..models.cfd import CadConfig
from ..models.rocket import BodyComponent, RocketModel

SCALE_M_TO_MM = 1000.0


@dataclass(frozen=True)
class ProfileSegmentMeta:
    """Axial interval and semantic wall zone associated with a body profile region."""

    axial_start_m: float
    axial_end_m: float
    zone_name: str  # "rocket_nosecone" | "rocket_body" | "rocket_boattail"


@dataclass(frozen=True)
class MeridianCurveSpan:
    """Continuous meridian span in mm: either linear endpoints or smooth curved spline."""

    points_mm: tuple[tuple[float, float], ...]
    is_curved: bool


def generate_component_profile_points(
    component: BodyComponent,
    num_points: int = 80,
    global_coords: bool = True,
) -> list[tuple[float, float]]:
    """Sample upper meridian (x, r) points in meters for a BodyComponent."""
    ctype = component.component_type.lower()
    shape = component.shape.lower() if component.shape else ""

    if ctype == "bodytube" or (ctype == "transition" and (not shape or "conical" in shape)):
        n = 2
    else:
        n = max(12, num_points)

    pts: list[tuple[float, float]] = []
    offset = component.axial_start if global_coords else 0.0
    for i in range(n + 1):
        local_x = (i / n) * component.length
        r = evaluate_profile_radius(
            component_type=component.component_type,
            shape=component.shape,
            shape_parameter=component.shape_parameter,
            length=component.length,
            fore_radius=component.fore_radius,
            aft_radius=component.aft_radius,
            local_x=local_x,
            shape_clipped=component.shape_clipped,
        )
        pts.append((offset + local_x, r))
    return pts


def _apply_nose_tip_blunting(
    pts_m: list[tuple[float, float]],
    nose: BodyComponent,
    tip_blunt_radius_m: float,
) -> list[tuple[float, float]]:
    """Replace sharp apex singularity at x=0 with a smooth elliptical dome cap."""
    if tip_blunt_radius_m <= 0.0 or nose.aft_radius <= 0.0 or nose.length <= 0.0:
        return pts_m

    r_target = min(tip_blunt_radius_m, 0.12 * nose.aft_radius, 0.15 * nose.length)
    if r_target <= 1e-6:
        return pts_m

    join_idx = None
    for idx, (x, r) in enumerate(pts_m):
        if idx > 0 and r >= r_target and (x - nose.axial_start) >= 0.5 * r_target:
            join_idx = idx
            break

    if join_idx is None:
        return pts_m

    x_start = pts_m[0][0]
    x_join, r_join = pts_m[join_idx]
    dx = x_join - x_start
    if dx <= 1e-6 or r_join <= 1e-6:
        return pts_m

    cap_samples = 10
    cap_pts: list[tuple[float, float]] = []
    for i in range(cap_samples):
        theta = (i / cap_samples) * (0.5 * math.pi)
        cx = x_start + dx * (1.0 - math.cos(theta))
        cr = r_join * math.sin(theta)
        cap_pts.append((cx, cr))

    return cap_pts + pts_m[join_idx:]


def _classify_component_zone(comp: BodyComponent) -> str:
    """Assign aerodynamic wall Named Selection zone for a body component."""
    ctype = comp.component_type.lower()
    if ctype == "nosecone":
        return "rocket_nosecone"
    if ctype == "transition" and comp.aft_radius < comp.fore_radius - 1e-5:
        return "rocket_boattail"
    return "rocket_body"


def _is_curved_component(comp: BodyComponent) -> bool:
    """Return True if component contour is non-linear and benefits from B-spline fitting."""
    ctype = comp.component_type.lower()
    shape = comp.shape.lower() if comp.shape else ""
    if ctype == "bodytube":
        return False
    return "conical" not in shape


def build_merged_meridian_points_mm(
    rocket: RocketModel,
    cad_config: CadConfig,
) -> tuple[list[tuple[float, float]], list[ProfileSegmentMeta]]:
    """Return flat list of merged (x_mm, r_mm) outer profile vertices and axial zone metadata."""
    spans, segment_meta = build_meridian_spans_mm(rocket, cad_config)
    pts: list[tuple[float, float]] = []
    for sp in spans:
        for pt in sp.points_mm:
            if not pts or math.hypot(pt[0] - pts[-1][0], pt[1] - pts[-1][1]) > 1e-4:
                pts.append(pt)
    return pts, segment_meta


def build_meridian_spans_mm(
    rocket: RocketModel,
    cad_config: CadConfig,
) -> tuple[list[MeridianCurveSpan], list[ProfileSegmentMeta]]:
    """
    Build merged meridian curve spans (in mm) grouping curved contours vs straight segments.

    Consecutive collinear straight segments (e.g., joined body tubes) are merged into one span
    to eliminate artificial seam edges, while curved nosecones/transitions become single smooth
    spline spans.
    """
    comps = rocket.all_components
    if not comps:
        return [], []

    segment_meta: list[ProfileSegmentMeta] = []
    raw_spans: list[MeridianCurveSpan] = []

    for idx, comp in enumerate(comps):
        c_pts_m = generate_component_profile_points(
            comp, num_points=cad_config.curve_samples, global_coords=True
        )
        if idx == 0 and comp.component_type == "nosecone":
            c_pts_m = _apply_nose_tip_blunting(c_pts_m, comp, cad_config.tip_blunt_radius_m)

        zone = _classify_component_zone(comp)
        if (
            segment_meta
            and segment_meta[-1].zone_name == zone
            and abs(segment_meta[-1].axial_end_m - comp.axial_start) < 1e-6
        ):
            prev = segment_meta[-1]
            segment_meta[-1] = ProfileSegmentMeta(
                axial_start_m=prev.axial_start_m,
                axial_end_m=comp.axial_end,
                zone_name=zone,
            )
        else:
            segment_meta.append(
                ProfileSegmentMeta(
                    axial_start_m=comp.axial_start,
                    axial_end_m=comp.axial_end,
                    zone_name=zone,
                )
            )

        pts_mm = [(x * SCALE_M_TO_MM, max(0.0, r * SCALE_M_TO_MM)) for x, r in c_pts_m]
        # Snap start vertex to exact end of previous span to avoid micro steps
        if raw_spans:
            prev_end = raw_spans[-1].points_mm[-1]
            if math.hypot(pts_mm[0][0] - prev_end[0], pts_mm[0][1] - prev_end[1]) < 0.05:
                pts_mm[0] = prev_end
            else:
                # Radial shoulder step between components
                raw_spans.append(
                    MeridianCurveSpan(points_mm=(prev_end, pts_mm[0]), is_curved=False)
                )

        is_curved = _is_curved_component(comp) and len(pts_mm) > 2
        if not is_curved and len(pts_mm) >= 2:
            pts_mm = [pts_mm[0], pts_mm[-1]]

        raw_spans.append(MeridianCurveSpan(points_mm=tuple(pts_mm), is_curved=is_curved))

    # Merge consecutive collinear straight spans (such as multiple equal-radius body tubes)
    merged_spans: list[MeridianCurveSpan] = []
    for sp in raw_spans:
        if (
            merged_spans
            and not sp.is_curved
            and not merged_spans[-1].is_curved
            and len(merged_spans[-1].points_mm) == 2
            and len(sp.points_mm) == 2
        ):
            p0, p1 = merged_spans[-1].points_mm
            q0, q1 = sp.points_mm
            if math.hypot(p1[0] - q0[0], p1[1] - q0[1]) < 1e-4:
                dx1, dr1 = p1[0] - p0[0], p1[1] - p0[1]
                dx2, dr2 = q1[0] - q0[0], q1[1] - q0[1]
                l1 = math.hypot(dx1, dr1)
                l2 = math.hypot(dx2, dr2)
                if l1 > 1e-6 and l2 > 1e-6:
                    cross = abs(dx1 * dr2 - dr1 * dx2) / (l1 * l2)
                    if cross < 1e-4:
                        merged_spans[-1] = MeridianCurveSpan(
                            points_mm=(p0, q1), is_curved=False
                        )
                        continue
        merged_spans.append(sp)

    return merged_spans, segment_meta
