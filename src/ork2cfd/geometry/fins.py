"""3D CFD-clean fin builders for Trapezoidal, Elliptical, Freeform, and Tube fin sets."""

from __future__ import annotations

import math

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from ..exceptions import GeometryBuildError
from ..models.cfd import CadConfig
from ..models.rocket import FinCrossSection, FinSet, RocketModel
from .profiles import SCALE_M_TO_MM


def build_fin_set_solids_mm(
    fin: FinSet,
    rocket: RocketModel,
    cad_config: CadConfig,
) -> list[TopoDS_Shape]:
    """Build all individual fin TopoDS_Shape solids (in mm) for a FinSet."""
    if fin.fin_count <= 0 or fin.span <= 0.0:
        return []

    if fin.fin_type == "tube":
        base_fin = _build_tube_fin_mm(fin, rocket, cad_config)
    else:
        base_fin = _build_planar_fin_mm(fin, rocket, cad_config)

    if abs(fin.cant_deg) > 1e-4:
        mid_x_mm = 0.5 * (fin.global_root_le_x + fin.global_root_te_x) * SCALE_M_TO_MM
        trsf_cant = gp_Trsf()
        trsf_cant.SetRotation(
            gp_Ax1(gp_Pnt(mid_x_mm, 0.0, 0.0), gp_Dir(0.0, 1.0, 0.0)),
            math.radians(fin.cant_deg),
        )
        base_fin = BRepBuilderAPI_Transform(base_fin, trsf_cant, True).Shape()

    solids: list[TopoDS_Shape] = []
    axis_x = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    for k in range(fin.fin_count):
        angle_deg = fin.rotation_deg + k * (360.0 / fin.fin_count)
        trsf = gp_Trsf()
        trsf.SetRotation(axis_x, math.radians(angle_deg))
        rotated = BRepBuilderAPI_Transform(base_fin, trsf, True).Shape()
        solids.append(rotated)
    return solids


def _root_radii_mm(fin: FinSet, rocket: RocketModel, embed_mm: float) -> tuple[float, float, float]:
    """Compute root radii and embedded base radius (mm) along the fin root."""
    r_le = rocket.radius_at_x(fin.global_root_le_x) * SCALE_M_TO_MM
    r_te = rocket.radius_at_x(fin.global_root_te_x) * SCALE_M_TO_MM
    r_mid = rocket.radius_at_x(0.5 * (fin.global_root_le_x + fin.global_root_te_x)) * SCALE_M_TO_MM
    r_min = min(r_le, r_te, r_mid)
    y_embed = max(1.0, r_min - embed_mm)
    return y_embed, max(r_le, r_te, r_mid), r_mid


def _build_planar_fin_mm(
    fin: FinSet,
    rocket: RocketModel,
    cad_config: CadConfig,
) -> TopoDS_Shape:
    """Construct a single fin solid positioned at angle=0 (+Y radial) in mm."""
    embed_mm = max(0.5, cad_config.fin_root_embed_m * SCALE_M_TO_MM)
    y_embed, r_root_max, r_mid = _root_radii_mm(fin, rocket, embed_mm)

    thickness_mm = max(0.5, (fin.thickness if fin.thickness > 0.0 else 0.003) * SCALE_M_TO_MM)
    span_mm = fin.span * SCALE_M_TO_MM
    y_tip = r_mid + span_mm

    x_le_root = fin.global_root_le_x * SCALE_M_TO_MM
    x_te_root = fin.global_root_te_x * SCALE_M_TO_MM

    if fin.fin_type == "trapezoid":
        x_le_tip = x_le_root + fin.sweep_length * SCALE_M_TO_MM
        tip_chord_mm = max(1.0, fin.tip_chord * SCALE_M_TO_MM)
        x_te_tip = x_le_tip + tip_chord_mm

        if fin.cross_section != FinCrossSection.SQUARE:
            return _build_lofted_trapezoid_mm(
                x_le_root=x_le_root,
                x_te_root=x_te_root,
                y_root=y_embed,
                x_le_tip=x_le_tip,
                x_te_tip=x_te_tip,
                y_tip=y_tip,
                thickness_mm=thickness_mm,
                te_thick_mm=max(0.3, cad_config.fin_te_thickness_m * SCALE_M_TO_MM),
                cross_section=fin.cross_section,
            )

        poly = BRepBuilderAPI_MakePolygon()
        for px, py in [
            (x_le_root, y_embed),
            (x_te_root, y_embed),
            (x_te_tip, y_tip),
            (x_le_tip, y_tip),
        ]:
            poly.Add(gp_Pnt(px, py, -0.5 * thickness_mm))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        solid = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, thickness_mm)).Shape()
        return solid

    if fin.fin_type == "elliptical":
        chord_mm = max(2.0, fin.root_chord * SCALE_M_TO_MM)
        x_mid = 0.5 * (x_le_root + x_te_root)
        half_c = 0.5 * chord_mm
        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(x_le_root, y_embed, -0.5 * thickness_mm))
        n_arc = 28
        for i in range(n_arc + 1):
            theta = math.pi * (1.0 - i / n_arc)
            px = x_mid + half_c * math.cos(theta)
            py = r_root_max + span_mm * math.sin(theta)
            poly.Add(gp_Pnt(px, py, -0.5 * thickness_mm))
        poly.Add(gp_Pnt(x_te_root, y_embed, -0.5 * thickness_mm))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        solid = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, thickness_mm)).Shape()
        return _maybe_round_fin_edges(solid, thickness_mm, fin.cross_section)

    if fin.fin_type == "freeform":
        if len(fin.points) < 3:
            raise GeometryBuildError(f"Freeform fin '{fin.name}' requires at least 3 points.")
        pts_mm: list[tuple[float, float]] = []
        for idx, (px_m, py_m) in enumerate(fin.points):
            gx = x_le_root + px_m * SCALE_M_TO_MM
            gy = y_embed if (idx in (0, len(fin.points) - 1) or py_m <= 1e-5) else (
                r_mid + py_m * SCALE_M_TO_MM
            )
            if not pts_mm or math.hypot(gx - pts_mm[-1][0], gy - pts_mm[-1][1]) > 0.05:
                pts_mm.append((gx, gy))
        if len(pts_mm) < 3:
            raise GeometryBuildError(f"Freeform fin '{fin.name}' collapsed to < 3 points.")

        poly = BRepBuilderAPI_MakePolygon()
        for px, py in pts_mm:
            poly.Add(gp_Pnt(px, py, -0.5 * thickness_mm))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        solid = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, thickness_mm)).Shape()
        return _maybe_round_fin_edges(solid, thickness_mm, fin.cross_section)

    raise GeometryBuildError(f"Unsupported fin type: {fin.fin_type}")


def _build_lofted_trapezoid_mm(
    x_le_root: float,
    x_te_root: float,
    y_root: float,
    x_le_tip: float,
    x_te_tip: float,
    y_tip: float,
    thickness_mm: float,
    te_thick_mm: float,
    cross_section: FinCrossSection,
) -> TopoDS_Shape:
    """Loft a ruled 3D trapezoidal fin with airfoil, rounded, or double-wedge section."""
    root_pts = _section_xz_points(x_le_root, x_te_root, thickness_mm, te_thick_mm, cross_section)
    tip_pts = _section_xz_points(x_le_tip, x_te_tip, thickness_mm, te_thick_mm, cross_section)

    root_wire = _make_xz_wire(root_pts, y_root)
    tip_wire = _make_xz_wire(tip_pts, y_tip)

    loft = BRepOffsetAPI_ThruSections(True, True)
    loft.AddWire(root_wire)
    loft.AddWire(tip_wire)
    loft.Build()
    if not loft.IsDone():
        raise GeometryBuildError("Failed to loft profiled trapezoidal fin.")
    return loft.Shape()


def _section_xz_points(
    x_le: float,
    x_te: float,
    thickness_mm: float,
    te_thick_mm: float,
    cross_section: FinCrossSection,
) -> list[tuple[float, float]]:
    """Generate consistent closed (x, z) cross-section polygon points with finite TE/LE."""
    chord = max(2.0, abs(x_te - x_le))
    x0 = min(x_le, x_te)
    half_t = 0.5 * min(thickness_mm, 0.35 * chord)
    half_te = 0.5 * min(te_thick_mm, 0.6 * thickness_mm, 0.15 * chord)
    half_le = max(0.15, 0.2 * half_te)

    if cross_section == FinCrossSection.DOUBLE_WEDGE:
        x_mid1 = x0 + 0.18 * chord
        x_mid2 = x0 + 0.82 * chord
        return [
            (x0, -half_le),
            (x0, half_le),
            (x_mid1, half_t),
            (x_mid2, half_t),
            (x0 + chord, half_te),
            (x0 + chord, -half_te),
            (x_mid2, -half_t),
            (x_mid1, -half_t),
        ]

    if cross_section == FinCrossSection.ROUNDED:
        r = min(half_t, 0.25 * chord)
        pts: list[tuple[float, float]] = []
        # LE semicircle (front)
        for i in range(6):
            ang = -0.5 * math.pi + (i / 5.0) * math.pi
            pts.append((x0 + r - r * math.cos(ang), r * math.sin(ang)))
        # TE semicircle (back)
        for i in range(6):
            ang = 0.5 * math.pi - (i / 5.0) * math.pi
            pts.append((x0 + chord - r + r * math.cos(ang), r * math.sin(ang)))
        return pts

    # Symmetric NACA-00xx-like airfoil with finite trailing-edge thickness
    n_side = 10
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    for i in range(n_side + 1):
        s = i / n_side
        if i == 0:
            z = half_le
        else:
            # Classical NACA 4-digit half-thickness normalized so max is ~1.0
            naca = 5.0 * (
                0.2969 * math.sqrt(s)
                - 0.1260 * s
                - 0.3516 * (s**2)
                + 0.2843 * (s**3)
                - 0.1015 * (s**4)
            )
            # Normalized peak of NACA is ~0.5 at s=0.3 -> scale to half_t, blend TE to half_te
            z_base = half_t * (naca / 0.5)
            z = max(half_te * s, z_base)
        px = x0 + s * chord
        upper.append((px, z))
        lower.append((px, -z))
    return lower[:1] + upper + list(reversed(lower[1:]))


def _make_xz_wire(pts_xz: list[tuple[float, float]], y_mm: float):
    """Create a closed TopoDS_Wire in the X-Z plane at constant radial coordinate y_mm."""
    poly = BRepBuilderAPI_MakePolygon()
    for px, pz in pts_xz:
        poly.Add(gp_Pnt(px, y_mm, pz))
    poly.Close()
    if poly.IsDone():
        return poly.Wire()
    wire_maker = BRepBuilderAPI_MakeWire()
    for i in range(len(pts_xz)):
        x1, z1 = pts_xz[i]
        x2, z2 = pts_xz[(i + 1) % len(pts_xz)]
        edge = BRepBuilderAPI_MakeEdge(gp_Pnt(x1, y_mm, z1), gp_Pnt(x2, y_mm, z2)).Edge()
        wire_maker.Add(edge)
    return wire_maker.Wire()


def _maybe_round_fin_edges(
    solid: TopoDS_Shape,
    thickness_mm: float,
    cross_section: FinCrossSection,
) -> TopoDS_Shape:
    """Apply safe edge rounding on extruded non-square fins with fallback to sharp."""
    if cross_section == FinCrossSection.SQUARE:
        return solid
    try:
        r_fillet = min(0.42 * thickness_mm, 1.2)
        if r_fillet <= 0.1:
            return solid
        fillet = BRepFilletAPI_MakeFillet(solid)
        exp = TopExp_Explorer(solid, TopAbs_EDGE)
        added = 0
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            fillet.Add(r_fillet, edge)
            added += 1
            exp.Next()
        if added > 0:
            fillet.Build()
            if fillet.IsDone():
                return fillet.Shape()
    except Exception:
        pass
    return solid


def _build_tube_fin_mm(
    fin: FinSet,
    rocket: RocketModel,
    cad_config: CadConfig,
) -> TopoDS_Shape:
    """Build a solid cylindrical tube fin shell (closed solid cylinder or ring)."""
    length_mm = max(2.0, fin.root_chord * SCALE_M_TO_MM)
    outer_r = max(2.0, 0.5 * fin.outer_diameter * SCALE_M_TO_MM)
    wall = max(0.5, fin.thickness * SCALE_M_TO_MM)
    inner_r = max(0.0, outer_r - wall)

    x_le = fin.global_root_le_x * SCALE_M_TO_MM
    r_mid = rocket.radius_at_x(0.5 * (fin.global_root_le_x + fin.global_root_te_x)) * SCALE_M_TO_MM
    embed_mm = max(0.5, cad_config.fin_root_embed_m * SCALE_M_TO_MM)
    y_center = r_mid + outer_r - embed_mm

    ax = gp_Ax2(gp_Pnt(x_le, y_center, 0.0), gp_Dir(1.0, 0.0, 0.0))
    outer_cyl = BRepPrimAPI_MakeCylinder(ax, outer_r, length_mm).Shape()
    if inner_r > 0.5:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

        inner_cyl = BRepPrimAPI_MakeCylinder(ax, inner_r, length_mm + 2.0).Shape()
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(-1.0, 0.0, 0.0))
        inner_cyl = BRepBuilderAPI_Transform(inner_cyl, trsf, True).Shape()
        cut = BRepAlgoAPI_Cut(outer_cyl, inner_cyl)
        cut.Build()
        if cut.IsDone():
            return cut.Shape()
    return outer_cyl
