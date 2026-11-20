"""Assembles the CFD-clean 3D Outer Mold Line (OML) rocket solid with seam rotation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.BRepTools import BRepTools_ReShape
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_INTERNAL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape

from ..exceptions import GeometryBuildError
from ..models.cfd import CadConfig
from ..models.rocket import RocketModel
from .appendages import build_appendage_solids_mm
from .fins import build_fin_set_solids_mm
from .profiles import (
    SCALE_M_TO_MM,
    MeridianCurveSpan,
    ProfileSegmentMeta,
    build_merged_meridian_points_mm,
    build_meridian_spans_mm,
)
from .validator import BrepValidationStats, validate_brep_shape


@dataclass(frozen=True)
class OmlBuildOutput:
    """Result of building the watertight external OML rocket solid (in mm)."""

    oml_shape: TopoDS_Shape
    body_only_shape: TopoDS_Shape
    body_stats: BrepValidationStats
    oml_stats: BrepValidationStats
    segment_meta: tuple[ProfileSegmentMeta, ...]
    warnings: tuple[str, ...]


def build_rocket_oml(
    rocket: RocketModel,
    cad_config: CadConfig,
    seam_angle_deg: float | None = None,
) -> OmlBuildOutput:
    """
    Build the single watertight 3D rocket OML solid (in mm) ready for fluid domain subtraction.

    Revolves the merged meridian profile with seam line rotated away from fin planes,
    fuses all fins and optional appendages, and runs BRepCheck validation.
    """
    spans, seg_meta = build_meridian_spans_mm(rocket, cad_config)
    if not spans:
        raise GeometryBuildError(f"Rocket '{rocket.name}' produced empty meridian profile.")

    warnings: list[str] = []
    # Rotate revolution seam so it bisects fins rather than intersecting fin roots
    if seam_angle_deg is None:
        n_dom = max(1, rocket.dominant_fin_count)
        first_rot = rocket.all_fins[0].rotation_deg if rocket.all_fins else 0.0
        seam_angle_deg = first_rot + 180.0 / n_dom

    body_shape = _revolve_spans_mm(spans, seam_angle_deg=seam_angle_deg)
    body_stats = validate_brep_shape(body_shape)
    if not body_stats.is_watertight:
        raise GeometryBuildError(
            f"Revolved OML body for '{rocket.name}' is not watertight: {body_stats.warnings}"
        )

    fin_solids: list[TopoDS_Shape] = []
    for fin in rocket.all_fins:
        fin_solids.extend(build_fin_set_solids_mm(fin, rocket, cad_config))

    app_solids = build_appendage_solids_mm(rocket, cad_config)
    all_attachments = fin_solids + app_solids

    oml_shape = body_shape
    if all_attachments:
        oml_shape = _fuse_shapes_robust(body_shape, all_attachments, warnings)
        oml_shape = _strip_internal_edges(oml_shape)

    if cad_config.apply_fin_fillets and fin_solids:
        oml_shape = _apply_root_fillets_safe(
            oml_shape, rocket, cad_config, warnings
        )

    oml_stats = validate_brep_shape(oml_shape)
    if not oml_stats.is_watertight:
        raise GeometryBuildError(
            f"Final OML solid for '{rocket.name}' failed watertightness check: "
            f"{oml_stats.warnings}"
        )

    warnings.extend(oml_stats.warnings)
    return OmlBuildOutput(
        oml_shape=oml_shape,
        body_only_shape=body_shape,
        body_stats=body_stats,
        oml_stats=oml_stats,
        segment_meta=tuple(seg_meta),
        warnings=tuple(warnings),
    )


def build_axisymmetric_2d_body_face_mm(
    rocket: RocketModel,
    cad_config: CadConfig,
) -> tuple[TopoDS_Face, tuple[ProfileSegmentMeta, ...]]:
    """Build planar XY (y >= 0) half-meridian face of the rocket body for 2D axisymmetric CFD."""
    pts_mm, seg_meta = build_merged_meridian_points_mm(rocket, cad_config)
    if len(pts_mm) < 2:
        raise GeometryBuildError(f"Rocket '{rocket.name}' produced empty 2D profile.")

    closed_pts: list[tuple[float, float]] = []
    if pts_mm[0][1] > 1e-4:
        closed_pts.append((pts_mm[0][0], 0.0))
    closed_pts.extend(pts_mm)
    if pts_mm[-1][1] > 1e-4:
        closed_pts.append((pts_mm[-1][0], 0.0))

    poly = BRepBuilderAPI_MakePolygon()
    for px, py in closed_pts:
        poly.Add(gp_Pnt(px, py, 0.0))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return face, tuple(seg_meta)


def _revolve_spans_mm(
    spans: list[MeridianCurveSpan],
    seam_angle_deg: float,
) -> TopoDS_Shape:
    """Build closed 2D meridian wire using smooth B-splines for curved sections and revolve."""
    wire_maker = BRepBuilderAPI_MakeWire()
    first_pt = spans[0].points_mm[0]
    last_pt = spans[-1].points_mm[-1]

    if first_pt[1] > 1e-4:
        e_front = BRepBuilderAPI_MakeEdge(
            gp_Pnt(first_pt[0], 0.0, 0.0), gp_Pnt(first_pt[0], first_pt[1], 0.0)
        ).Edge()
        wire_maker.Add(e_front)

    for sp in spans:
        pts = list(sp.points_mm)
        if len(pts) < 2:
            continue
        if sp.is_curved and len(pts) >= 4:
            dedup: list[tuple[float, float]] = [pts[0]]
            for p in pts[1:]:
                if math.hypot(p[0] - dedup[-1][0], p[1] - dedup[-1][1]) > 1e-3:
                    dedup.append(p)
            if len(dedup) >= 4:
                arr = TColgp_HArray1OfPnt(1, len(dedup))
                for idx, (px, py) in enumerate(dedup, start=1):
                    arr.SetValue(idx, gp_Pnt(px, py, 0.0))
                interp = GeomAPI_Interpolate(arr, False, 1e-4)
                interp.Perform()
                if interp.IsDone():
                    wire_maker.Add(BRepBuilderAPI_MakeEdge(interp.Curve()).Edge())
                    continue
        # Straight segment or fallback polyline
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            if math.hypot(x2 - x1, y2 - y1) > 1e-4:
                wire_maker.Add(
                    BRepBuilderAPI_MakeEdge(gp_Pnt(x1, y1, 0.0), gp_Pnt(x2, y2, 0.0)).Edge()
                )

    if last_pt[1] > 1e-4:
        wire_maker.Add(
            BRepBuilderAPI_MakeEdge(
                gp_Pnt(last_pt[0], last_pt[1], 0.0), gp_Pnt(last_pt[0], 0.0, 0.0)
            ).Edge()
        )
    if abs(last_pt[0] - first_pt[0]) > 1e-4:
        wire_maker.Add(
            BRepBuilderAPI_MakeEdge(
                gp_Pnt(last_pt[0], 0.0, 0.0), gp_Pnt(first_pt[0], 0.0, 0.0)
            ).Edge()
        )

    if not wire_maker.IsDone():
        raise GeometryBuildError("Failed to construct smooth meridian wire.")

    face = BRepBuilderAPI_MakeFace(wire_maker.Wire()).Face()
    axis_x = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    if abs(seam_angle_deg) > 1e-4:
        trsf = gp_Trsf()
        trsf.SetRotation(axis_x, math.radians(seam_angle_deg))
        face = TopoDS.Face_s(BRepBuilderAPI_Transform(face, trsf, True).Shape())

    revol = BRepPrimAPI_MakeRevol(face, axis_x, 2.0 * math.pi)
    revol.Build()
    if not revol.IsDone():
        raise GeometryBuildError("360-degree revolution of OML body failed.")

    unifier = ShapeUpgrade_UnifySameDomain(revol.Shape(), True, True, False)
    unifier.Build()
    return unifier.Shape()


def _fuse_shapes_robust(
    base: TopoDS_Shape,
    tools: list[TopoDS_Shape],
    warnings: list[str],
) -> TopoDS_Shape:
    """Boolean-fuse attachment solids onto body using batch fuse with sequential fallback."""
    try:
        from OCP.TopTools import TopTools_ListOfShape

        args = TopTools_ListOfShape()
        args.Append(base)
        tool_list = TopTools_ListOfShape()
        for t in tools:
            tool_list.Append(t)

        fuser = BRepAlgoAPI_Fuse()
        fuser.SetArguments(args)
        fuser.SetTools(tool_list)
        fuser.SetFuzzyValue(1e-3)
        fuser.Build()
        if fuser.IsDone():
            res = fuser.Shape()
            if validate_brep_shape(res).is_watertight:
                return res
    except Exception:
        pass

    # Sequential fuse fallback
    current = base
    for idx, tool in enumerate(tools):
        fuser = BRepAlgoAPI_Fuse(current, tool)
        fuser.SetFuzzyValue(2e-3)
        fuser.Build()
        if not fuser.IsDone():
            warnings.append(f"Attachment solid #{idx + 1} failed boolean fuse and was skipped.")
            continue
        current = fuser.Shape()
    return current


def _strip_internal_edges(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Remove redundant TopAbs_INTERNAL slit edges created by flush coplanar fin roots."""
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    internal_edges = []
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        if edge.Orientation() == TopAbs_INTERNAL:
            internal_edges.append(edge)
        exp.Next()
    if not internal_edges:
        return shape
    try:
        reshaper = BRepTools_ReShape()
        for e in internal_edges:
            reshaper.Remove(e)
        cleaned = reshaper.Apply(shape)
        if validate_brep_shape(cleaned).is_valid:
            return cleaned
    except Exception:
        pass
    return shape


def _apply_root_fillets_safe(
    oml_shape: TopoDS_Shape,
    rocket: RocketModel,
    cad_config: CadConfig,
    warnings: list[str],
) -> TopoDS_Shape:
    """Attempt CFD-safe root fillets on body-fin edges; gracefully fall back if rejected."""
    r_fillet_m = cad_config.fillet_radius_override_m
    if r_fillet_m is None:
        fin_fillets = [f.fillet_radius for f in rocket.all_fins if f.fillet_radius > 0.0]
        r_fillet_m = min(fin_fillets) if fin_fillets else 0.004

    r_mm = max(0.5, min(8.0, r_fillet_m * SCALE_M_TO_MM))
    body_r_max_mm = rocket.max_radius * SCALE_M_TO_MM

    try:
        fillet = BRepFilletAPI_MakeFillet(oml_shape)
        exp = TopExp_Explorer(oml_shape, TopAbs_EDGE)
        candidate_count = 0
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            props = GProp_GProps()
            BRepGProp.LinearProperties_s(edge, props)
            c = props.CentreOfMass()
            radial_dist = math.hypot(c.Y(), c.Z())
            # Heuristic: edges near body surface with length > 5 mm are likely fin roots
            if (
                radial_dist <= body_r_max_mm * 1.15
                and radial_dist >= 2.0
                and float(props.Mass()) >= 5.0
            ):
                fillet.Add(r_mm, edge)
                candidate_count += 1
            exp.Next()

        if candidate_count == 0:
            return oml_shape
        fillet.Build()
        if fillet.IsDone():
            candidate = fillet.Shape()
            if validate_brep_shape(candidate).is_watertight:
                return candidate
    except Exception:
        pass

    warnings.append("Fin root filleting could not complete cleanly; kept sharp CFD-clean roots.")
    return oml_shape


def count_faces(shape: TopoDS_Shape) -> int:
    """Return number of faces in a TopoDS_Shape."""
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n += 1
        exp.Next()
    return n
