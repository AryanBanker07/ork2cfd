"""Fluid enclosure subtraction, symmetry slicing, BOIs, and boundary face classification."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeRevol
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape

from ..exceptions import GeometryBuildError
from ..models.cfd import CadConfig, DomainConfig, DomainMode, EnclosureShape
from ..models.results import GeometryReport, SolverRecommendation
from ..models.rocket import RocketModel
from .oml_builder import OmlBuildOutput, build_axisymmetric_2d_body_face_mm, build_rocket_oml
from .profiles import SCALE_M_TO_MM, ProfileSegmentMeta, sanitize_zone_name, stage_zone_token
from .validator import validate_brep_shape


@dataclass(frozen=True)
class FaceZoneEntry:
    """Associates an OCCT TopoDS_Face with its Named Selection zone and metadata."""

    face: TopoDS_Face
    zone_name: str
    centroid_m: tuple[float, float, float]
    area_m2: float
    bbox_min_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DomainBuildOutput:
    """Complete CFD geometry package ready for XCAF STEP export."""

    main_shape: TopoDS_Shape
    main_name: str
    boi_solids: tuple[tuple[str, TopoDS_Shape], ...]
    face_zones: tuple[FaceZoneEntry, ...]
    report: GeometryReport
    bbox_domain_min_m: tuple[float, float, float]
    bbox_domain_max_m: tuple[float, float, float]


_GENERIC_FIN_NAMES = frozenset(
    {
        "trapezoidal_fin_set",
        "elliptical_fin_set",
        "freeform_fin_set",
        "tube_fin_set",
        "fin_set",
    }
)


def _wrap_oml_domain_output(
    oml_out: OmlBuildOutput,
    faces: list[FaceZoneEntry],
    rocket: RocketModel,
    warnings: list[str],
) -> DomainBuildOutput:
    """Package an OML solid and its classified faces into a DomainBuildOutput."""
    zone_counts = _count_zones(faces)
    report = GeometryReport(
        is_valid=oml_out.oml_stats.is_valid,
        is_watertight=oml_out.oml_stats.is_watertight,
        solid_count=oml_out.oml_stats.solid_count,
        free_edge_count=oml_out.oml_stats.free_edge_count,
        min_edge_length_m=oml_out.oml_stats.min_edge_length_m,
        sliver_face_count=oml_out.oml_stats.sliver_face_count,
        oml_volume_m3=oml_out.oml_stats.volume_m3,
        oml_wetted_area_m2=oml_out.oml_stats.surface_area_m2,
        oml_frontal_area_m2=rocket.reference_area,
        domain_volume_m3=oml_out.oml_stats.volume_m3,
        bbox_min_m=oml_out.oml_stats.bbox_min_m,
        bbox_max_m=oml_out.oml_stats.bbox_max_m,
        named_selections=zone_counts,
        boi_names=(),
        warnings=tuple(warnings),
    )
    return DomainBuildOutput(
        main_shape=oml_out.oml_shape,
        main_name="rocket_oml",
        boi_solids=(),
        face_zones=tuple(faces),
        report=report,
        bbox_domain_min_m=oml_out.oml_stats.bbox_min_m,
        bbox_domain_max_m=oml_out.oml_stats.bbox_max_m,
    )


def build_rocket_component_domain(
    rocket: RocketModel,
    cad_config: CadConfig,
) -> DomainBuildOutput:
    """Build watertight rocket-only B-Rep with distinct per-component Named Selections."""
    oml_out = build_rocket_oml(rocket, cad_config, preserve_component_faces=True)
    fin_intervals = _build_fin_zone_intervals(rocket)
    app_intervals = _build_appendage_intervals(rocket, cad_config)
    L_mm = rocket.total_length * SCALE_M_TO_MM

    faces = _classify_oml_faces(
        oml_out.oml_shape,
        rocket,
        lambda face, cx, r_max_surf, max_excess: _classify_component_rocket_face(
            face=face,
            cx=cx,
            r_max_surf=r_max_surf,
            max_excess_mm=max_excess,
            L_mm=L_mm,
            rocket=rocket,
            segment_meta=oml_out.segment_meta,
            fin_intervals=fin_intervals,
            app_intervals=app_intervals,
        ),
    )
    return _wrap_oml_domain_output(oml_out, faces, rocket, list(oml_out.warnings))


def build_cfd_domain(
    rocket: RocketModel,
    cad_config: CadConfig,
    domain_config: DomainConfig,
    solver_rec: SolverRecommendation,
) -> DomainBuildOutput:
    """Construct CFD-ready OML or fluid-domain geometry with BOIs and Named Selections."""
    warnings: list[str] = []
    mode = domain_config.mode

    if mode == DomainMode.AXISYMMETRIC_2D:
        return _build_2d_axisymmetric_domain(rocket, cad_config, domain_config, solver_rec)

    # Choose seam angle safely away from symmetry cut planes
    seam_deg = None
    if mode == DomainMode.SYMMETRY_HALF:
        seam_deg = -90.0
    elif mode in {DomainMode.SYMMETRY_SECTOR, DomainMode.SYMMETRY_HALF_SECTOR}:
        first_rot = rocket.all_fins[0].rotation_deg if rocket.all_fins else 0.0
        seam_deg = first_rot + 180.0

    oml_out = build_rocket_oml(rocket, cad_config, seam_angle_deg=seam_deg)
    warnings.extend(oml_out.warnings)

    if mode == DomainMode.OML_ONLY:
        L_mm = rocket.total_length * SCALE_M_TO_MM
        faces = _classify_oml_faces(
            oml_out.oml_shape,
            rocket,
            lambda face, cx, r_max_surf, max_excess: _classify_rocket_wall_face(
                face, cx, r_max_surf, max_excess, L_mm, rocket, oml_out.segment_meta
            ),
        )
        return _wrap_oml_domain_output(oml_out, faces, rocket, warnings)

    # Compute enclosure dimensions in mm
    L_m = rocket.total_length
    up_L = domain_config.upstream_L or solver_rec.recommended_upstream_L
    down_L = domain_config.downstream_L or solver_rec.recommended_downstream_L
    rad_L = domain_config.radial_L or solver_rec.recommended_radial_L

    r_far_m = max(rad_L * L_m, 4.5 * rocket.max_fin_span_diameter)
    x_min_mm = -up_L * L_m * SCALE_M_TO_MM
    x_max_mm = (1.0 + down_L) * L_m * SCALE_M_TO_MM
    r_far_mm = r_far_m * SCALE_M_TO_MM

    enclosure = _build_enclosure_solid_mm(
        shape_kind=domain_config.shape,
        x_min_mm=x_min_mm,
        x_max_mm=x_max_mm,
        r_far_mm=r_far_mm,
        seam_angle_deg=seam_deg or 45.0,
    )

    cutter = BRepAlgoAPI_Cut(enclosure, oml_out.oml_shape)
    cutter.SetFuzzyValue(1e-3)
    cutter.Build()
    if not cutter.IsDone():
        raise GeometryBuildError("Boolean subtraction of rocket OML from fluid enclosure failed.")
    fluid_shape = cutter.Shape()

    # Apply symmetry slicing if requested
    clip_solid = None
    if mode == DomainMode.SYMMETRY_HALF:
        if rocket.has_cant:
            warnings.append("Fin cant breaks bilateral symmetry; Symmetry Half is approximate.")
        clip_solid = BRepPrimAPI_MakeBox(
            gp_Pnt(x_min_mm - 500.0, -r_far_mm - 500.0, 0.0),
            gp_Pnt(x_max_mm + 500.0, r_far_mm + 500.0, r_far_mm + 500.0),
        ).Shape()
    elif mode in {DomainMode.SYMMETRY_SECTOR, DomainMode.SYMMETRY_HALF_SECTOR}:
        n_sym = rocket.rotational_symmetry_order
        if n_sym <= 1:
            warnings.append(
                "Incompatible fin counts across stages (rotational symmetry order = 1); "
                "keeping full 360° fluid domain."
            )
        else:
            first_rot = rocket.all_fins[0].rotation_deg if rocket.all_fins else 0.0
            if mode == DomainMode.SYMMETRY_SECTOR:
                if rocket.has_cant:
                    warnings.append(
                        "Fin cant breaks bilateral mirror symmetry; use rotational-periodic "
                        "BCs on sector cut planes in Fluent."
                    )
                elif not rocket.has_sector_mirror_symmetry:
                    warnings.append(
                        "Inter-stage fin rotation offset breaks bilateral mirror symmetry; "
                        "use rotational-periodic BCs on sector cut planes."
                    )
                start_angle_deg = first_rot - (180.0 / n_sym)
                sector_angle_rad = (2.0 * math.pi) / n_sym
            else:
                if rocket.has_cant or not rocket.has_sector_mirror_symmetry:
                    warnings.append(
                        "Fin cant or inter-stage fin stagger is incompatible with "
                        "half-sector mirror symmetry."
                    )
                start_angle_deg = first_rot
                sector_angle_rad = math.pi / n_sym

            clip_solid = _build_sector_wedge_mm(
                x_min_mm=x_min_mm - 500.0,
                x_max_mm=x_max_mm + 500.0,
                r_outer_mm=r_far_mm + 500.0,
                start_angle_deg=start_angle_deg,
                sweep_angle_rad=sector_angle_rad,
            )

    if clip_solid is not None:
        common = BRepAlgoAPI_Common(fluid_shape, clip_solid)
        common.SetFuzzyValue(1e-3)
        common.Build()
        if not common.IsDone():
            raise GeometryBuildError(f"Symmetry slicing failed for mode '{mode.value}'.")
        fluid_shape = common.Shape()

    dom_stats = validate_brep_shape(fluid_shape)
    if not dom_stats.is_watertight:
        raise GeometryBuildError(
            f"Constructed fluid domain failed watertightness check: {dom_stats.warnings}"
        )
    warnings.extend(dom_stats.warnings)

    # Build optional Bodies of Influence (BOIs) unclipped so BOI faces never lie coplanar on
    # symmetry cut planes and cannot trigger multi-connection overlaps in Fluent Meshing.
    boi_solids: list[tuple[str, TopoDS_Shape]] = (
        _build_bois_mm(rocket, domain_config) if domain_config.boi.enabled else []
    )

    faces = _classify_fluid_domain_faces(
        fluid_shape=fluid_shape,
        rocket=rocket,
        segment_meta=oml_out.segment_meta,
        x_min_mm=x_min_mm,
        x_max_mm=x_max_mm,
        r_far_mm=r_far_mm,
        shape_kind=domain_config.shape,
        mode=mode,
    )
    zone_counts = _count_zones(faces)

    report = GeometryReport(
        is_valid=dom_stats.is_valid,
        is_watertight=dom_stats.is_watertight,
        solid_count=dom_stats.solid_count,
        free_edge_count=dom_stats.free_edge_count,
        min_edge_length_m=dom_stats.min_edge_length_m,
        sliver_face_count=dom_stats.sliver_face_count,
        oml_volume_m3=oml_out.oml_stats.volume_m3,
        oml_wetted_area_m2=oml_out.oml_stats.surface_area_m2,
        oml_frontal_area_m2=rocket.reference_area,
        domain_volume_m3=dom_stats.volume_m3,
        bbox_min_m=dom_stats.bbox_min_m,
        bbox_max_m=dom_stats.bbox_max_m,
        named_selections=zone_counts,
        boi_names=tuple(name for name, _ in boi_solids),
        warnings=tuple(warnings),
    )

    return DomainBuildOutput(
        main_shape=fluid_shape,
        main_name="fluid_domain",
        boi_solids=tuple(boi_solids),
        face_zones=tuple(faces),
        report=report,
        bbox_domain_min_m=dom_stats.bbox_min_m,
        bbox_domain_max_m=dom_stats.bbox_max_m,
    )


def _build_2d_axisymmetric_domain(
    rocket: RocketModel,
    cad_config: CadConfig,
    domain_config: DomainConfig,
    solver_rec: SolverRecommendation,
) -> DomainBuildOutput:
    """Build planar XY (y >= 0) 2D axisymmetric fluid domain face."""
    body_face, seg_meta = build_axisymmetric_2d_body_face_mm(rocket, cad_config)
    L_m = rocket.total_length
    up_L = domain_config.upstream_L or solver_rec.recommended_upstream_L
    down_L = domain_config.downstream_L or solver_rec.recommended_downstream_L
    rad_L = domain_config.radial_L or solver_rec.recommended_radial_L

    r_far_mm = max(rad_L * L_m, 4.0 * rocket.max_diameter) * SCALE_M_TO_MM
    x_min_mm = -up_L * L_m * SCALE_M_TO_MM
    x_max_mm = (1.0 + down_L) * L_m * SCALE_M_TO_MM

    poly = BRepBuilderAPI_MakePolygon()
    for px, py in [
        (x_min_mm, 0.0),
        (x_max_mm, 0.0),
        (x_max_mm, r_far_mm),
        (x_min_mm, r_far_mm),
    ]:
        poly.Add(gp_Pnt(px, py, 0.0))
    poly.Close()
    outer_face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()

    cut = BRepAlgoAPI_Cut(outer_face, body_face)
    cut.Build()
    if not cut.IsDone():
        raise GeometryBuildError("Failed to subtract 2D body from axisymmetric domain.")
    dom_2d = cut.Shape()
    stats = validate_brep_shape(dom_2d, expect_2d=True)

    face_entries = [
        FaceZoneEntry(
            face=TopoDS.Face_s(Exp.Current()),
            zone_name="fluid_domain_2d",
            centroid_m=(0.0, 0.5 * r_far_mm / SCALE_M_TO_MM, 0.0),
            area_m2=stats.surface_area_m2,
            bbox_min_m=stats.bbox_min_m,
            bbox_max_m=stats.bbox_max_m,
        )
        for Exp in [TopExp_Explorer(dom_2d, TopAbs_FACE)]
        if Exp.More()
    ]

    report = GeometryReport(
        is_valid=stats.is_valid,
        is_watertight=True,
        solid_count=0,
        free_edge_count=0,
        min_edge_length_m=stats.min_edge_length_m,
        sliver_face_count=stats.sliver_face_count,
        oml_volume_m3=0.0,
        oml_wetted_area_m2=0.0,
        oml_frontal_area_m2=rocket.reference_area,
        domain_volume_m3=0.0,
        bbox_min_m=stats.bbox_min_m,
        bbox_max_m=stats.bbox_max_m,
        named_selections={"fluid_domain_2d": len(face_entries)},
        boi_names=(),
        warnings=(
            "2D axisymmetric mode omits 3D fins; verify axis/farfield edge zones in Fluent.",
        )
        + stats.warnings,
    )
    return DomainBuildOutput(
        main_shape=dom_2d,
        main_name="fluid_domain_2d",
        boi_solids=(),
        face_zones=tuple(face_entries),
        report=report,
        bbox_domain_min_m=stats.bbox_min_m,
        bbox_domain_max_m=stats.bbox_max_m,
    )


def _build_enclosure_solid_mm(
    shape_kind: EnclosureShape,
    x_min_mm: float,
    x_max_mm: float,
    r_far_mm: float,
    seam_angle_deg: float,
) -> TopoDS_Shape:
    """Create the outer farfield solid volume in mm."""
    if shape_kind == EnclosureShape.CYLINDER:
        ax = gp_Ax2(gp_Pnt(x_min_mm, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
        cyl = BRepPrimAPI_MakeCylinder(ax, r_far_mm, x_max_mm - x_min_mm).Shape()
        return _rotate_x_mm(cyl, seam_angle_deg)

    if shape_kind == EnclosureShape.BOX:
        return BRepPrimAPI_MakeBox(
            gp_Pnt(x_min_mm, -r_far_mm, -r_far_mm),
            gp_Pnt(x_max_mm, r_far_mm, r_far_mm),
        ).Shape()

    # BULLET: smooth quarter-ellipse spline dome from (x_min, 0) to (0, r_far) + cylinder to x_max
    n_arc = 24
    a = abs(x_min_mm)
    arr = TColgp_HArray1OfPnt(1, n_arc + 1)
    for i in range(n_arc + 1):
        theta = (i / n_arc) * (0.5 * math.pi)
        px = -a * math.cos(theta)
        py = r_far_mm * math.sin(theta)
        arr.SetValue(i + 1, gp_Pnt(px, py, 0.0))

    wire = BRepBuilderAPI_MakeWire()
    interp = GeomAPI_Interpolate(arr, False, 1e-3)
    interp.Perform()
    if interp.IsDone():
        wire.Add(BRepBuilderAPI_MakeEdge(interp.Curve()).Edge())
    else:
        for i in range(1, n_arc + 1):
            wire.Add(BRepBuilderAPI_MakeEdge(arr.Value(i), arr.Value(i + 1)).Edge())

    wire.Add(
        BRepBuilderAPI_MakeEdge(gp_Pnt(0.0, r_far_mm, 0.0), gp_Pnt(x_max_mm, r_far_mm, 0.0)).Edge()
    )
    wire.Add(
        BRepBuilderAPI_MakeEdge(
            gp_Pnt(x_max_mm, r_far_mm, 0.0), gp_Pnt(x_max_mm, 0.0, 0.0)
        ).Edge()
    )
    wire.Add(
        BRepBuilderAPI_MakeEdge(gp_Pnt(x_max_mm, 0.0, 0.0), gp_Pnt(x_min_mm, 0.0, 0.0)).Edge()
    )

    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    if abs(seam_angle_deg) > 1e-4:
        trsf = gp_Trsf()
        trsf.SetRotation(
            gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)),
            math.radians(seam_angle_deg),
        )
        face = TopoDS.Face_s(BRepBuilderAPI_Transform(face, trsf, True).Shape())

    revol = BRepPrimAPI_MakeRevol(
        face, gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), 2.0 * math.pi
    )
    revol.Build()
    if not revol.IsDone():
        raise GeometryBuildError("Bullet enclosure revolution failed.")
    return revol.Shape()


def _build_sector_wedge_mm(
    x_min_mm: float,
    x_max_mm: float,
    r_outer_mm: float,
    start_angle_deg: float,
    sweep_angle_rad: float,
) -> TopoDS_Shape:
    """Create a 3D pie-wedge solid spanning [start_angle_deg, start_angle_deg + sweep]."""
    poly = BRepBuilderAPI_MakePolygon()
    for px, py in [
        (x_min_mm, 0.0),
        (x_max_mm, 0.0),
        (x_max_mm, r_outer_mm),
        (x_min_mm, r_outer_mm),
    ]:
        poly.Add(gp_Pnt(px, py, 0.0))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    if abs(start_angle_deg) > 1e-4:
        face = TopoDS.Face_s(_rotate_x_mm(face, start_angle_deg))

    revol = BRepPrimAPI_MakeRevol(
        face, gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), sweep_angle_rad
    )
    revol.Build()
    if not revol.IsDone():
        raise GeometryBuildError("Symmetry sector wedge generation failed.")
    return revol.Shape()


def _build_bois_mm(
    rocket: RocketModel,
    domain_config: DomainConfig,
) -> list[tuple[str, TopoDS_Shape]]:
    """Construct nearfield and wake Body-of-Influence solids in mm."""
    boi = domain_config.boi
    L_m = rocket.total_length
    r_fin_max_m = 0.5 * rocket.max_fin_span_diameter

    # Nearfield cylinder enclosing entire rocket and fins
    x_nf_start = -boi.nearfield_upstream_L * L_m * SCALE_M_TO_MM
    x_nf_end = (1.0 + boi.nearfield_downstream_L) * L_m * SCALE_M_TO_MM
    r_nf = max(r_fin_max_m * boi.nearfield_radius_factor, rocket.max_radius * 2.5) * SCALE_M_TO_MM

    ax_nf = gp_Ax2(gp_Pnt(x_nf_start, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    nf_solid = BRepPrimAPI_MakeCylinder(ax_nf, r_nf, x_nf_end - x_nf_start).Shape()

    # Wake tapered cone extending downstream behind nearfield BOI with a positive axial gap
    gap_mm = max(5.0, 0.005 * L_m * SCALE_M_TO_MM)
    x_wk_start = x_nf_end + gap_mm
    x_wk_end = max(x_wk_start + 50.0, (1.0 + boi.wake_length_L) * L_m * SCALE_M_TO_MM)
    r_wk_start = r_nf
    r_wk_end = max(
        r_nf * 1.15,
        r_fin_max_m * boi.wake_end_radius_factor * SCALE_M_TO_MM,
    )

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone

    ax_wk = gp_Ax2(gp_Pnt(x_wk_start, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    wk_solid = BRepPrimAPI_MakeCone(ax_wk, r_wk_start, r_wk_end, x_wk_end - x_wk_start).Shape()

    return [("boi_nearfield", nf_solid), ("boi_wake", wk_solid)]


def _rotate_x_mm(shape: TopoDS_Shape, angle_deg: float) -> TopoDS_Shape:
    """Rotate shape around the X-axis by angle_deg."""
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), math.radians(angle_deg))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _face_metrics(
    face: TopoDS_Face,
    rocket: RocketModel | None = None,
) -> tuple[tuple[float, float, float], float, tuple[float, ...], float, float]:
    """Return centroid (mm), area (mm^2), bbox (mm), max_radial_mm, max_body_excess_mm."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    c = props.CentreOfMass()
    area = float(props.Mass())
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(face, box, False, False)

    r_max = math.hypot(c.Y(), c.Z())
    max_excess = 0.0
    if rocket is not None:
        cx_m = max(0.0, min(rocket.total_length, c.X() / SCALE_M_TO_MM))
        max_excess = max(0.0, r_max - rocket.radius_at_x(cx_m) * SCALE_M_TO_MM)

    e_exp = TopExp_Explorer(face, TopAbs_EDGE)
    while e_exp.More():
        edge = TopoDS.Edge_s(e_exp.Current())
        curve = BRepAdaptor_Curve(edge)
        t0, t1 = curve.FirstParameter(), curve.LastParameter()
        for i in range(5):
            t = t0 + (t1 - t0) * (i / 4.0)
            p = curve.Value(t)
            r = math.hypot(p.Y(), p.Z())
            r_max = max(r_max, r)
            if rocket is not None:
                px_m = max(0.0, min(rocket.total_length, p.X() / SCALE_M_TO_MM))
                r_body = rocket.radius_at_x(px_m) * SCALE_M_TO_MM
                max_excess = max(max_excess, r - r_body)
        e_exp.Next()

    return (c.X(), c.Y(), c.Z()), area, box.Get(), r_max, max_excess


def _classify_oml_faces(
    oml_shape: TopoDS_Shape,
    rocket: RocketModel,
    zone_fn: Callable[[TopoDS_Face, float, float, float], str],
) -> list[FaceZoneEntry]:
    """Classify all faces on an OML solid using a per-face classifier callback."""
    inv_s = 1.0 / SCALE_M_TO_MM
    entries: list[FaceZoneEntry] = []

    exp = TopExp_Explorer(oml_shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        (cx, cy, cz), area_mm2, bbox, r_max_surf, max_excess = _face_metrics(face, rocket)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        zone = zone_fn(face, cx, r_max_surf, max_excess)
        entries.append(
            FaceZoneEntry(
                face=face,
                zone_name=zone,
                centroid_m=(cx * inv_s, cy * inv_s, cz * inv_s),
                area_m2=area_mm2 * (inv_s**2),
                bbox_min_m=(xmin * inv_s, ymin * inv_s, zmin * inv_s),
                bbox_max_m=(xmax * inv_s, ymax * inv_s, zmax * inv_s),
            )
        )
        exp.Next()
    return entries


def _build_fin_zone_intervals(rocket: RocketModel) -> list[tuple[float, float, str]]:
    """Build axial (x_lo_m, x_hi_m, zone_name) envelopes for every FinSet across stages."""
    multi_or_many = rocket.is_multistage or len(rocket.all_fins) > 1
    entries: list[tuple[float, float, str, str]] = []
    raw_counts: dict[str, int] = {}

    for st_idx, stage in enumerate(rocket.stages, start=1):
        st_tok = stage_zone_token(stage.name, st_idx)
        for comp in stage.components:
            for fin in comp.fins:
                if fin.fin_type == "freeform" and fin.points:
                    xs = [fin.global_root_le_x + px for px, _ in fin.points]
                    x_lo = min(xs)
                    x_hi = max(xs)
                else:
                    tip_le = fin.global_root_le_x + fin.sweep_length
                    tip_te = tip_le + fin.tip_chord
                    x_lo = min(fin.global_root_le_x, tip_le)
                    x_hi = max(fin.global_root_te_x, tip_te)

                raw = sanitize_zone_name(fin.name, "fins")
                if raw in _GENERIC_FIN_NAMES:
                    base = f"{st_tok}_fins" if multi_or_many else "fins"
                else:
                    base = raw
                raw_counts[base] = raw_counts.get(base, 0) + 1
                entries.append((x_lo, x_hi, st_tok, base))

    candidates: list[tuple[float, float, str]] = []
    cand_counts: dict[str, int] = {}
    for x_lo, x_hi, st_tok, base in entries:
        name = (
            f"{st_tok}_{base}"
            if raw_counts[base] > 1 and not base.startswith(f"{st_tok}_")
            else base
        )
        candidates.append((x_lo, x_hi, name))
        cand_counts[name] = cand_counts.get(name, 0) + 1

    seen: dict[str, int] = {}
    resolved: list[tuple[float, float, str]] = []
    for x_lo, x_hi, name in candidates:
        if cand_counts[name] == 1:
            resolved.append((x_lo, x_hi, name))
        else:
            idx = seen.get(name, 0) + 1
            seen[name] = idx
            resolved.append((x_lo, x_hi, f"{name}_{idx}"))
    return resolved


def _build_appendage_intervals(
    rocket: RocketModel,
    cad_config: CadConfig,
) -> list[tuple[float, float, str]]:
    """Build axial intervals for enabled external rail buttons and launch lugs."""
    intervals: list[tuple[float, float, str]] = []
    if cad_config.include_rail_buttons:
        for comp in rocket.all_components:
            for rb in comp.rail_buttons:
                half_r = max(0.002, 0.5 * rb.outer_diameter)
                for idx in range(max(1, rb.instance_count)):
                    xm = rb.axial_x + idx * rb.instance_separation
                    intervals.append((xm - half_r, xm + half_r, "rail_buttons"))
    if cad_config.include_launch_lugs:
        for comp in rocket.all_components:
            for lug in comp.launch_lugs:
                for idx in range(max(1, lug.instance_count)):
                    xm = lug.axial_x + idx * lug.instance_separation
                    intervals.append((xm, xm + max(0.004, lug.length), "launch_lugs"))
    return intervals


def _match_axial_interval(
    cx_m: float,
    intervals: list[tuple[float, float, str]],
    tol_m: float,
) -> str | None:
    """Return zone name of the best-matching axial interval within tol_m."""
    hits = [
        (abs(cx_m - 0.5 * (x0 + x1)), name)
        for x0, x1, name in intervals
        if x0 - tol_m <= cx_m <= x1 + tol_m
    ]
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    return hits[0][1]


def _classify_component_rocket_face(
    face: TopoDS_Face,
    cx: float,
    r_max_surf: float,
    max_excess_mm: float,
    L_mm: float,
    rocket: RocketModel,
    segment_meta: tuple[ProfileSegmentMeta, ...],
    fin_intervals: list[tuple[float, float, str]],
    app_intervals: list[tuple[float, float, str]],
) -> str:
    """Classify an OML face into per-component body, fin-set, appendage, or cap zones."""
    cx_m = cx / SCALE_M_TO_MM
    r_body_mm = rocket.radius_at_x(max(0.0, min(rocket.total_length, cx_m))) * SCALE_M_TO_MM

    surf = BRepAdaptor_Surface(face)
    if surf.GetType() == GeomAbs_Plane:
        n = surf.Plane().Axis().Direction()
        if abs(n.X()) > 0.95 and r_max_surf <= r_body_mm + 2.0:
            if abs(cx - L_mm) < 2.0:
                return "rocket_base"
            if abs(cx) < 2.0:
                return "rocket_fore_cap"

    if max_excess_mm > 1.0:
        app_hit = _match_axial_interval(cx_m, app_intervals, tol_m=0.004)
        fin_hit = _match_axial_interval(cx_m, fin_intervals, tol_m=0.006)
        if max_excess_mm > 2.5 and fin_hit is not None:
            return fin_hit
        if app_hit is not None:
            return app_hit
        if fin_hit is not None:
            return fin_hit
        if max_excess_mm > 2.5 and fin_intervals:
            return min(
                fin_intervals,
                key=lambda item: abs(cx_m - 0.5 * (item[0] + item[1])),
            )[2]

    for seg in segment_meta:
        if seg.axial_start_m + 1e-4 <= cx_m <= seg.axial_end_m - 1e-4:
            return seg.zone_name
    for seg in segment_meta:
        if seg.axial_start_m - 1e-4 <= cx_m <= seg.axial_end_m + 1e-4:
            return seg.zone_name
    if segment_meta:
        return min(
            segment_meta,
            key=lambda seg: abs(cx_m - 0.5 * (seg.axial_start_m + seg.axial_end_m)),
        ).zone_name
    return "rocket_body"


def _classify_fluid_domain_faces(
    fluid_shape: TopoDS_Shape,
    rocket: RocketModel,
    segment_meta: tuple[ProfileSegmentMeta, ...],
    x_min_mm: float,
    x_max_mm: float,
    r_far_mm: float,
    shape_kind: EnclosureShape,
    mode: DomainMode,
) -> list[FaceZoneEntry]:
    """Classify every face of the fluid domain into farfield/inlet/outlet/symmetry/rocket zones."""
    L_mm = rocket.total_length * SCALE_M_TO_MM
    inv_s = 1.0 / SCALE_M_TO_MM
    entries: list[FaceZoneEntry] = []

    exp = TopExp_Explorer(fluid_shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        (cx, cy, cz), area_mm2, bbox, r_max_surf, max_excess = _face_metrics(face, rocket)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox

        zone = _classify_outer_or_symmetry_face(
            face=face,
            cx=cx,
            r_max_surf=r_max_surf,
            xmin=xmin,
            xmax=xmax,
            x_min_mm=x_min_mm,
            x_max_mm=x_max_mm,
            r_far_mm=r_far_mm,
            L_mm=L_mm,
            mode=mode,
        )
        if zone is None:
            zone = _classify_rocket_wall_face(
                face, cx, r_max_surf, max_excess, L_mm, rocket, segment_meta
            )

        entries.append(
            FaceZoneEntry(
                face=face,
                zone_name=zone,
                centroid_m=(cx * inv_s, cy * inv_s, cz * inv_s),
                area_m2=area_mm2 * (inv_s**2),
                bbox_min_m=(xmin * inv_s, ymin * inv_s, zmin * inv_s),
                bbox_max_m=(xmax * inv_s, ymax * inv_s, zmax * inv_s),
            )
        )
        exp.Next()
    return entries


def _classify_outer_or_symmetry_face(
    face: TopoDS_Face,
    cx: float,
    r_max_surf: float,
    xmin: float,
    xmax: float,
    x_min_mm: float,
    x_max_mm: float,
    r_far_mm: float,
    L_mm: float,
    mode: DomainMode,
) -> str | None:
    """Identify outlet, inlet, farfield, or symmetry cut faces on the outer boundary."""
    surf = BRepAdaptor_Surface(face)
    is_plane = surf.GetType() == GeomAbs_Plane

    if is_plane:
        pln = surf.Plane()
        n = pln.Axis().Direction()
        if abs(n.X()) > 0.95 and abs(cx - x_max_mm) < 10.0:
            return "outlet"
        if abs(n.X()) > 0.95 and abs(cx - x_min_mm) < 10.0:
            return "inlet"
        if (
            mode
            in {
                DomainMode.SYMMETRY_HALF,
                DomainMode.SYMMETRY_SECTOR,
                DomainMode.SYMMETRY_HALF_SECTOR,
            }
            and abs(n.X()) < 0.05
            and r_max_surf > 0.70 * r_far_mm
            and (xmax - xmin) > max(L_mm, 0.5 * r_far_mm)
        ):
            return "symmetry"

    if r_max_surf > 0.75 * r_far_mm:
        return "farfield"

    return None


def _classify_rocket_wall_face(
    face: TopoDS_Face,
    cx: float,
    r_max_surf: float,
    max_excess_mm: float,
    L_mm: float,
    rocket: RocketModel,
    segment_meta: tuple[ProfileSegmentMeta, ...],
) -> str:
    """Classify a face belonging to the rocket OML surface."""
    cx_m = cx / SCALE_M_TO_MM
    r_body_mm = rocket.radius_at_x(max(0.0, min(rocket.total_length, cx_m))) * SCALE_M_TO_MM

    surf = BRepAdaptor_Surface(face)
    if surf.GetType() == GeomAbs_Plane:
        n = surf.Plane().Axis().Direction()
        if abs(n.X()) > 0.95 and abs(cx - L_mm) < 2.0 and r_max_surf <= r_body_mm + 2.0:
            return "rocket_base"

    # Fin surfaces protrude radially beyond the local axisymmetric body profile
    if max_excess_mm > 2.5:
        return "rocket_fins"

    for seg in segment_meta:
        if seg.axial_start_m - 1e-4 <= cx_m <= seg.axial_end_m + 1e-4:
            return seg.zone_name
    return "rocket_body"


def _count_zones(faces: list[FaceZoneEntry]) -> dict[str, int]:
    """Tally face counts per Named Selection zone."""
    counts: dict[str, int] = {}
    for entry in faces:
        counts[entry.zone_name] = counts.get(entry.zone_name, 0) + 1
    return dict(sorted(counts.items()))
