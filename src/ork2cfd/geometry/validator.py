"""B-Rep validation checks for watertightness, free edges, slivers, and reference area."""

from __future__ import annotations

from dataclasses import dataclass

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_INTERNAL, TopAbs_SOLID
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from .profiles import SCALE_M_TO_MM


@dataclass(frozen=True)
class BrepValidationStats:
    """Low-level OCCT B-Rep quality metrics in SI meters."""

    is_valid: bool
    is_watertight: bool
    solid_count: int
    free_edge_count: int
    min_edge_length_m: float
    sliver_face_count: int
    volume_m3: float
    surface_area_m2: float
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    warnings: tuple[str, ...]


def validate_brep_shape(
    shape: TopoDS_Shape,
    expect_2d: bool = False,
    sliver_area_threshold_mm2: float = 0.02,
    micro_edge_threshold_mm: float = 0.02,
) -> BrepValidationStats:
    """Inspect an OCCT TopoDS_Shape (built in mm) and return SI metrics + warnings."""
    analyzer = BRepCheck_Analyzer(shape)
    is_valid = bool(analyzer.IsValid())

    # Count solids
    solid_count = 0
    s_exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while s_exp.More():
        solid_count += 1
        s_exp.Next()

    # Check free edges (watertight 3D solid has >= 2 faces per edge)
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    free_edges = 0
    min_edge_mm = float("inf")
    for idx in range(1, edge_face_map.Extent() + 1):
        edge = TopoDS.Edge_s(edge_face_map.FindKey(idx))
        if BRep_Tool.Degenerated_s(edge) or edge.Orientation() == TopAbs_INTERNAL:
            continue
        adj_faces = edge_face_map.FindFromIndex(idx).Extent()
        if not expect_2d and adj_faces < 2:
            # Verify edge is not solely an internal edge within its single adjacent face
            single_face = TopoDS.Face_s(edge_face_map.FindFromIndex(idx).First())
            is_internal_only = False
            fexp = TopExp_Explorer(single_face, TopAbs_EDGE)
            while fexp.More():
                cur_e = TopoDS.Edge_s(fexp.Current())
                if cur_e.IsSame(edge) and cur_e.Orientation() == TopAbs_INTERNAL:
                    is_internal_only = True
                    break
                fexp.Next()
            if not is_internal_only:
                free_edges += 1
        l_props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, l_props)
        elen = float(l_props.Mass())
        if elen > 1e-9:
            min_edge_mm = min(min_edge_mm, elen)

    if min_edge_mm == float("inf"):
        min_edge_mm = 0.0

    # Check sliver faces
    sliver_faces = 0
    f_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while f_exp.More():
        face = TopoDS.Face_s(f_exp.Current())
        f_props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, f_props)
        if float(f_props.Mass()) < sliver_area_threshold_mm2:
            sliver_faces += 1
        f_exp.Next()

    # Volume & surface area
    v_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, v_props)
    vol_mm3 = float(v_props.Mass())

    a_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, a_props)
    area_mm2 = float(a_props.Mass())

    # Bounding box
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    inv_s = 1.0 / SCALE_M_TO_MM
    warnings: list[str] = []
    is_watertight = is_valid and (
        expect_2d or (free_edges == 0 and solid_count >= 1 and vol_mm3 > 0)
    )

    if not is_valid:
        warnings.append("OCCT BRepCheck_Analyzer flagged topological anomalies.")
    if not expect_2d and free_edges > 0:
        warnings.append(f"Detected {free_edges} free/unsewn edge(s); shell is not watertight.")
    if 0.0 < min_edge_mm < micro_edge_threshold_mm:
        warnings.append(
            f"Detected micro-edge ({min_edge_mm:.4f} mm); check surface mesh min size."
        )
    if sliver_faces > 0:
        warnings.append(
            f"Detected {sliver_faces} tiny/sliver face(s) < {sliver_area_threshold_mm2} mm^2."
        )

    return BrepValidationStats(
        is_valid=is_valid,
        is_watertight=is_watertight,
        solid_count=solid_count,
        free_edge_count=free_edges,
        min_edge_length_m=min_edge_mm * inv_s,
        sliver_face_count=sliver_faces,
        volume_m3=vol_mm3 * (inv_s**3),
        surface_area_m2=area_mm2 * (inv_s**2),
        bbox_min_m=(xmin * inv_s, ymin * inv_s, zmin * inv_s),
        bbox_max_m=(xmax * inv_s, ymax * inv_s, zmax * inv_s),
        warnings=tuple(warnings),
    )
