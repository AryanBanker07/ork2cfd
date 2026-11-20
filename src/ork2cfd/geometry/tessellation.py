"""Shared OCCT surface tessellation into per-zone triangle arrays in SI meters."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Face

from .domain_builder import DomainBuildOutput

_MM_TO_M = 1e-3
_MM2_TO_M2 = 1e-6


@dataclass(frozen=True)
class ZoneMesh:
    """Triangulated surface mesh (SI meters) for a single Named Selection or BOI."""

    zone_name: str
    vertices_m: np.ndarray
    triangles: np.ndarray
    area_m2: float
    face_count: int
    is_boi: bool = False


def _extract_face_mesh(face: TopoDS_Face) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (vertices_m, triangles) for a tessellated face, filtering degenerate facets."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbTriangles() == 0:
        return None

    trsf = loc.Transformation()
    n_nodes = tri.NbNodes()
    coords = np.empty((n_nodes, 3), dtype=np.float64)
    for idx in range(1, n_nodes + 1):
        pnt = tri.Node(idx).Transformed(trsf)
        coords[idx - 1, 0] = pnt.X() * _MM_TO_M
        coords[idx - 1, 1] = pnt.Y() * _MM_TO_M
        coords[idx - 1, 2] = pnt.Z() * _MM_TO_M

    reversed_face = face.Orientation() == TopAbs_REVERSED
    n_tris = tri.NbTriangles()
    tris = np.empty((n_tris, 3), dtype=np.uint32)
    keep_count = 0
    for t_idx in range(1, n_tris + 1):
        i1, i2, i3 = tri.Triangle(t_idx).Get()
        if reversed_face:
            i2, i3 = i3, i2
        v0, v1, v2 = i1 - 1, i2 - 1, i3 - 1
        e1 = coords[v1] - coords[v0]
        e2 = coords[v2] - coords[v0]
        cross = np.cross(e1, e2)
        if float(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) <= 1e-36:
            continue
        tris[keep_count, 0] = v0
        tris[keep_count, 1] = v1
        tris[keep_count, 2] = v2
        keep_count += 1

    if keep_count == 0:
        return None
    return coords.astype(np.float32), tris[:keep_count]


def _aggregate_zone_faces(
    zone_name: str,
    faces_and_areas: list[tuple[TopoDS_Face, float]],
    is_boi: bool = False,
) -> ZoneMesh | None:
    """Combine multiple B-Rep faces belonging to the same zone into one ZoneMesh."""
    vert_chunks: list[np.ndarray] = []
    tri_chunks: list[np.ndarray] = []
    vertex_offset = 0
    total_area_m2 = 0.0

    for face, area_m2 in faces_and_areas:
        total_area_m2 += area_m2
        extracted = _extract_face_mesh(face)
        if extracted is None:
            continue
        verts, tris = extracted
        vert_chunks.append(verts)
        tri_chunks.append(tris + np.uint32(vertex_offset))
        vertex_offset += verts.shape[0]

    if not tri_chunks:
        return None

    return ZoneMesh(
        zone_name=zone_name,
        vertices_m=np.vstack(vert_chunks).astype(np.float32),
        triangles=np.vstack(tri_chunks).astype(np.uint32),
        area_m2=total_area_m2,
        face_count=len(faces_and_areas),
        is_boi=is_boi,
    )


def tessellate_domain_zones(
    domain_out: DomainBuildOutput,
    linear_deflection: float = 0.005,
    angular_deflection_deg: float = 15.0,
    relative: bool = True,
    include_bois: bool = True,
) -> dict[str, ZoneMesh]:
    """Tessellate domain and optional BOI solids into per-zone meshes in SI meters.

    Calls ``BRepTools.Clean_s`` before meshing so prior absolute/relative
    triangulations never leak between viewer and STL exports.
    """
    ang_rad = math.radians(max(angular_deflection_deg, 1.0))
    lin_arg = (
        max(linear_deflection, 1e-4)
        if relative
        else max(linear_deflection * 1000.0, 0.01)
    )

    BRepTools.Clean_s(domain_out.main_shape)
    BRepMesh_IncrementalMesh(domain_out.main_shape, lin_arg, relative, ang_rad, True)

    grouped: dict[str, list[tuple[TopoDS_Face, float]]] = {}
    for entry in domain_out.face_zones:
        grouped.setdefault(entry.zone_name, []).append((entry.face, entry.area_m2))

    meshes: dict[str, ZoneMesh] = {}
    for zone_name, items in grouped.items():
        zm = _aggregate_zone_faces(zone_name, items, is_boi=False)
        if zm is not None:
            meshes[zone_name] = zm

    if include_bois:
        boi_ang_rad = math.radians(18.0)
        for boi_name, boi_shape in domain_out.boi_solids:
            BRepTools.Clean_s(boi_shape)
            BRepMesh_IncrementalMesh(boi_shape, 0.01, True, boi_ang_rad, True)
            boi_faces: list[tuple[TopoDS_Face, float]] = []
            exp = TopExp_Explorer(boi_shape, TopAbs_FACE)
            while exp.More():
                face = TopoDS.Face_s(exp.Current())
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, props)
                boi_faces.append((face, float(props.Mass()) * _MM2_TO_M2))
                exp.Next()
            zm = _aggregate_zone_faces(boi_name, boi_faces, is_boi=True)
            if zm is not None:
                meshes[boi_name] = zm

    return meshes
