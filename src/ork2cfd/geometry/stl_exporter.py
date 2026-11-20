"""Multi-region ASCII STL mesh exporter for CFD-clean OCCT B-Rep domains."""

from __future__ import annotations

import math
from pathlib import Path

from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

from ..exceptions import GeometryBuildError
from ..models.cfd import CadConfig
from .domain_builder import DomainBuildOutput, FaceZoneEntry

_MM_TO_M = 1e-3


def _compute_unit_normal(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Return normalized triangle normal or None for degenerate facets."""
    ux, uy, uz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    vx, vy, vz = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-18:
        return None
    inv = 1.0 / norm
    return (nx * inv, ny * inv, nz * inv)


def _format_face_triangles(entry: FaceZoneEntry) -> list[str]:
    """Extract ASCII STL facet lines in SI meters for a single B-Rep face."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(entry.face, loc)
    if tri is None or tri.NbTriangles() == 0:
        return []

    trsf = loc.Transformation()
    n_nodes = tri.NbNodes()
    coords: list[tuple[float, float, float]] = []
    for idx in range(1, n_nodes + 1):
        pnt = tri.Node(idx).Transformed(trsf)
        coords.append((pnt.X() * _MM_TO_M, pnt.Y() * _MM_TO_M, pnt.Z() * _MM_TO_M))

    reversed_face = entry.face.Orientation() == TopAbs_REVERSED
    lines: list[str] = []
    for t_idx in range(1, tri.NbTriangles() + 1):
        i1, i2, i3 = tri.Triangle(t_idx).Get()
        if reversed_face:
            i2, i3 = i3, i2
        p1, p2, p3 = coords[i1 - 1], coords[i2 - 1], coords[i3 - 1]
        normal = _compute_unit_normal(p1, p2, p3)
        if normal is None:
            continue
        nx, ny, nz = normal
        lines.append(
            f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n"
            f"    outer loop\n"
            f"      vertex {p1[0]:.8e} {p1[1]:.8e} {p1[2]:.8e}\n"
            f"      vertex {p2[0]:.8e} {p2[1]:.8e} {p2[2]:.8e}\n"
            f"      vertex {p3[0]:.8e} {p3[1]:.8e} {p3[2]:.8e}\n"
            f"    endloop\n"
            f"  endfacet\n"
        )
    return lines


def export_stl_with_regions(
    domain_out: DomainBuildOutput,
    stl_path: Path,
    cad_config: CadConfig,
) -> tuple[Path, int]:
    """Tessellate domain faces and write a multi-solid ASCII STL file in SI meters.

    Each unique boundary zone name in ``domain_out.face_zones`` is written as its
    own ``solid <zone_name> ... endsolid <zone_name>`` region for Fluent / OpenFOAM.
    """
    lin_deflection_mm = max(cad_config.stl_linear_deflection_m * 1000.0, 0.01)
    ang_deflection_rad = math.radians(max(cad_config.stl_angular_deflection_deg, 1.0))

    BRepMesh_IncrementalMesh(
        domain_out.main_shape,
        lin_deflection_mm,
        False,
        ang_deflection_rad,
        True,
    )

    zone_facets: dict[str, list[str]] = {}
    for entry in domain_out.face_zones:
        facets = _format_face_triangles(entry)
        if facets:
            zone_facets.setdefault(entry.zone_name, []).extend(facets)

    total_triangles = sum(len(facets) for facets in zone_facets.values())
    if total_triangles == 0:
        raise GeometryBuildError("STL tessellation produced zero valid triangles.")

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    with stl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for zone_name, facets in zone_facets.items():
            handle.write(f"solid {zone_name}\n")
            handle.writelines(facets)
            handle.write(f"endsolid {zone_name}\n")

    return stl_path, total_triangles
