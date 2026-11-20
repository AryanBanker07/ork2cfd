"""Quantitative CAD verification engine and standalone STEP loader for ork2cfd."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool

from ..exceptions import GeometryBuildError
from ..models.cfd import CadConfig, DomainConfig, DomainMode, EnclosureShape
from ..models.results import GeometryReport, VerificationCheck, VerificationReport
from ..models.rocket import RocketModel
from .domain_builder import DomainBuildOutput, FaceZoneEntry
from .profiles import SCALE_M_TO_MM
from .tessellation import ZoneMesh
from .validator import validate_brep_shape

_STEP_READER_INITIALIZED = False


def _ensure_step_reader_init() -> None:
    """Initialize OCCT STEPCAF reader flags once per process."""
    global _STEP_READER_INITIALIZED
    if not _STEP_READER_INITIALIZED:
        STEPCAFControl_Controller.Init_s()
        _STEP_READER_INITIALIZED = True
    Interface_Static.SetIVal_s("read.stepcaf.subshapes.name", 1)


def _label_name(label) -> str:
    """Safely read an OCAF label name (guarded by IsAttribute to avoid OCP segfaults)."""
    if label.IsAttribute(TDataStd_Name.GetID_s()):
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            return attr.Get().ToExtString().strip()
    return ""


def _read_step_xcaf_document(step_path: Path) -> tuple[TDocStd_Document, object, TDF_LabelSequence]:
    """Read a STEP file into an XCAF document and return (doc, shape_tool, free_labels)."""
    _ensure_step_reader_init()
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)

    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise GeometryBuildError(f"Failed to read STEP file '{step_path}'.")

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XCAF"))
    app.InitDocument(doc)
    if not reader.Transfer(doc):
        raise GeometryBuildError(f"Failed to transfer XCAF entities from '{step_path}'.")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    free = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free)
    return doc, shape_tool, free


def _find_sibling_manifest(step_path: Path) -> dict | None:
    """Locate and parse an ork2cfd *_manifest.json file alongside step_path."""
    stem_base = step_path.stem.removesuffix("_cfd")
    candidates = [
        step_path.with_name(f"{stem_base}_manifest.json"),
        step_path.with_name(f"{step_path.stem}_manifest.json"),
        step_path.with_suffix(".json"),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "named_selections_centroids_m" in data:
                return data
    return None


def _match_zone_from_manifest(
    centroid_m: tuple[float, float, float],
    manifest_centroids: dict[str, list[list[float]]],
    tol_m: float = 0.02,
) -> str | None:
    """Return the zone name whose closest manifest centroid lies within tol_m."""
    best_zone: str | None = None
    best_dist = tol_m
    cx, cy, cz = centroid_m
    for zname, pts in manifest_centroids.items():
        for px, py, pz in pts:
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
            if dist <= best_dist:
                best_dist = dist
                best_zone = zname
    return best_zone


def _extract_subshape_face_names(shape_tool: object, main_label) -> list[tuple[TopoDS_Face, str]]:
    """Collect (face, name) pairs from XCAF subshape labels on a solid."""
    pairs: list[tuple[TopoDS_Face, str]] = []
    sub = TDF_LabelSequence()
    shape_tool.GetSubShapes_s(main_label, sub)
    for idx in range(1, sub.Length() + 1):
        sub_lbl = sub.Value(idx)
        name = _label_name(sub_lbl)
        if not name:
            continue
        sub_shp = shape_tool.GetShape_s(sub_lbl)
        if not sub_shp.IsNull() and sub_shp.ShapeType() == TopAbs_FACE:
            pairs.append((TopoDS.Face_s(sub_shp), name))
    return pairs


def load_step_for_verification(step_path: Path) -> DomainBuildOutput:
    """Load any STEP file into DomainBuildOutput with Named Selection zone recovery."""
    if not step_path.exists():
        raise GeometryBuildError(f"STEP file not found: '{step_path}'.")

    _doc, shape_tool, free = _read_step_xcaf_document(step_path)
    if free.Length() == 0:
        raise GeometryBuildError(f"STEP file '{step_path}' contains no free shapes.")

    boi_solids: list[tuple[str, TopoDS_Shape]] = []
    main_candidates: list[tuple[str, TopoDS_Shape, object]] = []

    for i in range(1, free.Length() + 1):
        lbl = free.Value(i)
        name = _label_name(lbl) or f"solid_{i}"
        shp = shape_tool.GetShape_s(lbl)
        if shp.IsNull():
            continue
        if name.lower().startswith("boi_"):
            boi_solids.append((name, shp))
        else:
            main_candidates.append((name, shp, lbl))

    if not main_candidates:
        raise GeometryBuildError(f"STEP file '{step_path}' contains no main solid shape.")

    preferred = {"fluid_domain", "rocket_oml", "fluid_domain_2d"}
    chosen = next((c for c in main_candidates if c[0] in preferred), main_candidates[0])
    main_name, main_shape, main_label = chosen

    prelim_stats = validate_brep_shape(main_shape, expect_2d=False)
    expect_2d = prelim_stats.solid_count == 0
    main_stats = (
        validate_brep_shape(main_shape, expect_2d=True) if expect_2d else prelim_stats
    )

    manifest = _find_sibling_manifest(step_path)
    manifest_centroids: dict[str, list[list[float]]] = (
        manifest.get("named_selections_centroids_m", {}) if manifest else {}
    )
    xcaf_face_names = _extract_subshape_face_names(shape_tool, main_label)

    inv_s = 1.0 / SCALE_M_TO_MM
    inv_s2 = inv_s * inv_s
    face_zones: list[FaceZoneEntry] = []
    zone_counts: dict[str, int] = {}

    f_exp = TopExp_Explorer(main_shape, TopAbs_FACE)
    while f_exp.More():
        face = TopoDS.Face_s(f_exp.Current())
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        cm = props.CentreOfMass()
        centroid_m = (cm.X() * inv_s, cm.Y() * inv_s, cm.Z() * inv_s)
        area_m2 = float(props.Mass()) * inv_s2

        zone = (
            _match_zone_from_manifest(centroid_m, manifest_centroids)
            if manifest_centroids
            else None
        )
        if zone is None:
            for ref_face, ref_name in xcaf_face_names:
                if face.IsSame(ref_face):
                    zone = ref_name
                    break
        if not zone:
            zone = main_name or "imported_cad"

        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        face_zones.append(
            FaceZoneEntry(
                face=face,
                zone_name=zone,
                centroid_m=centroid_m,
                area_m2=area_m2,
            )
        )
        f_exp.Next()

    oml_vol = float(manifest.get("validation", {}).get("oml_volume_m3", 0.0)) if manifest else 0.0
    ref_area = float(manifest.get("rocket_reference_area_m2", 0.0)) if manifest else 0.0
    report = GeometryReport(
        is_valid=main_stats.is_valid,
        is_watertight=main_stats.is_watertight,
        solid_count=main_stats.solid_count,
        free_edge_count=main_stats.free_edge_count,
        min_edge_length_m=main_stats.min_edge_length_m,
        sliver_face_count=main_stats.sliver_face_count,
        oml_volume_m3=oml_vol,
        oml_wetted_area_m2=main_stats.surface_area_m2 if oml_vol == 0.0 else 0.0,
        oml_frontal_area_m2=ref_area,
        domain_volume_m3=main_stats.volume_m3,
        bbox_min_m=main_stats.bbox_min_m,
        bbox_max_m=main_stats.bbox_max_m,
        named_selections=dict(sorted(zone_counts.items())),
        boi_names=tuple(name for name, _ in boi_solids),
        warnings=main_stats.warnings,
    )

    return DomainBuildOutput(
        main_shape=main_shape,
        main_name=main_name,
        boi_solids=tuple(boi_solids),
        face_zones=tuple(face_zones),
        report=report,
        bbox_domain_min_m=main_stats.bbox_min_m,
        bbox_domain_max_m=main_stats.bbox_max_m,
    )


def _compute_oml_extents(
    zone_meshes: dict[str, ZoneMesh],
    domain_out: DomainBuildOutput,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float, list[ZoneMesh]]:
    """Compute (bbox_min_m, bbox_max_m, max_radius_m, oml_meshes) over rocket_* zones."""
    oml_meshes = [m for n, m in zone_meshes.items() if n.startswith("rocket_")]
    if not oml_meshes:
        return domain_out.bbox_domain_min_m, domain_out.bbox_domain_max_m, 0.0, []

    all_verts = np.vstack([m.vertices_m for m in oml_meshes if m.vertices_m.size > 0])
    mins = tuple(float(v) for v in all_verts.min(axis=0))
    maxs = tuple(float(v) for v in all_verts.max(axis=0))
    max_r = float(np.max(np.hypot(all_verts[:, 1], all_verts[:, 2])))
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2]), max_r, oml_meshes


def _check_required_zones(
    zone_meshes: dict[str, ZoneMesh],
    rocket: RocketModel | None,
    domain_config: DomainConfig | None,
) -> VerificationCheck:
    """Verify Named Selection completeness and non-zero surface areas."""
    mode = domain_config.mode if domain_config is not None else None
    if mode == DomainMode.AXISYMMETRIC_2D:
        required = {"fluid_domain_2d"}
        missing = sorted(z for z in required if z not in zone_meshes)
        ok = len(missing) == 0 and zone_meshes["fluid_domain_2d"].area_m2 > 0
        return VerificationCheck(
            name="named_selections_complete",
            status="PASS" if ok else "FAIL",
            measured=", ".join(sorted(zone_meshes.keys())) or "none",
            expected="fluid_domain_2d (area > 0)",
            details=f"Missing: {missing}" if missing else "2D axisymmetric face verified.",
        )

    if rocket is None:
        empty = [z for z, m in zone_meshes.items() if m.face_count < 1 or m.area_m2 <= 0]
        ok = bool(zone_meshes) and not empty
        return VerificationCheck(
            name="named_selections_complete",
            status="PASS" if ok else "FAIL",
            measured=f"{len(zone_meshes)} zones",
            expected=">= 1 zone with positive area",
            details=f"Empty/zero-area zones: {empty}" if empty else "All zones have area > 0.",
        )

    missing: list[str] = []
    if "rocket_nosecone" not in zone_meshes and "rocket_body" not in zone_meshes:
        missing.append("rocket_nosecone|rocket_body")
    if rocket.all_fins and "rocket_fins" not in zone_meshes:
        missing.append("rocket_fins")
    if domain_config is not None and domain_config.mode != DomainMode.OML_ONLY:
        if "farfield" not in zone_meshes:
            missing.append("farfield")
        if "outlet" not in zone_meshes:
            missing.append("outlet")
        if (
            domain_config.mode in {DomainMode.SYMMETRY_HALF, DomainMode.SYMMETRY_SECTOR}
            and "symmetry" not in zone_meshes
        ):
            missing.append("symmetry")
        if (
            domain_config.shape in {EnclosureShape.CYLINDER, EnclosureShape.BOX}
            and "inlet" not in zone_meshes
        ):
            missing.append("inlet")

    zero_area = [z for z, m in zone_meshes.items() if not m.is_boi and m.area_m2 <= 0]
    ok = not missing and not zero_area
    return VerificationCheck(
        name="named_selections_complete",
        status="PASS" if ok else "FAIL",
        measured=", ".join(sorted(z for z, m in zone_meshes.items() if not m.is_boi)),
        expected="All rocket/boundary Named Selections present",
        details=(
            f"Missing={missing}, zero_area={zero_area}"
            if not ok
            else f"{len(zone_meshes)} zones/BOIs verified with positive surface area."
        ),
    )


def _check_step_roundtrip(
    step_path: Path,
    domain_out: DomainBuildOutput,
    is_2d: bool,
    rocket: RocketModel | None,
) -> tuple[VerificationCheck, bool]:
    """Re-import STEP via OCCT XCAF reader and verify SI meters, B-Rep, bbox, and labels."""
    txt = step_path.read_text(encoding="utf-8", errors="replace")
    unit_ok = "SI_UNIT($,.METRE.)" in txt
    face_names = {m for m in re.findall(r"ADVANCED_FACE\('([^']*)'", txt) if m}

    zones_ok = True
    if not is_2d and rocket is not None:
        expected_zones = set(domain_out.report.named_selections.keys())
        zones_ok = expected_zones.issubset(face_names)

    try:
        _doc, shape_tool, free = _read_step_xcaf_document(step_path)
    except Exception as exc:
        chk = VerificationCheck(
            name="step_file_roundtrip",
            status="FAIL",
            measured="reader error",
            expected="valid XCAF STEP transfer",
            details=str(exc),
        )
        return chk, False

    if free.Length() == 0:
        chk = VerificationCheck(
            name="step_file_roundtrip",
            status="FAIL",
            measured="0 solids",
            expected=">= 1 free shape",
            details="STEPCAFControl_Reader returned zero free shapes.",
        )
        return chk, False

    candidates: list[tuple[str, TopoDS_Shape]] = []
    for idx in range(1, free.Length() + 1):
        lbl = free.Value(idx)
        name = _label_name(lbl)
        shp = shape_tool.GetShape_s(lbl)
        if not shp.IsNull() and not name.lower().startswith("boi_"):
            candidates.append((name, shp))

    main_shp = candidates[0][1] if candidates else shape_tool.GetShape_s(free.Value(1))
    re_stats = validate_brep_shape(main_shp, expect_2d=is_2d)

    bbox_diff = max(
        max(
            abs(re_stats.bbox_min_m[k] - domain_out.bbox_domain_min_m[k]),
            abs(re_stats.bbox_max_m[k] - domain_out.bbox_domain_max_m[k]),
        )
        for k in range(3)
    )
    bbox_ok = bbox_diff <= 0.01
    ok = (
        unit_ok
        and zones_ok
        and re_stats.is_valid
        and (is_2d or re_stats.is_watertight)
        and bbox_ok
    )
    chk = VerificationCheck(
        name="step_file_roundtrip",
        status="PASS" if ok else "FAIL",
        measured=(
            f"free={free.Length()}, valid={re_stats.is_valid}, "
            f"wt={re_stats.is_watertight}, dBBox={bbox_diff:.4f}m"
        ),
        expected="SI_UNIT METRE, valid/watertight B-Rep, dBBox <= 0.01m",
        details=(
            f"SI_METRE={unit_ok}, labels_ok={zones_ok}, "
            f"STEP_faces={sorted(face_names)}"
        ),
    )
    return chk, ok


def _check_stl_integrity(stl_path: Path) -> VerificationCheck:
    """Validate ASCII STL region structure and numeric finiteness."""
    text = stl_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    solid_lines = sum(1 for ln in lines if ln.startswith("solid "))
    endsolid_lines = sum(1 for ln in lines if ln.startswith("endsolid "))
    facet_lines = sum(1 for ln in lines if ln.startswith("facet normal "))
    bad_num = any(
        ("nan" in ln.lower() or "inf" in ln.lower())
        for ln in lines
        if ln.startswith(("facet normal ", "vertex "))
    )
    ok = (
        bool(lines)
        and lines[0].startswith("solid ")
        and solid_lines >= 1
        and solid_lines == endsolid_lines
        and facet_lines >= 1
        and not bad_num
    )
    return VerificationCheck(
        name="stl_file_integrity",
        status="PASS" if ok else "FAIL",
        measured=f"regions={solid_lines}, facets={facet_lines}, bad_num={bad_num}",
        expected="Balanced multi-solid ASCII STL with finite coordinates",
        details=f"File size: {stl_path.stat().st_size:,} bytes",
    )


def verify_cad_model(
    domain_out: DomainBuildOutput,
    zone_meshes: dict[str, ZoneMesh],
    rocket: RocketModel | None = None,
    cad_config: CadConfig | None = None,
    domain_config: DomainConfig | None = None,
    step_path: Path | None = None,
    stl_path: Path | None = None,
) -> VerificationReport:
    """Run quantitative topology, dimensional fidelity, and STEP round-trip checks."""
    geom = domain_out.report
    mode = domain_config.mode if domain_config is not None else None
    is_2d = mode == DomainMode.AXISYMMETRIC_2D or geom.solid_count == 0
    oml_min, oml_max, oml_max_r, oml_meshes = _compute_oml_extents(zone_meshes, domain_out)

    checks: list[VerificationCheck] = []

    # 1. OCCT B-Rep Validity
    checks.append(
        VerificationCheck(
            name="occt_brep_validity",
            status="PASS" if geom.is_valid else "FAIL",
            measured=str(geom.is_valid),
            expected="True",
            details="OCCT BRepCheck_Analyzer topology & curve/surface check.",
        )
    )

    # 2. Watertight Manifold
    wt_ok = geom.is_watertight and geom.free_edge_count == 0
    checks.append(
        VerificationCheck(
            name="watertight_manifold",
            status="PASS" if wt_ok else "FAIL",
            measured=f"watertight={geom.is_watertight}, free_edges={geom.free_edge_count}",
            expected="watertight=True, free_edges=0",
            details=f"Solids={geom.solid_count}, domain_vol={geom.domain_volume_m3:.4f} m^3",
        )
    )

    # 3. Sliver Faces & Micro Edges
    edge_ok = geom.sliver_face_count == 0 and (
        geom.min_edge_length_m == 0.0 or geom.min_edge_length_m >= 2e-5
    )
    checks.append(
        VerificationCheck(
            name="sliver_and_micro_edges",
            status="PASS" if edge_ok else "WARN",
            measured=(
                f"slivers={geom.sliver_face_count}, "
                f"min_edge={geom.min_edge_length_m * 1000:.4f} mm"
            ),
            expected="slivers=0, min_edge >= 0.0200 mm",
            details="Tiny sliver faces or micro-edges can require finer Fluent min surface sizing.",
        )
    )

    # 4 & 5. Dimensional comparison vs OpenRocket model
    if rocket is not None and mode != DomainMode.AXISYMMETRIC_2D and oml_meshes:
        measured_len = oml_max[0] - oml_min[0]
        blunt_m = cad_config.tip_blunt_radius_m if cad_config else 0.001
        len_tol = max(0.004, 0.008 * rocket.total_length) + 2.0 * blunt_m
        len_ok = abs(measured_len - rocket.total_length) <= len_tol
        checks.append(
            VerificationCheck(
                name="oml_axial_length_vs_ork",
                status="PASS" if len_ok else "FAIL",
                measured=f"{measured_len:.4f} m",
                expected=f"{rocket.total_length:.4f} m (±{len_tol * 1000:.1f} mm)",
                details=f"Axial OML span X=[{oml_min[0]:.4f}, {oml_max[0]:.4f}] m",
            )
        )

        expected_r = 0.5 * rocket.max_fin_span_diameter
        if mode in {DomainMode.SYMMETRY_HALF, DomainMode.SYMMETRY_SECTOR}:
            rad_ok = (0.85 * rocket.max_radius) <= oml_max_r <= (expected_r + 0.006)
            rad_status = "PASS" if rad_ok else "WARN"
            exp_desc = f"[{0.85 * rocket.max_radius:.4f}, {expected_r:.4f}] m (sector)"
        else:
            rad_tol = max(0.004, 0.015 * expected_r)
            rad_ok = abs(oml_max_r - expected_r) <= rad_tol
            rad_status = "PASS" if rad_ok else "FAIL"
            exp_desc = f"{expected_r:.4f} m (±{rad_tol * 1000:.1f} mm)"
        checks.append(
            VerificationCheck(
                name="oml_radial_span_vs_ork",
                status=rad_status,
                measured=f"{oml_max_r:.4f} m",
                expected=exp_desc,
                details="Maximum radial distance sqrt(y^2 + z^2) over rocket_* faces.",
            )
        )

    # 6. Named Selection Completeness
    checks.append(_check_required_zones(zone_meshes, rocket, domain_config))

    # 7. Domain Clearance & Cavity Subtraction
    if domain_config is not None and domain_config.mode not in {
        DomainMode.OML_ONLY,
        DomainMode.AXISYMMETRIC_2D,
    }:
        ref_len = (
            rocket.total_length
            if rocket is not None
            else max(0.1, oml_max[0] - oml_min[0])
        )
        dmin, dmax = domain_out.bbox_domain_min_m, domain_out.bbox_domain_max_m
        clear_ok = (
            dmin[0] < oml_min[0] - 0.5 * ref_len
            and dmax[0] > oml_max[0] + 0.5 * ref_len
            and geom.domain_volume_m3 > geom.oml_volume_m3 > 0.0
        )
        checks.append(
            VerificationCheck(
                name="domain_clearance_and_cavity",
                status="PASS" if clear_ok else "FAIL",
                measured=(
                    f"X=[{dmin[0]:.2f}, {dmax[0]:.2f}] m, "
                    f"V_dom={geom.domain_volume_m3:.2f} m^3"
                ),
                expected=f"Upstream/downstream margin > {0.5 * ref_len:.2f} m, V_dom > V_oml",
                details=f"OML volume subtracted: {geom.oml_volume_m3:.5f} m^3",
            )
        )

    # 8. STEP Round-Trip Audit
    step_roundtrip_ok: bool | None = None
    if step_path is not None and step_path.exists():
        step_chk, step_roundtrip_ok = _check_step_roundtrip(
            step_path=step_path,
            domain_out=domain_out,
            is_2d=is_2d,
            rocket=rocket,
        )
        checks.append(step_chk)

    # 9. STL Integrity Audit
    if stl_path is not None and stl_path.exists():
        checks.append(_check_stl_integrity(stl_path))

    statuses = {c.status for c in checks}
    overall_status = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    return VerificationReport(
        overall_status=overall_status,
        checks=tuple(checks),
        zone_areas_m2={z: round(m.area_m2, 6) for z, m in sorted(zone_meshes.items())},
        zone_face_counts={z: m.face_count for z, m in sorted(zone_meshes.items())},
        zone_triangle_counts={z: int(m.triangles.shape[0]) for z, m in sorted(zone_meshes.items())},
        oml_bbox_min_m=tuple(round(v, 5) for v in oml_min),
        oml_bbox_max_m=tuple(round(v, 5) for v in oml_max),
        step_roundtrip_ok=step_roundtrip_ok,
        summary_notes=geom.warnings,
    )


def write_verification_reports(
    report: VerificationReport,
    json_path: Path,
    md_path: Path,
    model_name: str,
    step_path: Path | None = None,
    viewer_html_path: Path | None = None,
    views_png_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write machine-readable JSON and Markdown scorecard verification reports."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "overall_status": report.overall_status,
        "step_file": step_path.name if step_path else None,
        "viewer_html": viewer_html_path.name if viewer_html_path else None,
        "views_png": views_png_path.name if views_png_path else None,
        "step_roundtrip_ok": report.step_roundtrip_ok,
        "oml_bbox_min_m": list(report.oml_bbox_min_m),
        "oml_bbox_max_m": list(report.oml_bbox_max_m),
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "measured": c.measured,
                "expected": c.expected,
                "details": c.details,
            }
            for c in report.checks
        ],
        "zone_areas_m2": report.zone_areas_m2,
        "zone_face_counts": report.zone_face_counts,
        "zone_triangle_counts": report.zone_triangle_counts,
        "summary_notes": list(report.summary_notes),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_pass = sum(1 for c in report.checks if c.status == "PASS")
    lines = [
        f"# CAD Verification Report — `{model_name}`",
        "",
        (
            f"**Overall Status:** `{report.overall_status}` "
            f"({n_pass}/{len(report.checks)} checks PASS)"
        ),
        "",
        f"- **OML Bounding Box (m):** `{report.oml_bbox_min_m}` to `{report.oml_bbox_max_m}`",
    ]
    if step_path is not None:
        lines.append(f"- **STEP File:** `{step_path.name}`")
    if viewer_html_path is not None:
        lines.append(f"- **Interactive 3D Viewer:** `{viewer_html_path.name}`")
    if views_png_path is not None:
        lines.append(f"- **4-View Verification Sheet:** `{views_png_path.name}`")

    lines.extend(
        [
            "",
            "## Verification Checks",
            "",
            "| Check | Status | Measured | Expected | Details |",
            "|---|---|---|---|---|",
        ]
    )
    for c in report.checks:
        lines.append(
            f"| `{c.name}` | **{c.status}** | {c.measured} | {c.expected} | {c.details} |"
        )

    lines.extend(
        [
            "",
            "## Named Selections & Zone Mesh Summary",
            "",
            "| Zone | Faces | Triangles | Surface Area (m²) |",
            "|---|---:|---:|---:|",
        ]
    )
    for zone in sorted(report.zone_face_counts):
        fc = report.zone_face_counts.get(zone, 0)
        tc = report.zone_triangle_counts.get(zone, 0)
        area = report.zone_areas_m2.get(zone, 0.0)
        lines.append(f"| `{zone}` | {fc} | {tc:,} | {area:.6f} |")

    if report.summary_notes:
        lines.extend(["", "## Notes & Warnings", ""])
        for note in report.summary_notes:
            lines.append(f"- {note}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
