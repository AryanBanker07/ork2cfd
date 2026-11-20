"""OCCT XCAF STEP exporter preserving solid names, face Named Selections, colors, and manifest."""

from __future__ import annotations

import json
from pathlib import Path

from OCP.Interface import Interface_Static
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

from ..exceptions import GeometryBuildError
from ..models.cfd import CadConfig, DomainConfig
from ..models.rocket import RocketModel
from .domain_builder import DomainBuildOutput

# Distinct visual colors per CFD boundary zone for instant visual verification in CAD/ANSYS
_ZONE_COLORS_RGB: dict[str, tuple[float, float, float]] = {
    "farfield": (0.45, 0.70, 0.95),
    "inlet": (0.20, 0.80, 0.40),
    "outlet": (0.92, 0.35, 0.30),
    "symmetry": (0.75, 0.75, 0.78),
    "rocket_nosecone": (0.95, 0.65, 0.15),
    "rocket_body": (0.85, 0.88, 0.92),
    "rocket_boattail": (0.85, 0.45, 0.85),
    "rocket_fins": (0.95, 0.25, 0.25),
    "rocket_base": (0.35, 0.35, 0.40),
    "rocket_appendages": (0.95, 0.90, 0.20),
    "boi_nearfield": (0.25, 0.85, 0.85),
    "boi_wake": (0.55, 0.45, 0.95),
}


_STEP_INITIALIZED = False


def export_step_with_names(
    domain_out: DomainBuildOutput,
    step_path: Path,
    manifest_path: Path,
    rocket: RocketModel,
    cad_config: CadConfig,
    domain_config: DomainConfig,
) -> tuple[Path, Path]:
    """
    Export DomainBuildOutput to STEP (AP214/AP242) in meters with embedded XCAF names.

    Writes both top-level solid names (fluid_domain/rocket_oml, boi_nearfield, boi_wake)
    and individual face Named Selection labels, plus a JSON manifest for fallback selection.
    """
    global _STEP_INITIALIZED
    step_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize STEPCAF controller once so static flags exist before setting them
    if not _STEP_INITIALIZED:
        STEPCAFControl_Controller.Init_s()
        _STEP_INITIALIZED = True
    schema = cad_config.step_schema.upper() if cad_config.step_schema else "AP214"
    Interface_Static.SetCVal_s("write.step.schema", schema)
    Interface_Static.SetCVal_s("write.step.unit", "M")
    Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 1)
    Interface_Static.SetCVal_s("write.step.product.name", rocket.name)

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.InitDocument(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    main_shape = _unwrap_single_solid(domain_out.main_shape)
    main_label = shape_tool.AddShape(main_shape, False)
    _set_label_name(main_label, domain_out.main_name)

    for entry in domain_out.face_zones:
        sub_label = shape_tool.AddSubShape(main_label, entry.face)
        if not sub_label.IsNull():
            _set_label_name(sub_label, entry.zone_name)
            rgb = _ZONE_COLORS_RGB.get(entry.zone_name)
            if rgb is not None:
                qcol = Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB)
                color_tool.SetColor(sub_label, qcol, XCAFDoc_ColorSurf)

    for boi_name, boi_shape in domain_out.boi_solids:
        boi_unwrapped = _unwrap_single_solid(boi_shape)
        boi_label = shape_tool.AddShape(boi_unwrapped, False)
        _set_label_name(boi_label, boi_name)
        rgb = _ZONE_COLORS_RGB.get(boi_name, (0.3, 0.8, 0.8))
        qcol = Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB)
        color_tool.SetColor(boi_label, qcol, XCAFDoc_ColorGen)

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)

    if not writer.Transfer(doc, STEPControl_AsIs):
        raise GeometryBuildError("STEPCAFControl_Writer failed to transfer XCAF document.")

    status = writer.Write(str(step_path))
    if int(status) != 1 or not step_path.exists():
        raise GeometryBuildError(f"Failed to write STEP file to '{step_path}'.")

    manifest = _build_manifest_dict(domain_out, rocket, cad_config, domain_config, step_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return step_path, manifest_path


def _unwrap_single_solid(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Return the sole TopoDS_Solid if shape is a single-solid compound."""
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while exp.More():
        solids.append(TopoDS.Solid_s(exp.Current()))
        exp.Next()
    if len(solids) == 1:
        return solids[0]
    return shape


def _set_label_name(label, name: str) -> None:
    """Attach an OCAF TDataStd_Name attribute to a TDF_Label."""
    TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))


def _build_manifest_dict(
    domain_out: DomainBuildOutput,
    rocket: RocketModel,
    cad_config: CadConfig,
    domain_config: DomainConfig,
    step_path: Path,
) -> dict:
    """Construct structured geometry manifest describing zones, centroids, and reference metrics."""
    zones_summary: dict[str, list[list[float]]] = {}
    for entry in domain_out.face_zones:
        zones_summary.setdefault(entry.zone_name, []).append(
            [round(c, 6) for c in entry.centroid_m]
        )

    return {
        "rocket_name": rocket.name,
        "step_file": step_path.name,
        "units": "m",
        "step_schema": cad_config.step_schema,
        "domain_mode": domain_config.mode.value,
        "enclosure_shape": domain_config.shape.value,
        "stage_count": rocket.stage_count,
        "is_multistage": rocket.is_multistage,
        "rotational_symmetry_order": rocket.rotational_symmetry_order,
        "stages": [
            {
                "index": idx,
                "name": st.name,
                "axial_start_m": round(st.axial_start, 6),
                "axial_end_m": round(st.axial_end, 6),
                "fin_counts": [f.fin_count for c in st.components for f in c.fins],
            }
            for idx, st in enumerate(rocket.stages, start=1)
        ],
        "rocket_total_length_m": round(rocket.total_length, 6),
        "rocket_max_diameter_m": round(rocket.max_diameter, 6),
        "rocket_reference_area_m2": round(rocket.reference_area, 8),
        "domain_bbox_min_m": [round(v, 5) for v in domain_out.bbox_domain_min_m],
        "domain_bbox_max_m": [round(v, 5) for v in domain_out.bbox_domain_max_m],
        "solid_names": [domain_out.main_name] + [n for n, _ in domain_out.boi_solids],
        "named_selections_counts": domain_out.report.named_selections,
        "named_selections_centroids_m": zones_summary,
        "validation": {
            "is_valid": domain_out.report.is_valid,
            "is_watertight": domain_out.report.is_watertight,
            "free_edge_count": domain_out.report.free_edge_count,
            "oml_volume_m3": round(domain_out.report.oml_volume_m3, 8),
            "domain_volume_m3": round(domain_out.report.domain_volume_m3, 6),
            "warnings": list(domain_out.report.warnings),
        },
    }
