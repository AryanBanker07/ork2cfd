"""Optional external aerodynamic appendages (rail buttons and launch lugs)."""

from __future__ import annotations

import math

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.TopoDS import TopoDS_Shape

from ..models.cfd import CadConfig
from ..models.rocket import RocketModel
from .profiles import SCALE_M_TO_MM


def build_appendage_solids_mm(
    rocket: RocketModel,
    cad_config: CadConfig,
) -> list[TopoDS_Shape]:
    """Construct optional rail buttons and launch lugs (plugged for watertight CFD)."""
    solids: list[TopoDS_Shape] = []
    embed_mm = max(0.5, cad_config.fin_root_embed_m * SCALE_M_TO_MM)
    axis_x = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))

    if cad_config.include_rail_buttons:
        for comp in rocket.all_components:
            for rb in comp.rail_buttons:
                r_cyl = max(1.0, 0.5 * rb.outer_diameter * SCALE_M_TO_MM)
                h_mm = max(1.5, rb.height * SCALE_M_TO_MM)
                for idx in range(max(1, rb.instance_count)):
                    x_m = rb.axial_x + idx * rb.instance_separation
                    x_mm = x_m * SCALE_M_TO_MM
                    body_r = rocket.radius_at_x(x_m) * SCALE_M_TO_MM
                    if body_r <= 0.5:
                        continue
                    y_start = max(0.5, body_r - embed_mm)
                    ax = gp_Ax2(gp_Pnt(x_mm, y_start, 0.0), gp_Dir(0.0, 1.0, 0.0))
                    btn = BRepPrimAPI_MakeCylinder(ax, r_cyl, h_mm + embed_mm).Shape()
                    if abs(rb.angle_deg) > 1e-4:
                        trsf = gp_Trsf()
                        trsf.SetRotation(axis_x, math.radians(rb.angle_deg))
                        btn = BRepBuilderAPI_Transform(btn, trsf, True).Shape()
                    solids.append(btn)

    if cad_config.include_launch_lugs:
        for comp in rocket.all_components:
            for lug in comp.launch_lugs:
                r_lug = max(1.5, lug.outer_radius * SCALE_M_TO_MM)
                len_mm = max(4.0, lug.length * SCALE_M_TO_MM)
                for idx in range(max(1, lug.instance_count)):
                    x_m = lug.axial_x + idx * lug.instance_separation
                    x_mm = x_m * SCALE_M_TO_MM
                    body_r = rocket.radius_at_x(x_m + 0.5 * lug.length) * SCALE_M_TO_MM
                    if body_r <= 0.5:
                        continue
                    y_center = body_r + r_lug - 0.5 * embed_mm
                    ax = gp_Ax2(gp_Pnt(x_mm, y_center, 0.0), gp_Dir(1.0, 0.0, 0.0))
                    # Plugged solid cylinder to avoid tiny thin-walled interior mesh issues
                    solid_lug = BRepPrimAPI_MakeCylinder(ax, r_lug, len_mm).Shape()
                    if abs(lug.angle_deg) > 1e-4:
                        trsf = gp_Trsf()
                        trsf.SetRotation(axis_x, math.radians(lug.angle_deg))
                        solid_lug = BRepBuilderAPI_Transform(solid_lug, trsf, True).Shape()
                    solids.append(solid_lug)

    return solids
