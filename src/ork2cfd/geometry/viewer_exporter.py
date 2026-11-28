"""Offline WebGL 3D CAD viewer and 4-view verification image generator."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..models.results import VerificationReport
from ..models.rocket import RocketModel
from .domain_builder import DomainBuildOutput
from .profiles import generate_component_profile_points
from .step_exporter import zone_color_rgb
from .tessellation import ZoneMesh
from .verifier import _NON_ROCKET_ZONES

_EXTRA_COLORS_RGB: dict[str, tuple[float, float, float]] = {
    "fluid_domain_2d": (0.38, 0.65, 0.98),
    "imported_cad": (0.72, 0.78, 0.86),
}

_TRANSLUCENT_ZONES = frozenset({"farfield", "inlet", "outlet", "symmetry"})


def _zone_color(zone_name: str) -> tuple[float, float, float]:
    """Return consistent normalized RGB color for a boundary zone, component, or BOI."""
    if zone_name in _EXTRA_COLORS_RGB:
        return _EXTRA_COLORS_RGB[zone_name]
    return zone_color_rgb(zone_name)


def _mesh_extents(meshes: list[ZoneMesh]) -> tuple[list[float], list[float]]:
    """Compute overall [xmin, ymin, zmin] and [xmax, ymax, zmax] for a list of ZoneMesh."""
    valid = [m.vertices_m for m in meshes if m.vertices_m.size > 0]
    if not valid:
        return [0.0, -0.5, -0.5], [1.0, 0.5, 0.5]
    stacked = np.vstack(valid)
    return (
        [float(v) for v in stacked.min(axis=0)],
        [float(v) for v in stacked.max(axis=0)],
    )


def export_cad_viewer_html(
    zone_meshes: dict[str, ZoneMesh],
    verification_report: VerificationReport,
    html_path: Path,
    model_name: str,
    domain_out: DomainBuildOutput,
    rocket: RocketModel | None = None,
) -> Path:
    """Write a self-contained offline 3D WebGL CAD viewer & inspection HTML file."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("cad_viewer.html.j2")

    oml_list = [
        m for n, m in zone_meshes.items() if not m.is_boi and n not in _NON_ROCKET_ZONES
    ]
    all_list = list(zone_meshes.values())
    oml_bmin, oml_bmax = _mesh_extents(oml_list or all_list)
    dom_bmin = [float(v) for v in domain_out.bbox_domain_min_m]
    dom_bmax = [float(v) for v in domain_out.bbox_domain_max_m]

    total_tris = sum(int(m.triangles.shape[0]) for m in all_list)
    geom = domain_out.report
    oml_len = (
        rocket.total_length
        if rocket is not None
        else max(0.0, oml_bmax[0] - oml_bmin[0])
    )
    max_dia = (
        rocket.max_diameter
        if rocket is not None
        else max(0.0, oml_bmax[1] - oml_bmin[1], oml_bmax[2] - oml_bmin[2])
    )

    metrics = [
        ("OML Length", f"{oml_len:.3f} m"),
        ("Body Max Dia", f"{max_dia * 1000:.1f} mm"),
        ("Watertight", "YES" if geom.is_watertight else "NO"),
        ("Free Edges", str(geom.free_edge_count)),
        ("OML Vol (m³)", f"{geom.oml_volume_m3:.5f}"),
        ("Domain Vol (m³)", f"{geom.domain_volume_m3:.2f}"),
        ("Triangles", f"{total_tris:,}"),
        ("STEP Round-Trip", str(verification_report.step_roundtrip_ok)),
    ]

    zones_payload = []
    for zname, mesh in sorted(zone_meshes.items()):
        v_bytes = np.ascontiguousarray(mesh.vertices_m, dtype="<f4").tobytes()
        t_bytes = np.ascontiguousarray(mesh.triangles, dtype="<u4").tobytes()
        is_translucent = mesh.is_boi or (zname in _TRANSLUCENT_ZONES)
        zones_payload.append(
            {
                "name": zname,
                "color": [round(c, 4) for c in _zone_color(zname)],
                "translucent": is_translucent,
                "isBoi": mesh.is_boi,
                "faces": mesh.face_count,
                "tris": int(mesh.triangles.shape[0]),
                "area": round(mesh.area_m2, 6),
                "v": base64.b64encode(v_bytes).decode("ascii"),
                "t": base64.b64encode(t_bytes).decode("ascii"),
            }
        )

    scene_payload = {
        "omlBbox": [oml_bmin, oml_bmax],
        "domainBbox": [dom_bmin, dom_bmax],
        "metrics": metrics,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "measured": c.measured,
                "expected": c.expected,
                "details": c.details,
            }
            for c in verification_report.checks
        ],
        "zones": zones_payload,
    }
    scene_json = Markup(json.dumps(scene_payload).replace("</", "<\\/"))

    n_pass = sum(1 for c in verification_report.checks if c.status == "PASS")
    html_text = template.render(
        title=f"{model_name} — 3D CAD Viewer & Verification",
        overall_status=verification_report.overall_status,
        header_meta=(
            f"Mode: {domain_out.main_name} | "
            f"{n_pass}/{len(verification_report.checks)} Checks PASS | "
            f"{total_tris:,} Triangles"
        ),
        scene_json=scene_json,
    )
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def _subsample_tris(mesh: ZoneMesh, max_tris: int) -> np.ndarray:
    """Return uniformly strided triangle index subset across the full zone up to max_tris."""
    n = int(mesh.triangles.shape[0])
    if n <= max_tris:
        return mesh.triangles
    idx = np.linspace(0, n - 1, max_tris, dtype=np.intp)
    return mesh.triangles[idx]


def _style_2d_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark engineering theme styling to a Matplotlib 2D axis."""
    ax.set_facecolor("#111827")
    ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=8.5)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(True, color="#334155", alpha=0.35, lw=0.6)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _draw_3d_panel(ax, meshes: list[ZoneMesh]) -> None:
    """Render 3D isometric CAD triangles into an existing Axes3D subplot."""
    ax.set_facecolor("#111827")
    bmin, bmax = _mesh_extents(meshes)
    for mesh in meshes:
        sub_t = _subsample_tris(mesh, max_tris=2200)
        poly_verts = mesh.vertices_m[sub_t]
        col = _zone_color(mesh.zone_name)
        coll = Poly3DCollection(
            poly_verts,
            facecolors=[(*col, 0.92)],
            edgecolors=[(0.06, 0.09, 0.16, 0.25)],
            linewidths=0.25,
        )
        ax.add_collection3d(coll)

    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    cz = 0.5 * (bmin[2] + bmax[2])
    dx = max(0.05, bmax[0] - bmin[0])
    dy = max(0.05, bmax[1] - bmin[1])
    dz = max(0.05, bmax[2] - bmin[2])
    half = 0.55 * max(dx, dy, dz)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.08, 0.11, 0.18, 1.0))
        axis.pane.set_edgecolor("#334155")
        axis._axinfo["grid"]["color"] = (0.20, 0.25, 0.33, 0.45)
    ax.set_title("3D Isometric Rocket OML", color="#e2e8f0", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("X [m]", color="#cbd5e1", fontsize=8)
    ax.set_ylabel("Y [m]", color="#cbd5e1", fontsize=8)
    ax.set_zlabel("Z [m]", color="#cbd5e1", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)


def _draw_projection_panel(
    ax,
    meshes: list[ZoneMesh],
    axes_idx: tuple[int, int],
    max_tris_per_zone: int = 3500,
) -> None:
    """Render filled 2D orthographic triangle projection for selected axes."""
    u_i, v_i = axes_idx
    for mesh in meshes:
        sub_t = _subsample_tris(mesh, max_tris=max_tris_per_zone)
        pts_2d = mesh.vertices_m[sub_t][:, :, [u_i, v_i]]
        col = _zone_color(mesh.zone_name)
        coll = PolyCollection(
            pts_2d,
            facecolors=[(*col, 0.72)],
            edgecolors=[(*col, 0.95)],
            linewidths=0.3,
        )
        ax.add_collection(coll)


def _draw_ork_meridian_overlay(ax, rocket: RocketModel) -> None:
    """Overlay dashed analytical OpenRocket meridian contour in X-Y plane."""
    segments_upper: list[list[tuple[float, float]]] = []
    segments_lower: list[list[tuple[float, float]]] = []
    for comp in rocket.all_components:
        pts = generate_component_profile_points(comp, num_points=30, global_coords=True)
        if len(pts) >= 2:
            segments_upper.append(pts)
            segments_lower.append([(px, -py) for px, py in pts])
    if segments_upper:
        ax.add_collection(
            LineCollection(
                segments_upper + segments_lower,
                colors="#22d3ee",
                linestyles="--",
                linewidths=1.4,
                label="Analytical .ork contour",
            )
        )


def export_cad_views_png(
    zone_meshes: dict[str, ZoneMesh],
    verification_report: VerificationReport,
    png_path: Path,
    model_name: str,
    domain_out: DomainBuildOutput,
    rocket: RocketModel | None = None,
) -> Path:
    """Render a thread-safe 4-panel CAD verification PNG image in SI meters."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(14.0, 9.5), dpi=180, facecolor="#0b1120")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(
        2, 2, wspace=0.22, hspace=0.28, top=0.91, bottom=0.06, left=0.06, right=0.97
    )

    oml_meshes = [
        m for n, m in zone_meshes.items() if not m.is_boi and n not in _NON_ROCKET_ZONES
    ]
    primary_meshes = oml_meshes or [
        m for m in zone_meshes.values() if not m.is_boi
    ] or list(zone_meshes.values())
    bmin, bmax = _mesh_extents(primary_meshes)

    # Panel 1: 3D Isometric
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_3d_panel(ax3d, primary_meshes)

    # Panel 2: Side Elevation X-Y with .ork overlay
    ax_side = fig.add_subplot(gs[0, 1])
    _draw_projection_panel(ax_side, primary_meshes, axes_idx=(0, 1))
    if rocket is not None:
        _draw_ork_meridian_overlay(ax_side, rocket)
        ax_side.legend(loc="upper right", fontsize=8, framealpha=0.6)
    pad_x = max(0.05, 0.05 * (bmax[0] - bmin[0]))
    span_yz = max(0.06, 0.65 * max(bmax[1] - bmin[1], bmax[2] - bmin[2]))
    cy = 0.5 * (bmin[1] + bmax[1])
    ax_side.set_xlim(bmin[0] - pad_x, bmax[0] + pad_x)
    ax_side.set_ylim(cy - span_yz, cy + span_yz)
    _style_2d_axis(ax_side, "Side Elevation (X-Y) — CAD vs .ork Overlay", "Axial X [m]", "Y [m]")

    # Panel 3: Tail / Axial Y-Z
    ax_axial = fig.add_subplot(gs[1, 0])
    _draw_projection_panel(ax_axial, primary_meshes, axes_idx=(1, 2))
    cz = 0.5 * (bmin[2] + bmax[2])
    ax_axial.set_xlim(cy - span_yz, cy + span_yz)
    ax_axial.set_ylim(cz - span_yz, cz + span_yz)
    ax_axial.set_aspect("equal")
    _style_2d_axis(ax_axial, "Tail Axial Projection (Y-Z)", "Y [m]", "Z [m]")

    # Panel 4: Split Fluid Domain overview + Verification Scorecard
    sub_gs = gs[1, 1].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.22)
    ax_dom = fig.add_subplot(sub_gs[0, 0])
    all_meshes = list(zone_meshes.values())
    _draw_projection_panel(ax_dom, all_meshes, axes_idx=(0, 1), max_tris_per_zone=1200)
    dbmin, dbmax = domain_out.bbox_domain_min_m, domain_out.bbox_domain_max_m
    dpadx = max(0.1, 0.04 * (dbmax[0] - dbmin[0]))
    dpady = max(0.1, 0.08 * (dbmax[1] - dbmin[1]))
    ax_dom.set_xlim(dbmin[0] - dpadx, dbmax[0] + dpadx)
    ax_dom.set_ylim(dbmin[1] - dpady, dbmax[1] + dpady)
    _style_2d_axis(ax_dom, f"Full Domain & BOIs ({domain_out.main_name})", "X [m]", "Y [m]")

    ax_txt = fig.add_subplot(sub_gs[1, 0])
    ax_txt.set_facecolor("#111827")
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    for spine in ax_txt.spines.values():
        spine.set_color("#334155")

    n_pass = sum(1 for c in verification_report.checks if c.status == "PASS")
    status_col = (
        "#4ade80"
        if verification_report.overall_status == "PASS"
        else "#fbbf24"
        if verification_report.overall_status == "WARN"
        else "#f87171"
    )
    score_lines = [
        f"STATUS: {verification_report.overall_status} "
        f"({n_pass}/{len(verification_report.checks)} checks PASS)",
        "-" * 64,
    ]
    for chk in verification_report.checks:
        tag = f"[{chk.status:4s}]"
        meas = chk.measured if len(chk.measured) <= 36 else chk.measured[:33] + "..."
        score_lines.append(f"{tag} {chk.name:<27s} {meas}")

    ax_txt.text(
        0.02,
        0.95,
        "\n".join(score_lines[:11]),
        va="top",
        ha="left",
        transform=ax_txt.transAxes,
        color=status_col,
        fontsize=7.6,
        family="monospace",
    )

    fig.suptitle(
        f"ork2cfd CAD Verification Sheet — {model_name} "
        f"(Status: {verification_report.overall_status})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
    return png_path
