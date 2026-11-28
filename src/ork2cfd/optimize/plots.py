"""Visualization utilities: Fluent HDF5 CFD contour PNGs and Cd vs fin shape plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import griddata

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from ..models.rocket import FinCrossSection, FinSet, RocketModel  # noqa: E402
from .models import SimEvaluation  # noqa: E402

_CS_MARKERS: dict[FinCrossSection, str] = {
    FinCrossSection.SQUARE: "s",
    FinCrossSection.ROUNDED: "o",
    FinCrossSection.AIRFOIL: "^",
    FinCrossSection.DOUBLE_WEDGE: "D",
}


def _decode_zone_names(raw_names) -> list[str]:
    out: list[str] = []
    for item in raw_names:
        text = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
        out.extend(part.strip() for part in text.split(";") if part.strip())
    return out


def _draw_rocket_oml_polygon(
    ax: plt.Axes,
    rocket: RocketModel,
    *,
    facecolor: str = "#1f2937",
    edgecolor: str = "#e5e7eb",
) -> None:
    """Fill the axisymmetric rocket OML silhouette (+Y half-plane) on an X-Y axis."""
    xs: list[float] = []
    ys: list[float] = []
    for comp in rocket.all_components:
        n_pts = 30 if comp.component_type in {"nosecone", "transition"} else 4
        for i in range(n_pts + 1):
            lx = comp.length * (i / n_pts)
            xs.append(comp.axial_start + lx)
            ys.append(comp.radius_at_local_x(lx))
    if not xs:
        return
    pts = list(zip(xs, ys, strict=True)) + [(xs[-1], 0.0), (xs[0], 0.0)]
    ax.add_patch(
        Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor, lw=0.9, zorder=4)
    )


def _fin_polygon_xy(fin: FinSet, rocket: RocketModel) -> list[tuple[float, float]]:
    r_le = rocket.radius_at_x(fin.global_root_le_x)
    r_te = rocket.radius_at_x(fin.global_root_te_x)
    r_mid = rocket.radius_at_x(0.5 * (fin.global_root_le_x + fin.global_root_te_x))
    x_le_tip = fin.global_root_le_x + fin.sweep_length
    x_te_tip = x_le_tip + fin.tip_chord
    y_tip = r_mid + fin.span
    return [
        (fin.global_root_le_x, r_le),
        (fin.global_root_te_x, r_te),
        (x_te_tip, y_tip),
        (x_le_tip, y_tip),
    ]


def _grid_slice_field(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    vals: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rocket: RocketModel | None = None,
    nx: int = 340,
    ny: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate unstructured +Y slice cell centers onto a gap-free regular 2D grid."""
    gx = np.linspace(xlim[0], xlim[1], nx)
    gy = np.linspace(ylim[0], ylim[1], ny)
    xx, yy = np.meshgrid(gx, gy)
    if len(x_pts) < 4:
        return xx, yy, np.ma.masked_all_like(xx)

    pts_2d = np.column_stack([x_pts, y_pts])
    grid_near = griddata(pts_2d, vals, (xx, yy), method="nearest")
    try:
        grid_lin = griddata(pts_2d, vals, (xx, yy), method="linear")
        grid_vals = np.where(np.isnan(grid_lin), grid_near, grid_lin)
    except Exception:
        grid_vals = grid_near

    mask = np.isnan(grid_vals)
    if rocket is not None:
        oml_r = np.array(
            [
                rocket.radius_at_x(float(x)) if 0.0 <= float(x) <= rocket.total_length else -1.0
                for x in gx
            ],
            dtype=float,
        )
        mask = mask | ((xx >= 0.0) & (xx <= rocket.total_length) & (yy < oml_r[None, :] * 0.98))
    return xx, yy, np.ma.array(grid_vals, mask=mask)


def render_fluent_hdf5_contours(
    case_path: Path,
    data_path: Path,
    out_png: Path,
    *,
    title: str,
    rocket: RocketModel,
    target_fin: FinSet,
    mach_inf: float,
    p_inf_pa: float,
    rho_inf: float,
    v_inf: float,
) -> Path | None:
    """
    Render a 4-panel CFD contour figure (Mach & Cp slices + fin surface Cp) from Fluent HDF5.

    Works directly headless from `.cas.h5` + `.dat.h5` without needing an OpenGL display.
    Returns `None` if reading/plotting fails so solver automation never aborts.
    """
    try:
        import h5py
    except ImportError:
        return None

    fig = None
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(case_path, "r") as fc, h5py.File(data_path, "r") as fd:
            coords = fc["meshes/1/nodes/coords/1"][:]
            nnodes = fc["meshes/1/faces/nodes/1/nnodes"][:]
            fnodes = fc["meshes/1/faces/nodes/1/nodes"][:] - 1
            c0 = fc["meshes/1/faces/c0/1"][:] - 1
            c1 = fc["meshes/1/faces/c1/1"][:] - 1
            ncells = int(fc["meshes/1/cells/zoneTopology/maxId"][0])

            starts = np.zeros(len(nnodes) + 1, dtype=np.int64)
            np.cumsum(nnodes, out=starts[1:])
            fcent = np.add.reduceat(coords[fnodes], starts[:-1], axis=0) / nnodes[:, None]

            ccent = np.zeros((ncells, 3), dtype=np.float64)
            ccnt = np.zeros(ncells, dtype=np.int32)
            v0 = (c0 >= 0) & (c0 < ncells)
            np.add.at(ccent, c0[v0], fcent[v0])
            np.add.at(ccnt, c0[v0], 1)
            n_c1 = len(c1)
            v1 = (c1 >= 0) & (c1 < ncells)
            idx1 = np.nonzero(v1)[0]
            if len(idx1) > 0:
                np.add.at(ccent, c1[idx1], fcent[:n_c1][idx1])
                np.add.at(ccnt, c1[idx1], 1)
            ccent /= np.maximum(ccnt[:, None], 1)

            cells_grp = fd["results/1/phase-1/cells"]
            u = cells_grp["SV_U/1"][:]
            v = cells_grp["SV_V/1"][:]
            w = cells_grp["SV_W/1"][:]
            temp = cells_grp["SV_T/1"][:]
            pres = cells_grp["SV_P/1"][:]

            mach = np.sqrt(u * u + v * v + w * w) / np.sqrt(
                1.4 * 287.058 * np.maximum(temp, 50.0)
            )
            q_dyn = max(0.5 * rho_inf * v_inf * v_inf, 1.0)
            # In Fluent ideal-gas mode operating_pressure=0 so pres may be absolute static or gauge
            p_ref_shift = p_inf_pa if float(np.median(pres)) > 0.25 * p_inf_pa else 0.0
            cp_cell = np.clip((pres - p_ref_shift) / q_dyn, -1.2, 1.2)

            # Extract meridian +Y vertical slice (+Z >= 0 sector/full compatible)
            y_all = ccent[:, 1]
            z_all = ccent[:, 2]
            band = np.maximum(0.025, 0.28 * np.maximum(y_all, 0.02))
            slice_mask = (y_all >= -0.01) & (np.abs(z_all) <= band)
            sx = ccent[slice_mask, 0]
            sy = np.hypot(np.maximum(y_all[slice_mask], 0.0), z_all[slice_mask])
            smach = mach[slice_mask]
            scp = cp_cell[slice_mask]

            # Boundary faces on rocket_fins*
            zt_names = _decode_zone_names(fc["meshes/1/faces/zoneTopology/name"][:])
            zt_min = fc["meshes/1/faces/zoneTopology/minId"][:] - 1
            zt_max = fc["meshes/1/faces/zoneTopology/maxId"][:]
            fin_face_indices: list[np.ndarray] = []
            for zname, f0, f1 in zip(zt_names, zt_min, zt_max, strict=False):
                if "rocket_fins" in zname.lower():
                    fin_face_indices.append(np.arange(int(f0), int(f1), dtype=np.int64))
            if fin_face_indices:
                ff_idx = np.concatenate(fin_face_indices)
                ff_cent = fcent[ff_idx]
                ff_cp = cp_cell[np.clip(c0[ff_idx], 0, ncells - 1)]
                r_fin = np.hypot(ff_cent[:, 1], ff_cent[:, 2])
                x_hi_target = max(
                    target_fin.global_root_te_x,
                    target_fin.global_root_le_x + target_fin.sweep_length + target_fin.tip_chord,
                )
                target_mask = (
                    (ff_cent[:, 0] >= target_fin.global_root_le_x - 0.05)
                    & (ff_cent[:, 0] <= x_hi_target + 0.05)
                )
                fin_wx = ff_cent[target_mask, 0]
                fin_wr = r_fin[target_mask]
                fin_wcp = ff_cp[target_mask]
            else:
                fin_wx = np.empty(0)
                fin_wr = np.empty(0)
                fin_wcp = np.empty(0)

        l_tot = max(rocket.total_length, 0.5)
        d_max = max(rocket.max_diameter, 0.05)
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.2), dpi=160)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # Panel 1: Full-vehicle X-Y Mach contour
        ax = axes[0, 0]
        full_xlim = (-0.25 * l_tot, 1.30 * l_tot)
        full_ylim = (0.0, 4.5 * d_max)
        in_full = (
            (sx >= full_xlim[0] - 0.2)
            & (sx <= full_xlim[1] + 0.2)
            & (sy <= full_ylim[1] + 0.2)
        )
        xx, yy, zz = _grid_slice_field(
            sx[in_full], sy[in_full], smach[in_full], full_xlim, full_ylim, rocket=rocket
        )
        m_hi = max(
            mach_inf * 1.35,
            float(np.percentile(smach[in_full], 99)) if np.any(in_full) else 1.1,
        )
        im0 = ax.contourf(
            xx, yy, zz, levels=np.linspace(0.0, m_hi, 32), cmap="turbo", extend="both"
        )
        _draw_rocket_oml_polygon(ax, rocket)
        for fin in rocket.all_fins:
            ax.add_patch(
                Polygon(
                    _fin_polygon_xy(fin, rocket),
                    closed=True,
                    facecolor="#374151",
                    edgecolor="w",
                    lw=0.7,
                    zorder=5,
                )
            )
        fig.colorbar(im0, ax=ax, label="Mach Number", fraction=0.046, pad=0.04)
        ax.set_title("Full-Rocket Centerline Mach Number")
        ax.set_xlabel("Axial x (m)")
        ax.set_ylabel("Radial y (m)")
        ax.set_xlim(*full_xlim)
        ax.set_ylim(*full_ylim)

        # Panel 2: Zoomed Booster Fin & Wake Mach contour
        ax = axes[0, 1]
        x_tip_te = target_fin.global_root_le_x + target_fin.sweep_length + target_fin.tip_chord
        fin_xlim = (
            target_fin.global_root_le_x - 0.22,
            max(rocket.total_length, x_tip_te) + 0.35,
        )
        x_fin_mid = 0.5 * (target_fin.global_root_le_x + target_fin.global_root_te_x)
        r_body = rocket.radius_at_x(x_fin_mid)
        fin_ylim = (0.0, r_body + 2.0 * max(target_fin.span, 0.05))
        in_fin = (
            (sx >= fin_xlim[0] - 0.1)
            & (sx <= fin_xlim[1] + 0.1)
            & (sy <= fin_ylim[1] + 0.1)
        )
        xx_f, yy_f, zz_f = _grid_slice_field(
            sx[in_fin],
            sy[in_fin],
            smach[in_fin],
            fin_xlim,
            fin_ylim,
            rocket=rocket,
            nx=300,
            ny=200,
        )
        im1 = ax.contourf(
            xx_f, yy_f, zz_f, levels=np.linspace(0.0, m_hi, 32), cmap="turbo", extend="both"
        )
        _draw_rocket_oml_polygon(ax, rocket)
        ax.add_patch(
            Polygon(
                _fin_polygon_xy(target_fin, rocket),
                closed=True,
                facecolor="none",
                edgecolor="white",
                lw=1.4,
                zorder=6,
            )
        )
        fig.colorbar(im1, ax=ax, label="Mach Number", fraction=0.046, pad=0.04)
        ax.set_title("Booster Fin & Base Wake Mach Field")
        ax.set_xlabel("Axial x (m)")
        ax.set_ylabel("Radial y (m)")
        ax.set_xlim(*fin_xlim)
        ax.set_ylim(*fin_ylim)

        # Panel 3: Zoomed Booster Fin & Wake Cp contour
        ax = axes[1, 0]
        _, _, zz_cp = _grid_slice_field(
            sx[in_fin],
            sy[in_fin],
            scp[in_fin],
            fin_xlim,
            fin_ylim,
            rocket=rocket,
            nx=300,
            ny=200,
        )
        im2 = ax.contourf(
            xx_f, yy_f, zz_cp, levels=np.linspace(-0.8, 0.8, 33), cmap="coolwarm", extend="both"
        )
        _draw_rocket_oml_polygon(ax, rocket)
        ax.add_patch(
            Polygon(
                _fin_polygon_xy(target_fin, rocket),
                closed=True,
                facecolor="none",
                edgecolor="black",
                lw=1.3,
                zorder=6,
            )
        )
        fig.colorbar(im2, ax=ax, label="Cp = (p - p∞) / q∞", fraction=0.046, pad=0.04)
        ax.set_title("Booster Fin & Base Wake Pressure Coefficient (Cp)")
        ax.set_xlabel("Axial x (m)")
        ax.set_ylabel("Radial y (m)")
        ax.set_xlim(*fin_xlim)
        ax.set_ylim(*fin_ylim)

        # Panel 4: Fin Wall Surface Cp distribution
        ax = axes[1, 1]
        if len(fin_wx) > 0:
            sc = ax.scatter(
                fin_wx, fin_wr, c=fin_wcp, s=8, cmap="coolwarm", vmin=-0.6, vmax=0.8, alpha=0.85
            )
            fig.colorbar(sc, ax=ax, label="Surface Cp", fraction=0.046, pad=0.04)
        ax.add_patch(
            Polygon(
                _fin_polygon_xy(target_fin, rocket),
                closed=True,
                facecolor="none",
                edgecolor="#111827",
                lw=1.4,
            )
        )
        ax.set_title("Target Fin Wall Surface Cp Distribution")
        ax.set_xlabel("Axial x (m)")
        ax.set_ylabel("Radial Distance r (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        fig.savefig(out_png, dpi=160)
        return out_png
    except Exception:
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def render_optimization_cd_plot(
    evals: list[SimEvaluation],
    out_png: Path,
    *,
    rocket_name: str,
    target_stage_name: str,
    min_sm_cal: float,
) -> Path:
    """Render a 4-panel Cd vs fin shape & convergence summary plot."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not evals:
        raise ValueError("Cannot render Cd plot with zero evaluations.")

    baseline = evals[0]
    valid_evals = [e for e in evals if not e.diverged]
    feasible = [e for e in valid_evals if e.stability.feasible]
    best = min(feasible or valid_evals or evals, key=lambda e: e.cd_total)

    plot_evals = valid_evals or evals
    indices = [e.sim_index for e in plot_evals]
    running_best: list[float] = []
    cur_best = float("inf")
    for ev in plot_evals:
        if ev.stability.feasible or cur_best == float("inf"):
            cur_best = min(cur_best, ev.cd_total)
        running_best.append(cur_best)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.5), dpi=170)
    mode_tag = "DRY-RUN PROXY" if best.is_dry_run else "LIVE ANSYS FLUENT"
    div_count = sum(1 for e in evals if e.diverged)
    div_tag = f" | {div_count} diverged excluded" if div_count else ""
    fig.suptitle(
        f"{rocket_name} ({target_stage_name} Fins) — Whole-Stage/Rocket Cd ({mode_tag}{div_tag})",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Whole-Stage/Rocket Total Cd vs Simulation #
    ax = axes[0, 0]
    ax.plot(indices, running_best, "--", color="#059669", lw=1.6, label="Running Best (Feasible)")
    for ev in plot_evals:
        mk = _CS_MARKERS.get(ev.candidate.cross_section, "o")
        col = "#dc2626" if not ev.stability.feasible else (
            "#2563eb" if ev.phase in {"baseline", "cross_section_screen"} else "#059669"
        )
        ax.scatter(
            ev.sim_index,
            ev.cd_total,
            marker=mk,
            s=85,
            color=col,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )
        ax.annotate(
            f"#{ev.sim_index}",
            (ev.sim_index, ev.cd_total),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(baseline.cd_total, color="#6b7280", ls=":", lw=1.0, label="Baseline Total Cd")
    ax.set_title("Stage/Rocket Total Cd Convergence")
    ax.set_xlabel("Simulation #")
    ax.set_ylabel("Whole-Stage/Rocket Total Cd")
    ax.set_xticks(indices)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Whole-Stage/Rocket Total Cd vs Single-Fin Planform Area colored by Sweep Angle
    ax = axes[0, 1]
    areas_cm2 = [e.candidate.planform_area * 1e4 for e in plot_evals]
    sweeps = [e.candidate.sweep_angle_deg for e in plot_evals]
    vmin_s, vmax_s = min(sweeps), max(sweeps)
    if abs(vmax_s - vmin_s) < 1e-3:
        vmax_s = vmin_s + 1.0
    norm = plt.Normalize(vmin=vmin_s, vmax=vmax_s)
    cmap = plt.get_cmap("viridis")
    for ev, area_c, swp in zip(plot_evals, areas_cm2, sweeps, strict=True):
        mk = _CS_MARKERS.get(ev.candidate.cross_section, "o")
        ax.scatter(
            area_c,
            ev.cd_total,
            marker=mk,
            s=95,
            color=cmap(norm(swp)),
            edgecolors="#dc2626" if not ev.stability.feasible else "black",
            linewidths=1.2 if ev.sim_index == best.sim_index else 0.7,
            zorder=4,
        )
        ax.annotate(
            f"#{ev.sim_index}",
            (area_c, ev.cd_total),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    sm_obj = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm_obj, ax=ax, label="LE Sweep Angle (deg)")
    ax.set_title("Stage/Rocket Total Cd vs Fin Area")
    ax.set_xlabel("Single-Fin Planform Area (cm^2)")
    ax.set_ylabel("Whole-Stage/Rocket Total Cd")
    ax.grid(alpha=0.25)

    # Panel 3: Whole-Stage/Rocket Total Cd vs Barrowman Static Margin
    ax = axes[1, 0]
    sms = [e.stability.static_margin_cal for e in plot_evals]
    for ev, sm_val in zip(plot_evals, sms, strict=True):
        mk = _CS_MARKERS.get(ev.candidate.cross_section, "o")
        col = "#059669" if ev.sim_index == best.sim_index else (
            "#2563eb" if ev.stability.feasible else "#dc2626"
        )
        ax.scatter(
            sm_val,
            ev.cd_total,
            marker=mk,
            s=95,
            color=col,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )
        ax.annotate(
            f"#{ev.sim_index} ({ev.candidate.cross_section.value})",
            (sm_val, ev.cd_total),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.axvline(min_sm_cal, color="#dc2626", ls="--", lw=1.4, label=f"Min SM = {min_sm_cal:.2f} cal")
    ax.set_title("Stage/Rocket Total Cd vs Static Margin")
    ax.set_xlabel("Barrowman Static Margin (calibers)")
    ax.set_ylabel("Whole-Stage/Rocket Total Cd")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    # Panel 4: Baseline vs Optimized Fin Planform Overlay + OpenRocket Params
    ax = axes[1, 1]

    def _local_poly(cand) -> list[tuple[float, float]]:
        return [
            (-cand.root_chord, 0.0),
            (0.0, 0.0),
            (-cand.root_chord + cand.sweep_length + cand.tip_chord, cand.span),
            (-cand.root_chord + cand.sweep_length, cand.span),
        ]

    bc, oc = baseline.candidate, best.candidate
    ax.add_patch(
        Polygon(
            _local_poly(bc),
            closed=True,
            facecolor="#9ca3af",
            alpha=0.35,
            edgecolor="#374151",
            ls="--",
            lw=1.4,
            label=f"Baseline ({bc.cross_section.value})",
        )
    )
    ax.add_patch(
        Polygon(
            _local_poly(oc),
            closed=True,
            facecolor="#10b981",
            alpha=0.40,
            edgecolor="#047857",
            lw=2.0,
            label=f"Optimized #{best.sim_index} ({oc.cross_section.value})",
        )
    )
    delta_cd = best.cd_total - baseline.cd_total
    delta_pct = (delta_cd / max(baseline.cd_total, 1e-9)) * 100.0
    comp_line = ""
    if best.cd_fins is not None:
        airframe_cd = best.cd_total - best.cd_fins
        comp_line = f"\n  (Airframe={airframe_cd:.4f} + Fins={best.cd_fins:.4f})"
    summary_txt = (
        f"Lowest-Drag Fin (#{best.sim_index}, {oc.cross_section.value}):\n"
        f"  Stage/Rocket Cd: {best.cd_total:.5f} ({delta_pct:+.2f}%){comp_line}\n"
        f"  Root Chord:      {oc.root_chord*1000:.2f} mm\n"
        f"  Tip Chord:       {oc.tip_chord*1000:.2f} mm\n"
        f"  Height/Span:     {oc.span*1000:.2f} mm\n"
        f"  Sweep Len:       {oc.sweep_length*1000:.2f} mm ({oc.sweep_angle_deg:.1f}°)\n"
        f"  Thickness:       {oc.thickness*1000:.2f} mm (locked)\n"
        f"  Static Marg:     {best.stability.static_margin_cal:.2f} cal"
    )
    ax.text(
        0.03,
        0.97,
        summary_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cbd5e1",
        },
    )
    max_c = max(bc.root_chord, oc.root_chord, oc.sweep_length + oc.tip_chord) * 1.15
    max_s = max(bc.span, oc.span) * 1.45
    ax.set_xlim(-max_c, 0.08 * max_c)
    ax.set_ylim(-0.01, max_s)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Baseline vs Optimized Booster Fin Planform (TE Aligned)")
    ax.set_xlabel("Axial from Trailing Edge (m)")
    ax.set_ylabel("Radial Span (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png
