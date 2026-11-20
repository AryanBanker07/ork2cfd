"""Embedded Matplotlib live preview of rocket OML, fluid domain, BOIs, and zones."""

from __future__ import annotations

import math

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from ..geometry.profiles import generate_component_profile_points
from ..models.cfd import DomainConfig, DomainMode, EnclosureShape
from ..models.results import SolverRecommendation
from ..models.rocket import RocketModel

_ZONE_HEX: dict[str, str] = {
    "nosecone": "#f59e0b",
    "bodytube": "#38bdf8",
    "transition": "#c084fc",
    "fin": "#ef4444",
    "domain": "#60a5fa",
    "boi_nf": "#2dd4bf",
    "boi_wake": "#818cf8",
}


class PreviewCanvas(ctk.CTkFrame):
    """Two-panel live side-view visualizer (Fluid Domain + Zoomed Rocket OML)."""

    def __init__(self, master, dark: bool = True, **kwargs) -> None:
        super().__init__(master, corner_radius=8, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._dark = dark
        self.figure = Figure(figsize=(6.5, 4.6), dpi=100)
        self.ax_dom = self.figure.add_subplot(2, 1, 1)
        self.ax_oml = self.figure.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.set_theme(dark)
        self.render_empty()

    def set_theme(self, dark: bool) -> None:
        """Apply matching Dark or Light background and axis styling."""
        self._dark = dark
        bg = "#1e222a" if dark else "#f8fafc"
        fg = "#e2e8f0" if dark else "#1e293b"
        grid_c = "#334155" if dark else "#cbd5e1"
        self.figure.patch.set_facecolor(bg)
        for ax in (self.ax_dom, self.ax_oml):
            ax.set_facecolor("#151921" if dark else "#ffffff")
            ax.tick_params(colors=fg, labelsize=8)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_color(grid_c)
        self.canvas.draw_idle()

    def render_empty(self) -> None:
        """Show placeholder prompt before an .ork file is selected."""
        for ax in (self.ax_dom, self.ax_oml):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        fg = "#94a3b8" if self._dark else "#475569"
        self.ax_dom.text(
            0.5, 0.5, "Select an OpenRocket (.ork) file to preview\nCFD Domain, BOIs & Named Zones",
            ha="center", va="center", color=fg, fontsize=11, transform=self.ax_dom.transAxes,
        )
        self.ax_oml.text(
            0.5, 0.5, "Rocket OML Side Profile Preview",
            ha="center", va="center", color=fg, fontsize=10, transform=self.ax_oml.transAxes,
        )
        self.set_theme(self._dark)
        self.figure.tight_layout(pad=1.4)
        self.canvas.draw_idle()

    def update_preview(
        self,
        rocket: RocketModel,
        domain_cfg: DomainConfig,
        solver_rec: SolverRecommendation,
    ) -> None:
        """Redraw both Fluid Domain overview and zoomed Rocket OML profile."""
        self.ax_dom.clear()
        self.ax_oml.clear()

        L = max(0.1, rocket.total_length)
        up_L = domain_cfg.upstream_L or solver_rec.recommended_upstream_L
        down_L = domain_cfg.downstream_L or solver_rec.recommended_downstream_L
        rad_L = domain_cfg.radial_L or solver_rec.recommended_radial_L
        r_far = max(rad_L * L, 4.5 * rocket.max_fin_span_diameter)

        x_min = -up_L * L
        x_max = (1.0 + down_L) * L

        sym_modes = {
            DomainMode.SYMMETRY_HALF,
            DomainMode.SYMMETRY_SECTOR,
            DomainMode.AXISYMMETRIC_2D,
        }
        # Top subplot: Fluid domain & BOIs
        if domain_cfg.mode != DomainMode.OML_ONLY:
            if domain_cfg.shape == EnclosureShape.BULLET:
                dome_pts = [
                    (
                        -abs(x_min) * math.cos(0.5 * math.pi * i / 20),
                        r_far * math.sin(0.5 * math.pi * i / 20),
                    )
                    for i in range(21)
                ]
                if domain_cfg.mode in sym_modes:
                    poly_pts = dome_pts + [(x_max, r_far), (x_max, 0.0), (x_min, 0.0)]
                else:
                    bottom_dome = [(px, -py) for px, py in reversed(dome_pts)]
                    poly_pts = dome_pts + [(x_max, r_far), (x_max, -r_far)] + bottom_dome
                self.ax_dom.add_patch(
                    Polygon(
                        poly_pts,
                        closed=True,
                        fill=True,
                        facecolor=_ZONE_HEX["domain"],
                        alpha=0.14,
                        edgecolor=_ZONE_HEX["domain"],
                        lw=1.4,
                        label="farfield/outlet",
                    )
                )
            else:
                y0 = 0.0 if domain_cfg.mode in sym_modes else -r_far
                h = r_far if y0 == 0.0 else 2.0 * r_far
                self.ax_dom.add_patch(
                    Rectangle(
                        (x_min, y0),
                        x_max - x_min,
                        h,
                        facecolor=_ZONE_HEX["domain"],
                        alpha=0.14,
                        edgecolor=_ZONE_HEX["domain"],
                        lw=1.4,
                        label="domain",
                    )
                )

            if domain_cfg.boi.enabled and domain_cfg.mode != DomainMode.AXISYMMETRIC_2D:
                boi = domain_cfg.boi
                r_nf = max(
                    0.5 * rocket.max_fin_span_diameter * boi.nearfield_radius_factor,
                    rocket.max_radius * 2.5,
                )
                x_nf0 = -boi.nearfield_upstream_L * L
                x_nf1 = (1.0 + boi.nearfield_downstream_L) * L
                is_sym = domain_cfg.mode in {DomainMode.SYMMETRY_HALF, DomainMode.SYMMETRY_SECTOR}
                y_nf0 = 0.0 if is_sym else -r_nf
                h_nf = r_nf if y_nf0 == 0.0 else 2.0 * r_nf
                self.ax_dom.add_patch(
                    Rectangle(
                        (x_nf0, y_nf0),
                        x_nf1 - x_nf0,
                        h_nf,
                        facecolor=_ZONE_HEX["boi_nf"],
                        alpha=0.18,
                        edgecolor=_ZONE_HEX["boi_nf"],
                        ls="--",
                        lw=1.1,
                        label="boi_nearfield",
                    )
                )

                x_wk0, x_wk1 = 0.85 * L, (1.0 + boi.wake_length_L) * L
                r_wk0 = r_nf * 0.95
                r_wk1 = max(
                    r_nf * 1.15,
                    0.5 * rocket.max_fin_span_diameter * boi.wake_end_radius_factor,
                )
                if is_sym:
                    wk_pts = [(x_wk0, 0.0), (x_wk0, r_wk0), (x_wk1, r_wk1), (x_wk1, 0.0)]
                else:
                    wk_pts = [(x_wk0, -r_wk0), (x_wk0, r_wk0), (x_wk1, r_wk1), (x_wk1, -r_wk1)]
                self.ax_dom.add_patch(
                    Polygon(
                        wk_pts,
                        closed=True,
                        facecolor=_ZONE_HEX["boi_wake"],
                        alpha=0.15,
                        edgecolor=_ZONE_HEX["boi_wake"],
                        ls=":",
                        lw=1.1,
                        label="boi_wake",
                    )
                )

        # Draw rocket body silhouette in domain plot
        self._draw_rocket_patches(self.ax_dom, rocket, show_legend=False)
        if domain_cfg.mode != DomainMode.OML_ONLY:
            pad_x = 0.06 * (x_max - x_min)
            self.ax_dom.set_xlim(x_min - pad_x, x_max + pad_x)
            y_low = -0.1 * r_far if domain_cfg.mode in sym_modes else -1.1 * r_far
            self.ax_dom.set_ylim(y_low, 1.1 * r_far)
        else:
            self.ax_dom.set_xlim(-0.1 * L, 1.1 * L)
            r_lim = max(0.2, rocket.max_fin_span_diameter)
            self.ax_dom.set_ylim(-r_lim, r_lim)
        self.ax_dom.set_title(
            f"Fluid Domain & BOIs ({domain_cfg.mode.value} / {domain_cfg.shape.value})",
            fontsize=10,
        )
        self.ax_dom.set_xlabel("X [m]", fontsize=8)
        self.ax_dom.set_ylabel("R / Y [m]", fontsize=8)
        self.ax_dom.legend(loc="upper right", fontsize=7, framealpha=0.5)

        # Bottom subplot: Zoomed Rocket OML Side Profile
        self._draw_rocket_patches(self.ax_oml, rocket, show_legend=True)
        self.ax_oml.axhline(0.0, color="#94a3b8", ls="-.", lw=0.7, alpha=0.6)
        self.ax_oml.set_xlim(-0.06 * L, 1.06 * L)
        span_r = max(0.08, 0.72 * rocket.max_fin_span_diameter)
        self.ax_oml.set_ylim(-span_r, span_r)
        self.ax_oml.set_aspect("equal", adjustable="datalim")
        self.ax_oml.set_title(
            f"Rocket OML Side Profile — {rocket.name} "
            f"(L={L:.3f}m, D={rocket.max_diameter*1000:.1f}mm)",
            fontsize=10,
        )
        self.ax_oml.set_xlabel("Axial Station X [m]", fontsize=8)
        self.ax_oml.set_ylabel("Radial Y [m]", fontsize=8)

        self.set_theme(self._dark)
        self.figure.tight_layout(pad=1.3)
        self.canvas.draw_idle()

    def _draw_rocket_patches(self, ax, rocket: RocketModel, show_legend: bool) -> None:
        """Render OML body segments and fins color-coded by Named Selection zone."""
        seen_labels: set[str] = set()
        for comp in rocket.all_components:
            pts = generate_component_profile_points(comp, num_points=40, global_coords=True)
            poly_pts = pts + [(px, -pr) for px, pr in reversed(pts)]
            color = _ZONE_HEX.get(comp.component_type, "#38bdf8")
            lbl = None
            if show_legend:
                is_boattail = (
                    comp.component_type == "transition" and comp.aft_radius < comp.fore_radius
                )
                zone_lbl = (
                    "rocket_nosecone"
                    if comp.component_type == "nosecone"
                    else "rocket_boattail"
                    if is_boattail
                    else "rocket_body"
                )
                if zone_lbl not in seen_labels:
                    lbl = zone_lbl
                    seen_labels.add(zone_lbl)
            ax.add_patch(
                Polygon(
                    poly_pts,
                    closed=True,
                    facecolor=color,
                    alpha=0.75,
                    edgecolor=color,
                    lw=1.0,
                    label=lbl,
                )
            )

            for fin in comp.fins:
                r_le = rocket.radius_at_x(fin.global_root_le_x)
                r_te = rocket.radius_at_x(fin.global_root_te_x)
                r_mid = rocket.radius_at_x(0.5 * (fin.global_root_le_x + fin.global_root_te_x))
                x_le = fin.global_root_le_x
                x_te = fin.global_root_te_x
                y_tip = r_mid + fin.span

                if fin.fin_type == "trapezoid":
                    x_tip_le = x_le + fin.sweep_length
                    x_tip_te = x_tip_le + fin.tip_chord
                    fin_top = [(x_le, r_le), (x_tip_le, y_tip), (x_tip_te, y_tip), (x_te, r_te)]
                elif fin.fin_type == "elliptical":
                    xm = 0.5 * (x_le + x_te)
                    hc = 0.5 * fin.root_chord
                    fin_top = [
                        (
                            xm + hc * math.cos(math.pi * (1.0 - i / 18)),
                            r_mid + fin.span * math.sin(math.pi * (1.0 - i / 18)),
                        )
                        for i in range(19)
                    ]
                elif fin.fin_type == "freeform" and fin.points:
                    fin_top = [(x_le + px, r_mid + py) for px, py in fin.points]
                else:
                    fin_top = [(x_le, r_le), (x_le, y_tip), (x_te, y_tip), (x_te, r_te)]

                fin_bot = [(px, -py) for px, py in fin_top]
                f_lbl = (
                    "rocket_fins"
                    if (show_legend and "rocket_fins" not in seen_labels)
                    else None
                )
                if f_lbl:
                    seen_labels.add(f_lbl)
                ax.add_patch(
                    Polygon(
                        fin_top,
                        closed=True,
                        facecolor=_ZONE_HEX["fin"],
                        alpha=0.8,
                        edgecolor="#fca5a5",
                        lw=0.8,
                        label=f_lbl,
                    )
                )
                ax.add_patch(
                    Polygon(
                        fin_bot,
                        closed=True,
                        facecolor=_ZONE_HEX["fin"],
                        alpha=0.8,
                        edgecolor="#fca5a5",
                        lw=0.8,
                    )
                )

        if show_legend and seen_labels:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.55)
