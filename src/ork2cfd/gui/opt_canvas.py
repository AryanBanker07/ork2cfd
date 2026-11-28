"""Embedded Matplotlib live convergence & planform visualizer for fin optimization."""

from __future__ import annotations

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from ..models.rocket import FinCrossSection
from ..optimize.models import SimEvaluation

_PHASE_HEX: dict[str, str] = {
    "baseline": "#2563eb",
    "cross_section_screen": "#6366f1",
    "doe_lhs": "#0ea5e9",
    "bayesian_ei": "#10b981",
    "verification": "#f59e0b",
}

_CS_MARKERS: dict[FinCrossSection, str] = {
    FinCrossSection.SQUARE: "s",
    FinCrossSection.ROUNDED: "o",
    FinCrossSection.DOUBLE_WEDGE: "D",
    FinCrossSection.AIRFOIL: "^",
}


class OptimizationPlotCanvas(ctk.CTkFrame):
    """Live 2x2 Matplotlib convergence chart updating after every optimization sim."""

    def __init__(self, master, dark: bool = True, **kwargs) -> None:
        super().__init__(master, corner_radius=8, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._dark = dark
        self._last_evals: list[SimEvaluation] = []
        self._last_min_sm = 2.0
        self._last_stage = ""
        self._last_rocket = ""

        self.figure = Figure(figsize=(6.8, 4.6), dpi=100)
        self.axes = [
            self.figure.add_subplot(2, 2, 1),
            self.figure.add_subplot(2, 2, 2),
            self.figure.add_subplot(2, 2, 3),
            self.figure.add_subplot(2, 2, 4),
        ]
        self._ax_twin = None
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.set_theme(dark)
        self.render_empty()

    def set_theme(self, dark: bool) -> None:
        """Apply Dark or Light theme styling to all axes."""
        self._dark = dark
        if self._last_evals:
            self.update_evaluations(
                self._last_evals,
                min_sm_cal=self._last_min_sm,
                target_stage_name=self._last_stage,
                rocket_name=self._last_rocket,
            )
            return
        self._apply_axes_theme()
        self.canvas.draw_idle()

    def _apply_axes_theme(self) -> None:
        bg = "#1e222a" if self._dark else "#f8fafc"
        ax_bg = "#151921" if self._dark else "#ffffff"
        fg = "#e2e8f0" if self._dark else "#1e293b"
        grid_c = "#334155" if self._dark else "#cbd5e1"
        self.figure.patch.set_facecolor(bg)
        for ax in self.axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=fg, labelsize=7.5)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_color(grid_c)
        if self._ax_twin is not None:
            self._ax_twin.tick_params(colors="#a78bfa" if self._dark else "#7c3aed", labelsize=7.5)
            self._ax_twin.yaxis.label.set_color("#a78bfa" if self._dark else "#7c3aed")

    def render_empty(self) -> None:
        """Render placeholder state before any fin optimization run starts."""
        self._last_evals = []
        if self._ax_twin is not None:
            self._ax_twin.remove()
            self._ax_twin = None
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        fg = "#94a3b8" if self._dark else "#475569"
        self.axes[0].text(
            0.5,
            0.5,
            "Click 'Optimize Fins' to stream\nlive Cd Convergence History here",
            ha="center",
            va="center",
            color=fg,
            fontsize=9.5,
            transform=self.axes[0].transAxes,
        )
        self.axes[1].text(
            0.5,
            0.5,
            "Drag vs Static Margin\n(SM Feasibility Constraint)",
            ha="center",
            va="center",
            color=fg,
            fontsize=9,
            transform=self.axes[1].transAxes,
        )
        self.axes[2].text(
            0.5,
            0.5,
            "Cd vs Fin Planform Area\n& Sweep Exploration",
            ha="center",
            va="center",
            color=fg,
            fontsize=9,
            transform=self.axes[2].transAxes,
        )
        self.axes[3].text(
            0.5,
            0.5,
            "Baseline vs Lowest-Drag\nFin Planform Comparison",
            ha="center",
            va="center",
            color=fg,
            fontsize=9,
            transform=self.axes[3].transAxes,
        )
        self._apply_axes_theme()
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

    def update_evaluations(
        self,
        evals: list[SimEvaluation],
        *,
        min_sm_cal: float = 2.0,
        target_stage_name: str = "",
        rocket_name: str = "",
    ) -> None:
        """Redraw all four live optimization plots from completed SimEvaluation records."""
        if not evals:
            self.render_empty()
            return

        self._last_evals = list(evals)
        self._last_min_sm = min_sm_cal
        self._last_stage = target_stage_name
        self._last_rocket = rocket_name

        if self._ax_twin is not None:
            self._ax_twin.remove()
            self._ax_twin = None
        for ax in self.axes:
            ax.clear()

        fg = "#e2e8f0" if self._dark else "#1e293b"
        edge_col = "#f8fafc" if self._dark else "#0f172a"

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

        # Panel 1: Whole-Stage/Rocket Total Cd Convergence History
        ax0 = self.axes[0]
        ax0.plot(
            indices,
            running_best,
            "--",
            color="#22c55e",
            lw=1.5,
            label="Running Best",
            zorder=3,
        )
        ax0.axhline(
            baseline.cd_total,
            color="#94a3b8",
            ls=":",
            lw=1.0,
            label="Baseline Total Cd",
        )
        for ev in plot_evals:
            mk = _CS_MARKERS.get(ev.candidate.cross_section, "o")
            col = "#ef4444" if not ev.stability.feasible else _PHASE_HEX.get(ev.phase, "#10b981")
            ax0.scatter(
                ev.sim_index,
                ev.cd_total,
                marker=mk,
                s=55,
                color=col,
                edgecolors=edge_col,
                linewidths=0.6,
                zorder=4,
            )
            ax0.annotate(
                f"#{ev.sim_index}",
                (ev.sim_index, ev.cd_total),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=fg,
            )
        ax0.set_title("Stage/Rocket Total Cd Convergence", fontsize=9, fontweight="bold")
        ax0.set_xlabel("Simulation #", fontsize=8)
        ax0.set_ylabel("Whole-Stage/Rocket Total Cd", fontsize=8)
        ax0.set_xticks(indices)
        ax0.grid(alpha=0.2)
        ax0.legend(loc="upper right", fontsize=6.5, framealpha=0.5)

        # Panel 2: Whole-Stage/Rocket Total Cd vs Static Margin
        ax1 = self.axes[1]
        for ev in plot_evals:
            mk = _CS_MARKERS.get(ev.candidate.cross_section, "o")
            if ev.sim_index == best.sim_index:
                col, sz, mk_use = "#22c55e", 110, "*"
            else:
                col = (
                    "#ef4444"
                    if not ev.stability.feasible
                    else _PHASE_HEX.get(ev.phase, "#0ea5e9")
                )
                sz, mk_use = 55, mk
            ax1.scatter(
                ev.stability.static_margin_cal,
                ev.cd_total,
                marker=mk_use,
                s=sz,
                color=col,
                edgecolors=edge_col,
                linewidths=0.6,
                zorder=5 if ev.sim_index == best.sim_index else 4,
            )
            ax1.annotate(
                f"#{ev.sim_index}",
                (ev.stability.static_margin_cal, ev.cd_total),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=fg,
            )
        ax1.axvline(
            min_sm_cal,
            color="#ef4444",
            ls="--",
            lw=1.2,
            label=f"Min SM={min_sm_cal:.2f}",
        )
        ax1.set_title("Stage/Rocket Total Cd vs Static Margin", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Static Margin (cal)", fontsize=8)
        ax1.set_ylabel("Whole-Stage/Rocket Total Cd", fontsize=8)
        ax1.grid(alpha=0.2)
        ax1.legend(loc="upper left", fontsize=6.5, framealpha=0.5)

        # Panel 3: Whole-Stage/Rocket Total Cd vs Single-Fin Planform Area
        ax2 = self.axes[2]
        for ev in plot_evals:
            area_cm2 = ev.candidate.planform_area * 1e4
            col = (
                "#22c55e"
                if ev.sim_index == best.sim_index
                else _PHASE_HEX.get(ev.phase, "#38bdf8")
            )
            ax2.scatter(
                area_cm2,
                ev.cd_total,
                marker="*" if ev.sim_index == best.sim_index else "o",
                s=110 if ev.sim_index == best.sim_index else 55,
                color=col,
                edgecolors=edge_col,
                linewidths=0.6,
                zorder=4,
            )
            ax2.annotate(
                f"#{ev.sim_index} ({ev.candidate.sweep_angle_deg:.0f}°)",
                (area_cm2, ev.cd_total),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.8,
                color=fg,
            )
        ax2.set_title("Stage/Rocket Total Cd vs Fin Area", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Single-Fin Area (cm²)", fontsize=8)
        ax2.set_ylabel("Whole-Stage/Rocket Total Cd", fontsize=8)
        ax2.grid(alpha=0.2)

        # Panel 4: Baseline vs Best Fin Planform Overlay
        ax3 = self.axes[3]

        def _poly(cand):
            return [
                (-cand.root_chord, 0.0),
                (0.0, 0.0),
                (-cand.root_chord + cand.sweep_length + cand.tip_chord, cand.span),
                (-cand.root_chord + cand.sweep_length, cand.span),
            ]

        bc, oc = baseline.candidate, best.candidate
        ax3.add_patch(
            Polygon(
                _poly(bc),
                closed=True,
                facecolor="#94a3b8",
                alpha=0.25,
                edgecolor="#94a3b8",
                ls="--",
                lw=1.3,
                label=f"Base #{baseline.sim_index}",
            )
        )
        ax3.add_patch(
            Polygon(
                _poly(oc),
                closed=True,
                facecolor="#10b981",
                alpha=0.38,
                edgecolor="#10b981",
                lw=1.6,
                label=f"Best #{best.sim_index}",
            )
        )
        delta_pct = ((best.cd_total - baseline.cd_total) / max(baseline.cd_total, 1e-9)) * 100.0
        comp_txt = ""
        if best.cd_fins is not None:
            air_cd = best.cd_total - best.cd_fins
            comp_txt = f"\n(Airframe={air_cd:.4f} + Fins={best.cd_fins:.4f})"
        box_txt = (
            f"Best #{best.sim_index} ({oc.cross_section.value}):\n"
            f"Stage/Rocket Total Cd={best.cd_total:.5f} ({delta_pct:+.2f}%){comp_txt}\n"
            f"cr={oc.root_chord*1000:.1f} ct={oc.tip_chord*1000:.1f} mm\n"
            f"span={oc.span*1000:.1f} mm | sweep={oc.sweep_angle_deg:.1f}°\n"
            f"thick={oc.thickness*1000:.2f} mm (locked) | SM={best.stability.static_margin_cal:.2f}"
        )
        box_bg = "#0f172a" if self._dark else "#ffffff"
        ax3.text(
            0.03,
            0.97,
            box_txt,
            transform=ax3.transAxes,
            va="top",
            ha="left",
            fontsize=6.8,
            family="monospace",
            color=fg,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": box_bg, "alpha": 0.85},
        )
        all_pts = _poly(bc) + _poly(oc)
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        ax3.set_xlim(min(xs) - 0.06, max(xs) + 0.04)
        ax3.set_ylim(-0.01, max(ys) * 1.48 + 0.03)
        ax3.set_aspect("equal", adjustable="box")
        ax3.set_title("Baseline vs Optimized Planform", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Axial from TE (m)", fontsize=8)
        ax3.set_ylabel("Span (m)", fontsize=8)
        ax3.grid(alpha=0.2)
        ax3.legend(loc="lower right", fontsize=6.5, framealpha=0.5)

        self._apply_axes_theme()
        self.figure.tight_layout(pad=1.15)
        self.canvas.draw_idle()
