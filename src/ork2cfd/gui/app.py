"""Main CustomTkinter desktop window for ork2cfd."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ..ansys.detector import AnsysInstallation, select_best_installation
from ..ansys.fluent_launcher import (
    find_solved_files,
    launch_cfdpost_visualization,
    launch_fluent_run,
    launch_fluent_visualization,
    parse_cfd_results,
    run_fluent_solve_blocking,
)
from ..ansys.mesh_convergence import run_mesh_convergence_blocking
from ..ansys.next_steps import build_ansys_next_steps_markdown, build_ansys_next_steps_summary
from ..ansys.script_generator import _symmetry_multiplier
from ..models.cfd import (
    DEFAULT_SOLVER_ITERATIONS,
    BoiConfig,
    CadConfig,
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    EnclosureShape,
    FlightCondition,
    FlowRegime,
    MeshConvergenceConfig,
    MeshPreset,
    PipelineConfig,
)
from ..models.results import CfdMachPoint, CfdSolveResult, PackageResult
from ..models.rocket import RocketModel
from ..optimize import (
    FinOptimizationConfig,
    FinOptimizationResult,
    SimEvaluation,
    run_fin_optimization,
)
from ..parser.ork_parser import parse_ork_file
from ..physics.boundary_layer import compute_boundary_layer
from ..physics.regime import recommend_solver
from ..physics.resources import ResourceRecommendation, detect_system_hardware, recommend_resources
from ..pipeline import run_pipeline, sanitize_stem
from .opt_canvas import OptimizationPlotCanvas
from .preview_canvas import PreviewCanvas
from .progress_tracker import ProgressSnapshot, WorkflowProgressTracker, format_elapsed
from .widgets import (
    InfoIcon,
    SectionCard,
    add_labeled_checkbox,
    add_labeled_combo,
    add_labeled_entry,
)

_SPINNER_FRAMES = ("◐", "◓", "◑", "◒")

# phase_key -> (dark_badge_bg, light_badge_bg, dark_bar_fg, light_bar_fg)
_PHASE_COLORS: dict[str, tuple[str, str, str, str]] = {
    "idle": ("#334155", "#cbd5e1", "#3b82f6", "#2563eb"),
    "parse": ("#1d4ed8", "#2563eb", "#38bdf8", "#0284c7"),
    "cad": ("#1d4ed8", "#2563eb", "#38bdf8", "#0284c7"),
    "verify": ("#4338ca", "#4f46e5", "#818cf8", "#4f46e5"),
    "scripts": ("#0369a1", "#0284c7", "#38bdf8", "#0284c7"),
    "workbench": ("#6d28d9", "#7c3aed", "#a78bfa", "#7c3aed"),
    "fluent_launch": ("#0f766e", "#0d9488", "#14b8a6", "#0d9488"),
    "fluent_surface_mesh": ("#0f766e", "#0d9488", "#14b8a6", "#0d9488"),
    "fluent_volume_mesh": ("#0f766e", "#0d9488", "#14b8a6", "#0d9488"),
    "fluent_setup": ("#0f766e", "#0d9488", "#2dd4bf", "#0d9488"),
    "fluent_solve": ("#0f766e", "#0d9488", "#2dd4bf", "#0d9488"),
    "fluent_post": ("#0369a1", "#0284c7", "#38bdf8", "#0284c7"),
    "done": ("#15803d", "#16a34a", "#22c55e", "#16a34a"),
    "error": ("#b91c1c", "#dc2626", "#ef4444", "#dc2626"),
}


class Ork2CfdApp(ctk.CTk):
    """Simple, responsive desktop GUI bundling STEP, ANSYS scripts, and .wbpj generation."""

    def __init__(self, initial_ork: Path | None = None) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._base_title = "ork2cfd — OpenRocket to ANSYS Fluent CFD Converter"
        self.title(self._base_title)
        self.geometry("1240x820")
        self.minsize(1040, 680)

        self._rocket: RocketModel | None = None
        self._last_result: PackageResult | None = None
        self._last_cfd_result: CfdSolveResult | None = None
        self._last_opt_result: FinOptimizationResult | None = None
        self._ansys_install: AnsysInstallation | None = select_best_installation()
        self._is_running = False
        self._info_icons: list[InfoIcon] = []
        self._hw = detect_system_hardware()
        self._tracker = WorkflowProgressTracker(mode="package")
        self._last_snapshot = self._tracker.idle_snapshot()
        self._run_start_mono: float | None = None
        self._last_elapsed_text = ""
        self._ticker_after_id: str | None = None
        self._spinner_idx = 0
        self._stage_pills: list[ctk.CTkLabel] = []
        self._rendered_stage_labels: tuple[str, ...] = ()

        self._init_vars()
        self._build_layout()

        if initial_ork is not None and initial_ork.exists():
            self._load_ork(initial_ork)

    def _init_vars(self) -> None:
        self.var_ork_path = ctk.StringVar(value="")
        self.var_stage = ctk.StringVar(value="All Stages")
        self._stage_label_to_key: dict[str, str] = {"All Stages": "all"}
        self.var_out_dir = ctk.StringVar(value="")

        self.var_mach = ctk.StringVar(value="0.80")
        self.var_alt = ctk.StringVar(value="500")
        self.var_aoa = ctk.StringVar(value="0.0")
        self.var_yplus = ctk.StringVar(value="1.0")
        self.var_regime = ctk.StringVar(value=FlowRegime.AUTO.value)
        self.var_preset = ctk.StringVar(value=MeshPreset.STANDARD.value)
        self.var_mesh_conv = ctk.BooleanVar(value=False)
        self.var_mesh_conv_tol = ctk.StringVar(value="1.0")
        rec_hw = recommend_resources(hardware=self._hw)
        self.var_cores = ctk.StringVar(value=str(rec_hw.recommended_cores))
        self.var_ram_limit_gb = ctk.StringVar(value=f"{rec_hw.recommended_ram_limit_gb:.1f}")
        self.var_iters = ctk.StringVar(value=str(DEFAULT_SOLVER_ITERATIONS))

        self.var_mode = ctk.StringVar(value=DomainMode.FLUID_DOMAIN.value)
        self.var_shape = ctk.StringVar(value=EnclosureShape.BULLET.value)
        self.var_blunt_mm = ctk.StringVar(value="1.0")
        self.var_schema = ctk.StringVar(value="AP214")
        self.var_bois = ctk.BooleanVar(value=True)
        self.var_fillets = ctk.BooleanVar(value=False)
        self.var_rail_btns = ctk.BooleanVar(value=False)
        self.var_lugs = ctk.BooleanVar(value=False)

        self.var_export_preset = ctk.StringVar(value="Full CFD Package")
        self.var_step = ctk.BooleanVar(value=True)
        self.var_rocket_step = ctk.BooleanVar(value=True)
        self.var_stl = ctk.BooleanVar(value=False)
        self.var_verify = ctk.BooleanVar(value=True)
        self.var_scripts = ctk.BooleanVar(value=True)
        self.var_wbpj = ctk.BooleanVar(value=self._ansys_install is not None)
        self.var_wbpz = ctk.BooleanVar(value=False)
        self.var_auto_cfd = ctk.BooleanVar(value=False)
        self.var_result_mach = ctk.StringVar(value="—")
        self.var_dark = ctk.BooleanVar(value=True)

        self.var_opt_target = ctk.StringVar(value="Booster Stage Fins")
        self._opt_target_label_to_key: dict[str, str] = {
            "Booster Stage Fins": "booster",
            "Sustainer (Stage 1)": "sustainer",
        }
        self.var_opt_max_sims = ctk.StringVar(value="10")
        self.var_opt_min_sm = ctk.StringVar(value="2.0")
        self.var_opt_verify = ctk.BooleanVar(value=True)
        self.var_opt_dry_run = ctk.BooleanVar(value=False)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=475)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header bar
        header = ctk.CTkFrame(self, height=44, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="  ork2cfd — OpenRocket (.ork) to ANSYS Fluent Package Builder",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=8, sticky="w")

        ctk.CTkSwitch(
            header, text="Dark Mode", variable=self.var_dark, command=self._toggle_theme
        ).grid(row=0, column=1, padx=12, pady=8, sticky="e")

        # Left column: scrollable settings
        left = ctk.CTkScrollableFrame(self, width=465)
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_columnconfigure(0, weight=1)

        self._build_input_card(left, row=0)
        self._build_physics_card(left, row=1)
        self._build_cad_card(left, row=2)
        self._build_deliverables_card(left, row=3)
        self._build_fin_opt_card(left, row=4)
        self._build_next_steps_card(left, row=5)

        # Right column: Tabbed Previews / Convergence Plots + CFD Results Card + Log + Actions
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.viz_tabs = ctk.CTkTabview(right, corner_radius=8)
        self.viz_tabs.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        tab_prev = self.viz_tabs.add("Rocket & Domain Preview")
        tab_opt = self.viz_tabs.add("Fin Optimization Plots")
        tab_prev.grid_rowconfigure(0, weight=1)
        tab_prev.grid_columnconfigure(0, weight=1)
        tab_opt.grid_rowconfigure(1, weight=1)
        tab_opt.grid_columnconfigure(0, weight=1)

        self.preview = PreviewCanvas(tab_prev, dark=True)
        self.preview.grid(row=0, column=0, sticky="nsew")

        opt_bar = ctk.CTkFrame(tab_opt, fg_color="transparent")
        opt_bar.grid(row=0, column=0, padx=6, pady=(2, 4), sticky="ew")
        opt_bar.grid_columnconfigure(0, weight=1)
        self.lbl_opt_headline = ctk.CTkLabel(
            opt_bar,
            text="No fin optimization run yet.",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color="#94a3b8",
            anchor="w",
        )
        self.lbl_opt_headline.grid(row=0, column=0, sticky="ew")
        self.btn_opt_plot_png = ctk.CTkButton(
            opt_bar,
            text="Open PNG",
            width=78,
            height=24,
            state="disabled",
            fg_color="#0369a1",
            command=self._on_open_opt_plot_png,
        )
        self.btn_opt_plot_png.grid(row=0, column=1, padx=2)
        self.btn_opt_csv = ctk.CTkButton(
            opt_bar,
            text="CSV",
            width=54,
            height=24,
            state="disabled",
            fg_color="#334155",
            command=self._on_open_opt_csv,
        )
        self.btn_opt_csv.grid(row=0, column=2, padx=2)
        self.btn_opt_ork = ctk.CTkButton(
            opt_bar,
            text="Opt .ork",
            width=70,
            height=24,
            state="disabled",
            fg_color="#15803d",
            command=self._on_open_opt_ork,
        )
        self.btn_opt_ork.grid(row=0, column=3, padx=2)
        self.btn_opt_images = ctk.CTkButton(
            opt_bar,
            text="CFD Images",
            width=86,
            height=24,
            state="disabled",
            fg_color="#4338ca",
            command=self._on_open_opt_images,
        )
        self.btn_opt_images.grid(row=0, column=4, padx=(2, 0))

        self.opt_canvas = OptimizationPlotCanvas(tab_opt, dark=True)
        self.opt_canvas.grid(row=1, column=0, sticky="nsew")

        self._build_cfd_results_card(right, row=1)

        right.grid_rowconfigure(2, weight=2)
        bottom = ctk.CTkFrame(right, corner_radius=8)
        bottom.grid(row=2, column=0, sticky="nsew")
        bottom.grid_rowconfigure(3, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_row.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=2)
        btn_row.grid_columnconfigure((1, 2, 3, 4), weight=1)

        self.btn_generate = ctk.CTkButton(
            btn_row,
            text="Generate CFD Package",
            height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_generate,
        )
        self.btn_generate.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.btn_optimize_fins = ctk.CTkButton(
            btn_row,
            text="Optimize Fins",
            height=36,
            state="disabled",
            fg_color="#7c3aed",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_optimize_fins,
        )
        self.btn_optimize_fins.grid(row=0, column=1, padx=4, sticky="ew")

        self.btn_view_cad = ctk.CTkButton(
            btn_row,
            text="View 3D CAD",
            height=36,
            state="disabled",
            fg_color="#2563eb",
            command=self._on_view_cad,
        )
        self.btn_view_cad.grid(row=0, column=2, padx=4, sticky="ew")

        self.btn_open_dir = ctk.CTkButton(
            btn_row,
            text="Open Output Folder",
            height=36,
            state="disabled",
            fg_color="#334155",
            command=self._on_open_folder,
        )
        self.btn_open_dir.grid(row=0, column=3, padx=4, sticky="ew")

        self.btn_run_fluent = ctk.CTkButton(
            btn_row,
            text="Run in Fluent",
            height=36,
            state="disabled",
            fg_color="#0f766e",
            command=self._on_run_fluent,
        )
        self.btn_run_fluent.grid(row=0, column=4, padx=(6, 0), sticky="ew")

        self.status_card = ctk.CTkFrame(
            bottom,
            corner_radius=8,
            border_width=1,
            border_color=("#cbd5e1", "#334155"),
            fg_color=("#f1f5f9", "#0f172a"),
        )
        self.status_card.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        self.status_card.grid_columnconfigure(0, weight=1)

        status_hdr = ctk.CTkFrame(self.status_card, fg_color="transparent")
        status_hdr.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        status_hdr.grid_columnconfigure(1, weight=1)

        self.lbl_status_badge = ctk.CTkLabel(
            status_hdr,
            text="READY",
            width=124,
            height=22,
            corner_radius=6,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#334155",
            text_color="#f8fafc",
        )
        self.lbl_status_badge.grid(row=0, column=0, padx=(0, 8), sticky="w")

        self.lbl_status_headline = ctk.CTkLabel(
            status_hdr,
            text=self._last_snapshot.headline,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )
        self.lbl_status_headline.grid(row=0, column=1, sticky="ew")

        self.lbl_progress_meta = ctk.CTkLabel(
            status_hdr,
            text="0%",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            anchor="e",
            text_color="#94a3b8",
        )
        self.lbl_progress_meta.grid(row=0, column=2, padx=(8, 0), sticky="e")

        self.progress = ctk.CTkProgressBar(
            self.status_card,
            height=13,
            corner_radius=6,
            progress_color="#3b82f6",
        )
        self.progress.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        self.progress.set(0.0)

        self.stage_row = ctk.CTkFrame(self.status_card, fg_color="transparent")
        self.stage_row.grid(row=2, column=0, padx=10, pady=(0, 4), sticky="ew")

        self.lbl_status_detail = ctk.CTkLabel(
            self.status_card,
            text=self._last_snapshot.detail,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#94a3b8",
            anchor="w",
            justify="left",
        )
        self.lbl_status_detail.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="ew")

        log_hdr = ctk.CTkFrame(bottom, fg_color="transparent")
        log_hdr.grid(row=2, column=0, padx=10, pady=(0, 3), sticky="ew")
        log_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            log_hdr,
            text="Activity Log",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#475569", "#94a3b8"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.btn_clear_log = ctk.CTkButton(
            log_hdr,
            text="Clear Log",
            width=66,
            height=20,
            corner_radius=5,
            font=ctk.CTkFont(size=10),
            fg_color=("#e2e8f0", "#1e293b"),
            hover_color=("#cbd5e1", "#334155"),
            text_color=("#334155", "#cbd5e1"),
            command=self._clear_log,
        )
        self.btn_clear_log.grid(row=0, column=1, sticky="e")

        self.log_box = ctk.CTkTextbox(
            bottom, height=96, font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.log_box.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.log_box.configure(state="disabled")

        self._configure_log_tags()
        self._apply_progress_snapshot(self._last_snapshot)
        self._append_log("Ready. Choose an OpenRocket (.ork) file to begin.")

    def _build_input_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "1. Input OpenRocket (.ork) & Rocket Summary",
            help_key="sec_input",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        pick = ctk.CTkFrame(card.body, fg_color="transparent")
        pick.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        pick.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(
            pick, textvariable=self.var_ork_path, placeholder_text="Select .ork file..."
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(pick, text="Browse", width=80, command=self._browse_ork).grid(
            row=0, column=1
        )

        self.lbl_rocket_summary = ctk.CTkLabel(
            card.body,
            text="No file loaded.",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
        )
        self.lbl_rocket_summary.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.cmb_stage = add_labeled_combo(
            card.body,
            2,
            0,
            "Stage Configuration",
            ["All Stages"],
            self.var_stage,
            self._on_stage_change,
            help_key="stage_selection",
            icon_registry=self._info_icons,
        )
        self.cmb_stage.master.grid_configure(columnspan=2)

    def _build_physics_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "2. Flight Condition & Boundary Layer (y+)",
            help_key="sec_physics",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        add_labeled_entry(
            card.body, 0, 0, "Mach Number", self.var_mach,
            self._refresh_calculations, help_key="mach", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 0, 1, "Altitude (m)", self.var_alt,
            self._refresh_calculations, help_key="altitude", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 1, 0, "AoA (deg)", self.var_aoa,
            self._refresh_calculations, help_key="aoa", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 1, 1, "Target y+", self.var_yplus,
            self._refresh_calculations, help_key="yplus", icon_registry=self._info_icons,
        )
        add_labeled_combo(
            card.body, 2, 0, "Flow Regime", [r.value for r in FlowRegime],
            self.var_regime, self._refresh_calculations,
            help_key="regime", icon_registry=self._info_icons,
        )
        add_labeled_combo(
            card.body, 2, 1, "Mesh Preset", [p.value for p in MeshPreset],
            self.var_preset, self._refresh_calculations,
            help_key="mesh_preset", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 3, 0, "CPU Cores (-tN)", self.var_cores,
            self._refresh_calculations, help_key="cores", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 3, 1, "RAM Limit (GB)", self.var_ram_limit_gb,
            self._refresh_calculations, help_key="ram_limit", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 4, 0, "Solver Iterations", self.var_iters,
            self._refresh_calculations, help_key="iterations", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 4, 1, "Cd Conv. Tol (%)", self.var_mesh_conv_tol,
            self._refresh_calculations, help_key="mesh_conv_tol",
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 5, 0, "Auto Mesh Convergence", self.var_mesh_conv,
            help_key="mesh_convergence", command=self._on_mesh_conv_toggle,
            icon_registry=self._info_icons,
        )
        self.btn_apply_ram_rec = ctk.CTkButton(
            card.body,
            text="Apply RAM-Safe Rec.",
            height=26,
            fg_color="#0f766e",
            hover_color="#115e59",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_apply_recommended_settings,
        )
        self.btn_apply_ram_rec.grid(row=5, column=1, padx=4, pady=(2, 2), sticky="ew")

        self.lbl_hw_summary = ctk.CTkLabel(
            card.body,
            text="",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#4ade80",
        )
        self.lbl_hw_summary.grid(row=6, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        self.lbl_physics_summary = ctk.CTkLabel(
            card.body,
            text="Load an .ork file to compute Re, Δy1, and prism layers.",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#38bdf8",
        )
        self.lbl_physics_summary.grid(row=7, column=0, columnspan=2, pady=(4, 0), sticky="ew")
        self._update_hw_label(recommend_resources(hardware=self._hw), int(self.var_cores.get()))

    def _build_cad_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "3. CFD Geometry & Domain Settings",
            help_key="sec_cad",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        add_labeled_combo(
            card.body, 0, 0, "Domain Mode", [m.value for m in DomainMode],
            self.var_mode, self._refresh_calculations,
            help_key="domain_mode", icon_registry=self._info_icons,
        )
        add_labeled_combo(
            card.body, 0, 1, "Enclosure Shape", [s.value for s in EnclosureShape],
            self.var_shape, self._refresh_calculations,
            help_key="enclosure_shape", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 1, 0, "Nose Blunt Radius (mm)", self.var_blunt_mm,
            help_key="blunt_radius", icon_registry=self._info_icons,
        )
        add_labeled_combo(
            card.body, 1, 1, "STEP Schema", ["AP214", "AP242"], self.var_schema,
            help_key="step_schema", icon_registry=self._info_icons,
        )

        chk_frame = ctk.CTkFrame(card.body, fg_color="transparent")
        chk_frame.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="ew")
        chk_frame.grid_columnconfigure((0, 1), weight=1)
        add_labeled_checkbox(
            chk_frame, 0, 0, "Include BOIs", self.var_bois,
            help_key="include_bois", command=self._refresh_calculations,
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            chk_frame, 0, 1, "Fin Root Fillets", self.var_fillets,
            help_key="fin_fillets", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            chk_frame, 1, 0, "Include Rail Buttons", self.var_rail_btns,
            help_key="rail_buttons", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            chk_frame, 1, 1, "Include Launch Lugs", self.var_lugs,
            help_key="launch_lugs", icon_registry=self._info_icons,
        )

    def _build_deliverables_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "4. Export Options & Output Folder",
            help_key="sec_deliverables",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        out_row = ctk.CTkFrame(card.body, fg_color="transparent")
        out_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        out_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(
            out_row, textvariable=self.var_out_dir, placeholder_text="Output directory..."
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            out_row, text="Folder", width=70, command=self._browse_out_dir
        ).grid(row=0, column=1)

        ansys_text = (
            f"Detected: {self._ansys_install.label}"
            if self._ansys_install
            else "ANSYS not detected locally (scripts will still be generated)"
        )
        ansys_col = "#4ade80" if self._ansys_install else "#fbbf24"
        ctk.CTkLabel(
            card.body, text=ansys_text, font=ctk.CTkFont(size=11), text_color=ansys_col, anchor="w"
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))

        mode_row = ctk.CTkFrame(card.body, fg_color="transparent")
        mode_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        mode_row.grid_columnconfigure(1, weight=1)
        mode_hdr = ctk.CTkFrame(mode_row, fg_color="transparent")
        mode_hdr.grid(row=0, column=0, sticky="w", padx=(4, 8))
        ctk.CTkLabel(
            mode_hdr, text="Export Mode", font=ctk.CTkFont(size=11, weight="bold"), anchor="w"
        ).pack(side="left")
        preset_icon = InfoIcon(mode_hdr, help_key="export_preset")
        preset_icon.pack(side="left", padx=(5, 0))
        self._info_icons.append(preset_icon)

        self.seg_export_preset = ctk.CTkSegmentedButton(
            mode_row,
            values=["Full CFD Package", "Rocket .step Only"],
            variable=self.var_export_preset,
            command=self._on_export_preset_change,
        )
        self.seg_export_preset.grid(row=0, column=1, sticky="ew", padx=4)

        add_labeled_checkbox(
            card.body, 3, 0, "Rocket-Only .step (No Enclosure)", self.var_rocket_step,
            help_key="deliv_rocket_step", command=self._sync_generate_button_label,
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 3, 1, "CFD Enclosure .step + JSON", self.var_step,
            help_key="deliv_step", command=self._sync_generate_button_label,
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 4, 0, "ANSYS Scripts (.py/.jou)", self.var_scripts,
            help_key="deliv_scripts", command=self._sync_generate_button_label,
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 4, 1, "Build Workbench .wbpj", self.var_wbpj,
            help_key="deliv_wbpj", command=self._sync_generate_button_label,
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 5, 0, "Archive .wbpz", self.var_wbpz,
            help_key="deliv_wbpz", command=self._sync_generate_button_label,
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 5, 1, "Export Multi-Region .stl", self.var_stl,
            help_key="deliv_stl", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 6, 0, "CAD Viewer & Verify (.html/.png)", self.var_verify,
            help_key="deliv_verify", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 6, 1, "Auto-Run Fluent & Compute Cd", self.var_auto_cfd,
            help_key="deliv_auto_cfd", command=self._sync_generate_button_label,
            icon_registry=self._info_icons,
        )

    def _on_export_preset_change(self, value: str) -> None:
        self.var_export_preset.set(value)
        if value == "Rocket .step Only":
            self.var_rocket_step.set(True)
            self.var_verify.set(True)
            self.var_step.set(False)
            self.var_scripts.set(False)
            self.var_wbpj.set(False)
            self.var_wbpz.set(False)
            self.var_stl.set(False)
            self.var_auto_cfd.set(False)
        else:
            self.var_step.set(True)
            self.var_rocket_step.set(True)
            self.var_scripts.set(True)
            self.var_wbpj.set(self._ansys_install is not None)
            self.var_verify.set(True)
        self._sync_generate_button_label()

    def _on_mesh_conv_toggle(self) -> None:
        self._refresh_calculations()
        self._sync_generate_button_label()

    def _sync_generate_button_label(self) -> None:
        if self._is_running:
            return
        rocket_only = self.var_rocket_step.get() and not (
            self.var_step.get()
            or self.var_scripts.get()
            or self.var_wbpj.get()
            or self.var_wbpz.get()
            or self.var_auto_cfd.get()
        )
        if rocket_only:
            lbl = "Export Rocket .step"
        elif self.var_auto_cfd.get() and self.var_mesh_conv.get():
            lbl = "Generate & Converge Mesh"
        elif self.var_auto_cfd.get():
            lbl = "Generate & Solve CFD"
        else:
            lbl = "Generate CFD Package"
        self.btn_generate.configure(text=lbl)

    def _build_cfd_results_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "CFD Drag (Cd) Results & Direct Visualization",
            help_key="sec_cfd_results",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        card.body.grid_columnconfigure(0, weight=1)
        card.body.grid_columnconfigure(1, weight=0)

        top_row = ctk.CTkFrame(card.body, fg_color="transparent")
        top_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_row.grid_columnconfigure(0, weight=1)

        self.lbl_cd_headline = ctk.CTkLabel(
            top_row,
            text="Cd = —  |  Drag = —  |  Mach = —  |  Cells = —  |  y+ = —",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#94a3b8",
            anchor="w",
        )
        self.lbl_cd_headline.grid(row=0, column=0, sticky="ew")

        self.cmb_result_mach = ctk.CTkOptionMenu(
            top_row,
            values=["—"],
            variable=self.var_result_mach,
            width=110,
            height=24,
            state="disabled",
            command=self._on_result_mach_change,
        )
        self.cmb_result_mach.grid(row=0, column=1, padx=(8, 0), sticky="e")

        self.lbl_cd_zones = ctk.CTkLabel(
            card.body,
            text=(
                "Enable 'Auto-Run Fluent & Compute Cd' or click "
                "'Run CFD & Compute Cd' after generating files."
            ),
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#94a3b8",
            anchor="w",
            justify="left",
        )
        self.lbl_cd_zones.grid(row=1, column=0, columnspan=2, pady=(4, 6), sticky="ew")

        viz_row = ctk.CTkFrame(card.body, fg_color="transparent")
        viz_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        viz_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_solve_cfd = ctk.CTkButton(
            viz_row,
            text="Run CFD & Compute Cd",
            height=32,
            state="disabled",
            fg_color="#0f766e",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_solve_cfd,
        )
        self.btn_solve_cfd.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.btn_viz_fluent = ctk.CTkButton(
            viz_row,
            text="Open in Fluent (Viz)",
            height=32,
            state="disabled",
            fg_color="#4338ca",
            command=self._on_viz_fluent,
        )
        self.btn_viz_fluent.grid(row=0, column=1, padx=4, sticky="ew")

        self.btn_viz_cfdpost = ctk.CTkButton(
            viz_row,
            text="Open in CFD-Post",
            height=32,
            state="disabled",
            fg_color="#7c3aed",
            command=self._on_viz_cfdpost,
        )
        self.btn_viz_cfdpost.grid(row=0, column=2, padx=4, sticky="ew")

        self.btn_viz_contours = ctk.CTkButton(
            viz_row,
            text="View Contours",
            height=32,
            state="disabled",
            fg_color="#0369a1",
            command=self._on_viz_contours,
        )
        self.btn_viz_contours.grid(row=0, column=3, padx=(4, 0), sticky="ew")

    def _build_fin_opt_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "5. Fin Shape Optimization (Bayesian GP-EI)",
            help_key="sec_fin_opt",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        self.cmb_opt_target = add_labeled_combo(
            card.body,
            0,
            0,
            "Target Fin Stage",
            ["Booster Stage Fins", "Sustainer (Stage 1)"],
            self.var_opt_target,
            help_key="opt_target_stage",
            icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body,
            0,
            1,
            "Max Sims",
            self.var_opt_max_sims,
            help_key="opt_max_sims",
            icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body,
            1,
            0,
            "Min Static Margin (cal)",
            self.var_opt_min_sm,
            help_key="opt_min_sm",
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body,
            1,
            1,
            "Final High-Res Verify",
            self.var_opt_verify,
            help_key="opt_verify_final",
            icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body,
            2,
            0,
            "Fast Dry-Run (Analytic Proxy)",
            self.var_opt_dry_run,
            help_key="opt_dry_run",
            icon_registry=self._info_icons,
        )

        ctk.CTkLabel(
            card.body,
            text=(
                "Optimizes root/tip chord, span & sweep while keeping the .ork "
                "cross-section and thickness locked."
            ),
            justify="left",
            anchor="w",
            wraplength=410,
            font=ctk.CTkFont(size=11),
            text_color=("#475569", "#94a3b8"),
        ).grid(row=3, column=0, columnspan=2, padx=4, pady=(2, 6), sticky="ew")

        self.btn_optimize_fins_card = ctk.CTkButton(
            card.body,
            text="Run Fin Shape Optimization",
            height=30,
            state="disabled",
            fg_color="#7c3aed",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_optimize_fins,
        )
        self.btn_optimize_fins_card.grid(row=4, column=0, columnspan=2, pady=(2, 0), sticky="ew")

    def _build_next_steps_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "6. Next Steps — View Results in ANSYS",
            help_key="sec_next_steps",
            icon_registry=self._info_icons,
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        initial_text = build_ansys_next_steps_summary(
            stem="<rocket>",
            domain_mode=DomainMode.FLUID_DOMAIN,
            sym_multiplier=1.0,
            mach=0.8,
            aoa_deg=0.0,
            target_yplus=1.0,
            wbpj_filename="<rocket>_cfd.wbpj" if self._ansys_install else None,
        )
        self.lbl_next_steps = ctk.CTkLabel(
            card.body,
            text=initial_text,
            justify="left",
            anchor="w",
            wraplength=415,
            font=ctk.CTkFont(size=11),
            text_color="#cbd5e1",
        )
        self.lbl_next_steps.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.btn_ansys_guide = ctk.CTkButton(
            card.body,
            text="Open Full ANSYS Results Guide",
            height=30,
            fg_color="#1e40af",
            command=self._on_open_ansys_guide,
        )
        self.btn_ansys_guide.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew")

    def _refresh_next_steps(self) -> None:
        cfg = self._collect_config()
        active = self._active_rocket()
        stem = sanitize_stem(active.name) if active is not None else "<rocket>"
        stage_key = self._selected_stage_key()
        if (
            stage_key not in {"", "all"}
            and self._rocket is not None
            and self._rocket.is_multistage
        ):
            stem = f"{stem}_{sanitize_stem(stage_key)}"
        sym_mult = (
            _symmetry_multiplier(active, cfg.domain.mode)
            if active is not None
            else (
                2.0
                if cfg.domain.mode == DomainMode.SYMMETRY_HALF
                else 4.0
                if cfg.domain.mode == DomainMode.SYMMETRY_SECTOR
                else 8.0
                if cfg.domain.mode == DomainMode.SYMMETRY_HALF_SECTOR
                else 1.0
            )
        )
        wbpj_name = (
            self._last_result.wbpj_path.name
            if (self._last_result is not None and self._last_result.wbpj_path is not None)
            else (f"{stem}_cfd.wbpj" if cfg.deliverables.build_wbpj else None)
        )
        summary = build_ansys_next_steps_summary(
            stem=stem,
            domain_mode=cfg.domain.mode,
            sym_multiplier=sym_mult,
            mach=cfg.flight.mach,
            aoa_deg=cfg.flight.aoa_deg,
            target_yplus=cfg.flight.target_yplus,
            wbpj_filename=wbpj_name,
        )
        self.lbl_next_steps.configure(
            text=summary,
            text_color=("#cbd5e1" if self.var_dark.get() else "#1e293b"),
        )

    def _toggle_theme(self) -> None:
        dark = self.var_dark.get()
        ctk.set_appearance_mode("dark" if dark else "light")
        self.preview.set_theme(dark)
        self.opt_canvas.set_theme(dark)
        self.lbl_next_steps.configure(
            text_color=("#cbd5e1" if dark else "#1e293b"),
        )
        self._configure_log_tags()
        self._apply_progress_snapshot(self._last_snapshot)
        self._display_cfd_result(self._last_cfd_result)

    def destroy(self) -> None:
        self._stop_activity_ticker()
        super().destroy()

    def _browse_ork(self) -> None:
        path = filedialog.askopenfilename(
            title="Select OpenRocket File",
            filetypes=[("OpenRocket files", "*.ork *.ork.zip *.xml"), ("All files", "*.*")],
        )
        if path:
            self._load_ork(Path(path))

    def _browse_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Select CFD Package Output Folder")
        if path:
            self.var_out_dir.set(path)
            self._refresh_cfd_results_from_dir(Path(path))

    def _selected_stage_key(self) -> str:
        return self._stage_label_to_key.get(self.var_stage.get(), "all")

    def _active_rocket(self) -> RocketModel | None:
        if self._rocket is None:
            return None
        return self._rocket.select_stages(self._selected_stage_key())

    def _on_stage_change(self, _value: str | None = None) -> None:
        self._update_rocket_summary()
        self._refresh_calculations()

    def _update_rocket_summary(self) -> None:
        active = self._active_rocket()
        if self._rocket is None or active is None:
            return
        n_sym = active.rotational_symmetry_order
        if active.is_multistage:
            stage_desc = ", ".join(
                f"S{idx}:{name}({'+'.join(str(c) for c in fcs) or '0'} fins)"
                for idx, name, _length, fcs in active.stage_summaries
            )
            stage_hdr = f"{active.stage_count} Stages [{stage_desc}]"
        else:
            stage_hdr = f"1 Stage ({active.stages[0].name})" if active.stages else "0 Stages"
        fins_txt = ", ".join(
            f"{f.fin_count}x {f.fin_type} ({f.cross_section.value})" for f in active.all_fins
        ) or "None"
        summary = (
            f"Name: {active.name} | {stage_hdr} | Components: {len(active.all_components)}\n"
            f"L = {active.total_length:.3f} m | D_max = {active.max_diameter*1000:.1f} mm | "
            f"A_ref = {active.reference_area:.5f} m²\n"
            f"Symmetry: {n_sym}-fold ({360.0 / n_sym:.0f}° sector / "
            f"{180.0 / n_sym:.0f}° half) | Fins: {fins_txt}"
        )
        self.lbl_rocket_summary.configure(
            text=summary,
            text_color=("#e2e8f0" if self.var_dark.get() else "#0f172a"),
        )

    def _load_ork(self, ork_path: Path) -> None:
        try:
            rocket = parse_ork_file(ork_path)
        except Exception as exc:
            self._append_log(f"[ERROR] Failed to parse {ork_path.name}: {exc}")
            return

        self._rocket = rocket
        self.var_ork_path.set(str(ork_path))
        stem = sanitize_stem(rocket.name)
        if not self.var_out_dir.get():
            self.var_out_dir.set(str(ork_path.parent / f"{stem}_cfd"))

        label_map: dict[str, str] = {}
        if rocket.is_multistage:
            top_lbl = f"All Stages ({rocket.stage_count}-Stage Stack)"
            label_map[top_lbl] = "all"
            for idx, sname, slen, fcs in rocket.stage_summaries:
                fin_str = "+".join(str(c) for c in fcs) if fcs else "0"
                st_lbl = f"Stage {idx}: {sname} ({slen:.2f}m, {fin_str} fins)"
                label_map[st_lbl] = f"stage_{idx}"
            for k in range(2, rocket.stage_count):
                label_map[f"Stages 1..{k}"] = f"up_to_{k}"
        else:
            label_map["All Stages (Single Stage)"] = "all"

        self._stage_label_to_key = label_map
        labels = list(label_map.keys())
        self.cmb_stage.configure(values=labels)
        self.var_stage.set(labels[0])

        fin_stages = [(idx, sname) for idx, sname, _slen, fcs in rocket.stage_summaries if fcs]
        opt_map: dict[str, str] = {}
        if rocket.is_multistage and fin_stages:
            last_idx = fin_stages[-1][0]
            for idx, sname in reversed(fin_stages):
                low = sname.lower()
                if idx == last_idx and "booster" not in low:
                    lbl = f"Stage {idx}: {sname} (Booster)"
                elif idx == 1 and "sustainer" not in low:
                    lbl = f"Stage {idx}: {sname} (Sustainer)"
                else:
                    lbl = f"Stage {idx}: {sname}"
                opt_map[lbl] = f"stage_{idx}"
        elif fin_stages:
            idx0, sname0 = fin_stages[0]
            opt_map[f"Stage {idx0}: {sname0}"] = f"stage_{idx0}"
        else:
            sname0 = rocket.stages[0].name if rocket.stages else "Main"
            opt_map[f"Stage 1: {sname0}"] = "stage_1"
        self._opt_target_label_to_key = opt_map
        opt_labels = list(opt_map.keys())
        self.cmb_opt_target.configure(values=opt_labels)
        self.var_opt_target.set(opt_labels[0])
        if not self._is_running:
            self.btn_optimize_fins.configure(state="normal")
            self.btn_optimize_fins_card.configure(state="normal")

        self._update_rocket_summary()
        self._append_log(
            f"[OK] Loaded {ork_path.name} ({rocket.name}, "
            f"stages={rocket.stage_count}, sym={rocket.rotational_symmetry_order}-fold, "
            f"L={rocket.total_length:.3f}m)."
        )
        for w in rocket.warnings:
            self._append_log(f"[WARN] {w}")
        self._refresh_calculations()
        out_str = self.var_out_dir.get().strip()
        if out_str and Path(out_str).exists():
            self._refresh_cfd_results_from_dir(Path(out_str))

    def _collect_config(self) -> PipelineConfig:
        def _flt(var: ctk.StringVar, default: float) -> float:
            try:
                return float(var.get().strip())
            except ValueError:
                return default

        def _int(var: ctk.StringVar, default: int) -> int:
            try:
                return max(1, int(float(var.get().strip())))
            except ValueError:
                return default

        default_hw_rec = recommend_resources(hardware=self._hw)
        ram_limit = max(1.5, _flt(self.var_ram_limit_gb, default_hw_rec.recommended_ram_limit_gb))

        cad = CadConfig(
            tip_blunt_radius_m=max(0.0, _flt(self.var_blunt_mm, 1.0) / 1000.0),
            apply_fin_fillets=self.var_fillets.get(),
            include_rail_buttons=self.var_rail_btns.get(),
            include_launch_lugs=self.var_lugs.get(),
            step_schema=self.var_schema.get(),
        )
        domain = DomainConfig(
            mode=DomainMode(self.var_mode.get()),
            shape=EnclosureShape(self.var_shape.get()),
            boi=BoiConfig(enabled=self.var_bois.get()),
        )
        flight = FlightCondition(
            mach=max(0.01, _flt(self.var_mach, 0.8)),
            altitude_m=_flt(self.var_alt, 500.0),
            aoa_deg=_flt(self.var_aoa, 0.0),
            target_yplus=max(0.1, _flt(self.var_yplus, 1.0)),
            regime_override=FlowRegime(self.var_regime.get()),
            mesh_preset=MeshPreset(self.var_preset.get()),
            cores=_int(self.var_cores, default_hw_rec.recommended_cores),
            iterations=_int(self.var_iters, DEFAULT_SOLVER_ITERATIONS),
            ram_limit_gb=ram_limit,
        )
        active = self._active_rocket()
        solver_type = recommend_solver(active, flight).solver_type if active else "pressure-based"
        res_rec = recommend_resources(
            hardware=self._hw,
            rocket=active,
            mach=flight.mach,
            aoa_deg=flight.aoa_deg,
            solver_type=solver_type,
            requested_cores=flight.cores,
            ram_limit_gb=ram_limit,
        )
        deliverables = DeliverableConfig(
            export_step=self.var_step.get(),
            export_rocket_step=self.var_rocket_step.get(),
            export_stl=self.var_stl.get(),
            export_verification=self.var_verify.get(),
            generate_scripts=self.var_scripts.get(),
            build_wbpj=self.var_wbpj.get(),
            archive_wbpz=self.var_wbpz.get(),
            auto_run_cfd=self.var_auto_cfd.get(),
        )
        mesh_convergence = MeshConvergenceConfig(
            enabled=self.var_mesh_conv.get(),
            cd_rel_tol=max(0.05, _flt(self.var_mesh_conv_tol, 1.0)) / 100.0,
            max_cells=res_rec.max_safe_cells,
        )
        return PipelineConfig(
            cad=cad,
            domain=domain,
            flight=flight,
            deliverables=deliverables,
            mesh_convergence=mesh_convergence,
            stage_selection=self._selected_stage_key(),
        )

    def _update_hw_label(self, res_rec: ResourceRecommendation, active_cores: int) -> None:
        hw = res_rec.hardware
        txt = (
            f"HW: {hw.total_ram_gb:.1f} GB RAM ({hw.available_ram_gb:.1f} GB free) | "
            f"{hw.physical_cores} Cores / {hw.logical_threads} Threads\n"
            f"Allotted: {active_cores} cores, ≤ {res_rec.allotted_ram_gb:.1f} GB | "
            f"Safe Cap ≈ {res_rec.max_safe_cells:,} cells | "
            f"Est. Peak ≈ {res_rec.estimated_ram_gb:.1f} GB"
        )
        low_free = hw.available_ram_gb < res_rec.allotted_ram_gb * 0.65
        if res_rec.exceeds_ram_budget:
            col = "#f87171"
        elif active_cores > hw.physical_cores or low_free:
            col = "#fbbf24"
        else:
            col = "#4ade80"
        self.lbl_hw_summary.configure(text=txt, text_color=col)

    def _on_apply_recommended_settings(self) -> None:
        self._hw = detect_system_hardware()
        active = self._active_rocket()
        cfg = self._collect_config()
        solver_type = (
            recommend_solver(active, cfg.flight).solver_type if active else "pressure-based"
        )
        est_cells = (
            compute_boundary_layer(active, cfg.flight, cfg.domain.mode).estimated_cell_count
            if active is not None
            else 250_000
        )
        rec = recommend_resources(
            hardware=self._hw,
            rocket=active,
            mach=cfg.flight.mach,
            aoa_deg=cfg.flight.aoa_deg,
            solver_type=solver_type,
            estimated_cells=est_cells,
        )
        self.var_cores.set(str(rec.recommended_cores))
        self.var_ram_limit_gb.set(f"{rec.recommended_ram_limit_gb:.1f}")
        self.var_preset.set(rec.recommended_preset.value)
        self.var_yplus.set(f"{rec.recommended_yplus:g}")
        if active is not None:
            self.var_mode.set(rec.recommended_domain_mode.value)
        self._refresh_calculations()
        self._append_log(
            f"[OK] Applied RAM-recommended settings: cores={rec.recommended_cores}, "
            f"RAM<={rec.recommended_ram_limit_gb:.1f}GB, preset={rec.recommended_preset.value}, "
            f"y+={rec.recommended_yplus:g}, mode={rec.recommended_domain_mode.value} "
            f"(safe cap ~{rec.max_safe_cells:,} cells)."
        )

    def _refresh_calculations(self) -> None:
        self._hw = detect_system_hardware()
        active = self._active_rocket()
        if active is None:
            rec_idle = recommend_resources(hardware=self._hw)
            try:
                c_val = max(1, int(float(self.var_cores.get().strip())))
            except ValueError:
                c_val = rec_idle.recommended_cores
            self._update_hw_label(rec_idle, c_val)
            return
        cfg = self._collect_config()
        bl = compute_boundary_layer(active, cfg.flight, cfg.domain.mode)
        rec = recommend_solver(active, cfg.flight)
        res_rec = recommend_resources(
            hardware=self._hw,
            rocket=active,
            mach=cfg.flight.mach,
            aoa_deg=cfg.flight.aoa_deg,
            solver_type=rec.solver_type,
            estimated_cells=bl.estimated_cell_count,
            requested_cores=cfg.flight.cores,
            ram_limit_gb=cfg.flight.ram_limit_gb,
        )
        self._update_hw_label(res_rec, cfg.flight.cores)

        student_tag = " [>512k Student Limit!]" if bl.exceeds_student_limit else ""
        conv_line = (
            f"\nMesh Study: ON (up to {cfg.mesh_convergence.max_levels} levels, "
            f"r={cfg.mesh_convergence.refinement_factor:.2f}x, "
            f"stop |ΔCd|/Cd ≤ {cfg.mesh_convergence.cd_rel_tol * 100.0:.2f}%)"
            if cfg.mesh_convergence.enabled
            else ""
        )
        text = (
            f"Regime: {rec.regime.value.upper()} ({rec.solver_type}, {rec.fluid_model})\n"
            f"V={bl.velocity_m_s:.1f} m/s | Re_L={bl.reynolds_length:.2e} | "
            f"Δy1={bl.first_layer_thickness_m:.2e} m ({bl.prism_layers} layers)\n"
            f"Surf={bl.surface_min_size_m*1000:.1f}..{bl.surface_max_size_m*1000:.1f} mm | "
            f"Est.Cells≈{bl.estimated_cell_count:,}{student_tag}{conv_line}"
        )
        self.lbl_physics_summary.configure(
            text=text,
            text_color=("#fbbf24" if bl.exceeds_student_limit else "#38bdf8"),
        )
        self.preview.update_preview(active, cfg.domain, rec)
        self._refresh_next_steps()

    def _on_generate(self) -> None:
        if self._is_running:
            return
        ork_str = self.var_ork_path.get().strip()
        if not ork_str:
            self._append_log("[ERROR] Please select an .ork file first.")
            return
        out_str = self.var_out_dir.get().strip()
        if not out_str:
            out_str = str(Path(ork_str).parent / f"{sanitize_stem(Path(ork_str).stem)}_cfd")
            self.var_out_dir.set(out_str)

        cfg = self._collect_config()
        self._is_running = True
        self._run_start_mono = time.monotonic()
        self._last_elapsed_text = ""
        rocket_only = cfg.deliverables.export_rocket_step and not (
            cfg.deliverables.export_step
            or cfg.deliverables.generate_scripts
            or cfg.deliverables.build_wbpj
            or cfg.deliverables.archive_wbpz
            or cfg.deliverables.auto_run_cfd
        )
        mode = "package_and_cfd" if cfg.deliverables.auto_run_cfd else "package"
        self._tracker = WorkflowProgressTracker(
            mode=mode,
            rocket_only=rocket_only,
            include_verify=cfg.deliverables.export_verification,
            include_wbpj=cfg.deliverables.build_wbpj and self._ansys_install is not None,
            max_iters=cfg.flight.iterations,
            mesh_conv_levels=(
                cfg.mesh_convergence.max_levels if cfg.mesh_convergence.enabled else 1
            ),
        )
        gen_btn_text = (
            "Generating & Solving CFD..." if cfg.deliverables.auto_run_cfd else "Generating..."
        )
        self.btn_generate.configure(state="disabled", text=gen_btn_text)
        self.btn_optimize_fins.configure(state="disabled")
        self.btn_optimize_fins_card.configure(state="disabled")
        self.btn_solve_cfd.configure(state="disabled")
        self._apply_progress_snapshot(
            self._tracker.start_snapshot(
                headline=(
                    "Starting CAD generation & automated Fluent solve..."
                    if cfg.deliverables.auto_run_cfd
                    else "Starting CAD & CFD package generation..."
                ),
                detail=f"Output folder: {out_str}",
            )
        )
        self._start_activity_ticker()

        worker = threading.Thread(
            target=self._run_worker,
            args=(Path(ork_str), Path(out_str), cfg),
            daemon=True,
        )
        worker.start()

    def _safe_after(self, delay_ms: int, callback, *args) -> None:
        try:
            if self.winfo_exists():
                self.after(delay_ms, callback, *args)
        except (RuntimeError, tk.TclError):
            pass

    def _run_worker(self, ork_path: Path, out_dir: Path, cfg: PipelineConfig) -> None:
        try:
            fluent_cb = (
                (lambda line: self._safe_after(0, self._on_fluent_line, line))
                if cfg.deliverables.auto_run_cfd
                else None
            )
            res = run_pipeline(
                ork_path=ork_path,
                output_dir=out_dir,
                config=cfg,
                progress_callback=lambda msg, frac: self._safe_after(
                    0, self._on_progress, msg, frac
                ),
                fluent_line_callback=fluent_cb,
            )
            self._safe_after(0, self._on_complete, res)
        except Exception as exc:
            self._safe_after(0, self._on_error, exc)

    def _on_progress(self, msg: str, frac: float) -> None:
        snap = self._tracker.on_pipeline_progress(msg, frac)
        self._apply_progress_snapshot(snap)
        if snap.should_log and snap.log_line:
            self._append_log(snap.log_line)

    def _on_fluent_line(self, line: str) -> None:
        snap = self._tracker.on_fluent_line(line)
        if snap is None:
            return
        self._apply_progress_snapshot(snap)
        if snap.should_log and snap.log_line:
            self._append_log(snap.log_line)

    def _on_complete(self, res: PackageResult) -> None:
        self._is_running = False
        self._stop_activity_ticker()
        self._last_result = res
        self.btn_generate.configure(state="normal")
        self._sync_generate_button_label()
        if self._rocket is not None:
            self.btn_optimize_fins.configure(state="normal")
            self.btn_optimize_fins_card.configure(state="normal")
        if res.viewer_html_path and res.viewer_html_path.exists():
            self.btn_view_cad.configure(state="normal")
        else:
            self.btn_view_cad.configure(state="disabled")
        self.btn_open_dir.configure(state="normal")
        if self._ansys_install and self._ansys_install.fluent_exe and res.script_paths:
            self.btn_run_fluent.configure(state="normal")
        else:
            self.btn_run_fluent.configure(state="disabled")

        if res.cfd_result is not None and res.cfd_result.primary_run is not None:
            r0 = res.cfd_result.primary_run
            done_head = f"CFD Complete — Cd = {r0.total_cd:.4f} (M={r0.mach:.2f})"
            done_detail = f"Drag = {r0.total_drag_n:.1f} N | Saved to {res.output_dir.name}/"
        elif res.rocket_step_path and not res.script_paths:
            done_head = "Rocket-Only .step Export Complete"
            done_detail = f"Saved {res.rocket_step_path.name} cleanly."
        else:
            done_head = "CFD Package Generation Complete"
            done_detail = (
                f"Ready in {res.output_dir.name}/ — click 'Run CFD & Compute Cd' "
                "or inspect scripts."
            )
        self._apply_progress_snapshot(
            self._tracker.complete_snapshot(headline=done_head, detail=done_detail)
        )
        self.title(self._base_title)

        if res.geometry_report:
            gr = res.geometry_report
            self._append_log(
                f"[OK] Geometry validated: watertight={gr.is_watertight}, "
                f"free_edges={gr.free_edge_count}, zones={list(gr.named_selections.keys())}"
            )
        if res.step_path:
            self._append_log(f"[OK] STEP: {res.step_path.name}")
        if res.rocket_step_path:
            self._append_log(f"[OK] Rocket STEP: {res.rocket_step_path.name}")
        if res.stl_path:
            self._append_log(f"[OK] STL: {res.stl_path.name}")
        if res.viewer_html_path:
            self._append_log(f"[OK] CAD Viewer: {res.viewer_html_path.name}")
        if res.views_png_path:
            self._append_log(f"[OK] Views PNG: {res.views_png_path.name}")
        if res.verification_report:
            vr = res.verification_report
            n_pass = sum(1 for c in vr.checks if c.status == "PASS")
            self._append_log(
                f"[OK] Verification: {vr.overall_status} ({n_pass}/{len(vr.checks)} checks PASS)"
            )
        if res.script_paths:
            self._append_log(f"[OK] Scripts: {', '.join(res.script_paths.keys())}")
        if res.wbpj_path:
            self._append_log(f"[OK] Workbench: {res.wbpj_path.name}")
        if res.wbpz_path:
            self._append_log(f"[OK] Archive: {res.wbpz_path.name}")
        if res.cfd_result is not None:
            for r in res.cfd_result.runs:
                self._append_log(
                    f"[OK] CFD Solved (M={r.mach:.2f}): Cd = {r.total_cd:.4f}, "
                    f"Drag = {r.total_drag_n:.2f} N"
                )
            if res.cfd_result.mesh_convergence is not None:
                mc = res.cfd_result.mesh_convergence
                gci_s = (
                    f", GCI={mc.gci_fine_pct:.2f}%" if mc.gci_fine_pct is not None else ""
                )
                self._append_log(
                    f"[OK] Mesh Convergence ({mc.stop_reason}): Optimal L{mc.optimal_level} / "
                    f"Finest L{mc.finest_level}{gci_s}"
                )
            self._display_cfd_result(res.cfd_result)
        else:
            self._refresh_cfd_results_from_dir(res.output_dir)
        for w in res.warnings:
            self._append_log(f"[WARN] {w}")
        self._refresh_next_steps()
        if res.rocket_step_path and not res.script_paths:
            self._append_log(
                f"[OK] Rocket-only .step ready (no enclosure): {res.rocket_step_path}"
            )
        elif res.cfd_result is not None:
            self._append_log(
                "[OK] Cd computed! Use 'Open in Fluent (Viz)' or "
                "'Open in CFD-Post' to view contours."
            )
        else:
            self._append_log(
                "[NEXT] Click 'Run CFD & Compute Cd' for automated Fluent solve, "
                "or see README_RUN.md."
            )

    def _on_error(self, exc: Exception) -> None:
        self._is_running = False
        self._stop_activity_ticker()
        self.btn_generate.configure(state="normal")
        self._sync_generate_button_label()
        if self._rocket is not None:
            self.btn_optimize_fins.configure(state="normal")
            self.btn_optimize_fins_card.configure(state="normal")
        self._display_cfd_result(self._last_cfd_result)
        self._apply_progress_snapshot(self._tracker.error_snapshot(str(exc)))
        self.title(self._base_title)
        self._append_log(f"[ERROR] {exc}")

    def _on_open_ansys_guide(self) -> ctk.CTkToplevel | None:
        cfg = self._collect_config()
        active = self._active_rocket()
        stem = sanitize_stem(active.name) if active is not None else "<rocket>"
        stage_key = self._selected_stage_key()
        if (
            stage_key not in {"", "all"}
            and self._rocket is not None
            and self._rocket.is_multistage
        ):
            stem = f"{stem}_{sanitize_stem(stage_key)}"
        bl = compute_boundary_layer(active, cfg.flight, cfg.domain.mode) if active else None
        rec = recommend_solver(active, cfg.flight) if active else None
        sym_mult = (
            _symmetry_multiplier(active, cfg.domain.mode)
            if active is not None
            else 1.0
        )
        wbpj_name = (
            self._last_result.wbpj_path.name
            if (self._last_result is not None and self._last_result.wbpj_path is not None)
            else (f"{stem}_cfd.wbpj" if cfg.deliverables.build_wbpj else None)
        )
        guide_text = build_ansys_next_steps_markdown(
            rocket=active,
            stem=stem,
            config=cfg,
            bl=bl,
            solver_rec=rec,
            sym_multiplier=sym_mult,
            wbpj_filename=wbpj_name,
        )

        dlg = ctk.CTkToplevel(self)
        dlg.title("Next Steps — Viewing Results in ANSYS Fluent & CFD-Post")
        dlg.geometry("700x560")
        dlg.minsize(520, 400)
        dlg.grid_columnconfigure(0, weight=1)
        dlg.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            dlg,
            text="Next Steps: Viewing & Verifying Results in ANSYS Fluent / CFD-Post",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(12, 6), sticky="ew")

        textbox = ctk.CTkTextbox(
            dlg,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        textbox.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="nsew")
        textbox.insert("1.0", guide_text)
        textbox.configure(state="disabled")

        btn_bar = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_bar.grid(row=2, column=0, padx=14, pady=(0, 12), sticky="ew")
        btn_bar.grid_columnconfigure(0, weight=1)

        def _copy_guide() -> None:
            self.clipboard_clear()
            self.clipboard_append(guide_text)
            self._append_log("[OK] Copied ANSYS results walkthrough to clipboard.")

        ctk.CTkButton(btn_bar, text="Copy Guide", width=110, command=_copy_guide).grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(btn_bar, text="Close", width=90, command=dlg.destroy).grid(
            row=0, column=1, sticky="e"
        )
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        return dlg

    def _on_view_cad(self) -> None:
        if (
            self._last_result is not None
            and self._last_result.viewer_html_path is not None
            and self._last_result.viewer_html_path.exists()
        ):
            webbrowser.open(self._last_result.viewer_html_path.resolve().as_uri())
            self._append_log("[OK] Opened 3D CAD viewer in browser.")

    def _on_open_folder(self) -> None:
        out_str = self.var_out_dir.get().strip()
        if out_str and Path(out_str).exists():
            os.startfile(out_str)

    def _resolve_active_out_dir(self) -> Path | None:
        if self._last_result is not None:
            return self._last_result.output_dir
        out_str = self.var_out_dir.get().strip()
        return Path(out_str) if out_str else None

    def _selected_mach_point(self, cfd_res: CfdSolveResult | None) -> CfdMachPoint | None:
        if cfd_res is None or not cfd_res.runs:
            return None
        chosen = self.var_result_mach.get().strip()
        for run in cfd_res.runs:
            if f"M={run.mach:.2f}" == chosen:
                return run
        return cfd_res.primary_run

    def _on_result_mach_change(self, _value: str | None = None) -> None:
        self._display_cfd_result(self._last_cfd_result)

    def _display_cfd_result(self, cfd_res: CfdSolveResult | None) -> None:
        self._last_cfd_result = cfd_res
        dark = self.var_dark.get()
        muted = "#94a3b8" if dark else "#475569"
        body_fg = "#e2e8f0" if dark else "#1e293b"
        cd_fg = "#4ade80" if dark else "#15803d"

        out_dir = (
            cfd_res.output_dir if cfd_res is not None else self._resolve_active_out_dir()
        )
        run_pt: CfdMachPoint | None = None
        if cfd_res is not None and cfd_res.runs:
            mach_labels = [f"M={r.mach:.2f}" for r in cfd_res.runs]
            self.cmb_result_mach.configure(
                values=mach_labels,
                state=("normal" if len(mach_labels) > 1 else "disabled"),
            )
            if self.var_result_mach.get() not in mach_labels:
                self.var_result_mach.set(mach_labels[0])
            run_pt = self._selected_mach_point(cfd_res)
            if run_pt is not None:
                cells_txt = (
                    f"{cfd_res.mesh_cells:,}" if cfd_res.mesh_cells is not None else "—"
                )
                yp_txt = (
                    f"{run_pt.wall_yplus_avg:.2f}" if run_pt.wall_yplus_avg is not None else "—"
                )
                scope_lbl = "Whole-Rocket" if self._selected_stage_key() == "all" else "Whole-Stage"
                div_suffix = "  [DIVERGED]" if run_pt.diverged else ""
                self.lbl_cd_headline.configure(
                    text=(
                        f"{scope_lbl} Cd = {run_pt.total_cd:.4f}  |  "
                        f"Drag = {run_pt.total_drag_n:.2f} N  |  "
                        f"M = {run_pt.mach:.2f}  |  Cells = {cells_txt}  |  "
                        f"y+ = {yp_txt}{div_suffix}"
                    ),
                    text_color=("#f87171" if run_pt.diverged else cd_fg),
                )
                zones_txt = "  |  ".join(
                    f"{z.zone.removeprefix('rocket_')}: {z.cd:.4f} ({z.drag_force_n:.1f} N)"
                    for z in run_pt.zones
                )
                mc_line = ""
                if cfd_res.mesh_convergence is not None:
                    mc = cfd_res.mesh_convergence
                    status_lbl = "Converged" if mc.converged else mc.stop_reason.replace("_", " ")
                    gci_s = (
                        f" | GCI={mc.gci_fine_pct:.2f}%"
                        if mc.gci_fine_pct is not None
                        else ""
                    )
                    rich_s = (
                        f" | Rich.Cd={mc.richardson_cd:.5f}"
                        if mc.richardson_cd is not None
                        else ""
                    )
                    mc_line = (
                        f"\nMesh Study ({status_lbl}): Optimal L{mc.optimal_level} / "
                        f"Finest L{mc.finest_level}{gci_s}{rich_s}"
                    )
                self.lbl_cd_zones.configure(
                    text=(zones_txt or "Total aerodynamic drag computed.") + mc_line,
                    text_color=body_fg,
                )
        else:
            self.cmb_result_mach.configure(values=["—"], state="disabled")
            self.var_result_mach.set("—")
            self.lbl_cd_headline.configure(
                text="Cd = —  |  Drag = —  |  Mach = —  |  Cells = —  |  y+ = —",
                text_color=muted,
            )
            self.lbl_cd_zones.configure(
                text=(
                    "Enable 'Auto-Run Fluent & Compute Cd' or click "
                    "'Run CFD & Compute Cd' after generating files."
                ),
                text_color=muted,
            )

        has_fluent = bool(self._ansys_install and self._ansys_install.fluent_exe)
        has_cfdpost = bool(self._ansys_install and self._ansys_install.cfdpost_exe)
        has_run_py = bool(out_dir is not None and (out_dir / "run_fluent.py").exists())

        self.btn_solve_cfd.configure(
            state=("normal" if (not self._is_running and has_run_py and has_fluent) else "disabled")
        )

        f_case, f_data, f_pngs = (
            find_solved_files(out_dir) if out_dir is not None else (None, None, [])
        )
        case_target = (run_pt.case_file if run_pt is not None else None) or f_case
        data_target = (run_pt.data_file if run_pt is not None else None) or f_data or case_target

        run_pngs: list[Path] = []
        if run_pt is not None:
            if run_pt.mach_contour_png is not None and run_pt.mach_contour_png.exists():
                run_pngs.append(run_pt.mach_contour_png)
            if run_pt.pressure_contour_png is not None and run_pt.pressure_contour_png.exists():
                run_pngs.append(run_pt.pressure_contour_png)
        if not run_pngs:
            run_pngs = [p for p in f_pngs if p.exists()]
        if (
            cfd_res is not None
            and cfd_res.mesh_convergence is not None
            and cfd_res.mesh_convergence.plot_png is not None
            and cfd_res.mesh_convergence.plot_png.exists()
        ):
            run_pngs.append(cfd_res.mesh_convergence.plot_png)

        can_fluent = has_fluent and case_target is not None and case_target.exists()
        can_cfdpost = has_cfdpost and data_target is not None and data_target.exists()
        self.btn_viz_fluent.configure(state="normal" if can_fluent else "disabled")
        self.btn_viz_cfdpost.configure(state="normal" if can_cfdpost else "disabled")
        self.btn_viz_contours.configure(
            state=("normal" if run_pngs else "disabled")
        )

    def _refresh_cfd_results_from_dir(self, out_dir: Path) -> None:
        parsed = parse_cfd_results(out_dir)
        self._display_cfd_result(parsed)

    def _on_solve_cfd(self) -> None:
        if self._is_running:
            return
        out_dir = self._resolve_active_out_dir()
        if out_dir is None or not (out_dir / "run_fluent.py").exists():
            self._append_log("[ERROR] Generate the CFD package (with ANSYS scripts) first.")
            return
        if self._ansys_install is None or self._ansys_install.fluent_exe is None:
            self._append_log("[ERROR] Local ANSYS Fluent (fluent.exe) not detected.")
            return

        cfg = self._collect_config()
        self._is_running = True
        self._run_start_mono = time.monotonic()
        self._last_elapsed_text = ""
        self._tracker = WorkflowProgressTracker(
            mode="cfd_only",
            max_iters=cfg.flight.iterations,
            mesh_conv_levels=(
                cfg.mesh_convergence.max_levels if cfg.mesh_convergence.enabled else 1
            ),
        )
        self.btn_generate.configure(state="disabled")
        self.btn_optimize_fins.configure(state="disabled")
        self.btn_optimize_fins_card.configure(state="disabled")
        self.btn_solve_cfd.configure(
            state="disabled",
            text=("Converging Mesh..." if cfg.mesh_convergence.enabled else "Solving CFD..."),
        )
        self._apply_progress_snapshot(
            self._tracker.start_snapshot(
                headline=(
                    f"Starting Fluent Mesh Convergence ({cfg.flight.cores} cores)..."
                    if cfg.mesh_convergence.enabled
                    else f"Launching Headless ANSYS Fluent ({cfg.flight.cores} cores)..."
                ),
                detail=f"Working directory: {out_dir}",
            )
        )
        self._start_activity_ticker()
        if cfg.mesh_convergence.enabled:
            self._append_log(
                f"[INFO] Starting progressive mesh convergence ({cfg.flight.cores} cores, "
                f"tol={cfg.mesh_convergence.cd_rel_tol * 100.0:.2f}%) in {out_dir}..."
            )
        else:
            self._append_log(
                f"[INFO] Starting headless ANSYS Fluent ({cfg.flight.cores} cores) in {out_dir}..."
            )

        worker = threading.Thread(
            target=self._solve_worker,
            args=(out_dir, cfg.flight.cores, cfg.mesh_convergence, cfg.flight.mesh_preset),
            daemon=True,
        )
        worker.start()

    def _solve_worker(
        self,
        out_dir: Path,
        cores: int,
        mesh_conv: MeshConvergenceConfig,
        mesh_preset: MeshPreset,
    ) -> None:
        try:
            line_cb = lambda line: self._safe_after(0, self._on_fluent_line, line)  # noqa: E731
            if mesh_conv.enabled:
                cfd_res = run_mesh_convergence_blocking(
                    output_dir=out_dir,
                    ansys_install=self._ansys_install,
                    conv_config=mesh_conv,
                    base_preset=mesh_preset,
                    cores=cores,
                    line_callback=line_cb,
                )
            else:
                cfd_res = run_fluent_solve_blocking(
                    output_dir=out_dir,
                    ansys_install=self._ansys_install,
                    cores=cores,
                    line_callback=line_cb,
                )
            self._safe_after(0, self._on_solve_complete, cfd_res)
        except Exception as exc:
            self._safe_after(0, self._on_solve_error, exc)

    def _on_solve_complete(self, cfd_res: CfdSolveResult) -> None:
        self._is_running = False
        self._stop_activity_ticker()
        self.btn_generate.configure(state="normal")
        self._sync_generate_button_label()
        if self._rocket is not None:
            self.btn_optimize_fins.configure(state="normal")
            self.btn_optimize_fins_card.configure(state="normal")
        self.btn_solve_cfd.configure(text="Run CFD & Compute Cd")
        self._display_cfd_result(cfd_res)
        r0 = cfd_res.primary_run
        done_head = (
            f"Fluent Complete — Cd = {r0.total_cd:.4f} (M={r0.mach:.2f})"
            if r0 is not None
            else "Fluent Solve Complete"
        )
        cells_s = f"{cfd_res.mesh_cells:,} cells" if cfd_res.mesh_cells is not None else "Solved"
        done_detail = (
            f"{cells_s} | Drag = {r0.total_drag_n:.1f} N | Results loaded above."
            if r0 is not None
            else "Results parsed from output directory."
        )
        self._apply_progress_snapshot(
            self._tracker.complete_snapshot(headline=done_head, detail=done_detail)
        )
        self.title(self._base_title)
        for r in cfd_res.runs:
            self._append_log(
                f"[OK] Solved Mach {r.mach:.2f}: "
                f"Cd = {r.total_cd:.4f}, Drag = {r.total_drag_n:.2f} N"
            )
        if cfd_res.mesh_convergence is not None:
            mc = cfd_res.mesh_convergence
            gci_s = f", GCI={mc.gci_fine_pct:.2f}%" if mc.gci_fine_pct is not None else ""
            rich_s = (
                f", Richardson Cd={mc.richardson_cd:.5f}"
                if mc.richardson_cd is not None
                else ""
            )
            self._append_log(
                f"[OK] Mesh Convergence ({mc.stop_reason}): Optimal L{mc.optimal_level} / "
                f"Finest L{mc.finest_level}{gci_s}{rich_s}"
            )

    def _on_solve_error(self, exc: Exception) -> None:
        self._is_running = False
        self._stop_activity_ticker()
        self.btn_generate.configure(state="normal")
        self._sync_generate_button_label()
        if self._rocket is not None:
            self.btn_optimize_fins.configure(state="normal")
            self.btn_optimize_fins_card.configure(state="normal")
        self.btn_solve_cfd.configure(text="Run CFD & Compute Cd")
        self._display_cfd_result(self._last_cfd_result)
        self._apply_progress_snapshot(self._tracker.error_snapshot(str(exc)))
        self.title(self._base_title)
        self._append_log(f"[ERROR] CFD solve failed: {exc}")

    def _on_optimize_fins(self) -> None:
        if self._is_running:
            return
        ork_str = self.var_ork_path.get().strip()
        if not ork_str:
            self._append_log("[ERROR] Please select an .ork file first.")
            return
        ork_path = Path(ork_str)

        dry_run = bool(self.var_opt_dry_run.get())
        if not dry_run and (self._ansys_install is None or self._ansys_install.fluent_exe is None):
            self._append_log(
                "[ERROR] Local ANSYS Fluent not detected. Enable 'Fast Dry-Run' to run without "
                "ANSYS Fluent."
            )
            return

        cfg = self._collect_config()
        try:
            max_sims = max(3, int(float(self.var_opt_max_sims.get().strip())))
        except ValueError:
            max_sims = 10
        try:
            min_sm = max(0.2, float(self.var_opt_min_sm.get().strip()))
        except ValueError:
            min_sm = 2.0

        active = self._active_rocket()
        stem = sanitize_stem(active.name if active is not None else ork_path.stem)
        out_base = self.var_out_dir.get().strip()
        opt_dir = (
            (Path(out_base).parent / f"{Path(out_base).name}_fin_opt")
            if out_base
            else (ork_path.parent / f"{stem}_fin_opt")
        )
        target_stage_key = self._opt_target_label_to_key.get(
            self.var_opt_target.get(), "booster"
        )
        active_out = self._resolve_active_out_dir()
        conv_src: Path | None = None
        if active_out is not None and (active_out / "mesh_convergence.json").exists():
            conv_src = active_out / "mesh_convergence.json"

        is_booster_target = (
            "stage_2" in target_stage_key or "booster" in target_stage_key.lower()
        )
        if is_booster_target and cfg.flight.mach >= 1.2:
            self._append_log(
                f"[WARN] Optimizing '{target_stage_key}' fins at Mach {cfg.flight.mach:.2f} "
                "(supersonic). Verify Mach in Section 2 if booster stays subsonic (e.g. M=0.80)."
            )

        opt_cfg = FinOptimizationConfig(
            max_sims=max_sims,
            stage_selection=cfg.stage_selection,
            target_fin_stage=target_stage_key,
            screen_cross_sections=(),
            mach=cfg.flight.mach,
            altitude_m=cfg.flight.altitude_m,
            aoa_deg=cfg.flight.aoa_deg,
            min_static_margin_cal=min_sm,
            domain_mode="auto",
            enclosure_shape=EnclosureShape.CYLINDER,
            mesh_preset=cfg.flight.mesh_preset,
            target_yplus=cfg.flight.target_yplus,
            mesh_convergence_baseline=True,
            mesh_conv_tol=cfg.mesh_convergence.cd_rel_tol,
            mesh_convergence_source=conv_src,
            verify_final=bool(self.var_opt_verify.get()),
            blunt_radius_m=cfg.cad.tip_blunt_radius_m,
            cores=cfg.flight.cores,
            iterations=cfg.flight.iterations,
            ram_limit_gb=cfg.flight.ram_limit_gb,
            max_cells=cfg.mesh_convergence.max_cells,
            dry_run=dry_run,
        )
        if conv_src is not None:
            self._append_log(
                f"[INFO] Locking optimal mesh settings from prior convergence study: {conv_src}"
            )
        else:
            self._append_log(
                "[INFO] Baseline Sim 1 will run initial mesh convergence study and lock optimal "
                "mesh sizing for Sims 2+."
            )

        self._is_running = True
        self._last_opt_result = None
        self._run_start_mono = time.monotonic()
        self._last_elapsed_text = ""
        self._tracker = WorkflowProgressTracker(
            mode="fin_opt",
            total_sims=opt_cfg.max_sims,
            include_verify=(opt_cfg.verify_final and not opt_cfg.dry_run),
            max_iters=opt_cfg.iterations,
        )
        self.btn_generate.configure(state="disabled")
        self.btn_optimize_fins.configure(state="disabled", text="Optimizing...")
        self.btn_optimize_fins_card.configure(state="disabled", text="Optimizing Fins...")
        self.btn_solve_cfd.configure(state="disabled")
        for b in (
            self.btn_opt_plot_png,
            self.btn_opt_csv,
            self.btn_opt_ork,
            self.btn_opt_images,
        ):
            b.configure(state="disabled")

        self.viz_tabs.set("Fin Optimization Plots")
        self.opt_canvas.render_empty()
        self.lbl_opt_headline.configure(
            text=f"Running GP-EI fin optimization ({max_sims} sims, SM >= {min_sm:.2f} cal)...",
            text_color=("#38bdf8" if self.var_dark.get() else "#0284c7"),
        )
        self._apply_progress_snapshot(
            self._tracker.start_snapshot(
                headline=f"Starting Fin Optimization ({max_sims} sims, target={target_stage_key})",
                detail=f"Output folder: {opt_dir}",
            )
        )
        self._start_activity_ticker()
        self._append_log(
            f"[INFO] Starting fin optimization ({max_sims} sims, target={target_stage_key}, "
            f"SM>={min_sm:.2f} cal, dry_run={dry_run}) in {opt_dir}"
        )

        worker = threading.Thread(
            target=self._opt_worker,
            args=(ork_path, opt_dir, opt_cfg, stem),
            daemon=True,
        )
        worker.start()

    def _opt_worker(
        self,
        ork_path: Path,
        opt_dir: Path,
        opt_cfg: FinOptimizationConfig,
        stem: str,
    ) -> None:
        try:
            res = run_fin_optimization(
                ork_path=ork_path,
                output_dir=opt_dir,
                config=opt_cfg,
                progress_callback=lambda msg, frac: self._safe_after(
                    0, self._on_opt_progress, msg, frac
                ),
                eval_callback=lambda evals: self._safe_after(
                    0,
                    self._on_opt_eval_update,
                    evals,
                    opt_cfg.min_static_margin_cal,
                    opt_cfg.target_fin_stage,
                    stem,
                ),
                fluent_line_callback=(
                    None
                    if opt_cfg.dry_run
                    else (lambda line: self._safe_after(0, self._on_fluent_line, line))
                ),
            )
            self._safe_after(0, self._on_opt_complete, res, opt_cfg.min_static_margin_cal)
        except Exception as exc:
            self._safe_after(0, self._on_opt_error, exc)

    def _on_opt_progress(self, msg: str, frac: float) -> None:
        snap = self._tracker.on_fin_opt_progress(msg, frac)
        self._apply_progress_snapshot(snap)
        if snap.should_log and snap.log_line:
            self._append_log(snap.log_line)

    def _on_opt_eval_update(
        self,
        evals: list[SimEvaluation],
        min_sm: float,
        target_stage: str,
        rocket_name: str,
    ) -> None:
        self.opt_canvas.update_evaluations(
            evals,
            min_sm_cal=min_sm,
            target_stage_name=target_stage,
            rocket_name=rocket_name,
        )
        if evals:
            base = evals[0]
            valid = [e for e in evals if not e.diverged]
            feasible = [e for e in valid if e.stability.feasible]
            best = min(feasible or valid or evals, key=lambda e: e.cd_total)
            latest = evals[-1]
            dpct = ((best.cd_total - base.cd_total) / max(base.cd_total, 1e-9)) * 100.0
            split_txt = (
                f" (Airframe={best.cd_total - best.cd_fins:.4f} + Fins={best.cd_fins:.4f})"
                if best.cd_fins is not None
                else ""
            )
            self.lbl_opt_headline.configure(
                text=(
                    f"Eval {len(evals)} | Best #{best.sim_index}: "
                    f"Whole-Stage/Rocket Cd={best.cd_total:.5f} ({dpct:+.2f}%){split_txt} | "
                    f"SM={best.stability.static_margin_cal:.2f} cal"
                ),
                text_color=("#4ade80" if self.var_dark.get() else "#15803d"),
            )
            self._append_log(
                f"[OK] Opt Sim {latest.sim_index} [{latest.phase}]: Whole-Stage/Rocket Cd = "
                f"{latest.cd_total:.5f}"
                + (f" (fins={latest.cd_fins:.5f})" if latest.cd_fins is not None else "")
                + f", SM={latest.stability.static_margin_cal:.2f} cal"
                + (" [DIVERGED-EXCLUDED]" if latest.diverged else "")
            )
            if best.run_dir:
                best_parsed = parse_cfd_results(Path(best.run_dir))
                if best_parsed is not None and best_parsed.runs:
                    self._display_cfd_result(best_parsed)

    def _on_opt_complete(
        self, res: FinOptimizationResult, min_sm_cal: float | None = None
    ) -> None:
        self._is_running = False
        self._stop_activity_ticker()
        self._last_opt_result = res
        self.btn_generate.configure(state="normal")
        self._sync_generate_button_label()
        self.btn_optimize_fins.configure(state="normal", text="Optimize Fins")
        self.btn_optimize_fins_card.configure(
            state="normal", text="Run Fin Shape Optimization"
        )
        self._display_cfd_result(self._last_cfd_result)

        if min_sm_cal is None:
            try:
                resolved_min_sm = max(0.2, float(self.var_opt_min_sm.get().strip() or 2.0))
            except ValueError:
                resolved_min_sm = 2.0
        else:
            resolved_min_sm = float(min_sm_cal)
        self.opt_canvas.update_evaluations(
            list(res.evaluations),
            min_sm_cal=resolved_min_sm,
            target_stage_name=res.target_stage_name,
            rocket_name=res.optimized_ork_path.stem.removesuffix("_optimized"),
        )
        self.btn_opt_plot_png.configure(
            state=("normal" if res.cd_plot_path and res.cd_plot_path.exists() else "disabled")
        )
        self.btn_opt_csv.configure(
            state=("normal" if res.summary_csv_path.exists() else "disabled")
        )
        self.btn_opt_ork.configure(
            state=("normal" if res.optimized_ork_path.exists() else "disabled")
        )
        has_imgs = bool(
            res.cfd_images_dir
            and res.cfd_images_dir.exists()
            and any(res.cfd_images_dir.glob("*.png"))
        )
        self.btn_opt_images.configure(state=("normal" if has_imgs else "disabled"))

        best = res.best_eval
        oc = best.candidate
        self.lbl_opt_headline.configure(
            text=(
                f"Done ({len(res.evaluations)} sims) | Best #{best.sim_index}: "
                f"Whole-Stage/Rocket Cd={best.cd_total:.5f} ({res.delta_cd_pct:+.2f}%) | "
                f"cr={oc.root_chord*1000:.1f} ct={oc.tip_chord*1000:.1f} "
                f"span={oc.span*1000:.1f}mm swp={oc.sweep_angle_deg:.1f}°"
            ),
            text_color=("#4ade80" if self.var_dark.get() else "#15803d"),
        )
        self._apply_progress_snapshot(
            self._tracker.complete_snapshot(
                headline=(
                    f"Fin Opt Complete — Best Whole-Stage/Rocket Cd = {best.cd_total:.5f} "
                    f"({res.delta_cd_pct:+.2f}%)"
                ),
                detail=(
                    f"Best: cr={oc.root_chord*1000:.1f}mm ct={oc.tip_chord*1000:.1f}mm "
                    f"span={oc.span*1000:.1f}mm sweep={oc.sweep_angle_deg:.1f}° "
                    f"SM={best.stability.static_margin_cal:.2f} cal"
                ),
            )
        )
        self.title(self._base_title)
        self._append_log(
            f"[OK] Fin Optimization finished: Whole-Stage/Rocket Cd "
            f"{res.baseline_eval.cd_total:.5f} -> {best.cd_total:.5f} ({res.delta_cd_pct:+.2f}%), "
            f"saved {res.optimized_ork_path.name}"
        )
        if best.run_dir:
            best_parsed = parse_cfd_results(Path(best.run_dir))
            if best_parsed is not None and best_parsed.runs:
                self._display_cfd_result(best_parsed)

    def _on_opt_error(self, exc: Exception) -> None:
        self._is_running = False
        self._stop_activity_ticker()
        self.btn_generate.configure(state="normal")
        self._sync_generate_button_label()
        self.btn_optimize_fins.configure(state="normal", text="Optimize Fins")
        self.btn_optimize_fins_card.configure(
            state="normal", text="Run Fin Shape Optimization"
        )
        self._display_cfd_result(self._last_cfd_result)
        self.lbl_opt_headline.configure(
            text="Fin optimization failed.",
            text_color="#f87171",
        )
        self._apply_progress_snapshot(self._tracker.error_snapshot(str(exc)))
        self.title(self._base_title)
        self._append_log(f"[ERROR] Fin optimization failed: {exc}")

    def _on_open_opt_plot_png(self) -> None:
        if self._last_opt_result and self._last_opt_result.cd_plot_path:
            os.startfile(str(self._last_opt_result.cd_plot_path))

    def _on_open_opt_csv(self) -> None:
        if self._last_opt_result and self._last_opt_result.summary_csv_path.exists():
            os.startfile(str(self._last_opt_result.summary_csv_path))

    def _on_open_opt_ork(self) -> None:
        if self._last_opt_result and self._last_opt_result.optimized_ork_path.exists():
            self._load_ork(self._last_opt_result.optimized_ork_path)
            self._append_log(
                f"[OK] Loaded optimized rocket: {self._last_opt_result.optimized_ork_path.name}"
            )

    def _on_open_opt_images(self) -> None:
        if self._last_opt_result and self._last_opt_result.cfd_images_dir:
            os.startfile(str(self._last_opt_result.cfd_images_dir))

    def _on_viz_fluent(self) -> None:
        out_dir = self._resolve_active_out_dir()
        if out_dir is None:
            return
        run_pt = self._selected_mach_point(self._last_cfd_result)
        case_path = run_pt.case_file if run_pt is not None else None
        cfg = self._collect_config()
        try:
            launch_fluent_visualization(
                output_dir=out_dir,
                ansys_install=self._ansys_install,
                case_path=case_path,
                cores=cfg.flight.cores,
            )
            self._append_log("[OK] Opened solved case & data directly in ANSYS Fluent GUI.")
        except Exception as exc:
            self._append_log(f"[ERROR] {exc}")

    def _on_viz_cfdpost(self) -> None:
        out_dir = self._resolve_active_out_dir()
        if out_dir is None:
            return
        run_pt = self._selected_mach_point(self._last_cfd_result)
        target = (run_pt.data_file or run_pt.case_file) if run_pt is not None else None
        try:
            launch_cfdpost_visualization(
                output_dir=out_dir,
                ansys_install=self._ansys_install,
                data_path=target,
            )
            self._append_log("[OK] Opened solved results directly in ANSYS CFD-Post.")
        except Exception as exc:
            self._append_log(f"[ERROR] {exc}")

    def _on_viz_contours(self) -> None:
        out_dir = self._resolve_active_out_dir()
        if out_dir is None:
            return
        run_pt = self._selected_mach_point(self._last_cfd_result)
        pngs: list[Path] = []
        if run_pt is not None:
            for p in (run_pt.mach_contour_png, run_pt.pressure_contour_png):
                if p is not None and p.exists():
                    pngs.append(p)
        if not pngs:
            _c, _d, pngs = find_solved_files(out_dir)
        if (
            self._last_cfd_result is not None
            and self._last_cfd_result.mesh_convergence is not None
            and self._last_cfd_result.mesh_convergence.plot_png is not None
            and self._last_cfd_result.mesh_convergence.plot_png.exists()
            and self._last_cfd_result.mesh_convergence.plot_png not in pngs
        ):
            pngs.append(self._last_cfd_result.mesh_convergence.plot_png)
        for p in pngs:
            os.startfile(str(p))
            self._append_log(f"[OK] Opened contour image: {p.name}")

    def _on_run_fluent(self) -> None:
        if self._last_result is None:
            return
        cfg = self._collect_config()
        try:
            launch_fluent_run(
                output_dir=self._last_result.output_dir,
                ansys_install=self._ansys_install,
                cores=cfg.flight.cores,
                show_gui=True,
            )
            self._append_log("[OK] Launched ANSYS Fluent in background.")
        except Exception as exc:
            self._append_log(f"[ERROR] Failed to launch Fluent: {exc}")

    def _rebuild_stage_pills(self, labels: tuple[str, ...]) -> None:
        if labels == self._rendered_stage_labels and self._stage_pills:
            return
        for pill in self._stage_pills:
            pill.destroy()
        self._stage_pills.clear()
        for col in range(max(1, len(labels))):
            self.stage_row.grid_columnconfigure(col, weight=1)
        for col, text in enumerate(labels):
            pill = ctk.CTkLabel(
                self.stage_row,
                text=f"○ {text}",
                height=20,
                corner_radius=5,
                font=ctk.CTkFont(size=10, weight="bold"),
                padx=6,
            )
            pill.grid(
                row=0,
                column=col,
                padx=(0 if col == 0 else 3, 0 if col == len(labels) - 1 else 3),
                sticky="ew",
            )
            self._stage_pills.append(pill)
        self._rendered_stage_labels = labels

    def _update_stage_pills(self, snap: ProgressSnapshot) -> None:
        dark = self.var_dark.get()
        is_err = snap.phase_key == "error"
        for idx, (pill, label) in enumerate(zip(self._stage_pills, snap.stage_labels, strict=True)):
            if snap.active_stage_index >= len(snap.stage_labels) or idx < snap.active_stage_index:
                prefix = "✓ "
                fg = "#14532d" if dark else "#dcfce7"
                txt_col = "#86efac" if dark else "#166534"
            elif idx == snap.active_stage_index and not is_err:
                prefix = "▶ "
                fg = "#0369a1" if dark else "#0284c7"
                txt_col = "#ffffff"
            else:
                prefix = "○ "
                fg = "#1e293b" if dark else "#e2e8f0"
                txt_col = "#94a3b8" if dark else "#475569"
            pill.configure(text=f"{prefix}{label}", fg_color=fg, text_color=txt_col)

    def _current_elapsed_text(self) -> str:
        if self._run_start_mono is not None:
            self._last_elapsed_text = format_elapsed(time.monotonic() - self._run_start_mono)
        return self._last_elapsed_text

    def _apply_progress_snapshot(self, snap: ProgressSnapshot) -> None:
        self._last_snapshot = snap
        dark = self.var_dark.get()
        d_badge, l_badge, d_bar, l_bar = _PHASE_COLORS.get(
            snap.phase_key, _PHASE_COLORS["idle"]
        )
        badge_bg = d_badge if dark else l_badge
        bar_fg = d_bar if dark else l_bar
        badge_fg = "#0f172a" if (snap.phase_key == "idle" and not dark) else "#ffffff"
        frame_char = _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]
        spin = f"{frame_char} " if snap.is_busy else ""
        pct = int(round(max(0.0, min(1.0, snap.fraction)) * 100.0))
        elapsed = self._current_elapsed_text()
        meta_txt = f"{pct}%  •  {elapsed}" if elapsed else f"{pct}%"

        if snap.phase_key == "done":
            border_col = "#16a34a"
        elif snap.phase_key == "error":
            border_col = "#ef4444"
        elif snap.is_busy:
            border_col = bar_fg
        else:
            border_col = "#334155" if dark else "#cbd5e1"

        self.status_card.configure(border_color=border_col)
        self.lbl_status_badge.configure(
            text=f"{spin}{snap.badge_text}",
            fg_color=badge_bg,
            text_color=badge_fg,
        )
        self.lbl_status_headline.configure(
            text=snap.headline,
            text_color=("#f8fafc" if dark else "#0f172a"),
        )
        meta_idle_col = "#94a3b8" if dark else "#475569"
        self.lbl_progress_meta.configure(
            text=meta_txt,
            text_color=(
                bar_fg if (snap.is_busy or snap.phase_key == "done") else meta_idle_col
            ),
        )
        self.lbl_status_detail.configure(
            text=snap.detail,
            text_color=("#cbd5e1" if dark else "#334155"),
        )
        self.progress.configure(progress_color=bar_fg)
        self.progress.set(max(0.0, min(1.0, snap.fraction)))

        self._rebuild_stage_pills(snap.stage_labels)
        self._update_stage_pills(snap)
        if snap.is_busy:
            win_el = f" ({elapsed})" if elapsed else ""
            self.title(f"[{pct}%] {snap.badge_text}{win_el} — ork2cfd")

    def _start_activity_ticker(self) -> None:
        self._stop_activity_ticker()
        self._spinner_idx = 0
        self._ticker_after_id = self.after(400, self._tick_activity)

    def _stop_activity_ticker(self) -> None:
        if self._ticker_after_id is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._ticker_after_id)
            self._ticker_after_id = None

    def _tick_activity(self) -> None:
        self._ticker_after_id = None
        if not self._is_running:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER_FRAMES)
        self._apply_progress_snapshot(self._last_snapshot)
        self._ticker_after_id = self.after(400, self._tick_activity)

    def _configure_log_tags(self) -> None:
        dark = self.var_dark.get()
        palette = {
            "ok": "#4ade80" if dark else "#15803d",
            "warn": "#fbbf24" if dark else "#b45309",
            "error": "#f87171" if dark else "#b91c1c",
            "fluent": "#2dd4bf" if dark else "#0f766e",
            "info": "#7dd3fc" if dark else "#0369a1",
            "next": "#c4b5fd" if dark else "#6d28d9",
        }
        with contextlib.suppress(Exception):
            tb = self.log_box._textbox
            for tag_name, col in palette.items():
                tb.tag_config(tag_name, foreground=col)

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}] {text}\n"
        tag = None
        if text.startswith("[OK]"):
            tag = "ok"
        elif text.startswith("[WARN]"):
            tag = "warn"
        elif text.startswith("[ERROR]"):
            tag = "error"
        elif text.startswith("[FLUENT]"):
            tag = "fluent"
        elif text.startswith("[INFO]"):
            tag = "info"
        elif text.startswith("[NEXT]"):
            tag = "next"

        self.log_box.configure(state="normal")
        if tag is not None:
            try:
                self.log_box._textbox.insert("end", formatted, (tag,))
            except Exception:
                self.log_box.insert("end", formatted)
        else:
            self.log_box.insert("end", formatted)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


def launch_gui(initial_ork: Path | None = None) -> None:
    """Launch the ork2cfd CustomTkinter desktop application."""
    app = Ork2CfdApp(initial_ork=initial_ork)
    app.mainloop()
