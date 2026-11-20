"""Main CustomTkinter desktop window for ork2cfd."""

from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ..ansys.detector import AnsysInstallation, select_best_installation
from ..ansys.fluent_launcher import launch_fluent_run
from ..ansys.next_steps import build_ansys_next_steps_markdown, build_ansys_next_steps_summary
from ..ansys.script_generator import _symmetry_multiplier
from ..models.cfd import (
    BoiConfig,
    CadConfig,
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    EnclosureShape,
    FlightCondition,
    FlowRegime,
    MeshPreset,
    PipelineConfig,
)
from ..models.results import PackageResult
from ..models.rocket import RocketModel
from ..parser.ork_parser import parse_ork_file
from ..physics.boundary_layer import compute_boundary_layer
from ..physics.regime import recommend_solver
from ..pipeline import run_pipeline, sanitize_stem
from .preview_canvas import PreviewCanvas
from .widgets import (
    InfoIcon,
    SectionCard,
    add_labeled_checkbox,
    add_labeled_combo,
    add_labeled_entry,
)


class Ork2CfdApp(ctk.CTk):
    """Simple, responsive desktop GUI bundling STEP, ANSYS scripts, and .wbpj generation."""

    def __init__(self, initial_ork: Path | None = None) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("ork2cfd — OpenRocket to ANSYS Fluent CFD Converter")
        self.geometry("1240x800")
        self.minsize(1040, 680)

        self._rocket: RocketModel | None = None
        self._last_result: PackageResult | None = None
        self._ansys_install: AnsysInstallation | None = select_best_installation()
        self._is_running = False
        self._info_icons: list[InfoIcon] = []

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
        self.var_cores = ctk.StringVar(value="4")
        self.var_iters = ctk.StringVar(value="400")

        self.var_mode = ctk.StringVar(value=DomainMode.FLUID_DOMAIN.value)
        self.var_shape = ctk.StringVar(value=EnclosureShape.BULLET.value)
        self.var_blunt_mm = ctk.StringVar(value="1.0")
        self.var_schema = ctk.StringVar(value="AP214")
        self.var_bois = ctk.BooleanVar(value=True)
        self.var_fillets = ctk.BooleanVar(value=False)
        self.var_rail_btns = ctk.BooleanVar(value=False)
        self.var_lugs = ctk.BooleanVar(value=False)

        self.var_step = ctk.BooleanVar(value=True)
        self.var_rocket_step = ctk.BooleanVar(value=True)
        self.var_stl = ctk.BooleanVar(value=False)
        self.var_verify = ctk.BooleanVar(value=True)
        self.var_scripts = ctk.BooleanVar(value=True)
        self.var_wbpj = ctk.BooleanVar(value=self._ansys_install is not None)
        self.var_wbpz = ctk.BooleanVar(value=False)
        self.var_dark = ctk.BooleanVar(value=True)

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
        self._build_next_steps_card(left, row=4)

        # Right column: Preview + Log + Actions
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.preview = PreviewCanvas(right, dark=True)
        self.preview.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        bottom = ctk.CTkFrame(right, corner_radius=8)
        bottom.grid(row=1, column=0, sticky="nsew")
        bottom.grid_rowconfigure(1, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_row.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=2)
        btn_row.grid_columnconfigure((1, 2, 3), weight=1)

        self.btn_generate = ctk.CTkButton(
            btn_row,
            text="Generate CFD Package",
            height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_generate,
        )
        self.btn_generate.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.btn_view_cad = ctk.CTkButton(
            btn_row,
            text="View 3D CAD",
            height=36,
            state="disabled",
            fg_color="#2563eb",
            command=self._on_view_cad,
        )
        self.btn_view_cad.grid(row=0, column=1, padx=4, sticky="ew")

        self.btn_open_dir = ctk.CTkButton(
            btn_row,
            text="Open Output Folder",
            height=36,
            state="disabled",
            fg_color="#334155",
            command=self._on_open_folder,
        )
        self.btn_open_dir.grid(row=0, column=2, padx=4, sticky="ew")

        self.btn_run_fluent = ctk.CTkButton(
            btn_row,
            text="Run in Fluent",
            height=36,
            state="disabled",
            fg_color="#0f766e",
            command=self._on_run_fluent,
        )
        self.btn_run_fluent.grid(row=0, column=3, padx=(6, 0), sticky="ew")

        self.log_box = ctk.CTkTextbox(
            bottom, height=130, font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.log_box.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="nsew")
        self.log_box.configure(state="disabled")

        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.progress.set(0.0)

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
            card.body, 3, 0, "Fluent Cores", self.var_cores,
            self._refresh_calculations, help_key="cores", icon_registry=self._info_icons,
        )
        add_labeled_entry(
            card.body, 3, 1, "Solver Iterations", self.var_iters,
            self._refresh_calculations, help_key="iterations", icon_registry=self._info_icons,
        )

        self.lbl_physics_summary = ctk.CTkLabel(
            card.body,
            text="Load an .ork file to compute Re, Δy1, and prism layers.",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#38bdf8",
        )
        self.lbl_physics_summary.grid(row=4, column=0, columnspan=2, pady=(6, 0), sticky="ew")

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
            "4. Deliverables & Output Folder",
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

        add_labeled_checkbox(
            card.body, 2, 0, "CFD Domain .step + JSON", self.var_step,
            help_key="deliv_step", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 2, 1, "ANSYS Scripts (.py/.jou)", self.var_scripts,
            help_key="deliv_scripts", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 3, 0, "Build Workbench .wbpj", self.var_wbpj,
            help_key="deliv_wbpj", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 3, 1, "Archive .wbpz", self.var_wbpz,
            help_key="deliv_wbpz", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 4, 0, "Export Multi-Region .stl", self.var_stl,
            help_key="deliv_stl", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 4, 1, "CAD Viewer & Verify (.html/.png)", self.var_verify,
            help_key="deliv_verify", icon_registry=self._info_icons,
        )
        add_labeled_checkbox(
            card.body, 5, 0, "Rocket-Only .step (Named Parts)", self.var_rocket_step,
            help_key="deliv_rocket_step", icon_registry=self._info_icons,
        )

    def _build_next_steps_card(self, parent, row: int) -> None:
        card = SectionCard(
            parent,
            "5. Next Steps — View Results in ANSYS",
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
        self.lbl_next_steps.configure(
            text_color=("#cbd5e1" if dark else "#1e293b"),
        )

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

        self._update_rocket_summary()
        self._append_log(
            f"[OK] Loaded {ork_path.name} ({rocket.name}, "
            f"stages={rocket.stage_count}, sym={rocket.rotational_symmetry_order}-fold, "
            f"L={rocket.total_length:.3f}m)."
        )
        for w in rocket.warnings:
            self._append_log(f"[WARN] {w}")
        self._refresh_calculations()

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
            cores=_int(self.var_cores, 4),
            iterations=_int(self.var_iters, 400),
        )
        deliverables = DeliverableConfig(
            export_step=self.var_step.get(),
            export_rocket_step=self.var_rocket_step.get(),
            export_stl=self.var_stl.get(),
            export_verification=self.var_verify.get(),
            generate_scripts=self.var_scripts.get(),
            build_wbpj=self.var_wbpj.get(),
            archive_wbpz=self.var_wbpz.get(),
        )
        return PipelineConfig(
            cad=cad,
            domain=domain,
            flight=flight,
            deliverables=deliverables,
            stage_selection=self._selected_stage_key(),
        )

    def _refresh_calculations(self) -> None:
        active = self._active_rocket()
        if active is None:
            return
        cfg = self._collect_config()
        bl = compute_boundary_layer(active, cfg.flight, cfg.domain.mode)
        rec = recommend_solver(active, cfg.flight)

        student_tag = " [>512k Student Limit!]" if bl.exceeds_student_limit else ""
        text = (
            f"Regime: {rec.regime.value.upper()} ({rec.solver_type}, {rec.fluid_model})\n"
            f"V={bl.velocity_m_s:.1f} m/s | Re_L={bl.reynolds_length:.2e} | "
            f"Δy1={bl.first_layer_thickness_m:.2e} m ({bl.prism_layers} layers)\n"
            f"Surf={bl.surface_min_size_m*1000:.1f}..{bl.surface_max_size_m*1000:.1f} mm | "
            f"Est.Cells≈{bl.estimated_cell_count:,}{student_tag}"
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
        self.btn_generate.configure(state="disabled", text="Generating...")
        self.progress.set(0.02)

        worker = threading.Thread(
            target=self._run_worker,
            args=(Path(ork_str), Path(out_str), cfg),
            daemon=True,
        )
        worker.start()

    def _run_worker(self, ork_path: Path, out_dir: Path, cfg: PipelineConfig) -> None:
        try:
            res = run_pipeline(
                ork_path=ork_path,
                output_dir=out_dir,
                config=cfg,
                progress_callback=lambda msg, frac: self.after(0, self._on_progress, msg, frac),
            )
            self.after(0, self._on_complete, res)
        except Exception as exc:
            self.after(0, self._on_error, exc)

    def _on_progress(self, msg: str, frac: float) -> None:
        self.progress.set(max(0.0, min(1.0, frac)))
        self._append_log(f"[INFO] {msg}")

    def _on_complete(self, res: PackageResult) -> None:
        self._is_running = False
        self._last_result = res
        self.btn_generate.configure(state="normal", text="Generate CFD Package")
        if res.viewer_html_path and res.viewer_html_path.exists():
            self.btn_view_cad.configure(state="normal")
        else:
            self.btn_view_cad.configure(state="disabled")
        self.btn_open_dir.configure(state="normal")
        if self._ansys_install and self._ansys_install.fluent_exe:
            self.btn_run_fluent.configure(state="normal")
        self.progress.set(1.0)

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
        for w in res.warnings:
            self._append_log(f"[WARN] {w}")
        self._refresh_next_steps()
        self._append_log(
            "[NEXT] See README_RUN.md (Section 4) or click 'Open Full ANSYS Results Guide' "
            "for Fluent / CFD-Post result steps."
        )

    def _on_error(self, exc: Exception) -> None:
        self._is_running = False
        self.btn_generate.configure(state="normal", text="Generate CFD Package")
        self.progress.set(0.0)
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

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


def launch_gui(initial_ork: Path | None = None) -> None:
    """Launch the ork2cfd CustomTkinter desktop application."""
    app = Ork2CfdApp(initial_ork=initial_ork)
    app.mainloop()
