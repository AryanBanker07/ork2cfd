"""Live workflow progress state machine and Fluent stdout telemetry parser."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CELLS_CREATED_RE = re.compile(r"(\d+)\s+cells were created", re.IGNORECASE)
_VOL_CELLS_RE = re.compile(r"Volume mesh total cell count:\s*(\d+)", re.IGNORECASE)
_ORTH_QUALITY_RE = re.compile(
    r"minimum\s+Orthogonal Quality\s*(?:of:|=)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_YPLUS_RE = re.compile(r"wall_yplus_avg\s*=\s*([0-9.]+)")
_MACH_REPORTS_RE = re.compile(r"Mach\s+([0-9.]+)\s+reports:")
_CONV_LEVEL_RE = re.compile(
    r"Mesh Convergence Level\s+(\d+)/(\d+)(?:\s*\(mesh_scale=([0-9.]+)\))?"
)
_CONV_SUMMARY_RE = re.compile(
    r"Level\s+(\d+):\s*cells=([^,]+),\s*Cd=([0-9.]+)(?:,\s*(?:ΔCd|Delta\s*Cd)=([0-9.]+%))?"
)
_FIN_OPT_SIM_RE = re.compile(r"Sim\s+(\d+)/(\d+)\s+\[([^\]]+)\]:\s*(.*)$")
_ITER_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"
    r"((?:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s+){5,9})"
    r"(\d+:\d{2}:\d{2})\s+"
    r"(\d+)\s*$"
)
_FLOAT_TOKEN_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_PACKAGE_END_FRAC = 0.20


@dataclass(frozen=True)
class ProgressSnapshot:
    """UI-agnostic snapshot describing current workflow phase, progress, and telemetry."""

    phase_key: str
    badge_text: str
    headline: str
    detail: str
    fraction: float
    active_stage_index: int
    stage_labels: tuple[str, ...]
    is_busy: bool
    is_indeterminate_substep: bool = False
    should_log: bool = False
    log_line: str | None = None


def format_elapsed(seconds: float) -> str:
    """Format elapsed wall-clock seconds cleanly as MM:SS or H:MM:SS."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_stage_labels(
    mode: str,
    *,
    rocket_only: bool = False,
    include_verify: bool = True,
    include_wbpj: bool = True,
) -> tuple[str, ...]:
    """Return ordered human-readable stage pill labels for a workflow mode."""
    if mode == "fin_opt":
        if include_verify:
            return ("Baseline", "LHS DoE", "GP-EI Opt", "Verify", "Export")
        return ("Baseline", "LHS DoE", "GP-EI Opt", "Export")
    if mode == "cfd_only":
        return ("Launch", "Surface Mesh", "Volume Mesh", "Setup", "Solve", "Results")
    if mode == "package_and_cfd":
        return ("CAD & Export", "Scripts / WB", "Surface Mesh", "Volume Mesh", "Solve", "Results")
    if rocket_only:
        labels = ["Parse", "Rocket CAD"]
        if include_verify:
            labels.append("Verify")
        return tuple(labels)
    labels = ["Parse", "3D CAD"]
    if include_verify:
        labels.append("Verify")
    labels.append("Scripts")
    if include_wbpj:
        labels.append("Workbench")
    return tuple(labels)


class WorkflowProgressTracker:
    """Translate coarse pipeline events and raw Fluent stdout lines into ProgressSnapshot."""

    def __init__(
        self,
        mode: str = "package",
        *,
        rocket_only: bool = False,
        include_verify: bool = True,
        include_wbpj: bool = True,
        max_iters: int = 300,
        mesh_conv_levels: int = 1,
        total_sims: int = 10,
    ) -> None:
        if mode not in {"package", "package_and_cfd", "cfd_only", "fin_opt"}:
            raise ValueError(f"Unsupported tracker mode: {mode!r}")
        self.mode = mode
        self.stage_labels = build_stage_labels(
            mode,
            rocket_only=rocket_only,
            include_verify=include_verify,
            include_wbpj=include_wbpj,
        )
        self.max_iters = max(1, int(max_iters))
        self.total_levels = max(1, int(mesh_conv_levels))
        self.total_sims = max(1, int(total_sims))
        self.current_sim = 0
        self.current_sim_phase = "baseline"
        self.current_level = 1
        self.current_scale: float | None = None
        self.mesh_cells: int | None = None
        self.min_orth_quality: float | None = None
        self.wall_yplus_avg: float | None = None
        self.last_mach: float | None = None
        self.last_cd: float | None = None
        self._saw_cd_col = False
        self._max_fraction = 0.0

    def idle_snapshot(self) -> ProgressSnapshot:
        """Return initial ready state before any run starts."""
        return ProgressSnapshot(
            phase_key="idle",
            badge_text="READY",
            headline="Ready — select an OpenRocket (.ork) file to begin.",
            detail="Live stage progress, Fluent meshing counts, and solver ETA appear here.",
            fraction=0.0,
            active_stage_index=-1,
            stage_labels=self.stage_labels,
            is_busy=False,
        )

    def start_snapshot(self, headline: str, detail: str) -> ProgressSnapshot:
        """Return initial busy snapshot right as a worker thread starts."""
        self._max_fraction = 0.02 if self.mode != "cfd_only" else 0.04
        phase = "fluent_launch" if self.mode == "cfd_only" else "parse"
        badge = "FLUENT LAUNCH" if self.mode == "cfd_only" else "STARTING"
        return ProgressSnapshot(
            phase_key=phase,
            badge_text=badge,
            headline=headline,
            detail=detail,
            fraction=self._max_fraction,
            active_stage_index=0,
            stage_labels=self.stage_labels,
            is_busy=True,
            is_indeterminate_substep=True,
        )

    def complete_snapshot(self, headline: str, detail: str) -> ProgressSnapshot:
        """Return terminal completed snapshot at 100%."""
        self._max_fraction = 1.0
        return ProgressSnapshot(
            phase_key="done",
            badge_text="COMPLETE",
            headline=headline,
            detail=detail,
            fraction=1.0,
            active_stage_index=len(self.stage_labels),
            stage_labels=self.stage_labels,
            is_busy=False,
        )

    def error_snapshot(self, message: str) -> ProgressSnapshot:
        """Return terminal error snapshot."""
        short = str(message).strip().splitlines()[0] if str(message).strip() else "Unknown error"
        return ProgressSnapshot(
            phase_key="error",
            badge_text="FAILED",
            headline="Run failed — check Activity Log for details",
            detail=short[:140],
            fraction=self._max_fraction,
            active_stage_index=-1,
            stage_labels=self.stage_labels,
            is_busy=False,
        )

    def _stage_index(self, candidate: str, fallback: int = 0) -> int:
        low = candidate.lower()
        for idx, lbl in enumerate(self.stage_labels):
            if low in lbl.lower():
                return idx
        return min(max(fallback, 0), max(0, len(self.stage_labels) - 1))

    def _map_cfd_fraction(self, sub_frac: float) -> float:
        sub = min(1.0, max(0.0, sub_frac))
        if self.mode == "fin_opt":
            sim_idx = max(1, self.current_sim)
            raw = ((sim_idx - 1) + sub) / max(1, self.total_sims)
            self._max_fraction = max(self._max_fraction, min(0.98, raw))
            return self._max_fraction
        level_span = 1.0 / self.total_levels
        level_frac = ((self.current_level - 1) + sub) * level_span
        if self.mode == "package_and_cfd":
            raw = _PACKAGE_END_FRAC + (1.0 - _PACKAGE_END_FRAC) * level_frac
        else:
            raw = level_frac
        self._max_fraction = max(self._max_fraction, min(0.99, raw))
        return self._max_fraction

    def _level_prefix(self) -> str:
        if self.mode == "fin_opt" and self.current_sim > 0:
            return f"Sim {self.current_sim}/{self.total_sims} • "
        if self.total_levels > 1:
            return f"L{self.current_level}/{self.total_levels} • "
        return ""

    def on_fin_opt_progress(self, msg: str, frac: float) -> ProgressSnapshot:
        """Update snapshot from run_fin_optimization(msg, frac) callback."""
        low = msg.lower()
        sim_m = _FIN_OPT_SIM_RE.search(msg)
        if sim_m:
            self.current_sim = int(sim_m.group(1))
            self.total_sims = max(self.total_sims, int(sim_m.group(2)))
            self.current_sim_phase = sim_m.group(3).strip().lower()
            cand_desc = sim_m.group(4).strip()
            self.mesh_cells = None
            self.min_orth_quality = None
            self._saw_cd_col = False

            phase_map = {
                "baseline": (0, "BASELINE", "fluent_launch"),
                "cross_section_screen": (0, "SCREEN", "fluent_launch"),
                "doe_lhs": (1, "LHS DOE", "fluent_setup"),
                "bayesian_ei": (2, "GP-EI OPT", "fluent_solve"),
                "verification": (3, "VERIFYING", "verify"),
            }
            stage_idx, badge, phase_k = phase_map.get(
                self.current_sim_phase,
                (1, "OPTIMIZING", "fluent_solve"),
            )
            stage_idx = min(stage_idx, max(0, len(self.stage_labels) - 1))
            raw = max(0.02, (self.current_sim - 1) / max(1, self.total_sims))
            self._max_fraction = max(self._max_fraction, min(0.95, raw))
            return ProgressSnapshot(
                phase_key=phase_k,
                badge_text=badge,
                headline=f"Fin Opt Sim {self.current_sim}/{self.total_sims} [{badge}]",
                detail=cand_desc[:140],
                fraction=self._max_fraction,
                active_stage_index=stage_idx,
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=f"[INFO] {msg}",
            )

        if "complete" in low or float(frac) >= 0.999:
            return self.complete_snapshot(
                headline="Fin Shape Optimization Complete",
                detail=msg,
            )

        self._max_fraction = max(self._max_fraction, min(0.99, float(frac)))
        return ProgressSnapshot(
            phase_key="fluent_post",
            badge_text="EXPORTING",
            headline="Exporting Optimized .ork, CAD & Report",
            detail=msg[:140],
            fraction=self._max_fraction,
            active_stage_index=max(0, len(self.stage_labels) - 1),
            stage_labels=self.stage_labels,
            is_busy=True,
            should_log=True,
            log_line=f"[INFO] {msg}",
        )

    def _mesh_stats_suffix(self) -> str:
        parts: list[str] = []
        if self.mesh_cells is not None:
            parts.append(f"{self.mesh_cells:,} cells")
        if self.min_orth_quality is not None:
            parts.append(f"min orth={self.min_orth_quality:.3f}")
        return f" | {', '.join(parts)}" if parts else ""

    def on_pipeline_progress(self, msg: str, frac: float) -> ProgressSnapshot:
        """Update snapshot from run_pipeline(msg, frac) callback."""
        low = msg.lower()
        clamped = min(1.0, max(0.0, float(frac)))
        if self.mode == "package_and_cfd":
            mapped = min(_PACKAGE_END_FRAC, (clamped / 0.90) * _PACKAGE_END_FRAC)
        else:
            mapped = clamped
        self._max_fraction = max(self._max_fraction, mapped)

        if "parsing" in low:
            phase, badge = "parse", "PARSING"
            stage_idx = self._stage_index("parse", 0)
        elif (
            "atmosphere" in low
            or "watertight" in low
            or "rocket-only" in low
            or "step" in low
            or "stl" in low
        ):
            phase, badge = "cad", "3D CAD"
            stage_idx = self._stage_index("cad", 1)
        elif "verif" in low or "viewer" in low:
            phase, badge = "verify", "VERIFYING"
            stage_idx = self._stage_index("verify", 1)
        elif "generating ansys" in low or "script" in low:
            phase, badge = "scripts", "SCRIPTS"
            stage_idx = self._stage_index("scripts", 1)
        elif "workbench" in low:
            phase, badge = "workbench", "WORKBENCH"
            stage_idx = self._stage_index("workbench", 1)
        elif "fluent" in low or "mesh convergence" in low:
            phase, badge = "fluent_launch", "FLUENT LAUNCH"
            stage_idx = 2 if self.mode == "package_and_cfd" else 0
        elif "complete" in low or clamped >= 0.999:
            return self.complete_snapshot(
                headline=msg,
                detail="All requested files generated cleanly.",
            )
        else:
            phase, badge = "cad", "WORKING"
            stage_idx = min(1, max(0, len(self.stage_labels) - 1))

        return ProgressSnapshot(
            phase_key=phase,
            badge_text=badge,
            headline=msg,
            detail=f"Stage {stage_idx + 1} of {len(self.stage_labels)} in progress...",
            fraction=self._max_fraction,
            active_stage_index=stage_idx,
            stage_labels=self.stage_labels,
            is_busy=True,
            is_indeterminate_substep=("workbench" in low or "fluent" in low),
            should_log=True,
            log_line=f"[INFO] {msg}",
        )

    def on_fluent_line(self, raw_line: str) -> ProgressSnapshot | None:
        """Parse one stripped line from Fluent/PyFluent stdout and update snapshot if relevant."""
        line = raw_line.strip()
        if not line:
            return None
        # Fluent echoes Python script source lines with leading >>> or ... during `-py -i`
        if line.startswith(">>>") or line.startswith("..."):
            return None

        lvl_match = _CONV_LEVEL_RE.search(line)
        if lvl_match:
            self.current_level = int(lvl_match.group(1))
            self.total_levels = max(self.total_levels, int(lvl_match.group(2)))
            try:
                self.current_scale = float(lvl_match.group(3)) if lvl_match.group(3) else None
            except ValueError:
                self.current_scale = None
            self.mesh_cells = None
            self.min_orth_quality = None
            self._saw_cd_col = False
            scale_txt = f" (mesh_scale={self.current_scale:.3f})" if self.current_scale else ""
            frac = self._map_cfd_fraction(0.02)
            return ProgressSnapshot(
                phase_key="fluent_launch",
                badge_text="MESH STUDY",
                headline=(
                    f"Mesh Convergence Level {self.current_level}/{self.total_levels}{scale_txt}"
                ),
                detail="Starting Fluent Meshing session for next refinement level...",
                fraction=frac,
                active_stage_index=0 if self.mode == "cfd_only" else 2,
                stage_labels=self.stage_labels,
                is_busy=True,
                is_indeterminate_substep=True,
                should_log=True,
                log_line=f"[FLUENT] {line}",
            )

        sum_match = _CONV_SUMMARY_RE.search(line)
        if sum_match:
            try:
                lvl_num = int(sum_match.group(1))
                cells_s = sum_match.group(2).strip()
                cd_val = float(sum_match.group(3))
            except ValueError:
                return None
            self.last_cd = cd_val
            chg_s = f", ΔCd={sum_match.group(4)}" if sum_match.group(4) else ""
            frac = self._map_cfd_fraction(0.99)
            return ProgressSnapshot(
                phase_key="fluent_post",
                badge_text="LEVEL DONE",
                headline=f"Mesh Study L{lvl_num}/{self.total_levels}: Cd = {cd_val:.5f}{chg_s}",
                detail=f"Cells: {cells_s}{chg_s}",
                fraction=frac,
                active_stage_index=len(self.stage_labels) - 1,
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=f"[FLUENT] {line}",
            )

        if "iter" in line.lower() and "continuity" in line.lower():
            self._saw_cd_col = "cd_total" in line.lower()
            return None

        iter_match = _ITER_ROW_RE.match(line)
        if iter_match:
            cur_iter = int(iter_match.group(1))
            num_tokens = _FLOAT_TOKEN_RE.findall(iter_match.group(2))
            eta_str = iter_match.group(3)
            rem_iters = int(iter_match.group(4))
            total_iters = cur_iter + rem_iters if (cur_iter + rem_iters) > 0 else self.max_iters
            cont = float(num_tokens[0])
            cd_conv: float | None = None
            if len(num_tokens) >= 8 or (self._saw_cd_col and len(num_tokens) >= 7):
                cd_conv = float(num_tokens[-1])

            sub_frac = 0.44 + 0.50 * min(1.0, max(0.0, cur_iter / max(1, total_iters)))
            frac = self._map_cfd_fraction(sub_frac)
            detail_parts = [f"Continuity {cont:.2e}"]
            if cd_conv is not None:
                detail_parts.append(f"cd_conv {cd_conv:.2e}")
            detail_parts.append(f"Fluent ETA {eta_str}")
            if self.mesh_cells is not None:
                detail_parts.append(f"{self.mesh_cells:,} cells")

            should_log = cur_iter == 1 or cur_iter % 10 == 0 or rem_iters == 0
            log_msg = (
                f"[FLUENT] {self._level_prefix()}Iter {cur_iter}/{total_iters} | "
                f"{' | '.join(detail_parts[:3])}"
            )
            return ProgressSnapshot(
                phase_key="fluent_solve",
                badge_text="SOLVING",
                headline=f"{self._level_prefix()}Fluent Solving — Iter {cur_iter} / {total_iters}",
                detail=" | ".join(detail_parts),
                fraction=frac,
                active_stage_index=self._stage_index("solve", 4),
                stage_labels=self.stage_labels,
                is_busy=True,
                is_indeterminate_substep=False,
                should_log=should_log,
                log_line=log_msg if should_log else None,
            )

        cells_m = _CELLS_CREATED_RE.search(line) or _VOL_CELLS_RE.search(line)
        if cells_m:
            self.mesh_cells = int(cells_m.group(1))
            frac = self._map_cfd_fraction(0.32)
            return ProgressSnapshot(
                phase_key="fluent_volume_mesh",
                badge_text="VOLUME MESH",
                headline=f"{self._level_prefix()}Volume Mesh Built ({self.mesh_cells:,} cells)",
                detail=f"Checking orthogonal quality and writing mesh{self._mesh_stats_suffix()}",
                fraction=frac,
                active_stage_index=self._stage_index("volume mesh", 3),
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=f"[FLUENT] {line}",
            )

        orth_m = _ORTH_QUALITY_RE.search(line)
        if orth_m:
            try:
                self.min_orth_quality = float(orth_m.group(1))
            except ValueError:
                return None
            low_orth = self.min_orth_quality < 0.05
            frac = self._map_cfd_fraction(0.35)
            log_line = (
                f"[WARN] Low mesh orthogonal quality (min orth={self.min_orth_quality:.4f} "
                "< 0.05) — thin prisms on sharp edges may cause solver divergence."
                if low_orth
                else f"[FLUENT] {line}"
            )
            return ProgressSnapshot(
                phase_key="fluent_volume_mesh",
                badge_text="LOW QUALITY" if low_orth else "VOLUME MESH",
                headline=(
                    f"{self._level_prefix()}Mesh Quality Checked "
                    f"(min orth={self.min_orth_quality:.4f})"
                ),
                detail=f"Ready to switch to Fluent Solver{self._mesh_stats_suffix()}",
                fraction=frac,
                active_stage_index=self._stage_index("volume mesh", 3),
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=log_line,
            )

        yp_m = _YPLUS_RE.search(line)
        if yp_m:
            try:
                self.wall_yplus_avg = float(yp_m.group(1))
            except ValueError:
                return None
            frac = self._map_cfd_fraction(0.97)
            return ProgressSnapshot(
                phase_key="fluent_post",
                badge_text="EXPORTING",
                headline=(
                    f"{self._level_prefix()}Extracting Drag & Wall y+ "
                    f"({self.wall_yplus_avg:.2f})"
                ),
                detail=f"Saving CSV/JSON drag breakdown{self._mesh_stats_suffix()}",
                fraction=frac,
                active_stage_index=self._stage_index("results", len(self.stage_labels) - 1),
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=f"[FLUENT] {line}",
            )

        mach_m = _MACH_REPORTS_RE.search(line)
        if mach_m:
            self.last_mach = float(mach_m.group(1))
            frac = self._map_cfd_fraction(0.96)
            return ProgressSnapshot(
                phase_key="fluent_post",
                badge_text="EXPORTING",
                headline=f"{self._level_prefix()}Computing Drag Forces (Mach {self.last_mach:.2f})",
                detail=f"Evaluating per-zone Cd reports{self._mesh_stats_suffix()}",
                fraction=frac,
                active_stage_index=self._stage_index("results", len(self.stage_labels) - 1),
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=f"[FLUENT] {line}",
            )

        # Ordered milestone table
        milestones: tuple[tuple[tuple[str, ...], str, str, str, str, float, str], ...] = (
            (
                ("Launching ANSYS Fluent", "Imported CAD objects"),
                "fluent_launch",
                "FLUENT LAUNCH",
                "Fluent Launching & Importing CAD",
                "Reading STEP geometry and reconstructing Named Selections...",
                0.05,
                "launch",
            ),
            (
                ("Surface Meshing", "Generate the Surface Mesh", "Import complete"),
                "fluent_surface_mesh",
                "SURFACE MESH",
                "Fluent Surface Meshing",
                "Tessellating rocket OML, fins, nosecone, and farfield faces...",
                0.11,
                "surface mesh",
            ),
            (
                (
                    "Separated remeshed zones",
                    "Verified labels on remeshed",
                    "Update Boundaries",
                    "Update Regions",
                ),
                "fluent_surface_mesh",
                "SURFACE MESH",
                "Labeling Boundaries & Fluid Regions",
                "Assigning pressure-far-field, wall, symmetry, and dead void zones...",
                0.18,
                "surface mesh",
            ),
            (
                ("Boundary layers scoped", "auto meshing object", "Generating Prisms"),
                "fluent_volume_mesh",
                "VOLUME MESH",
                "Fluent Volume Meshing & Prisms",
                "Growing y+ inflation layers and poly-hexcore volume mesh...",
                0.26,
                "volume mesh",
            ),
            (
                ("Saved volume mesh",),
                "fluent_volume_mesh",
                "VOLUME MESH",
                "Volume Mesh Saved — Switching to Solver",
                f"Opening Fluent 3D Pressure/Density Solver{self._mesh_stats_suffix()}",
                0.36,
                "volume mesh",
            ),
            (
                (
                    "Active solver boundary zones",
                    "Configured cd_total convergence",
                    "Hybrid initialization",
                ),
                "fluent_setup",
                "SOLVER SETUP",
                "Solver Setup & Hybrid Initialization",
                f"Initializing flow field and drag convergence monitors{self._mesh_stats_suffix()}",
                0.42,
                "setup",
            ),
            (
                ("solution is converged", "convergence condition"),
                "fluent_post",
                "CONVERGED",
                "Cd Stabilized — Converged Early",
                "Writing solved .cas.h5 / .dat.h5 files and drag reports...",
                0.95,
                "results",
            ),
            (
                ("Solved and saved case/data", "Wrote aero_results.csv"),
                "fluent_post",
                "EXPORTING",
                "Finalizing CFD Case & Drag Outputs",
                "Saved .cas.h5, .dat.h5, aero_results.csv, and cfd_summary.json.",
                0.98,
                "results",
            ),
        )
        for tokens, phase_k, badge_t, head_t, det_t, sub_f, stage_kw in milestones:
            if any(tok in line for tok in tokens):
                frac = self._map_cfd_fraction(sub_f)
                fallback_idx = 2 if (self.mode == "package_and_cfd" and stage_kw == "launch") else 0
                stage_idx = self._stage_index(stage_kw, fallback_idx)
                return ProgressSnapshot(
                    phase_key=phase_k,
                    badge_text=badge_t,
                    headline=f"{self._level_prefix()}{head_t}",
                    detail=det_t,
                    fraction=frac,
                    active_stage_index=stage_idx,
                    stage_labels=self.stage_labels,
                    is_busy=True,
                    is_indeterminate_substep=(phase_k in {"fluent_launch", "fluent_volume_mesh"}),
                    should_log=True,
                    log_line=f"[FLUENT] {line}",
                )

        if "Error:" in line or "[WARN]" in line or "retrying" in line.lower():
            return ProgressSnapshot(
                phase_key="fluent_setup",
                badge_text="NOTICE",
                headline=f"{self._level_prefix()}Fluent Notice / Warning",
                detail=line[:140],
                fraction=self._max_fraction,
                active_stage_index=min(2, max(0, len(self.stage_labels) - 1)),
                stage_labels=self.stage_labels,
                is_busy=True,
                should_log=True,
                log_line=f"[FLUENT] {line}",
            )

        return None
