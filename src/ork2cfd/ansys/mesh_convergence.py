"""Progressive coarse-to-fine CFD mesh-convergence (grid-independence) engine."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import math
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from ..exceptions import AnsysError, Ork2CfdError
from ..models.cfd import MeshConvergenceConfig, MeshPreset
from ..models.results import (
    CfdSolveResult,
    MeshConvergenceReport,
    MeshRefinementLevel,
)
from ..physics.boundary_layer import _PRESET_FACTORS
from .detector import AnsysInstallation
from .fluent_launcher import run_fluent_solve_blocking

_MESH_PRESET_RE = re.compile(r'MESH_PRESET\s*=\s*"([^"]+)"')
_SURF_MAX_RE = re.compile(r"^\s*SURF_MAX_M\s*=\s*([0-9eE.+_-]+)", re.MULTILINE)
_SURF_MIN_RE = re.compile(r"^\s*SURF_MIN_M\s*=\s*([0-9eE.+_-]+)", re.MULTILINE)


def is_valid_aerodynamic_cd(
    total_cd: float,
    wall_yplus_avg: float | None = None,
    min_cd: float = 0.02,
    max_cd: float = 5.0,
) -> tuple[bool, str]:
    """Validate that solved Cd and wall y+ are finite and physically plausible."""
    if not math.isfinite(total_cd):
        return False, f"non-finite Cd={total_cd}"
    if total_cd <= min_cd:
        return False, f"unphysical low/negative Cd={total_cd:.4f} (<= {min_cd:.2f})"
    if total_cd > max_cd:
        return False, f"diverged high Cd={total_cd:.4f} (> {max_cd:.2f})"
    if wall_yplus_avg is not None:
        if not math.isfinite(wall_yplus_avg) or wall_yplus_avg <= 0.0:
            return False, f"invalid wall y+={wall_yplus_avg}"
        if wall_yplus_avg > 2000.0:
            return False, f"diverged wall y+={wall_yplus_avg:.1f} (> 2000)"
    return True, "ok"


def compute_mesh_convergence_scales(
    base_preset: MeshPreset,
    conv_config: MeshConvergenceConfig,
) -> list[float]:
    """Build a coarse-to-fine geometric element scale schedule anchored at MeshPreset.COARSE."""
    if conv_config.max_levels < 2:
        raise Ork2CfdError(
            f"Mesh convergence `max_levels` must be >= 2 (got {conv_config.max_levels})."
        )
    if conv_config.refinement_factor <= 1.05:
        raise Ork2CfdError(
            "Mesh convergence `refinement_factor` must be > 1.05 "
            f"(got {conv_config.refinement_factor})."
        )
    if conv_config.cd_rel_tol <= 0.0:
        raise Ork2CfdError(
            f"Mesh convergence `cd_rel_tol` must be > 0 (got {conv_config.cd_rel_tol})."
        )

    coarse_f_max = _PRESET_FACTORS[MeshPreset.COARSE][1]
    base_f_max = _PRESET_FACTORS[base_preset][1]
    start_scale = max(1.0, coarse_f_max / base_f_max)
    return [
        round(start_scale / (conv_config.refinement_factor**idx), 6)
        for idx in range(conv_config.max_levels)
    ]


def _effective_ratio(
    fine: MeshRefinementLevel,
    coarse: MeshRefinementLevel,
    fallback_factor: float,
) -> float:
    """Compute effective 3D linear grid refinement ratio r = h_coarse / h_fine."""
    if (
        fine.mesh_cells is not None
        and coarse.mesh_cells is not None
        and coarse.mesh_cells > 0
        and fine.mesh_cells > int(coarse.mesh_cells * 1.05)
    ):
        return max(1.05, (fine.mesh_cells / coarse.mesh_cells) ** (1.0 / 3.0))
    if fine.mesh_scale > 0.0 and coarse.mesh_scale > fine.mesh_scale * 1.01:
        return max(1.05, coarse.mesh_scale / fine.mesh_scale)
    return max(1.05, float(fallback_factor))


def compute_gci_metrics(
    levels: Sequence[MeshRefinementLevel],
    refinement_factor: float,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute Roache Grid Convergence Index (GCI_fine %), Richardson Cd, and observed order p.

    Uses 3-grid monotonic order extraction (Fs = 1.25, p clamped to [0.5, 4.0]) when 3+
    monotonically converging levels are available, falling back to formal p = 2.0 (with
    Fs = 3.0 for 2 grids or Fs = 1.25 for 3+ non-monotonic grids).
    """
    if len(levels) < 2:
        return None, None, None

    fine = levels[-1]
    med = levels[-2]
    f1 = float(fine.total_cd)
    f2 = float(med.total_cd)
    if abs(f1) < 1e-12:
        return None, None, None

    r21 = _effective_ratio(fine, med, refinement_factor)
    eps21 = f2 - f1

    if len(levels) >= 3:
        coarse = levels[-3]
        f3 = float(coarse.total_cd)
        r32 = _effective_ratio(med, coarse, refinement_factor)
        eps32 = f3 - f2
        fs = 1.25

        if abs(eps21) <= 1e-12:
            return 0.0, round(f1, 6), 2.0

        if (eps21 * eps32) > 0.0 and abs(eps32) > abs(eps21) * 1.01:
            p = 2.0
            ln_r21 = math.log(r21)
            ratio_term = math.log(abs(eps32 / eps21))
            for _ in range(25):
                q = (
                    math.log(max(1e-9, (r21**p - 1.0) / max(r32**p - 1.0, 1e-9)))
                    if abs(r21 - r32) > 1e-4
                    else 0.0
                )
                p_next = max(0.5, min(4.0, abs(ratio_term + q) / ln_r21))
                if abs(p_next - p) < 1e-6:
                    p = p_next
                    break
                p = p_next
        else:
            p = 2.0
    else:
        fs = 3.0
        p = 2.0

    e_a = abs(f1 - f2) / max(abs(f1), 1e-9)
    denom = max(r21**p - 1.0, 1e-6)
    gci_fine_pct = fs * (e_a / denom) * 100.0
    richardson_cd = f1 + (f1 - f2) / denom
    return round(gci_fine_pct, 4), round(richardson_cd, 6), round(p, 4)


def mesh_convergence_to_dict(report: MeshConvergenceReport) -> dict[str, object]:
    """Serialize a MeshConvergenceReport to JSON-safe dictionary with relative file names."""
    return {
        "converged": report.converged,
        "cd_rel_tol": report.cd_rel_tol,
        "refinement_factor": report.refinement_factor,
        "optimal_level": report.optimal_level,
        "finest_level": report.finest_level,
        "gci_fine_pct": report.gci_fine_pct,
        "richardson_cd": report.richardson_cd,
        "observed_order_p": report.observed_order_p,
        "stop_reason": report.stop_reason,
        "report_json": report.report_json.name if report.report_json else None,
        "report_csv": report.report_csv.name if report.report_csv else None,
        "report_md": report.report_md.name if report.report_md else None,
        "plot_png": report.plot_png.name if report.plot_png else None,
        "levels": [
            {
                "level": lv.level,
                "mesh_scale": lv.mesh_scale,
                "surf_max_m": lv.surf_max_m,
                "mesh_cells": lv.mesh_cells,
                "min_orth_quality": lv.min_orth_quality,
                "total_cd": lv.total_cd,
                "total_drag_n": lv.total_drag_n,
                "wall_yplus_avg": lv.wall_yplus_avg,
                "rel_cd_change": lv.rel_cd_change,
            }
            for lv in report.levels
        ],
    }


def _resolve_optional_file(out_dir: Path, raw_path: object) -> Path | None:
    if not raw_path:
        return None
    cand = Path(str(raw_path))
    if not cand.is_absolute():
        cand = out_dir / cand
    return cand if cand.exists() else None


def parse_mesh_convergence_dict(
    raw: dict | None,
    out_dir: Path,
) -> MeshConvergenceReport | None:
    """Deserialize a MeshConvergenceReport dictionary if valid."""
    if not isinstance(raw, dict):
        return None
    raw_levels = raw.get("levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        return None

    levels: list[MeshRefinementLevel] = []
    for item in raw_levels:
        if not isinstance(item, dict) or "level" not in item or "total_cd" not in item:
            continue
        cells_raw = item.get("mesh_cells")
        orth_raw = item.get("min_orth_quality")
        yp_raw = item.get("wall_yplus_avg")
        rel_raw = item.get("rel_cd_change")
        levels.append(
            MeshRefinementLevel(
                level=int(item["level"]),
                mesh_scale=float(item.get("mesh_scale", 1.0)),
                surf_max_m=float(item.get("surf_max_m", 0.0)),
                mesh_cells=int(cells_raw) if cells_raw is not None else None,
                min_orth_quality=float(orth_raw) if orth_raw is not None else None,
                total_cd=float(item["total_cd"]),
                total_drag_n=float(item.get("total_drag_n", 0.0)),
                wall_yplus_avg=float(yp_raw) if yp_raw is not None else None,
                rel_cd_change=float(rel_raw) if rel_raw is not None else None,
            )
        )
    if not levels:
        return None

    gci_raw = raw.get("gci_fine_pct")
    rich_raw = raw.get("richardson_cd")
    p_raw = raw.get("observed_order_p")
    return MeshConvergenceReport(
        converged=bool(raw.get("converged", False)),
        cd_rel_tol=float(raw.get("cd_rel_tol", 0.01)),
        refinement_factor=float(raw.get("refinement_factor", 1.35)),
        levels=tuple(levels),
        optimal_level=int(raw.get("optimal_level", levels[-1].level)),
        finest_level=int(raw.get("finest_level", levels[-1].level)),
        gci_fine_pct=float(gci_raw) if gci_raw is not None else None,
        richardson_cd=float(rich_raw) if rich_raw is not None else None,
        observed_order_p=float(p_raw) if p_raw is not None else None,
        stop_reason=str(raw.get("stop_reason", "converged")),
        report_json=_resolve_optional_file(out_dir, raw.get("report_json")),
        report_csv=_resolve_optional_file(out_dir, raw.get("report_csv")),
        report_md=_resolve_optional_file(out_dir, raw.get("report_md")),
        plot_png=_resolve_optional_file(out_dir, raw.get("plot_png")),
    )


def render_mesh_convergence_plot(
    report: MeshConvergenceReport,
    out_png: Path,
    rocket_name: str = "Rocket",
    mach: float | None = None,
) -> Path | None:
    """Render an offscreen Cd vs mesh resolution convergence chart."""
    if not report.levels:
        return None

    levels = report.levels
    has_cells = all(lv.mesh_cells is not None and lv.mesh_cells > 0 for lv in levels)
    x_vals = [
        float(lv.mesh_cells) if has_cells else (1.0 / max(lv.mesh_scale, 1e-6))
        for lv in levels
    ]
    y_vals = [float(lv.total_cd) for lv in levels]

    fig = plt.Figure(figsize=(8.0, 4.8), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.plot(x_vals, y_vals, color="#0284c7", linewidth=2.0, zorder=2, label="Total $C_D$")
    ax.scatter(x_vals, y_vals, color="#0369a1", s=55, zorder=3)

    fine_cd = y_vals[-1]
    band = abs(fine_cd) * report.cd_rel_tol
    ax.axhspan(
        fine_cd - band,
        fine_cd + band,
        color="#22c55e",
        alpha=0.14,
        zorder=1,
        label=f"±{report.cd_rel_tol * 100.0:.2f}% Tolerance Band",
    )

    if report.richardson_cd is not None:
        ax.axhline(
            report.richardson_cd,
            color="#f59e0b",
            linestyle="--",
            linewidth=1.3,
            zorder=1,
            label=f"Richardson Extrap. ($C_D={report.richardson_cd:.5f}$)",
        )

    opt_lv = next((lv for lv in levels if lv.level == report.optimal_level), None)
    if opt_lv is not None:
        idx_opt = levels.index(opt_lv)
        ax.scatter(
            [x_vals[idx_opt]],
            [y_vals[idx_opt]],
            color="#16a34a",
            marker="*",
            s=190,
            zorder=5,
            label=f"Optimal (L{opt_lv.level})",
        )

    ax.scatter(
        [x_vals[-1]],
        [y_vals[-1]],
        color="#dc2626",
        marker="D",
        s=75,
        zorder=4,
        label=f"Finest (L{levels[-1].level})",
    )

    for xv, yv, lv in zip(x_vals, y_vals, levels, strict=True):
        pct_txt = f" ({lv.rel_cd_change * 100.0:.2f}%)" if lv.rel_cd_change is not None else ""
        ax.annotate(
            f"L{lv.level}{pct_txt}",
            xy=(xv, yv),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1e293b",
        )

    mach_txt = f" | Mach {mach:.2f}" if mach is not None else ""
    status_txt = "CONVERGED" if report.converged else report.stop_reason.replace("_", " ").upper()
    gci_txt = f" | GCI={report.gci_fine_pct:.2f}%" if report.gci_fine_pct is not None else ""
    ax.set_title(
        f"Mesh Convergence Study — {rocket_name}{mach_txt}\n[{status_txt}{gci_txt}]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Mesh Cell Count" if has_cells else "Relative Linear Density (1 / mesh_scale)")
    ax.set_ylabel("Total Drag Coefficient ($C_D$)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def write_mesh_convergence_outputs(
    output_dir: Path,
    report: MeshConvergenceReport,
    rocket_name: str,
    mach: float | None = None,
    base_preset: MeshPreset | None = None,
    target_yplus: float | None = None,
) -> MeshConvergenceReport:
    """Persist mesh_convergence.{json,csv,md}, convergence plot PNG, and update cfd_summary.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "mesh_convergence.json"
    csv_path = out_dir / "mesh_convergence.csv"
    md_path = out_dir / "mesh_convergence.md"
    png_path = out_dir / "mesh_convergence_plot.png"

    rendered_png: Path | None = None
    with contextlib.suppress(Exception):
        rendered_png = render_mesh_convergence_plot(
            report, png_path, rocket_name=rocket_name, mach=mach
        )

    updated = dataclasses.replace(
        report,
        report_json=json_path,
        report_csv=csv_path,
        report_md=md_path,
        plot_png=rendered_png if (rendered_png and rendered_png.exists()) else None,
    )

    payload = mesh_convergence_to_dict(updated)
    if base_preset is not None:
        payload["base_preset"] = base_preset.value
    if target_yplus is not None:
        payload["target_yplus"] = round(float(target_yplus), 4)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "level",
            "mesh_scale",
            "surf_max_m",
            "mesh_cells",
            "min_orth_quality",
            "wall_yplus_avg",
            "total_cd",
            "total_drag_n",
            "rel_cd_change_pct",
            "role",
        ])
        for lv in updated.levels:
            roles: list[str] = []
            if lv.level == updated.optimal_level:
                roles.append("optimal")
            if lv.level == updated.finest_level:
                roles.append("finest")
            if not roles and lv.level == 1:
                roles.append("coarse_baseline")
            writer.writerow([
                lv.level,
                f"{lv.mesh_scale:.6f}",
                f"{lv.surf_max_m:.6f}",
                lv.mesh_cells if lv.mesh_cells is not None else "",
                f"{lv.min_orth_quality:.5f}" if lv.min_orth_quality is not None else "",
                f"{lv.wall_yplus_avg:.4f}" if lv.wall_yplus_avg is not None else "",
                f"{lv.total_cd:.6f}",
                f"{lv.total_drag_n:.6f}",
                f"{lv.rel_cd_change * 100.0:.4f}" if lv.rel_cd_change is not None else "",
                "+".join(roles) or "intermediate",
            ])

    mach_hdr = f" (Mach {mach:.2f})" if mach is not None else ""
    conv_lbl = "YES" if updated.converged else "NO"
    lines = [
        f"# Mesh Convergence Study — {rocket_name}{mach_hdr}",
        "",
        f"- **Converged:** `{conv_lbl}` (`stop_reason={updated.stop_reason}`)",
        f"- **Tolerance Threshold:** `|ΔCd| / Cd <= {updated.cd_rel_tol * 100.0:.2f}%`",
        f"- **Refinement Factor:** `r = {updated.refinement_factor:.2f}x`",
        (
            f"- **Optimal Mesh Level:** `Level {updated.optimal_level}` "
            f"(verification finest: `Level {updated.finest_level}`)"
        ),
    ]
    if updated.gci_fine_pct is not None:
        lines.append(f"- **Roache Fine-Grid GCI:** `{updated.gci_fine_pct:.3f}%`")
    if updated.richardson_cd is not None:
        lines.append(f"- **Richardson Extrapolated $C_D$:** `{updated.richardson_cd:.6f}`")
    if updated.observed_order_p is not None:
        lines.append(f"- **Observed Order of Accuracy ($p$):** `{updated.observed_order_p:.3f}`")
    lines.extend([
        "",
        (
            "| Level | Scale | Surf Max (mm) | Mesh Cells | Wall $y^+$ | Total $C_D$ "
            "| Drag (N) | $|\\Delta C_D|/C_D$ | Role |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ])
    for lv in updated.levels:
        cells_s = f"{lv.mesh_cells:,}" if lv.mesh_cells is not None else "—"
        yp_s = f"{lv.wall_yplus_avg:.2f}" if lv.wall_yplus_avg is not None else "—"
        chg_s = f"{lv.rel_cd_change * 100.0:.3f}%" if lv.rel_cd_change is not None else "—"
        role_tags: list[str] = []
        if lv.level == updated.optimal_level:
            role_tags.append("Optimal")
        if lv.level == updated.finest_level:
            role_tags.append("Finest")
        role_s = " / ".join(role_tags) if role_tags else ("Baseline" if lv.level == 1 else "—")
        lines.append(
            f"| {lv.level} | `{lv.mesh_scale:.4f}` | `{lv.surf_max_m * 1000.0:.2f}` | "
            f"{cells_s} | {yp_s} | `{lv.total_cd:.6f}` | `{lv.total_drag_n:.2f}` | "
            f"{chg_s} | {role_s} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = out_dir / "cfd_summary.json"
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary_data, dict):
                summary_data["mesh_convergence"] = payload
                summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

    return updated


def _read_script_mesh_metadata(
    out_dir: Path,
    base_preset: MeshPreset | None,
    surf_max_base_m: float | None,
) -> tuple[MeshPreset, float, float]:
    """Infer base MeshPreset, SURF_MAX_M, and SURF_MIN_M from generated run_fluent.py."""
    preset = base_preset or MeshPreset.STANDARD
    surf_max = surf_max_base_m if surf_max_base_m is not None else 0.0
    surf_min = 0.0002

    run_py = out_dir / "run_fluent.py"
    if run_py.exists():
        text = run_py.read_text(encoding="utf-8", errors="replace")
        if base_preset is None:
            m_preset = _MESH_PRESET_RE.search(text)
            if m_preset:
                with contextlib.suppress(ValueError):
                    preset = MeshPreset(m_preset.group(1))
        if surf_max_base_m is None:
            m_max = _SURF_MAX_RE.search(text)
            if m_max:
                with contextlib.suppress(ValueError):
                    surf_max = float(m_max.group(1))
        m_min = _SURF_MIN_RE.search(text)
        if m_min:
            with contextlib.suppress(ValueError):
                surf_min = float(m_min.group(1))
    return preset, surf_max, surf_min


def _effective_surf_max_m(surf_max_base: float, surf_min_base: float, scale: float) -> float:
    if surf_max_base <= 0.0:
        return 0.0
    if abs(scale - 1.0) <= 1e-9:
        return round(surf_max_base, 6)
    s_min = round(max(1e-4, surf_min_base * scale), 6)
    return round(max(s_min * 2.5, surf_max_base * scale), 6)


def _solved_artifact_paths(out_dir: Path) -> list[Path]:
    """Return per-run CFD output files in out_dir suitable for level archiving/restoration."""
    candidates: list[Path] = []
    for fixed in ("cfd_summary.json", "aero_results.csv", "rocketpy_drag.csv", "fluent_run.log"):
        p = out_dir / fixed
        if p.exists():
            candidates.append(p)
    for pattern in ("*.cas.h5", "*.dat.h5", "*_mach.png", "*_pressure.png"):
        for p in sorted(out_dir.glob(pattern)):
            if not p.name.startswith("_") and p.is_file():
                candidates.append(p)
    return candidates


def _archive_level_artifacts(out_dir: Path, level_dir: Path) -> None:
    level_dir.mkdir(parents=True, exist_ok=True)
    for src in _solved_artifact_paths(out_dir):
        shutil.copy2(src, level_dir / src.name)


def _restore_level_artifacts(level_dir: Path, out_dir: Path) -> None:
    for stale in _solved_artifact_paths(out_dir):
        with contextlib.suppress(OSError):
            stale.unlink()
    for saved in sorted(level_dir.iterdir()):
        if saved.is_file():
            shutil.copy2(saved, out_dir / saved.name)


def run_mesh_convergence_blocking(
    output_dir: Path,
    ansys_install: AnsysInstallation | None,
    conv_config: MeshConvergenceConfig | None = None,
    base_preset: MeshPreset | None = None,
    cores: int | None = None,
    mach_list: list[float] | None = None,
    surf_max_base_m: float | None = None,
    rocket_name: str | None = None,
    mach: float | None = None,
    target_yplus: float | None = None,
    log_callback: Callable[[str], None] | None = None,
    line_callback: Callable[[str], None] | None = None,
    mpi: str | None = None,
    solve_fn: Callable[..., CfdSolveResult] | None = None,
) -> CfdSolveResult:
    """
    Execute progressive coarse-to-fine Fluent solves until |ΔCd|/Cd stabilizes within tolerance.

    Each level spawns a fresh Fluent Meshing -> Solver session using `ORK2CFD_MESH_SCALE`,
    archives artifacts into `mesh_convergence/level_XX`, computes Roache GCI metrics, and
    gracefully restores the previous valid mesh if a finer level fails (e.g., 512k Student limit).
    """
    out_dir = Path(output_dir)
    cfg = conv_config or MeshConvergenceConfig()
    resolved_preset, base_smax, base_smin = _read_script_mesh_metadata(
        out_dir, base_preset, surf_max_base_m
    )
    scales = compute_mesh_convergence_scales(resolved_preset, cfg)

    def _emit(msg: str) -> None:
        if log_callback is not None:
            with contextlib.suppress(Exception):
                log_callback(msg)
        if line_callback is not None:
            with contextlib.suppress(Exception):
                line_callback(msg)

    active_mpi: list[str | None] = [mpi]

    def _invoke_solve(scale: float) -> CfdSolveResult:
        if solve_fn is not None:
            extra = {"line_callback": line_callback} if line_callback is not None else {}
            return solve_fn(
                output_dir=out_dir,
                ansys_install=ansys_install,
                cores=cores,
                mach_list=mach_list,
                log_callback=log_callback,
                mpi=active_mpi[0],
                mesh_scale=scale,
                **extra,
            )
        try:
            return run_fluent_solve_blocking(
                output_dir=out_dir,
                ansys_install=ansys_install,
                cores=cores,
                mach_list=mach_list,
                log_callback=log_callback,
                line_callback=line_callback,
                mpi=active_mpi[0],
                mesh_scale=scale,
            )
        except AnsysError as exc:
            if active_mpi[0] is not None:
                raise
            _emit(f"[ork2cfd] Default MPI failed ({exc}); retrying with -mpi=msmpi...")
            res_msmpi = run_fluent_solve_blocking(
                output_dir=out_dir,
                ansys_install=ansys_install,
                cores=cores,
                mach_list=mach_list,
                log_callback=log_callback,
                line_callback=line_callback,
                mpi="msmpi",
                mesh_scale=scale,
            )
            active_mpi[0] = "msmpi"
            return res_msmpi

    conv_root = out_dir / "mesh_convergence"
    conv_root.mkdir(parents=True, exist_ok=True)

    levels: list[MeshRefinementLevel] = []
    last_valid_result: CfdSolveResult | None = None
    prev_cd: float | None = None
    prev_cells: int | None = None
    converged = False
    stop_reason = "running"

    _emit(
        f"[ork2cfd] Starting progressive mesh convergence study: "
        f"{len(scales)} max levels, r={cfg.refinement_factor:.2f}x, "
        f"tol={cfg.cd_rel_tol * 100.0:.2f}%"
    )

    for idx, scale in enumerate(scales, start=1):
        _emit(f"[ork2cfd] Mesh Convergence Level {idx}/{len(scales)} (mesh_scale={scale:.4f})...")
        try:
            solved = _invoke_solve(scale)
        except AnsysError as exc:
            if idx == 1 or last_valid_result is None:
                raise
            failed_log = out_dir / "fluent_run.log"
            if failed_log.exists():
                shutil.copy2(failed_log, conv_root / f"level_{idx:02d}_failed.log")
            prev_dir = conv_root / f"level_{idx - 1:02d}"
            if prev_dir.exists():
                _restore_level_artifacts(prev_dir, out_dir)
            stop_reason = "finer_level_failed_recovered"
            _emit(
                f"[ork2cfd] [WARN] Level {idx} failed ({exc}); restored Level {idx - 1} solution."
            )
            break

        run0 = solved.primary_run
        if run0 is None:
            if idx == 1 or last_valid_result is None:
                raise AnsysError(
                    f"Mesh Convergence Level {idx} returned no Mach results in '{out_dir}'."
                )
            prev_dir = conv_root / f"level_{idx - 1:02d}"
            if prev_dir.exists():
                _restore_level_artifacts(prev_dir, out_dir)
            stop_reason = "finer_level_failed_recovered"
            _emit(
                f"[ork2cfd] [WARN] Level {idx} returned empty runs; restored Level {idx - 1}."
            )
            break

        curr_cd = float(run0.total_cd)
        curr_drag = float(run0.total_drag_n)
        curr_yplus = run0.wall_yplus_avg
        curr_cells = solved.mesh_cells
        curr_orth = solved.min_orth_quality

        ok_cd, bad_reason = is_valid_aerodynamic_cd(curr_cd, curr_yplus)
        if not ok_cd:
            if idx == 1 or last_valid_result is None:
                raise AnsysError(
                    f"Mesh Convergence Level {idx} diverged ({bad_reason}). "
                    "Check Mach number, y+, or mesh quality."
                )
            div_log = out_dir / "fluent_run.log"
            if div_log.exists():
                shutil.copy2(div_log, conv_root / f"level_{idx:02d}_diverged.log")
            prev_dir = conv_root / f"level_{idx - 1:02d}"
            if prev_dir.exists():
                _restore_level_artifacts(prev_dir, out_dir)
            stop_reason = "finer_level_diverged_recovered"
            _emit(
                f"[ork2cfd] [WARN] Level {idx} diverged ({bad_reason}); "
                f"restored Level {idx - 1} solution."
            )
            break

        rel_change = (
            abs(curr_cd - prev_cd) / max(abs(curr_cd), 1e-9)
            if prev_cd is not None
            else None
        )

        levels.append(
            MeshRefinementLevel(
                level=idx,
                mesh_scale=scale,
                surf_max_m=_effective_surf_max_m(base_smax, base_smin, scale),
                mesh_cells=curr_cells,
                min_orth_quality=curr_orth,
                total_cd=curr_cd,
                total_drag_n=curr_drag,
                wall_yplus_avg=curr_yplus,
                rel_cd_change=rel_change,
            )
        )
        last_valid_result = solved
        _archive_level_artifacts(out_dir, conv_root / f"level_{idx:02d}")

        cells_txt = f"{curr_cells:,}" if curr_cells is not None else "unknown"
        chg_txt = f", ΔCd={rel_change * 100.0:.3f}%" if rel_change is not None else ""
        _emit(f"[ork2cfd] Level {idx}: cells={cells_txt}, Cd={curr_cd:.6f}{chg_txt}")

        if rel_change is not None and rel_change <= cfg.cd_rel_tol:
            converged = True
            stop_reason = "converged"
            _emit(
                f"[ork2cfd] Mesh converged at Level {idx} "
                f"(|ΔCd|/Cd={rel_change * 100.0:.3f}% <= {cfg.cd_rel_tol * 100.0:.2f}%)."
            )
            break

        if cfg.max_cells is not None and curr_cells is not None and curr_cells >= cfg.max_cells:
            stop_reason = "max_cells_reached"
            _emit(
                f"[ork2cfd] Stopping at Level {idx}: mesh_cells={curr_cells:,} "
                f"reached cap ({cfg.max_cells:,})."
            )
            break

        if (
            idx < len(scales)
            and cfg.max_cells is not None
            and curr_cells is not None
            and curr_cells > int(cfg.max_cells * 0.65)
        ):
            stop_reason = "ram_cell_cap_reached"
            _emit(
                f"[ork2cfd] Stopping at Level {idx}: refining {curr_cells:,} cells by "
                f"{cfg.refinement_factor:.2f}x would exceed safe RAM cell cap "
                f"({cfg.max_cells:,})."
            )
            break

        if (
            prev_cells is not None
            and curr_cells is not None
            and prev_cells > 0
            and curr_cells <= int(prev_cells * 1.05)
        ):
            stop_reason = "cell_count_saturated"
            _emit(
                f"[ork2cfd] Stopping at Level {idx}: cell count saturated "
                f"({prev_cells:,} -> {curr_cells:,})."
            )
            break

        prev_cd = curr_cd
        prev_cells = curr_cells

    if stop_reason == "running":
        stop_reason = "max_levels_reached"

    assert last_valid_result is not None
    optimal_level = (len(levels) - 1) if (converged and len(levels) >= 2) else len(levels)
    finest_level = len(levels)
    gci_pct, richardson_cd, obs_p = compute_gci_metrics(levels, cfg.refinement_factor)

    resolved_rocket = rocket_name or last_valid_result.rocket_name
    resolved_mach = (
        mach
        if mach is not None
        else (last_valid_result.primary_run.mach if last_valid_result.primary_run else None)
    )

    report = MeshConvergenceReport(
        converged=converged,
        cd_rel_tol=cfg.cd_rel_tol,
        refinement_factor=cfg.refinement_factor,
        levels=tuple(levels),
        optimal_level=optimal_level,
        finest_level=finest_level,
        gci_fine_pct=gci_pct,
        richardson_cd=richardson_cd,
        observed_order_p=obs_p,
        stop_reason=stop_reason,
    )
    final_report = write_mesh_convergence_outputs(
        out_dir,
        report,
        rocket_name=resolved_rocket,
        mach=resolved_mach,
        base_preset=resolved_preset,
        target_yplus=target_yplus,
    )
    return dataclasses.replace(last_valid_result, mesh_convergence=final_report)
