"""Local ANSYS Fluent & CFD-Post runners, result parsers, and visualization launchers."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..exceptions import AnsysError
from ..models.cfd import default_fluent_cores
from ..models.results import CfdMachPoint, CfdSolveResult, CfdZoneResult
from .detector import AnsysInstallation

_MILESTONE_TOKENS = (
    "[ork2cfd]",
    "Import complete",
    "Surface Meshing",
    "auto meshing object",
    "Generating Prisms",
    "Volume mesh",
    "cells were created",
    "Orthogonal Quality",
    "Hybrid initialization",
    "Mach ",
    "solution is converged",
    "converged",
    "convergence",
    "Error:",
)


def launch_fluent_run(
    output_dir: Path,
    ansys_install: AnsysInstallation | None = None,
    cores: int | None = None,
    show_gui: bool = True,
) -> subprocess.Popen:
    """
    Launch local ANSYS Fluent run in a detached process.

    Prefers PyFluent (`run_fluent.py`) when `ansys.fluent.core` is installed; otherwise
    launches `fluent.exe` directly in meshing mode reading `run_fluent.py` or `fluent_meshing.jou`.
    """
    resolved_cores = max(1, cores if cores is not None else default_fluent_cores())
    run_py = output_dir / "run_fluent.py"
    meshing_jou = output_dir / "fluent_meshing.jou"

    try:
        has_pyfluent = importlib.util.find_spec("ansys.fluent.core") is not None
    except ModuleNotFoundError:
        has_pyfluent = False
    if has_pyfluent and run_py.exists():
        code = f"import run_fluent; run_fluent.run_cfd(show_gui={show_gui!r})"
        try:
            return subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=str(output_dir),
            )
        except OSError as exc:
            raise AnsysError(f"Failed to launch PyFluent runner: {exc}") from exc

    if ansys_install is None or ansys_install.fluent_exe is None:
        raise AnsysError(
            "Neither `ansys-fluent-core` nor a local `fluent.exe` installation was found."
        )

    cmd = [
        str(ansys_install.fluent_exe),
        "3ddp",
        "-meshing",
        f"-t{resolved_cores}",
    ]
    if not show_gui:
        cmd.append("-g")
    if run_py.exists():
        cmd.extend(["-py", "-i", str(run_py)])
    elif meshing_jou.exists():
        cmd.extend(["-i", str(meshing_jou)])

    try:
        return subprocess.Popen(cmd, cwd=str(output_dir))
    except OSError as exc:
        raise AnsysError(f"Failed to launch fluent.exe: {exc}") from exc


def find_solved_files(output_dir: Path) -> tuple[Path | None, Path | None, list[Path]]:
    """Discover primary solved .cas.h5, .dat.h5, and contour PNGs in output_dir."""
    if not output_dir.exists():
        return None, None, []

    def _sort_key(path: Path) -> tuple[int, float, str]:
        is_mach_specific = (
            1 if re.search(r"_M\d+(?:\.\d+)?(?:\.[^./\\]+)+$", path.name) else 0
        )
        try:
            mtime = -path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (is_mach_specific, mtime, path.name)

    cases = sorted(
        (p for p in output_dir.glob("*.cas.h5") if not p.name.startswith("_")),
        key=_sort_key,
    )
    case_path = cases[0] if cases else None

    data_path: Path | None = None
    if case_path is not None:
        paired = case_path.with_name(case_path.name.removesuffix(".cas.h5") + ".dat.h5")
        if paired.exists():
            data_path = paired
    if data_path is None:
        datas = sorted(
            (p for p in output_dir.glob("*.dat.h5") if not p.name.startswith("_")),
            key=_sort_key,
        )
        data_path = datas[0] if datas else None

    contour_pngs = sorted(
        list(output_dir.glob("*_mach.png")) + list(output_dir.glob("*_pressure.png"))
    )
    return case_path, data_path, contour_pngs


def _resolve_optional(output_dir: Path, rel_name: str | None) -> Path | None:
    """Resolve an optional filename relative to output_dir and verify it exists."""
    if not rel_name:
        return None
    cand = Path(rel_name)
    if not cand.is_absolute():
        cand = output_dir / cand
    return cand if cand.exists() else None


def parse_cfd_results(output_dir: Path) -> CfdSolveResult | None:
    """Parse solved CFD drag metrics from cfd_summary.json with aero_results.csv fallback."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return None

    default_case, default_data, default_pngs = find_solved_files(out_dir)
    csv_path = out_dir / "aero_results.csv"
    summary_path = out_dir / "cfd_summary.json"

    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("runs"):
            runs: list[CfdMachPoint] = []
            for raw_run in payload.get("runs", []):
                if not isinstance(raw_run, dict):
                    continue
                zones = tuple(
                    CfdZoneResult(
                        zone=str(z["zone"]),
                        drag_force_n=float(z.get("drag_force_n", 0.0)),
                        cd=float(z.get("cd", 0.0)),
                    )
                    for z in (raw_run.get("zones") or [])
                    if isinstance(z, dict) and "zone" in z
                )
                c_file = _resolve_optional(out_dir, raw_run.get("case_file")) or default_case
                d_file = _resolve_optional(out_dir, raw_run.get("data_file")) or default_data
                m_png = _resolve_optional(out_dir, raw_run.get("mach_contour_png"))
                p_png = _resolve_optional(out_dir, raw_run.get("pressure_contour_png"))
                yplus_raw = raw_run.get("wall_yplus_avg")
                runs.append(
                    CfdMachPoint(
                        mach=float(raw_run.get("mach", 0.0)),
                        velocity_m_s=float(raw_run.get("velocity_m_s", 0.0)),
                        total_cd=float(raw_run.get("total_cd", 0.0)),
                        total_drag_n=float(raw_run.get("total_drag_n", 0.0)),
                        wall_yplus_avg=float(yplus_raw) if yplus_raw is not None else None,
                        case_file=c_file,
                        data_file=d_file,
                        mach_contour_png=m_png,
                        pressure_contour_png=p_png,
                        zones=zones,
                    )
                )
            if runs:
                from .mesh_convergence import parse_mesh_convergence_dict

                cells_raw = payload.get("mesh_cells")
                orth_raw = payload.get("min_orth_quality")
                conv_raw = payload.get("mesh_convergence")
                if conv_raw is None and (out_dir / "mesh_convergence.json").exists():
                    try:
                        conv_raw = json.loads(
                            (out_dir / "mesh_convergence.json").read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        conv_raw = None
                return CfdSolveResult(
                    rocket_name=str(payload.get("rocket_name", "rocket")),
                    output_dir=out_dir,
                    altitude_m=float(payload.get("altitude_m", 0.0)),
                    aoa_deg=float(payload.get("aoa_deg", 0.0)),
                    sym_factor=float(payload.get("sym_factor", 1.0)),
                    s_ref_m2=float(payload.get("s_ref_m2", 0.0)),
                    mesh_cells=int(cells_raw) if cells_raw is not None else None,
                    min_orth_quality=float(orth_raw) if orth_raw is not None else None,
                    runs=tuple(runs),
                    results_csv=csv_path if csv_path.exists() else None,
                    summary_json=summary_path,
                    mesh_convergence=parse_mesh_convergence_dict(conv_raw, out_dir),
                )

    if not csv_path.exists():
        return None

    grouped: dict[float, list[dict[str, str]]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    m_val = float(row["mach"])
                except (KeyError, TypeError, ValueError):
                    continue
                grouped.setdefault(m_val, []).append(row)
    except OSError:
        return None

    if not grouped:
        return None

    s_ref = 0.0
    for man_p in sorted(out_dir.glob("*_manifest.json")):
        if man_p.name.endswith("_rocket_manifest.json"):
            continue
        try:
            man_data = json.loads(man_p.read_text(encoding="utf-8"))
            if isinstance(man_data, dict) and "rocket_reference_area_m2" in man_data:
                s_ref = float(man_data["rocket_reference_area_m2"])
                break
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    first_row = next(iter(grouped.values()))[0]
    rocket_name = str(first_row.get("rocket", "rocket"))
    altitude_m = float(first_row.get("altitude_m", 0.0) or 0.0)
    aoa_deg = float(first_row.get("aoa_deg", 0.0) or 0.0)

    mach_png_fallback = next((p for p in default_pngs if p.name.endswith("_mach.png")), None)
    pres_png_fallback = next((p for p in default_pngs if p.name.endswith("_pressure.png")), None)

    csv_runs: list[CfdMachPoint] = []
    for m_val, rows in grouped.items():
        zones_list: list[CfdZoneResult] = []
        tot_cd: float | None = None
        tot_drag: float | None = None
        for r in rows:
            zname = str(r.get("zone", ""))
            d_n = float(r.get("drag_force_n", 0.0) or 0.0)
            cd_val = float(r.get("cd", 0.0) or 0.0)
            if zname == "total":
                tot_drag = d_n
                tot_cd = cd_val
            elif zname:
                zones_list.append(CfdZoneResult(zone=zname, drag_force_n=d_n, cd=cd_val))

        if tot_cd is None:
            tot_cd = sum(z.cd for z in zones_list)
        if tot_drag is None:
            tot_drag = sum(z.drag_force_n for z in zones_list)

        mach_tag_case = next(
            (p for p in out_dir.glob(f"*_M{m_val:.2f}.cas.h5") if p.exists()),
            default_case,
        )
        mach_tag_data = next(
            (p for p in out_dir.glob(f"*_M{m_val:.2f}.dat.h5") if p.exists()),
            default_data,
        )
        mach_tag_mpng = next(
            (p for p in out_dir.glob(f"*_M{m_val:.2f}_mach.png") if p.exists()),
            mach_png_fallback if len(grouped) == 1 else None,
        )
        mach_tag_ppng = next(
            (p for p in out_dir.glob(f"*_M{m_val:.2f}_pressure.png") if p.exists()),
            pres_png_fallback if len(grouped) == 1 else None,
        )
        csv_runs.append(
            CfdMachPoint(
                mach=m_val,
                velocity_m_s=0.0,
                total_cd=tot_cd,
                total_drag_n=tot_drag,
                wall_yplus_avg=None,
                case_file=mach_tag_case,
                data_file=mach_tag_data,
                mach_contour_png=mach_tag_mpng,
                pressure_contour_png=mach_tag_ppng,
                zones=tuple(zones_list),
            )
        )

    return CfdSolveResult(
        rocket_name=rocket_name,
        output_dir=out_dir,
        altitude_m=altitude_m,
        aoa_deg=aoa_deg,
        sym_factor=1.0,
        s_ref_m2=s_ref,
        mesh_cells=None,
        min_orth_quality=None,
        runs=tuple(csv_runs),
        results_csv=csv_path,
        summary_json=None,
    )


def run_fluent_solve_blocking(
    output_dir: Path,
    ansys_install: AnsysInstallation | None,
    cores: int | None = None,
    mach_list: list[float] | None = None,
    log_callback: Callable[[str], None] | None = None,
    line_callback: Callable[[str], None] | None = None,
    mpi: str | None = None,
    mesh_scale: float | None = None,
) -> CfdSolveResult:
    """Run headless ANSYS Fluent natively via `-py -i run_fluent.py` and return Cd results."""
    resolved_cores = max(1, cores if cores is not None else default_fluent_cores())
    out_dir = Path(output_dir)
    if ansys_install is None or ansys_install.fluent_exe is None:
        raise AnsysError(
            "Local ANSYS Fluent executable (`fluent.exe`) was not detected."
        )
    run_py = out_dir / "run_fluent.py"
    if not run_py.exists():
        raise AnsysError(f"Generated Fluent script not found: '{run_py}'.")

    cmd = [
        str(ansys_install.fluent_exe),
        "3ddp",
        "-meshing",
        "-g",
        f"-t{resolved_cores}",
    ]
    if mpi:
        cmd.append(f"-mpi={mpi}")
    cmd.extend([
        "-py",
        "-i",
        run_py.name,
    ])
    env = os.environ.copy()
    if mach_list:
        env["ORK2CFD_MACH_LIST"] = ",".join(str(float(m)) for m in mach_list)
    if mesh_scale is not None:
        env["ORK2CFD_MESH_SCALE"] = f"{float(mesh_scale):.6f}"

    log_path = out_dir / "fluent_run.log"
    tail_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(out_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise AnsysError(f"Failed to start fluent.exe: {exc}") from exc

    with log_path.open("w", encoding="utf-8", newline="\n", buffering=1) as log_file:
        if proc.stdout is not None:
            for line in proc.stdout:
                log_file.write(line)
                stripped = line.strip()
                if stripped:
                    tail_lines.append(stripped)
                    if len(tail_lines) > 25:
                        tail_lines.pop(0)
                    if line_callback is not None:
                        with contextlib.suppress(Exception):
                            line_callback(stripped)
                    if log_callback is not None and any(
                        tok in stripped for tok in _MILESTONE_TOKENS
                    ):
                        with contextlib.suppress(Exception):
                            log_callback(stripped)
        rc = proc.wait()

    if rc != 0:
        err_ctx = "\n".join(tail_lines[-8:])
        raise AnsysError(
            f"ANSYS Fluent exited with code {rc}. Check '{log_path.name}'.\n{err_ctx}"
        )

    parsed = parse_cfd_results(out_dir)
    if parsed is None or not parsed.runs:
        raise AnsysError(
            f"ANSYS Fluent completed, but no results were found in '{out_dir}'."
        )
    return parsed


def launch_fluent_visualization(
    output_dir: Path,
    ansys_install: AnsysInstallation | None,
    case_path: Path | None = None,
    cores: int | None = None,
) -> subprocess.Popen:
    """Open solved .cas.h5 / .dat.h5 directly inside the ANSYS Fluent Solution GUI."""
    resolved_cores = max(1, cores if cores is not None else default_fluent_cores())
    out_dir = Path(output_dir)
    if ansys_install is None or ansys_install.fluent_exe is None:
        raise AnsysError(
            "Local ANSYS Fluent executable (`fluent.exe`) was not detected."
        )
    resolved_case = case_path
    if resolved_case is None:
        resolved_case, _dat, _pngs = find_solved_files(out_dir)
    if resolved_case is None or not resolved_case.exists():
        raise AnsysError(
            f"No solved Fluent .cas.h5 file found in '{out_dir}'. Run CFD first."
        )

    dat_partner = resolved_case.with_name(
        resolved_case.name.removesuffix(".cas.h5") + ".dat.h5"
    )
    jou_path = out_dir / "_open_viz.jou"
    cmd_line = (
        f'/file/read-case-data "{resolved_case.name}"\n'
        if dat_partner.exists()
        else f'/file/read-case "{resolved_case.name}"\n'
    )
    jou_path.write_text(cmd_line, encoding="utf-8")

    cmd = [
        str(ansys_install.fluent_exe),
        "3ddp",
        f"-t{resolved_cores}",
        "-i",
        jou_path.name,
    ]
    try:
        return subprocess.Popen(cmd, cwd=str(out_dir))
    except OSError as exc:
        raise AnsysError(f"Failed to launch Fluent visualization GUI: {exc}") from exc


def launch_cfdpost_visualization(
    output_dir: Path,
    ansys_install: AnsysInstallation | None,
    data_path: Path | None = None,
) -> subprocess.Popen:
    """Open solved .dat.h5 / .cas.h5 directly inside ANSYS CFD-Post."""
    out_dir = Path(output_dir)
    if ansys_install is None or ansys_install.cfdpost_exe is None:
        raise AnsysError(
            "Local ANSYS CFD-Post executable (`cfdpost.exe`) was not detected."
        )
    target = data_path
    if target is None:
        found_case, found_data, _pngs = find_solved_files(out_dir)
        target = found_data or found_case
    if target is None or not target.exists():
        raise AnsysError(
            f"No solved .dat.h5 or .cas.h5 results file found in '{out_dir}'. Run CFD first."
        )

    cmd = [
        str(ansys_install.cfdpost_exe),
        "-results",
        str(target.resolve()),
    ]
    try:
        return subprocess.Popen(cmd, cwd=str(out_dir))
    except OSError as exc:
        raise AnsysError(f"Failed to launch ANSYS CFD-Post: {exc}") from exc
