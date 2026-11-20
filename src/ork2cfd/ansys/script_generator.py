"""Jinja2 template renderer generating runnable ANSYS scripts, journals, and README."""

from __future__ import annotations

import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..models.cfd import DomainMode, PipelineConfig
from ..models.results import BoundaryLayerReport, SolverRecommendation
from ..models.rocket import RocketModel
from .detector import AnsysInstallation
from .next_steps import build_ansys_next_steps_markdown

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _symmetry_multiplier(rocket: RocketModel, mode: DomainMode) -> float:
    """Return domain reduction factor (1 full, 2 half, N_fins sector, 2*N_fins half-sector)."""
    if mode == DomainMode.SYMMETRY_HALF:
        return 2.0
    if mode == DomainMode.SYMMETRY_SECTOR:
        return float(max(1, rocket.rotational_symmetry_order))
    if mode == DomainMode.SYMMETRY_HALF_SECTOR:
        return float(2 * max(1, rocket.rotational_symmetry_order))
    return 1.0


def generate_ansys_scripts(
    output_dir: Path,
    stem: str,
    rocket: RocketModel,
    config: PipelineConfig,
    bl: BoundaryLayerReport,
    solver_rec: SolverRecommendation,
    step_path: Path,
    manifest_path: Path,
    ansys_install: AnsysInstallation | None = None,
    wbpj_path: Path | None = None,
    wbpz_path: Path | None = None,
) -> dict[str, Path]:
    """Render all ANSYS automation scripts and journals into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    version_code = ansys_install.version_code if ansys_install else "25.2"
    release_tag = ansys_install.release_tag if ansys_install else "v252"
    wb_script_ver = ansys_install.wb_script_version if ansys_install else "25.2.170"
    runwb2_hint = (
        str(ansys_install.runwb2_exe)
        if (ansys_install and ansys_install.runwb2_exe)
        else r"C:\Program Files\ANSYS Inc\v252\Framework\bin\Win64\RunWB2.exe"
    )

    wbpj_target = wbpj_path or (output_dir / f"{stem}_cfd.wbpj")
    wbpz_target = wbpz_path or (output_dir / f"{stem}_cfd.wbpz")
    sym_mult = _symmetry_multiplier(rocket, config.domain.mode)
    wbpj_name = wbpj_target.name if config.deliverables.build_wbpj else None

    context = {
        "rocket": rocket,
        "stem": stem,
        "display_name": f"{stem}_CFD",
        "cad": config.cad,
        "domain": config.domain,
        "flight": config.flight,
        "bl": bl,
        "solver_rec": solver_rec,
        "step_path": str(step_path.resolve()),
        "step_filename": step_path.name,
        "manifest_filename": manifest_path.name,
        "wbpj_path": str(wbpj_target.resolve()),
        "wbpj_filename": wbpj_name,
        "wbpz_path": str(wbpz_target.resolve()),
        "archive_wbpz": config.deliverables.archive_wbpz,
        "ansys_version": version_code,
        "release_tag": release_tag,
        "wb_script_version": wb_script_ver,
        "runwb2_hint": runwb2_hint,
        "wbjn_filename": "create_project.wbjn",
        "meshing_jou_filename": "fluent_meshing.jou",
        "solver_jou_filename": "fluent_solver.jou",
        "symmetry_multiplier": sym_mult,
        "next_steps_md": build_ansys_next_steps_markdown(
            rocket=rocket,
            stem=stem,
            config=config,
            bl=bl,
            solver_rec=solver_rec,
            sym_multiplier=sym_mult,
            wbpj_filename=wbpj_name,
        ),
        "math": math,
    }

    outputs: dict[str, Path] = {}
    mapping = {
        "run_fluent.py": "run_fluent.py.j2",
        "fluent_meshing.jou": "fluent_meshing.jou.j2",
        "fluent_solver.jou": "fluent_solver.jou.j2",
        "discovery_prep.py": "discovery_prep.py.j2",
        "create_project.wbjn": "create_project.wbjn.j2",
        "README_RUN.md": "README_RUN.md.j2",
    }

    for out_name, tpl_name in mapping.items():
        rendered = env.get_template(tpl_name).render(**context)
        out_path = output_dir / out_name
        out_path.write_text(rendered, encoding="utf-8")
        outputs[out_name] = out_path

    return outputs
