"""ANSYS environment detection, script/journal generation, and Workbench automation."""

from .detector import AnsysInstallation, detect_ansys_installations
from .fluent_launcher import (
    find_solved_files,
    launch_cfdpost_visualization,
    launch_fluent_run,
    launch_fluent_visualization,
    parse_cfd_results,
    run_fluent_solve_blocking,
)
from .mesh_convergence import (
    compute_gci_metrics,
    compute_mesh_convergence_scales,
    run_mesh_convergence_blocking,
)
from .next_steps import build_ansys_next_steps_markdown, build_ansys_next_steps_summary
from .script_generator import generate_ansys_scripts
from .workbench_builder import build_workbench_project

__all__ = [
    "AnsysInstallation",
    "build_ansys_next_steps_markdown",
    "build_ansys_next_steps_summary",
    "build_workbench_project",
    "compute_gci_metrics",
    "compute_mesh_convergence_scales",
    "detect_ansys_installations",
    "find_solved_files",
    "generate_ansys_scripts",
    "launch_cfdpost_visualization",
    "launch_fluent_run",
    "launch_fluent_visualization",
    "parse_cfd_results",
    "run_fluent_solve_blocking",
    "run_mesh_convergence_blocking",
]
