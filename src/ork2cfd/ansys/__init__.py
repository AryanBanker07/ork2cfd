"""ANSYS environment detection, script/journal generation, and Workbench automation."""

from .detector import AnsysInstallation, detect_ansys_installations
from .fluent_launcher import launch_fluent_run
from .next_steps import build_ansys_next_steps_markdown, build_ansys_next_steps_summary
from .script_generator import generate_ansys_scripts
from .workbench_builder import build_workbench_project

__all__ = [
    "AnsysInstallation",
    "build_ansys_next_steps_markdown",
    "build_ansys_next_steps_summary",
    "build_workbench_project",
    "detect_ansys_installations",
    "generate_ansys_scripts",
    "launch_fluent_run",
]
