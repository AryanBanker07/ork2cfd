"""Optional local ANSYS Fluent launcher for generated run_fluent.py or TUI journals."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from ..exceptions import AnsysError
from .detector import AnsysInstallation


def launch_fluent_run(
    output_dir: Path,
    ansys_install: AnsysInstallation | None = None,
    cores: int = 4,
    show_gui: bool = True,
) -> subprocess.Popen:
    """
    Launch local ANSYS Fluent run in a detached process.

    Prefers PyFluent (`run_fluent.py`) when `ansys.fluent.core` is installed; otherwise
    launches `fluent.exe` directly in meshing GUI mode reading `fluent_meshing.jou`.
    """
    run_py = output_dir / "run_fluent.py"
    meshing_jou = output_dir / "fluent_meshing.jou"

    has_pyfluent = importlib.util.find_spec("ansys.fluent.core") is not None
    if has_pyfluent and run_py.exists():
        code = (
            f"import run_fluent; run_fluent.run_cfd(show_gui={show_gui!r})"
        )
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
        f"-t{max(1, cores)}",
    ]
    if not show_gui:
        cmd.append("-g")
    if meshing_jou.exists():
        cmd.extend(["-i", str(meshing_jou)])

    try:
        return subprocess.Popen(cmd, cwd=str(output_dir))
    except OSError as exc:
        raise AnsysError(f"Failed to launch fluent.exe: {exc}") from exc
