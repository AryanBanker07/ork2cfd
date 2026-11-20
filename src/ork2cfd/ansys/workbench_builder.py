"""Automated ANSYS Workbench (.wbpj / .wbpz) project generator via RunWB2.exe."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..exceptions import AnsysError
from .detector import AnsysInstallation


def build_workbench_project(
    wbjn_path: Path,
    expected_wbpj_path: Path,
    ansys_install: AnsysInstallation,
    expected_wbpz_path: Path | None = None,
    timeout_s: int = 240,
) -> tuple[Path, Path | None]:
    """
    Execute RunWB2.exe in batch mode (-B -R) to create a .wbpj (and optional .wbpz) file.

    Returns (wbpj_path, wbpz_path_or_none). Raises AnsysError on failure or timeout.
    """
    if ansys_install.runwb2_exe is None or not ansys_install.runwb2_exe.exists():
        raise AnsysError(
            f"RunWB2.exe not found under {ansys_install.root_dir}; cannot build .wbpj."
        )
    if not wbjn_path.exists():
        raise AnsysError(f"Workbench journal script not found: {wbjn_path}")

    cmd = [str(ansys_install.runwb2_exe), "-B", "-R", str(wbjn_path.resolve())]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(wbjn_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AnsysError(
            f"ANSYS Workbench batch execution timed out after {timeout_s}s."
        ) from exc
    except OSError as exc:
        raise AnsysError(f"Failed to launch RunWB2.exe: {exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise AnsysError(
            f"RunWB2.exe exited with code {proc.returncode}. Output tail:\n{tail}"
        )

    if not expected_wbpj_path.exists():
        raise AnsysError(
            f"Workbench finished without producing expected project file: {expected_wbpj_path}"
        )

    wbpz_out = None
    if expected_wbpz_path is not None and expected_wbpz_path.exists():
        wbpz_out = expected_wbpz_path

    return expected_wbpj_path, wbpz_out
