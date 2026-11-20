"""Auto-detect local ANSYS installations (Workbench, Fluent, Discovery)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnsysInstallation:
    """Discovered local ANSYS release and executable paths."""

    version_code: str  # e.g. "25.2"
    release_tag: str  # e.g. "v252"
    wb_script_version: str  # e.g. "25.2.170"
    root_dir: Path
    runwb2_exe: Path | None
    fluent_exe: Path | None
    discovery_exe: Path | None
    is_student: bool = False

    @property
    def label(self) -> str:
        """Human-readable summary for GUI display."""
        tag = f"ANSYS {self.version_code} ({self.release_tag})"
        return f"{tag} [Student]" if self.is_student else tag


def detect_ansys_installations() -> list[AnsysInstallation]:
    """Scan environment variables and standard Windows paths for ANSYS releases."""
    candidates: list[tuple[str, Path, bool]] = []

    # Check AWP_ROOTxxx env vars (e.g. AWP_ROOT252 -> v252)
    for key, val in os.environ.items():
        m = re.match(r"^AWP_ROOT(\d{3})$", key)
        if m and val:
            code = m.group(1)
            root = Path(val)
            if root.exists():
                candidates.append((f"v{code}", root, "student" in str(root).lower()))

    # Standard Program Files locations
    base_dir = Path(r"C:\Program Files\ANSYS Inc")
    if base_dir.exists():
        for child in sorted(base_dir.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            m = re.match(r"^v(\d{3})$", child.name, re.IGNORECASE)
            if m:
                candidates.append((f"v{m.group(1)}", child, False))
            elif "student" in child.name.lower():
                for sub in sorted(child.iterdir(), reverse=True):
                    sm = re.match(r"^v(\d{3})$", sub.name, re.IGNORECASE)
                    if sm and sub.is_dir():
                        candidates.append((f"v{sm.group(1)}", sub, True))

    seen_roots: set[Path] = set()
    installs: list[AnsysInstallation] = []
    for tag, root, is_student in candidates:
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)

        digits = tag.lstrip("vV")
        ver_code = f"{digits[:2]}.{digits[2:]}" if len(digits) >= 3 else digits
        runwb2 = resolved / "Framework" / "bin" / "Win64" / "RunWB2.exe"
        fluent = resolved / "fluent" / "ntbin" / "win64" / "fluent.exe"
        discovery = resolved / "Discovery" / "Discovery.exe"

        wb_script_ver = _detect_wb_build_version(resolved, ver_code)
        installs.append(
            AnsysInstallation(
                version_code=ver_code,
                release_tag=tag,
                wb_script_version=wb_script_ver,
                root_dir=resolved,
                runwb2_exe=runwb2 if runwb2.exists() else None,
                fluent_exe=fluent if fluent.exists() else None,
                discovery_exe=discovery if discovery.exists() else None,
                is_student=is_student,
            )
        )

    # Sort newest version first, non-student preferred when same version
    installs.sort(key=lambda x: (x.version_code, not x.is_student), reverse=True)
    return installs


def select_best_installation(preferred_version: str | None = None) -> AnsysInstallation | None:
    """Return preferred or newest local ANSYS installation, or None if unavailable."""
    installs = detect_ansys_installations()
    if not installs:
        return None
    if preferred_version:
        norm = preferred_version.strip().lower().lstrip("v")
        for inst in installs:
            if inst.version_code.lower() == norm or inst.release_tag.lstrip("v").lower() == norm:
                return inst
    return installs[0]


def _detect_wb_build_version(root_dir: Path, ver_code: str) -> str:
    """Infer Workbench SetScriptVersion string (e.g. '25.2.170') from installation."""
    # Check Framework config or known release defaults
    ver_file = root_dir / "Framework" / "bin" / "Win64" / "Ans.Core.Version.txt"
    if ver_file.exists():
        text = ver_file.read_text(encoding="utf-8", errors="ignore").strip()
        m = re.search(r"(\d+\.\d+\.\d+)", text)
        if m:
            return m.group(1)
    if ver_code == "25.2":
        return "25.2.170"
    if ver_code == "24.2":
        return "24.2.133"
    return f"{ver_code}.0"
