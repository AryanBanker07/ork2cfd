"""Live ANSYS 2025 R2 integration tests (run with: pytest -v -m ansys)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ork2cfd.ansys.detector import detect_ansys_installations
from ork2cfd.models import (
    DeliverableConfig,
    DomainConfig,
    DomainMode,
    FlightCondition,
    MeshPreset,
    PipelineConfig,
)
from ork2cfd.pipeline import run_pipeline
from tests.conftest import REAL_ORK_CANDIDATES


@pytest.mark.ansys
def test_live_workbench_wbpj_and_wbpz_build(synthetic_ork_file: Path, tmp_path: Path) -> None:
    installs = detect_ansys_installations()
    wb_installs = [inst for inst in installs if inst.runwb2_exe is not None]
    if not wb_installs:
        pytest.skip("No local ANSYS Workbench (RunWB2.exe) detected.")

    # Use spaceshot.ork if available locally, otherwise synthetic fixture
    spaceshot = next(
        (p for p in REAL_ORK_CANDIDATES if p.name == "spaceshot.ork" and p.exists()),
        None,
    )
    ork_input = spaceshot or synthetic_ork_file

    cfg = PipelineConfig(
        domain=DomainConfig(mode=DomainMode.FLUID_DOMAIN),
        flight=FlightCondition(mach=0.8, mesh_preset=MeshPreset.COARSE),
        deliverables=DeliverableConfig(
            export_step=True,
            generate_scripts=True,
            build_wbpj=True,
            archive_wbpz=True,
        ),
    )

    out_dir = tmp_path / "live_wb_pkg"
    res = run_pipeline(ork_path=ork_input, output_dir=out_dir, config=cfg)

    assert res.step_path is not None and res.step_path.exists()
    assert res.wbpj_path is not None and res.wbpj_path.exists()
    assert res.wbpj_path.stat().st_size > 10_000
    assert res.wbpz_path is not None and res.wbpz_path.exists()
    assert res.wbpz_path.stat().st_size > 10_000
    wb_txt = res.wbpj_path.read_text(encoding="utf-8", errors="ignore")
    assert (
        "Fluid Flow with Fluent Meshing" in wb_txt
        or "Fluid Flow (Fluent with Fluent Meshing)" in wb_txt
    )
