"""System CPU/RAM hardware detection and memory-safe CFD resource recommendations."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from dataclasses import dataclass

from ..models.cfd import DomainMode, MeshPreset
from ..models.rocket import RocketModel


@dataclass(frozen=True)
class SystemHardware:
    """Detected host memory and CPU topology."""

    total_ram_gb: float
    available_ram_gb: float
    physical_cores: int
    logical_threads: int


@dataclass(frozen=True)
class ResourceRecommendation:
    """Memory- and core-budgeted CFD solver and mesh recommendations."""

    hardware: SystemHardware
    allotted_ram_gb: float
    recommended_ram_limit_gb: float
    recommended_cores: int
    max_safe_cells: int
    estimated_ram_gb: float
    recommended_preset: MeshPreset
    recommended_yplus: float
    recommended_domain_mode: DomainMode
    recommended_mpi: str
    exceeds_ram_budget: bool
    warnings: tuple[str, ...] = ()


def _windows_memory_gb() -> tuple[float, float] | None:
    """Query total and available physical RAM (GiB) on Windows via GlobalMemoryStatusEx."""
    if not sys.platform.startswith("win"):
        return None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            gib = float(1024**3)
            return stat.ullTotalPhys / gib, stat.ullAvailPhys / gib
    except Exception:
        return None
    return None


def _posix_memory_gb() -> tuple[float, float] | None:
    """Query total and available RAM (GiB) from /proc/meminfo on Linux."""
    meminfo = "/proc/meminfo"
    if not os.path.exists(meminfo):
        return None
    vals_kb: dict[str, int] = {}
    with contextlib.suppress(OSError), open(meminfo, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) < 2:
                continue
            num_toks = parts[1].strip().split()
            if num_toks and num_toks[0].isdigit():
                vals_kb[parts[0].strip()] = int(num_toks[0])
    if "MemTotal" not in vals_kb:
        return None
    total_gb = vals_kb["MemTotal"] / (1024.0 * 1024.0)
    avail_kb = vals_kb.get("MemAvailable", vals_kb.get("MemFree", vals_kb["MemTotal"] // 2))
    avail_gb = avail_kb / (1024.0 * 1024.0)
    return total_gb, avail_gb


def detect_system_hardware() -> SystemHardware:
    """Detect physical/available RAM and CPU cores/threads cleanly without hard dependencies."""
    logical = max(1, os.cpu_count() or 4)
    physical: int | None = None
    total_gb: float | None = None
    avail_gb: float | None = None

    with contextlib.suppress(Exception):
        import psutil

        vm = psutil.virtual_memory()
        gib = float(1024**3)
        total_gb = vm.total / gib
        avail_gb = vm.available / gib
        p_cores = psutil.cpu_count(logical=False)
        if p_cores is not None and p_cores >= 1:
            physical = int(p_cores)
        l_threads = psutil.cpu_count(logical=True)
        if l_threads is not None and l_threads >= 1:
            logical = int(l_threads)

    if total_gb is None or avail_gb is None:
        mem_pair = _windows_memory_gb() or _posix_memory_gb()
        if mem_pair is not None:
            total_gb, avail_gb = mem_pair

    if total_gb is None:
        total_gb = 16.0
    if avail_gb is None:
        avail_gb = total_gb * 0.65

    total_clean = round(max(1.0, float(total_gb)), 2)
    avail_clean = round(max(0.5, min(total_clean, float(avail_gb))), 2)
    default_phys = physical if physical is not None else max(1, logical // 2)
    phys_clean = max(1, min(logical, int(default_phys)))
    return SystemHardware(
        total_ram_gb=total_clean,
        available_ram_gb=avail_clean,
        physical_cores=phys_clean,
        logical_threads=logical,
    )


def recommend_resources(
    *,
    hardware: SystemHardware | None = None,
    rocket: RocketModel | None = None,
    mach: float = 0.8,
    aoa_deg: float = 0.0,
    solver_type: str = "pressure-based",
    estimated_cells: int = 250_000,
    requested_cores: int | None = None,
    ram_limit_gb: float | None = None,
) -> ResourceRecommendation:
    """Compute RAM-safe MPI core count, max cell cap, mesh preset, and y+ recommendations."""
    hw = hardware or detect_system_hardware()

    rec_ram = round(
        max(
            2.0,
            min(
                hw.total_ram_gb * 0.70,
                max(hw.available_ram_gb * 0.90, hw.total_ram_gb * 0.50),
            ),
        ),
        1,
    )
    if ram_limit_gb is not None:
        allotted_ram = round(max(1.5, min(hw.total_ram_gb * 0.92, float(ram_limit_gb))), 1)
    else:
        allotted_ram = rec_ram

    max_core_cap = 8 if hw.total_ram_gb < 24.0 else 16
    rec_cores = max(
        1,
        min(
            hw.physical_cores,
            int(max(1.0, allotted_ram // 1.6)),
            max_core_cap,
        ),
    )
    active_cores = max(1, int(requested_cores) if requested_cores is not None else rec_cores)

    gb_per_million = 9.0 if solver_type == "density-based" else 6.5
    rank_overhead_gb = 0.35 * active_cores
    est_ram = round(
        rank_overhead_gb + (max(0, int(estimated_cells)) / 1_000_000.0) * gb_per_million,
        2,
    )
    usable_gb = max(0.8, allotted_ram - rank_overhead_gb)
    max_safe_cells = max(
        100_000,
        min(20_000_000, int((usable_gb / gb_per_million) * 1_000_000)),
    )

    rec_yplus = 30.0 if (allotted_ram < 16.0 or (mach >= 1.2 and allotted_ram < 24.0)) else 1.0

    if rocket is not None:
        if abs(aoa_deg) < 1e-6 and rocket.rotational_symmetry_order >= 2:
            rec_mode = DomainMode.SYMMETRY_SECTOR
        elif abs(aoa_deg) >= 1e-6 and (
            rocket.has_sector_mirror_symmetry or rocket.rotational_symmetry_order % 2 == 0
        ):
            rec_mode = DomainMode.SYMMETRY_HALF
        else:
            rec_mode = DomainMode.FLUID_DOMAIN
    else:
        rec_mode = DomainMode.SYMMETRY_SECTOR if abs(aoa_deg) < 1e-6 else DomainMode.SYMMETRY_HALF

    if max_safe_cells < 350_000:
        rec_preset = MeshPreset.COARSE
    elif max_safe_cells < 950_000:
        rec_preset = MeshPreset.STUDENT_SAFE
    elif max_safe_cells < 2_500_000:
        rec_preset = MeshPreset.STANDARD
    else:
        rec_preset = MeshPreset.FINE

    rec_mpi = (
        "msmpi"
        if (os.name == "nt" and (hw.total_ram_gb < 24.0 or est_ram > 0.70 * allotted_ram))
        else "default"
    )
    exceeds = (int(estimated_cells) > max_safe_cells) or (est_ram > allotted_ram)

    warnings: list[str] = []
    if requested_cores is not None and int(requested_cores) > hw.physical_cores:
        warnings.append(
            f"Requested {requested_cores} Fluent cores exceeds {hw.physical_cores} physical CPU "
            f"cores ({hw.logical_threads} logical threads); hyperthreading adds MPI RAM overhead "
            "without scaling."
        )
    if exceeds:
        warnings.append(
            f"Estimated mesh (~{estimated_cells:,} cells, ~{est_ram:.1f} GB RAM) exceeds "
            f"allotted budget ({allotted_ram:.1f} GB / {max_safe_cells:,} safe cells); consider "
            "fewer cores, Symmetry Sector, y+=30, or Student Safe/Coarse preset."
        )
    if hw.available_ram_gb < allotted_ram * 0.65:
        warnings.append(
            f"Only {hw.available_ram_gb:.1f} GB free RAM currently available (allotted "
            f"{allotted_ram:.1f} GB); close heavy background apps or lower RAM Limit."
        )

    return ResourceRecommendation(
        hardware=hw,
        allotted_ram_gb=allotted_ram,
        recommended_ram_limit_gb=rec_ram,
        recommended_cores=rec_cores,
        max_safe_cells=max_safe_cells,
        estimated_ram_gb=est_ram,
        recommended_preset=rec_preset,
        recommended_yplus=rec_yplus,
        recommended_domain_mode=rec_mode,
        recommended_mpi=rec_mpi,
        exceeds_ram_budget=exceeds,
        warnings=tuple(warnings),
    )
