"""Context-aware ANSYS Fluent & CFD-Post post-processing walkthrough generator."""

from __future__ import annotations

import math

from ..models.cfd import DomainMode, PipelineConfig
from ..models.results import BoundaryLayerReport, SolverRecommendation
from ..models.rocket import RocketModel


def _symmetry_instancing_hint(domain_mode: DomainMode, sym_multiplier: float) -> str:
    """Return CFD-Post Instancing Transform guidance tailored to the active symmetry mode."""
    copies = max(1, int(round(sym_multiplier)))
    if domain_mode == DomainMode.SYMMETRY_HALF:
        return (
            "Right-click **Default Transform → Edit → Instancing Transform**, enable **Apply "
            "Reflection** across the **XY plane (Z = 0)** to mirror into the full 360° vehicle."
        )
    if domain_mode in {DomainMode.SYMMETRY_SECTOR, DomainMode.SYMMETRY_HALF_SECTOR}:
        return (
            f"Right-click **Default Transform → Edit → Instancing Transform**, set **Number of "
            f"Copies = {copies}** around the **X-axis** ({360.0 / copies:.1f}° step"
            + (" + mirror" if domain_mode == DomainMode.SYMMETRY_HALF_SECTOR else "")
            + ") to display the complete 360° rocket."
        )
    return "Full 360° geometry is already solved — no Instancing Transform copy is needed."


def build_ansys_next_steps_summary(
    stem: str,
    domain_mode: DomainMode,
    sym_multiplier: float,
    mach: float,
    aoa_deg: float,
    target_yplus: float,
    wbpj_filename: str | None = None,
) -> str:
    """Build a concise 5-step summary for the GUI Section 5 card and CLI Convert panel."""
    aoa_rad = math.radians(aoa_deg)
    cos_a = math.cos(aoa_rad)
    sin_a = math.sin(aoa_rad)
    wb_hint = (
        f"Double-click {wbpj_filename} in Workbench (Update Mesh/Setup/Solution)"
        if wbpj_filename
        else "Run python run_fluent.py (or fluent_meshing.jou + fluent_solver.jou)"
    )
    sym_mult_int = max(1, int(round(sym_multiplier)))
    force_scale_note = (
        f"Cd is full-rocket; multiply N by {sym_mult_int}"
        if sym_multiplier > 1.01
        else "1:1 full-rocket forces & Cd"
    )
    plane_hint = (
        "pick symmetry face or create XY Plane at Point (0,0,0), Normal (0,0,1)"
        if domain_mode
        in {
            DomainMode.SYMMETRY_HALF,
            DomainMode.SYMMETRY_SECTOR,
            DomainMode.SYMMETRY_HALF_SECTOR,
        }
        else "Domain → Surface → Create → Plane, Point (0,0,0), Normal (0,0,1)"
    )
    inst_short = (
        f"mirror/pattern {sym_mult_int}x via Default Transform → Instancing Transform"
        if sym_multiplier > 1.01
        else "view full 360° solution directly"
    )

    return (
        f"1. Run & Open Solution: {wb_hint}; in Fluent open "
        f"File → Read → Case & Data... → {stem}.cas.h5 (Mach {mach:.2f}).\n"
        f"2. Check Residuals & Wall y+: Confirm residuals < 1e-4, then plot "
        f"Results → Graphics → Contours → Turbulence → Wall Yplus on rocket_* walls "
        f"(target y+ ≈ {target_yplus:.1f}).\n"
        f"3. Read Drag & Lift (Cd, Cl): Results → Reports → Forces → Forces... → "
        f"select all rocket_* walls → Force Vector ({cos_a:.4f}, {sin_a:.4f}, 0) → Print "
        f"({force_scale_note}).\n"
        f"4. Plot Shock & Wake Contours: Slice plane ({plane_hint}) → Contours → "
        f"Velocity → Mach Number & Static Pressure; add Pathlines for fin/base wake.\n"
        f"5. 3D Visuals in CFD-Post: Open Results (or load {stem}.dat.h5 in CFD-Post) "
        f"and {inst_short}."
    )


def build_ansys_next_steps_markdown(
    rocket: RocketModel | None,
    stem: str,
    config: PipelineConfig,
    bl: BoundaryLayerReport | None,
    solver_rec: SolverRecommendation | None,
    sym_multiplier: float,
    wbpj_filename: str | None = None,
) -> str:
    """Generate a detailed, flight-point-specific ANSYS post-processing walkthrough."""
    mach = config.flight.mach
    aoa_deg = config.flight.aoa_deg
    target_yplus = config.flight.target_yplus
    aoa_rad = math.radians(aoa_deg)
    cos_a = math.cos(aoa_rad)
    sin_a = math.sin(aoa_rad)
    lift_x = -sin_a
    lift_y = cos_a

    mode = config.domain.mode
    sym_mult_int = max(1, int(round(sym_multiplier)))
    yplus_guideline = (
        "Wall-resolved mesh (`y+ < 5`, ideally `~1`) for accurate skin-friction drag and "
        "shock-induced separation."
        if target_yplus <= 5.0
        else "Wall-function mesh (`30 <= y+ <= 100`) suited for coarser Student-limit runs."
    )

    if rocket is not None and bl is not None:
        ref_area_sym = rocket.reference_area / max(1.0, sym_multiplier)
        ref_table = (
            "| Reference Quantity | Expected Value in Fluent |\n"
            "|---|---|\n"
            f"| **Area (`A_ref / SYM_FACTOR`)** | `{ref_area_sym:.8f} m²` "
            f"(full `S_ref = {rocket.reference_area:.8f} m²`, "
            f"divisor `{sym_multiplier:g}`) |\n"
            f"| **Length (`L_ref`)** | `{rocket.total_length:.4f} m` |\n"
            f"| **Velocity (`V_inf`)** | `{bl.velocity_m_s:.2f} m/s` (Mach `{mach}`) |\n"
            f"| **Density (`rho_inf`)** | `{bl.atmosphere.density_kg_m3:.6f} kg/m³` |\n"
        )
    else:
        ref_table = (
            "| Reference Quantity | Expected Value in Fluent |\n"
            "|---|---|\n"
            "| **Area (`A_ref / SYM_FACTOR`)** | Generated in `run_fluent.py` "
            "and `fluent_solver.jou` |\n"
            f"| **Flight Mach / AoA** | Mach `{mach}`, AoA `{aoa_deg}°` |\n"
        )

    sym_force_note = (
        f"- **Symmetry Factor Active (`{sym_multiplier:g}×`):** Because Reference **Area** is "
        f"already divided by `{sym_multiplier:g}`, Fluent's printed **`Cd`** and **`Cl`** "
        f"are already **full-rocket aerodynamic coefficients**. Multiply raw printed forces "
        f"(Newtons) and moments (N·m) by **`{sym_mult_int}`** to obtain full-vehicle loads."
        if sym_multiplier > 1.01
        else "- **Full 360° Domain (`1×`):** Reported **`Cd`**, **`Cl`**, forces (N), and "
        "moments (N·m) directly represent the complete rocket."
    )

    if wbpj_filename:
        wb_open = (
            f"1. **ANSYS Workbench Project:** Double-click **`{wbpj_filename}`** in this "
            f"folder (**Fluid Flow (Fluent with Fluent Meshing)**), then open/update **Mesh** "
            f"to run Fluent Meshing and **Solution** to launch the Fluent Solver."
        )
    else:
        wb_open = (
            "1. **Run Solver First:** Execute `python run_fluent.py` (or run "
            "`fluent_meshing.jou` then `fluent_solver.jou`) to produce the mesh (`*.msh.h5`) "
            "and solved case/data (`*.cas.h5`, `*.dat.h5`)."
        )

    instancing_text = _symmetry_instancing_hint(mode, sym_multiplier)
    sym_slice_modes = {
        DomainMode.SYMMETRY_HALF,
        DomainMode.SYMMETRY_SECTOR,
        DomainMode.SYMMETRY_HALF_SECTOR,
    }
    slice_tip = (
        "Select the existing **`symmetry`** boundary directly, OR create a custom "
        "slice plane: **Domain → Surface → Create → Plane...** → **Point** `(0, 0, 0)`, "
        "**Normal** `(0, 0, 1)`."
        if mode in sym_slice_modes
        else "**Domain → Surface → Create → Plane...** → **Method = Point and Normal**, "
        "**Point** `(0, 0, 0)`, **Normal** `(0, 0, 1)`, click **Create**."
    )
    regime_label = solver_rec.regime.value.upper() if solver_rec is not None else "AERODYNAMIC"

    sections = [
        "### Step 1 — Open Your Solution in ANSYS Workbench or Fluent GUI",
        wb_open,
        (
            "2. **Standalone Fluent GUI:** Open **ANSYS Fluent** (Solution Mode, 3D "
            "double precision), click **`File → Read → Case & Data...`** and select "
            f"**`{stem}.cas.h5`** (loads **`{stem}.dat.h5`** automatically if solved)."
        ),
        "",
        "---",
        "",
        "### Step 2 — Verify Convergence & Wall `y+` Before Reading Drag",
        (
            "1. **Check Residual Monitors:** Confirm continuity, momentum (`x/y/z`), energy "
            "(if compressible), and `k`/`omega` residuals dropped below **`1e-4`** and leveled."
        ),
        (
            "2. **Inspect Wall `y+`:** Go to **`Results → Graphics → Contours`** "
            "(double-click **New**):"
        ),
        "   - **Contours of:** `Turbulence... → Wall Yplus`",
        (
            "   - **Surfaces:** select `rocket_nosecone`, `rocket_body`, "
            "`rocket_boattail`, `rocket_fins`, `rocket_base`"
        ),
        (
            f"   - Click **Save/Display** and verify against target "
            f"`y+ = {target_yplus:.1f}` ({yplus_guideline})."
        ),
        "",
        "---",
        "",
        "### Step 3 — Extract Aerodynamic Drag, Lift & Center of Pressure (`Cd`, `Cl`, `X_CP`)",
        f"1. Confirm **`Setup → Reference Values`** matches your flight point ({regime_label}):",
        "",
        ref_table.rstrip(),
        "2. **Drag (`Cd`) & Axial Force:** **`Results → Reports → Forces → Forces...`**:",
        (
            "   - **Wall Zones:** select `rocket_nosecone`, `rocket_body`, "
            "`rocket_boattail`, `rocket_fins`, `rocket_base`."
        ),
        (
            f"   - **Force Vector:** **`({cos_a:.4f}, {sin_a:.4f}, 0)`** "
            f"(`cos α, sin α, 0` for `α = {aoa_deg:.1f}°`)."
        ),
        (
            "   - Click **Print** — read **`Total`** Pressure + Viscous Drag (`N`) "
            "and Drag Coefficient **`Cd`**."
        ),
        "3. **Lift / Normal Force (`Cl`) & Center of Pressure (`X_CP`):**",
        (
            f"   - For Lift (`AoA != 0°`), set Force Vector **`({lift_x:.4f}, {lift_y:.4f}, 0)`** "
            "(`-sin α, cos α, 0`) and click **Print** to read **`Cl`** and **`F_N`**."
        ),
        (
            "   - Switch to **`Moments...`**, Center **`(0, 0, 0)`**, Axis **`(0, 0, 1)`**: "
            "Center of Pressure `X_CP = -M_z / F_N` (m aft of nose tip)."
        ),
        sym_force_note,
        "",
        "---",
        "",
        "### Step 4 — Plot Contours, Shock Waves & Wake Streamlines in Fluent",
        f"1. **Create Centerline Slice Plane:** {slice_tip}",
        "2. **Mach & Shock Contours:** **`Results → Graphics → Contours → New`**:",
        (
            "   - **Contours of:** `Velocity... → Mach Number` (check bow shock, "
            "fin leading-edge oblique shocks, and boattail expansion)."
        ),
        (
            "   - Repeat with **`Pressure... → Static Pressure`** (or `Pressure Coefficient`) "
            "on the slice plane and `rocket_*` walls."
        ),
        (
            "3. **Wake Streamlines:** **`Results → Graphics → Pathlines → New`** → "
            "**Release from:** `inlet` / `farfield` (or vertical rake aft of `rocket_base`) → "
            "**Color by:** `Velocity Magnitude` or `Mach Number`."
        ),
        "",
        "---",
        "",
        "### Step 5 — Full-360° 3D Visualization & Plots in ANSYS CFD-Post",
        (
            "1. In Workbench, double-click **`Results`** (or launch **CFD-Post** and "
            f"open **`File → Load Results... → {stem}.dat.h5`**)."
        ),
        f"2. **Reconstruct Full 360° Vehicle:** {instancing_text}",
        (
            "3. **Axial `Cp(x)` Curve:** **`Insert → Location → Polyline`** (intersect Symmetry "
            "plane with body wall) → **`Insert → Chart`** (`X = X [m]`, `Y = Cp`)."
        ),
        (
            "4. **High-Res Export:** Use **`File → Save Picture`** (PNG) for design reviews "
            "and CDR slides."
        ),
        "",
        "---",
        "",
        "### Step 6 — Build a Full Drag Curve for OpenRocket / RocketPy (`rocketpy_drag.csv`)",
        (
            "- `run_fluent.py` writes starter **`aero_results.csv`** and "
            "**`rocketpy_drag.csv`** (`mach,cd`) in this folder."
        ),
        (
            "- Sweep Mach (`0.2, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0`), record each "
            "converged `Cd` from Step 3, and load into **RocketPy** (`power_off_drag` / "
            "`power_on_drag`) or **OpenRocket**."
        ),
    ]
    return "\n".join(sections) + "\n"
