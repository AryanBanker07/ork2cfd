"""In-depth aerospace & ANSYS Fluent help text registry for GUI (i) info icons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HelpEntry:
    """Title and detailed aerospace/CFD explanation shown on hover and click."""

    title: str
    body: str


HELP_TEXTS: dict[str, HelpEntry] = {
    # Section headers
    "sec_input": HelpEntry(
        title="OpenRocket (.ork) Input & OML Filtering",
        body=(
            "Loads plain XML, ZIP-compressed, or GZIP-compressed OpenRocket (.ork) files.\n\n"
            "• Automatic Multistage Detection: Reads all axial stages (e.g., Sustainer + "
            "Booster) and computes the true GCD rotational symmetry order across all fin sets.\n"
            "• External OML Only: Internal structural parts (bulkheads, centering rings, tube "
            "couplers, inner motor tubes, parachutes, shock cords, and mass components) are "
            "automatically stripped so the fluid mesh has zero internal cavities.\n"
            "• Multi-Tube Merging: Equal-radius body tubes are collapsed into a single smooth "
            "cylinder to eliminate artificial seam rings in Fluent Meshing."
        ),
    ),
    "stage_selection": HelpEntry(
        title="Multistage Configuration Selector",
        body=(
            "Automatically detects multistage vehicles (e.g., Sustainer + Booster) in the "
            "loaded .ork file and lets you choose which flight configuration to simulate:\n\n"
            "• All Stages: Full-stack liftoff configuration combining every axial stage. Per-stage "
            "fin sets share rotational symmetry via GCD (e.g., 2 stages × 4 fins = 4-fold "
            "rotational symmetry → 90° quadrants, NOT 8).\n"
            "• Stage 1 / Stage 2 / Up to K: Isolates a specific upper stage or post-separation "
            "vehicle and re-zeros its axial X coordinate to 0.0 m."
        ),
    ),
    "sec_physics": HelpEntry(
        title="Flight Condition & Boundary-Layer (y+) Calculator",
        body=(
            "Uses the ISA 1976 atmosphere model and Sutherland viscosity law to compute "
            "freestream speed, Reynolds number (Re_L), skin-friction coefficient (Cf), and "
            "first prism layer thickness (Δy1) for your target y+.\n\n"
            "These pre-computed physics parameters are embedded directly into run_fluent.py, "
            "fluent_meshing.jou, and fluent_solver.jou."
        ),
    ),
    "sec_cad": HelpEntry(
        title="CFD Geometry, Symmetry & Fluid Domain Settings",
        body=(
            "Builds a watertight, sliver-free B-Rep assembly using OpenCASCADE (OCCT):\n"
            "• Rotates body revolution seams midway between fins (π / N_fins).\n"
            "• Embeds fin roots slightly into the body tube to prevent tangent-edge slivers.\n"
            "• Tags solid bodies and boundary faces with XCAF Named Selections (farfield, "
            "inlet, outlet, symmetry, rocket_nosecone, rocket_body, rocket_boattail, "
            "rocket_fins, rocket_base) plus a geometric fallback manifest."
        ),
    ),
    "sec_deliverables": HelpEntry(
        title="Export Options & ANSYS Integration",
        body=(
            "Controls which CAD and ANSYS artifacts are generated in the output folder:\n"
            "• Switch Export Mode to 'Rocket .step Only' to export ONLY the watertight rocket "
            "CAD file (<rocket>_rocket.step) with no fluid enclosure or Workbench wait.\n"
            "• Keep 'Full CFD Package' to also export the fluid enclosure (<rocket>_cfd.step), "
            "runnable ANSYS scripts (run_fluent.py, .jou journals, discovery_prep.py, "
            "create_project.wbjn), and an auto-built Workbench project (.wbpj/.wbpz)."
        ),
    ),
    "export_preset": HelpEntry(
        title="Export Preset: Full CFD Package vs Rocket .step Only",
        body=(
            "Quick 1-click switch for what ork2cfd exports:\n\n"
            "• Full CFD Package: Exports the fluid enclosure <rocket>_cfd.step, standalone "
            "<rocket>_rocket.step, runnable ANSYS scripts, Workbench .wbpj, and 3D viewer.\n"
            "• Rocket .step Only: Exports ONLY the standalone rocket CAD file "
            "<rocket>_rocket.step (the actual rocket shape with per-component Named Selections "
            "— NO fluid enclosure, NO scripts, NO Workbench wait) plus the 3D CAD viewer."
        ),
    ),
    # Section 2: Flight & Physics
    "mach": HelpEntry(
        title="Mach Number (M = V / a)",
        body=(
            "Freestream Mach number at the simulated flight point.\n\n"
            "• Drives velocity (V = M × a), Reynolds number, boundary-layer height Δy1, and "
            "automatic Flow Regime selection when Regime is set to 'auto':\n"
            "  - M < 0.30 → Subsonic (incompressible pressure-based)\n"
            "  - 0.30 ≤ M < 1.20 → Transonic (compressible ideal-gas pressure-based)\n"
            "  - M ≥ 1.20 → Supersonic (density-based implicit + Mach-cone domain check)\n"
            "• Tip: For a drag curve, evaluate M = 0.2, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0."
        ),
    ),
    "altitude": HelpEntry(
        title="Flight Altitude (m above MSL)",
        body=(
            "Geometric altitude evaluated against the ISA 1976 standard atmosphere (0–50 km).\n\n"
            "• Determines ambient temperature T∞, static pressure P∞, air density ρ∞, speed "
            "of sound a, and dynamic viscosity μ (via Sutherland's law).\n"
            "• Higher altitude lowers ρ∞ and Re_L, thickening the boundary layer (larger Δy1).\n"
            "• Tip: Choose the altitude near Max-Q or peak velocity from your OpenRocket sim."
        ),
    ),
    "aoa": HelpEntry(
        title="Angle of Attack α (deg)",
        body=(
            "Pitch angle between the freestream velocity vector and the rocket +X axis in the "
            "X–Y pitch plane.\n\n"
            "• α = 0°: Pure axial zero-lift drag (Cd0). Compatible with all domain modes, "
            "including Symmetry Sector.\n"
            "• α > 0° (e.g. 2°–8°): Computes normal force (Cl / Cn), pitching moment (Cm), "
            "and Center of Pressure (X_CP). Must use Full 360° or Half Symmetry (180°), NOT "
            "Symmetry Sector."
        ),
    ),
    "yplus": HelpEntry(
        title="Target Wall y+ (Dimensionless Wall Distance)",
        body=(
            "Controls first prism layer cell thickness Δy1 = 2 × y+ × μ / (ρ × u_τ) for the "
            "k-ω SST turbulence model:\n\n"
            "• y+ ≈ 1.0 (Recommended): Wall-resolved viscous sublayer mesh. Most accurate for "
            "skin-friction drag, flow separation, and boattail shock-boundary interaction.\n"
            "• y+ = 30–50: Wall-function mode. Uses much thicker Δy1 and fewer prism layers "
            "to keep total mesh count low (useful for ANSYS Student 512k limit).\n"
            "• Avoid 5 < y+ < 30 (buffer layer), where wall shear predictions degrade."
        ),
    ),
    "regime": HelpEntry(
        title="Flow Regime & Solver Formulation",
        body=(
            "Selects the ANSYS Fluent solver physics and boundary types:\n\n"
            "• auto (Recommended): Chooses regime automatically from Mach number.\n"
            "• subsonic (M < 0.30): Pressure-based coupled, constant ρ/μ, energy OFF, "
            "velocity-inlet + pressure-outlet.\n"
            "• transonic (0.30 ≤ M < 1.20): Pressure-based coupled, ideal-gas + Sutherland, "
            "energy ON, pressure-far-field.\n"
            "• moderate supersonic (1.20 ≤ M < 2.00): Pressure-based coupled ideal-gas + "
            "High-Speed Numerics for better unstructured shock stability and ~30% lower RAM.\n"
            "• high supersonic (M ≥ 2.00): Density-based implicit with AUSM flux, farfield init, "
            "and low-CFL warmup + Mach-cone radial clearance."
        ),
    ),
    "mesh_preset": HelpEntry(
        title="Mesh Sizing Preset",
        body=(
            "Scales surface min/max sizes, BOI refinement sizes, and farfield size relative "
            "to rocket diameter D and length L:\n\n"
            "• coarse: Fast smoke test (~2× coarser surface & BOI).\n"
            "• student_safe: Tuned to help keep cell counts near/under the ANSYS Student "
            "512k CFD limit (combine with Symmetry Half or Sector if needed).\n"
            "• standard (Recommended): Balanced production aerodynamics sizing.\n"
            "• fine: High-resolution mesh for final validation or grid-independence studies."
        ),
    ),
    "mesh_convergence": HelpEntry(
        title="Auto Mesh Convergence (Grid Independence Study)",
        body=(
            "Automatically sweeps from a coarse baseline mesh to progressively finer meshes "
            "(dividing surface, fin, nose, BOI, and farfield sizes by r = 1.35× per level, "
            "roughly 2.5× cells per step) while keeping first prism height Δy1 fixed at your "
            "target y+.\n\n"
            "• Stops as soon as |ΔCd| / Cd between consecutive grids drops within your Cd "
            "Convergence Tolerance (default 1.0%) or max levels (4) is reached.\n"
            "• Reports the optimal production mesh level, Roache Grid Convergence Index (GCI), "
            "Richardson continuum Cd estimate, and a convergence plot PNG."
        ),
    ),
    "mesh_conv_tol": HelpEntry(
        title="Mesh Convergence Cd Relative Tolerance (%)",
        body=(
            "Stopping threshold for Auto Mesh Convergence:\n\n"
            "• |Cd_k - Cd_{k-1}| / Cd_k × 100% ≤ Tolerance.\n"
            "• 1.0% (Recommended): Engineering grid independence for rocket drag studies.\n"
            "• 0.5%: Tighter academic/publication convergence "
            "(may require 3–4 refinement levels).\n"
            "• 2.0%: Fast preliminary screening under tighter compute or license budgets."
        ),
    ),
    "cores": HelpEntry(
        title="CPU Cores (-tN) vs Logical Threads",
        body=(
            "Number of MPI parallel worker processes passed to Fluent (-tN flag).\n\n"
            "• Defaults automatically based on detected physical cores and RAM allotment "
            "(~0.35 GB overhead per MPI rank, scaling best on physical cores rather than "
            "hyperthreads).\n"
            "• Exceeding physical cores adds MPI memory overhead with little speedup.\n"
            "• ANSYS Student licenses typically cap solver cores at 4."
        ),
    ),
    "ram_limit": HelpEntry(
        title="RAM Allotment Limit (GB) & Safe Cell Budget",
        body=(
            "Target maximum host RAM (GB) allotted to ANSYS Fluent meshing and solving.\n\n"
            "• ork2cfd leaves OS headroom (typically ~50–70% of total RAM or 90% of free RAM) "
            "to prevent Windows MPI rank crashes (EXIT STATUS -1) on large poly-hexcore meshes.\n"
            "• Estimates peak Fluent memory (~6.5 GB/M cells pressure-based, ~9.0 GB/M cells "
            "density-based + 0.35 GB/rank) and caps Auto Mesh Convergence and Fin Optimization "
            "verification cell counts safely below your limit.\n"
            "• Use 'Apply RAM-Safe Rec.' to auto-set cores, RAM limit, mesh preset, y+, and mode."
        ),
    ),
    "iterations": HelpEntry(
        title="Steady-State Solver Iterations",
        body=(
            "Maximum steady-state solver iterations written into run_fluent.py and "
            "fluent_solver.jou (default: 300).\n\n"
            "• Automatic Early Stop: run_fluent.py configures a report-based convergence "
            "condition on cd_total (stabilized within 1e-4 over 25 consecutive iterations, "
            "after ignoring the first 50) across subsonic, transonic, and supersonic runs "
            "so converged runs stop early even if wake residuals stall near 1e-3.\n"
            "• Increase to 600–1000+ if high-AoA or complex transonic shocks need longer "
            "before cd_total settles."
        ),
    ),
    # Section 3: CAD & Domain
    "domain_mode": HelpEntry(
        title="CFD Domain Mode",
        body=(
            "Selects how the rocket OML and surrounding fluid volume are built in STEP:\n\n"
            "• fluid_domain: Full 360° farfield enclosure minus rocket OML. Required if fins "
            "have full-vehicle asymmetry.\n"
            "• symmetry_half: Cuts domain at z ≥ 0 (180°). Halves cell count while still "
            "supporting non-zero Angle of Attack (α ≠ 0) in the X–Y pitch plane.\n"
            "• symmetry_sector: 1/N_fins sector (360°/N_fins, e.g. 90° quadrant = 4 sectors for "
            "4 fins per stage). Keeps 1 complete fin per stage intact and cuts through open air.\n"
            "• symmetry_half_sector: Minimal 1/(2×N_fins) mirror wedge (180°/N_fins, e.g. "
            "45° = 8 wedges for 4 fins), slicing along a fin centerplane. Valid ONLY for α = 0° "
            "and aligned zero-cant fins.\n"
            "• axisymmetric_2d: Planar X–Y (y ≥ 0) fluid face without 3D fins.\n"
            "• oml_only: Exports only the closed rocket solid (build your own wind tunnel)."
        ),
    ),
    "enclosure_shape": HelpEntry(
        title="Farfield Enclosure Shape",
        body=(
            "Outer boundary geometry of the fluid domain:\n\n"
            "• bullet (Recommended): Smooth elliptical nose dome + cylindrical barrel + flat "
            "outlet disk. Eliminates sharp inlet corners and works cleanly with "
            "pressure-far-field BCs at all angles of attack and Mach numbers.\n"
            "• cylinder: Cylindrical tunnel with flat circular inlet and outlet planes.\n"
            "• box: Rectangular wind-tunnel domain with flat inlet, outlet, and 4 side walls."
        ),
    ),
    "blunt_radius": HelpEntry(
        title="Nose Tip Blunting Radius (mm)",
        body=(
            "Replaces the infinitely sharp mathematical point at x = 0 with a tiny smooth "
            "elliptical dome cap of this radius (clamped safely to ≤ 12% of nose base radius).\n\n"
            "• 0.5–1.5 mm (Recommended): Eliminates degenerate zero-area needle triangles "
            "and high skewness (>0.95) at the nose tip in Fluent Meshing while preserving "
            "total vehicle length and having negligible effect on overall Cd.\n"
            "• Set to 0.0 only if you explicitly want an unblunted sharp apex."
        ),
    ),
    "step_schema": HelpEntry(
        title="STEP Export Protocol (AP214 vs AP242)",
        body=(
            "ISO 10303 STEP application protocol used by OpenCASCADE XCAF:\n\n"
            "• AP214 (Recommended): Broadest compatibility across ANSYS Workbench, Fluent "
            "Meshing, Discovery, and SpaceClaim for carrying ADVANCED_FACE Named Selections, "
            "body names, and face colors.\n"
            "• AP242: Newer Model-Based Definition (MBD) schema; use if a downstream CAD "
            "pipeline explicitly requires STEP AP242."
        ),
    ),
    "include_bois": HelpEntry(
        title="Include Bodies of Influence (BOIs)",
        body=(
            "Adds two separate named solid zones to the STEP assembly (without subtracting "
            "them from the fluid volume):\n\n"
            "• boi_nearfield: Cylinder tightly enclosing the nosecone, body, and fins to "
            "resolve bow shocks, boundary-layer transition, and fin-tip vortices.\n"
            "• boi_wake: Expanding frustum extending 5×L downstream to capture base drag "
            "and turbulent wake recirculation accurately."
        ),
    ),
    "fin_fillets": HelpEntry(
        title="Fin Root Fillets (Body–Fin Junction)",
        body=(
            "Attempts to add small CFD fillets along the body-to-fin intersection edges.\n\n"
            "• OFF (Default): Clean, sharp fin-body intersections that mesh reliably and "
            "match Barrowman/OpenRocket assumptions.\n"
            "• ON: Models epoxy fillet interference drag. Note: tight swept or beveled roots "
            "can be rejected by OpenCASCADE; if a fillet fails, ork2cfd safely falls back "
            "to sharp roots and logs a warning."
        ),
    ),
    "rail_buttons": HelpEntry(
        title="Include Rail Buttons",
        body=(
            "Models external rail buttons defined in the .ork file as small solid cylinders "
            "embedded into the airframe.\n\n"
            "• OFF (Default): Clean baseline airframe drag.\n"
            "• ON: Captures parasitic drag of rail buttons. Requires 'fluid_domain' (360°) "
            "or 'symmetry_half' if buttons break rotational symmetry."
        ),
    ),
    "launch_lugs": HelpEntry(
        title="Include Launch Lugs",
        body=(
            "Models external launch lugs defined in the .ork file.\n\n"
            "• Lug interiors are plugged into solid aerodynamic protrusions so tiny hollow "
            "tube bores do not explode cell counts during meshing.\n"
            "• OFF (Default) for clean OML drag; ON for full-fidelity parasitic drag."
        ),
    ),
    # Section 4: Deliverables
    "deliv_step": HelpEntry(
        title="Deliverable: CFD Domain .step + Manifest",
        body=(
            "Exports <rocket>_cfd.step in SI meters with embedded XCAF Named Selections "
            "and BOI solids, plus <rocket>_manifest.json containing exact validation stats "
            "(watertightness, free-edge count = 0, bounding box, reference area) and face "
            "centroids for automated fallback selection."
        ),
    ),
    "deliv_rocket_step": HelpEntry(
        title="Export Option: Rocket-Only .step (No Enclosure)",
        body=(
            "Exports <rocket>_rocket.step + <rocket>_rocket_manifest.json containing a "
            "watertight 3D solid of the physical rocket shape alone (no fluid enclosure), "
            "with clean axial seams and embedded XCAF Named Selections + distinct colors "
            "for every main OpenRocket component:\n\n"
            "• Nosecones & Transitions (e.g., nose_cone, transition)\n"
            "• Individual Body Tubes (e.g., recovery_tube, avionics_tube, motor_tube, "
            "booster_tube, body_tube) — kept as separate faces instead of merged\n"
            "• Per-Stage Fin Sets (e.g., sustainer_fins, stage2_fins / booster_fins)\n"
            "• Base Cap (rocket_base) & enabled Rail Buttons / Launch Lugs\n\n"
            "Ideal for opening directly in SolidWorks, ANSYS SpaceClaim/Discovery, Fusion "
            "360, or custom structural/CFD setups where you want the rocket CAD alone."
        ),
    ),
    "deliv_scripts": HelpEntry(
        title="Deliverable: Runnable ANSYS Scripts & Journals",
        body=(
            "Generates 6 pre-configured files tailored to your rocket and flight point:\n"
            "• run_fluent.py — End-to-end PyFluent Watertight Meshing (with fin/nose local "
            "face sizing, BOIs, y+ prisms, poly-hexcore) + Mach-sweep Solver + CSV.\n"
            "• fluent_meshing.jou & fluent_solver.jou — Fluent Meshing & Solver TUI journals.\n"
            "• discovery_prep.py — ANSYS Discovery/SpaceClaim Named Selection fallback.\n"
            "• create_project.wbjn — Workbench journal wiring .step into Fluid Flow "
            "(Fluent with Fluent Meshing).\n"
            "• README_RUN.md — Workflow status breakdown and copy-paste run commands."
        ),
    ),
    "deliv_wbpj": HelpEntry(
        title="Deliverable: Build Workbench Project (.wbpj)",
        body=(
            "Automatically runs your locally installed ANSYS Workbench (RunWB2.exe -B -R "
            "create_project.wbjn) in the background (~15–40 s) to produce a ready-to-open "
            "<rocket>_cfd.wbpj project + _files/ folder with the .step geometry already "
            "linked into a Fluid Flow (Fluent with Fluent Meshing) system.\n\n"
            "Simply double-click the .wbpj file to open it directly in ANSYS Workbench."
        ),
    ),
    "deliv_wbpz": HelpEntry(
        title="Deliverable: Archive Single-File .wbpz",
        body=(
            "Instructs ANSYS Workbench to pack the .wbpj project, _files/ directory, and "
            "imported .step file into a single self-contained <rocket>_cfd.wbpz archive.\n\n"
            "Ideal for sharing with teammates, backing up, or transferring to another "
            "workstation without broken relative file paths."
        ),
    ),
    "deliv_stl": HelpEntry(
        title="Deliverable: Multi-Region .stl Mesh",
        body=(
            "Exports an ASCII .stl surface tessellation in SI meters with separate named "
            "solid regions (e.g., rocket_nosecone, rocket_body, rocket_fins, inlet, "
            "outlet, farfield).\n\n"
            "Useful for ANSYS Fluent Fault-Tolerant Meshing, quick geometry checks, or "
            "external CFD meshers such as OpenFOAM snappyHexMesh and cfMesh."
        ),
    ),
    "deliv_verify": HelpEntry(
        title="Deliverable: 3D CAD Viewer & Verification Pack",
        body=(
            "Automatically runs quantitative CAD checks (watertightness, free edges, sliver "
            "faces, .ork axial/radial dimensions, Named Selections, and XCAF STEP "
            "round-trip re-import) and generates four inspection files:\n"
            "• <rocket>_cad_viewer.html — Offline WebGL 3D viewer with orbit/pan/zoom, "
            "X/Y/Z section clip plane, wireframe toggle, and Named Selection isolation.\n"
            "• <rocket>_cad_views.png — 4-view sheet (3D iso, X-Y CAD vs .ork overlay, "
            "Y-Z tail projection, and domain + verification scorecard).\n"
            "• <rocket>_verification.json & .md — Machine- and human-readable PASS/WARN/FAIL "
            "verification audit."
        ),
    ),
    "deliv_auto_cfd": HelpEntry(
        title="Auto-Run Fluent & Compute Cd (Headless)",
        body=(
            "When enabled, ork2cfd automatically executes your local ANSYS Fluent "
            "installation headlessly right after generating the CAD package "
            "(`fluent 3ddp -meshing -g -tN -py -i run_fluent.py`) — zero external "
            "`pip install` required.\n\n"
            "• Performs Watertight Geometry meshing (`rocket_fins` / `rocket_nosecone` "
            "local sizing, BOIs, uniform y+ prisms, `poly-hexcore`), solves at your "
            "configured Mach point, writes `cfd_summary.json` + `aero_results.csv` + "
            "solved `.cas.h5`/`.dat.h5` + centerline contour PNGs, and displays Total Cd "
            "and per-component breakdown directly inside the GUI."
        ),
    ),
    "sec_cfd_results": HelpEntry(
        title="CFD Drag (Cd) Results & Direct Visualization Launchers",
        body=(
            "Displays solved ANSYS Fluent aerodynamic drag results directly inside ork2cfd "
            "and provides 1-click launchers to visualize the flowfield:\n\n"
            "• Run CFD & Compute Cd: Runs headless ANSYS Fluent on the active package "
            "on demand, streaming live meshing/solver progress into the log and populating "
            "Total Cd, Drag (N), Wall y+, Mesh Cells, and per-zone Cd breakdown.\n"
            "• Open in Fluent (Viz): Launches ANSYS Fluent GUI with your solved `.cas.h5` "
            "and `.dat.h5` pre-loaded so you can immediately plot contours and pathlines.\n"
            "• Open in CFD-Post: Launches ANSYS CFD-Post directly loading `.dat.h5` for "
            "full-360° Instancing Transform and publication renders.\n"
            "• View Contours: Opens the Mach & Static Pressure contour PNGs rendered after solve."
        ),
    ),
    "sec_fin_opt": HelpEntry(
        title="Fin Shape Optimization (Bayesian GP-EI)",
        body=(
            "Optimizes the target stage's fin planform (root_chord, tip_chord, span, and "
            "sweep_angle_deg) to minimize whole-stage/whole-rocket total drag coefficient (Cd) "
            "under strict simulation budgets:\n\n"
            "• Whole-Vehicle Cd: Always optimizes and reports total vehicle/stage Cd (nosecone, "
            "body tubes, boattail, all fins, and base) rather than isolated fin Cd alone.\n"
            "• Divergence & RAM Guards: Automatically filters diverged runs and keeps final "
            "verification within your allotted RAM cell budget.\n"
            "• Fixed Geometry Rules: Keeps the .ork fin cross-section and thickness locked.\n"
            "• Algorithm: Feasible maximin LHS DoE followed by ARD Matérn-5/2 GP + Expected "
            "Improvement (GP-EI) infill."
        ),
    ),
    "opt_target_stage": HelpEntry(
        title="Target Fin Stage to Optimize",
        body=(
            "Selects which stage's fin set is parameterized and reshaped during optimization "
            "while simulating the full stack selected in Section 1:\n\n"
            "• On multi-stage rockets, optimizing Booster fins keeps Sustainer fins fixed "
            "at their baseline .ork dimensions.\n"
            "• On single-stage rockets, selects that stage's main fin set."
        ),
    ),
    "opt_max_sims": HelpEntry(
        title="Maximum CFD Simulation Budget (Max Sims)",
        body=(
            "Total number of ANSYS Fluent (or dry-run) evaluations allocated to this "
            "optimization campaign (minimum 3):\n\n"
            "• ~10 sims: Fast campaign (1 Baseline + 4 LHS DoE + 4 GP-EI + 1 Verify) "
            "capturing the main low-drag trend.\n"
            "• 18–25 sims: Recommended for tight 4D Bayesian convergence.\n"
            "• Checkpoints every simulation to optimization_history.json so you can stop "
            "or increase Max Sims later and resume without re-running prior points."
        ),
    ),
    "opt_min_sm": HelpEntry(
        title="Minimum Barrowman Static Stability Margin (calibers)",
        body=(
            "Minimum aerodynamic static stability margin SM = (X_cp - X_cg) / D_ref "
            "required across all evaluated fin candidates:\n\n"
            "• Analytical Barrowman stability checks run inside the Bayesian optimizer "
            "before any CAD/Fluent run, preventing wasted CFD solves on unstable fins.\n"
            "• Typical subsonic/transonic flight guideline: ≥ 1.5 to 2.0 calibers."
        ),
    ),
    "opt_verify_final": HelpEntry(
        title="Final High-Resolution Verification Run",
        body=(
            "When enabled (and Max Sims ≥ 8 in live Fluent mode), reserves the final "
            "simulation slot to re-simulate the winning lowest-drag fin geometry on a "
            "finer verification mesh (Standard preset / y+ = 1.0 or one refinement step "
            "finer than the locked baseline mesh)."
        ),
    ),
    "opt_dry_run": HelpEntry(
        title="Fast Dry-Run (Analytical Drag Proxy)",
        body=(
            "Uses an aerodynamic analytical Cd proxy (skin friction + form drag + "
            "compressibility sweep relief) instead of launching ANSYS Fluent:\n\n"
            "• Completes a full 10-step optimization campaign in a few seconds.\n"
            "• Useful for previewing feasibility constraints, live convergence plots, and "
            "OpenRocket .ork parameter export before launching multi-hour Fluent runs."
        ),
    ),
    "opt_params": HelpEntry(
        title="Parameters to Optimize vs Lock",
        body=(
            "Choose which fin shape parameters Bayesian GP-EI optimization may vary and "
            "which stay fixed at their baseline .ork dimensions:\n\n"
            "• Continuous Planform Variables: Root Chord, Tip Chord, Span (Height), and "
            "LE Sweep Angle — unchecking any box locks that dimension strictly to the "
            "baseline .ork value across all simulations.\n"
            "• Cross-Section: When checked, screens Airfoil, Double-Wedge, and Rounded "
            "profiles against baseline before optimizing geometry; when unchecked, keeps "
            "the .ork cross-section fixed.\n"
            "• Fin Thickness always remains locked for structural consistency.\n\n"
            "Tip: Optimizing fewer active variables converges tighter within small "
            "simulation budgets (e.g., locking Root Chord or Span when bay length or "
            "rail clearance is fixed)."
        ),
    ),
    "sec_next_steps": HelpEntry(
        title="Next Steps — Viewing & Verifying Results in ANSYS",
        body=(
            "Provides exact, rocket-specific post-processing instructions after solving in "
            "ANSYS Workbench (.wbpj), ANSYS Fluent GUI (.cas.h5 / .dat.h5), and ANSYS "
            "CFD-Post:\n"
            "• Verify solver residuals and Wall y+ contours on rocket_* boundary zones.\n"
            "• Extract Cd, Cl, and X_CP using pre-computed force vectors and exact "
            "symmetry-sector scaling factors.\n"
            "• Slice centerline Mach/Pressure contours, plot wake pathlines, and enable "
            "CFD-Post Instancing Transform to view the full 360° vehicle."
        ),
    ),
}
