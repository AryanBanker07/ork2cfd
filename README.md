# `ork2cfd` — OpenRocket (`.ork`) to ANSYS Fluent GUI & CFD Package Builder

Convert OpenRocket (`.ork`) rocket designs into watertight, ANSYS Fluent–ready `.step` CAD geometries, pre-configured ANSYS scripts, and ready-to-open ANSYS Workbench (`.wbpj` / `.wbpz`) projects from a simple desktop GUI or CLI.

## Features

- **Simple Desktop GUI (`launch_gui.bat` / `ork2cfd`)**:
  - Load any `.ork` file (XML, ZIP, or GZIP) and see live 2D side-profile previews of the rocket OML, fluid enclosure, and Bodies of Influence (BOIs).
  - Real-time ISA atmosphere, Reynolds number, and $y^+$ boundary-layer prism calculator ($\Delta y_1$, prism count, growth rate, cell-count budget).
  - One-click generation of the complete CFD package + direct button to open the output folder or launch local ANSYS Fluent.
- **CFD-Clean Watertight `.step` Export**:
  - Automatically strips internal parts (bulkheads, centering rings, couplers, parachutes, motors).
  - Merges consecutive collinear body tubes into single smooth faces and fits curved nosecones/transitions (Haack / Von Kármán, Ogive, Ellipsoid, Parabolic, Power) to smooth B-splines.
  - Rotates the body revolution seam away from fin planes, embeds fin roots cleanly, blunts nose tips, and supports `square`, `rounded`, `airfoil`, and `double-wedge` fin profiles + cant.
  - Exports full 3D Fluid Domain, Half Symmetry (180°), Sector Symmetry ($1/N_{\text{fins}}$), 2D Axisymmetric, or OML-only STEP files with embedded OCCT XCAF Named Selections (`farfield`, `inlet`, `outlet`, `symmetry`, `rocket_nosecone`, `rocket_body`, `rocket_boattail`, `rocket_fins`, `rocket_base`) and `boi_nearfield` / `boi_wake` solids.
- **Multi-Region CFD `.stl` Export (Optional)**:
  - Export an ASCII `.stl` surface tessellation in SI meters with separate named solid regions (`solid <zone_name> ... endsolid <zone_name>`) for ANSYS Fluent Fault-Tolerant Meshing or OpenFOAM (`--stl` in CLI or GUI checkbox).
- **Interactive 3D CAD Viewer & Automated Verification Pack**:
  - Generates `<stem>_cad_viewer.html` (offline WebGL 3D viewer with orbit/pan/zoom, X/Y/Z section clip plane, wireframe overlay, domain X-ray opacity slider, and Named Selection toggles), `<stem>_cad_views.png` (4-view CAD vs. `.ork` overlay diagnostic sheet), and `<stem>_verification.{json,md}` (`PASS`/`WARN`/`FAIL` B-Rep, dimensional fidelity, and STEP round-trip audit).
  - Open directly via the **View 3D CAD** button in the GUI or `ork2cfd verify <rocket.ork | model.step> --open`.
- **Runnable ANSYS Scripts & Automated `.wbpj` Project**:
  - `run_fluent.py` (PyFluent Watertight Meshing → Poly-Hexcore + Prism → Solver → CSV)
  - `fluent_meshing.jou` & `fluent_solver.jou` (Fluent TUI journals for GUI or HPC)
  - `discovery_prep.py` (ANSYS Discovery / SpaceClaim geometry & fallback Named Selection script)
  - `create_project.wbjn` + automatic batch `.wbpj` (and optional `.wbpz`) creation via local `RunWB2.exe`.

## Installation

```bash
pip install -e ".[dev]"
# Optional: for live PyFluent script execution
pip install -e ".[fluent]"
```

## Usage

### Desktop GUI
Double-click `launch_gui.bat` or run:
```bash
ork2cfd
```

### CLI
```bash
# Inspect an .ork model
ork2cfd inspect rocket.ork

# Check y+ and boundary-layer sizing
ork2cfd calc-bl rocket.ork --mach 0.85 --alt 500 --yplus 1.0

# Generate STEP (+ optional STL) + CAD viewer/verification + ANSYS scripts + .wbpj
ork2cfd convert rocket.ork -o ./rocket_cfd --mach 0.85 --mode fluid_domain --shape bullet --stl

# Verify & inspect any .ork design or exported .step file in 3D HTML + 4-view PNG
ork2cfd verify rocket.ork -o ./verify_out --mode fluid_domain --open
```

## Testing

```bash
ruff check src tests
pytest -v -m "not ansys" --cov=ork2cfd
pytest -v -m ansys   # Live ANSYS Workbench .wbpj verification (requires local ANSYS)
```
