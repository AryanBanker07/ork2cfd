"""Immutable data models representing OpenRocket aerodynamic OML assemblies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

from ._curves import evaluate_profile_radius


class SurfaceFinish(str, Enum):
    """OpenRocket surface finish grades mapped to equivalent sand-grain roughness ks (m)."""

    UNFINISHED = "unfinished"
    ROUGH = "rough"
    NORMAL = "normal"
    SMOOTH = "smooth"
    POLISHED = "polished"

    @property
    def roughness_m(self) -> float:
        """Equivalent sand-grain roughness height k_s in meters."""
        mapping = {
            SurfaceFinish.UNFINISHED: 500e-6,
            SurfaceFinish.ROUGH: 200e-6,
            SurfaceFinish.NORMAL: 60e-6,
            SurfaceFinish.SMOOTH: 20e-6,
            SurfaceFinish.POLISHED: 2e-6,
        }
        return mapping[self]

    @classmethod
    def from_str(cls, value: str | None) -> SurfaceFinish:
        """Parse an OpenRocket finish string safely with NORMAL default."""
        if not value:
            return cls.NORMAL
        norm = value.strip().lower()
        for member in cls:
            if member.value == norm:
                return member
        return cls.NORMAL


class FinCrossSection(str, Enum):
    """Fin cross-section airfoil styles supported by OpenRocket."""

    SQUARE = "square"
    ROUNDED = "rounded"
    AIRFOIL = "airfoil"
    DOUBLE_WEDGE = "double_wedge"

    @classmethod
    def from_str(cls, value: str | None) -> FinCrossSection:
        """Parse an OpenRocket crosssection string safely."""
        if not value:
            return cls.SQUARE
        norm = value.strip().lower().replace("-", "_").replace(" ", "_")
        if norm in {"airfoil", "naca"}:
            return cls.AIRFOIL
        if norm in {"rounded", "round"}:
            return cls.ROUNDED
        if norm in {"hexagonal", "double_wedge", "doublewedge", "wedge"}:
            return cls.DOUBLE_WEDGE
        return cls.SQUARE


@dataclass(frozen=True)
class FinSet:
    """Represents a circumferential set of aerodynamic fins attached to a parent body."""

    name: str
    fin_type: str  # trapezoid | elliptical | freeform | tube
    fin_count: int
    root_chord: float
    tip_chord: float
    span: float
    sweep_length: float
    thickness: float
    cross_section: FinCrossSection = FinCrossSection.SQUARE
    cant_deg: float = 0.0
    rotation_deg: float = 0.0
    fillet_radius: float = 0.0
    points: tuple[tuple[float, float], ...] = ()
    root_le_x: float = 0.0
    root_te_x: float = 0.0
    global_root_le_x: float = 0.0
    global_root_te_x: float = 0.0
    outer_diameter: float = 0.0


@dataclass(frozen=True)
class RailButton:
    """External rail button attachment on a body component."""

    name: str
    outer_diameter: float
    height: float
    axial_x: float
    angle_deg: float = 0.0
    instance_count: int = 1
    instance_separation: float = 0.0


@dataclass(frozen=True)
class LaunchLug:
    """External launch lug tube attachment on a body component."""

    name: str
    outer_radius: float
    length: float
    axial_x: float
    angle_deg: float = 0.0
    instance_count: int = 1
    instance_separation: float = 0.0


@dataclass(frozen=True)
class BodyComponent:
    """An axisymmetric external rocket body segment (nosecone, bodytube, transition)."""

    name: str
    component_type: str  # nosecone | bodytube | transition
    length: float
    fore_radius: float
    aft_radius: float
    shape: str = ""
    shape_parameter: float = 0.0
    shape_clipped: bool = False
    finish: SurfaceFinish = SurfaceFinish.NORMAL
    axial_start: float = 0.0
    axial_end: float = 0.0
    fins: tuple[FinSet, ...] = ()
    rail_buttons: tuple[RailButton, ...] = ()
    launch_lugs: tuple[LaunchLug, ...] = ()

    @property
    def max_radius(self) -> float:
        """Maximum outer radius of this segment in meters."""
        return max(self.fore_radius, self.aft_radius)

    def radius_at_local_x(self, local_x: float) -> float:
        """Exact analytical radius in meters at local axial distance from component start."""
        return evaluate_profile_radius(
            component_type=self.component_type,
            shape=self.shape,
            shape_parameter=self.shape_parameter,
            length=self.length,
            fore_radius=self.fore_radius,
            aft_radius=self.aft_radius,
            local_x=local_x,
            shape_clipped=self.shape_clipped,
        )


@dataclass(frozen=True)
class Stage:
    """An inline stage containing sequential aerodynamic body components."""

    name: str
    components: tuple[BodyComponent, ...] = ()
    axial_start: float = 0.0
    axial_end: float = 0.0

    @property
    def length(self) -> float:
        """Axial length of the stage in meters."""
        return max(0.0, self.axial_end - self.axial_start)


@dataclass(frozen=True)
class RocketModel:
    """Complete parsed OpenRocket vehicle model containing only external OML geometry."""

    name: str
    stages: tuple[Stage, ...] = ()
    source_path: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_length(self) -> float:
        """Total vehicle axial length in meters."""
        if not self.stages:
            return 0.0
        return max(stage.axial_end for stage in self.stages)

    @property
    def max_radius(self) -> float:
        """Maximum body radius in meters."""
        max_r = 0.0
        for comp in self.all_components:
            max_r = max(max_r, comp.fore_radius, comp.aft_radius)
        return max_r

    @property
    def max_diameter(self) -> float:
        """Maximum body diameter in meters."""
        return self.max_radius * 2.0

    @property
    def reference_area(self) -> float:
        """Aerodynamic reference frontal area S_ref = pi * R_max^2 in m^2."""
        r = self.max_radius
        return math.pi * r * r

    @property
    def max_fin_span_diameter(self) -> float:
        """Maximum tip-to-tip span diameter including fins in meters."""
        max_d = self.max_diameter
        for comp in self.all_components:
            for fin in comp.fins:
                local_r = max(
                    comp.radius_at_local_x(max(0.0, fin.root_le_x)),
                    comp.radius_at_local_x(min(comp.length, fin.root_te_x)),
                    comp.aft_radius,
                )
                max_d = max(max_d, 2.0 * (local_r + fin.span))
        return max_d

    @property
    def all_components(self) -> tuple[BodyComponent, ...]:
        """Flat tuple of all external body components from nose to tail."""
        comps: list[BodyComponent] = []
        for stage in self.stages:
            comps.extend(stage.components)
        return tuple(comps)

    @property
    def all_fins(self) -> tuple[FinSet, ...]:
        """Flat tuple of all fin sets across all stages."""
        fins: list[FinSet] = []
        for comp in self.all_components:
            fins.extend(comp.fins)
        return tuple(fins)

    @property
    def stage_count(self) -> int:
        """Number of axial stages in the rocket."""
        return len(self.stages)

    @property
    def is_multistage(self) -> bool:
        """Whether the rocket consists of more than one axial stage."""
        return len(self.stages) > 1

    @property
    def stage_summaries(self) -> tuple[tuple[int, str, float, tuple[int, ...]], ...]:
        """Return (1-based index, name, length_m, fin_counts) summary per stage."""
        return tuple(
            (
                idx,
                stage.name,
                stage.length,
                tuple(fin.fin_count for comp in stage.components for fin in comp.fins),
            )
            for idx, stage in enumerate(self.stages, start=1)
        )

    @property
    def rotational_symmetry_order(self) -> int:
        """GCD of all fin counts across active stages (default 4 if no fins)."""
        counts = [fin.fin_count for fin in self.all_fins if fin.fin_count > 0]
        if not counts:
            return 4
        g = counts[0]
        for c in counts[1:]:
            g = math.gcd(g, c)
        return max(1, g)

    @property
    def dominant_fin_count(self) -> int:
        """Backward-compatible alias for rotational_symmetry_order."""
        return self.rotational_symmetry_order

    @property
    def has_sector_mirror_symmetry(self) -> bool:
        """True if fin cant is zero and all fin sets align with the primary mirror planes."""
        if self.has_cant:
            return False
        fins = self.all_fins
        if len(fins) <= 1:
            return True
        n = self.rotational_symmetry_order
        if n < 2:
            return False
        half_pitch = 180.0 / n
        base_rot = fins[0].rotation_deg
        for fin in fins[1:]:
            rem = (fin.rotation_deg - base_rot) % half_pitch
            if min(rem, half_pitch - rem) > 0.5:
                return False
        return True

    def resolve_stage_indices(self, selection: str = "all") -> list[int]:
        """Resolve a stage selection key to 0-based stage indices in `self.stages`."""
        sel = selection.strip().lower()
        n_stages = len(self.stages)
        if n_stages == 0:
            return []
        if sel in {"", "all"} or n_stages <= 1:
            return list(range(n_stages))
        if sel == "sustainer":
            return [0]
        if sel == "booster":
            return [n_stages - 1]
        if sel.startswith("stage_") and sel[6:].isdigit():
            k = int(sel[6:])
            if not 1 <= k <= n_stages:
                raise ValueError(f"Unknown stage index {k} for {n_stages}-stage rocket.")
            return [k - 1]
        if sel.isdigit():
            k = int(sel)
            if not 1 <= k <= n_stages:
                raise ValueError(f"Unknown stage index {k} for {n_stages}-stage rocket.")
            return [k - 1]
        if sel.startswith("up_to_") and sel[6:].isdigit():
            k = int(sel[6:])
            if not 1 <= k <= n_stages:
                raise ValueError(f"Unknown stage prefix count {k} for {n_stages}-stage rocket.")
            return list(range(k))
        matches = [idx for idx, st in enumerate(self.stages) if st.name.strip().lower() == sel]
        if not matches:
            raise ValueError(
                f"Unknown stage selection '{selection}' for {n_stages}-stage rocket."
            )
        return [matches[0]]

    def select_stages(self, selection: str = "all") -> RocketModel:
        """Return a sub-model with selected stages and re-zeroed axial coordinates."""
        sel = selection.strip().lower()
        if sel in {"", "all"} or len(self.stages) <= 1:
            return self

        chosen = [self.stages[idx] for idx in self.resolve_stage_indices(selection)]

        offset = chosen[0].axial_start
        shifted_stages: list[Stage] = []
        for st in chosen:
            new_comps: list[BodyComponent] = []
            for comp in st.components:
                new_fins = tuple(
                    replace(
                        fin,
                        global_root_le_x=fin.global_root_le_x - offset,
                        global_root_te_x=fin.global_root_te_x - offset,
                    )
                    for fin in comp.fins
                )
                new_rb = tuple(
                    replace(rb, axial_x=rb.axial_x - offset) for rb in comp.rail_buttons
                )
                new_ll = tuple(
                    replace(ll, axial_x=ll.axial_x - offset) for ll in comp.launch_lugs
                )
                new_comps.append(
                    replace(
                        comp,
                        axial_start=comp.axial_start - offset,
                        axial_end=comp.axial_end - offset,
                        fins=new_fins,
                        rail_buttons=new_rb,
                        launch_lugs=new_ll,
                    )
                )
            shifted_stages.append(
                replace(
                    st,
                    components=tuple(new_comps),
                    axial_start=st.axial_start - offset,
                    axial_end=st.axial_end - offset,
                )
            )

        return RocketModel(
            name=self.name,
            stages=tuple(shifted_stages),
            source_path=self.source_path,
            warnings=self.warnings,
        )

    @property
    def has_cant(self) -> bool:
        """Whether any fin set has a non-zero cant angle."""
        return any(abs(fin.cant_deg) > 1e-4 for fin in self.all_fins)

    @property
    def dominant_finish(self) -> SurfaceFinish:
        """Length-weighted surface finish (defaults to coarsest significant body finish)."""
        comps = self.all_components
        if not comps:
            return SurfaceFinish.NORMAL
        return max(comps, key=lambda c: c.finish.roughness_m * max(c.length, 0.01)).finish

    def radius_at_x(self, x: float) -> float:
        """Evaluate exact outer body radius in meters at global axial station x."""
        comps = self.all_components
        if not comps:
            return 0.0
        for comp in comps:
            if comp.axial_start - 1e-9 <= x <= comp.axial_end + 1e-9:
                return comp.radius_at_local_x(x - comp.axial_start)
        if x < comps[0].axial_start:
            return comps[0].fore_radius
        return comps[-1].aft_radius
