"""Validation checks for parsed OpenRocket models before CFD CAD generation."""

from __future__ import annotations

from ..exceptions import OrkParseError
from ..models.rocket import RocketModel


def validate_rocket_model(rocket: RocketModel) -> list[str]:
    """Validate structural soundness of a parsed RocketModel and collect non-fatal warnings."""
    comps = rocket.all_components
    if not comps:
        raise OrkParseError(f"Rocket '{rocket.name}' contains no external body components.")

    if rocket.total_length <= 0.0:
        raise OrkParseError(f"Rocket '{rocket.name}' has non-positive total length.")

    if rocket.max_radius <= 0.0:
        raise OrkParseError(f"Rocket '{rocket.name}' has non-positive maximum radius.")

    warnings: list[str] = list(rocket.warnings)

    for comp in comps:
        if comp.length <= 0.0:
            warnings.append(f"Component '{comp.name}' has zero/negative length and was skipped.")
        if comp.component_type == "nosecone" and comp.aft_radius <= 0.0:
            warnings.append(f"NoseCone '{comp.name}' resolved to zero aft radius.")
        if comp.component_type == "bodytube" and comp.aft_radius <= 0.0:
            warnings.append(f"BodyTube '{comp.name}' resolved to zero radius.")

        for fin in comp.fins:
            if fin.fin_count <= 0:
                warnings.append(f"FinSet '{fin.name}' has invalid fin_count={fin.fin_count}.")
            if fin.span <= 0.0:
                warnings.append(f"FinSet '{fin.name}' has zero/negative span.")
            if fin.thickness <= 0.0:
                warnings.append(
                    f"FinSet '{fin.name}' has zero thickness; fallback thickness will be applied."
                )
            if fin.root_te_x > comp.length + 0.02 or fin.root_le_x < -0.02:
                warnings.append(
                    f"FinSet '{fin.name}' extends beyond parent '{comp.name}' axial bounds "
                    f"(LE={fin.root_le_x:.3f}m, TE={fin.root_te_x:.3f}m, "
                    f"parent L={comp.length:.3f}m)."
                )
            if fin.fin_type == "freeform" and len(fin.points) < 3:
                warnings.append(
                    f"Freeform FinSet '{fin.name}' has fewer than 3 fin points."
                )

    return warnings
