"""Pure mathematical aerodynamic profile functions for OpenRocket components."""

from __future__ import annotations

import math


def evaluate_profile_radius(
    component_type: str,
    shape: str,
    shape_parameter: float,
    length: float,
    fore_radius: float,
    aft_radius: float,
    local_x: float,
    shape_clipped: bool = False,
) -> float:
    """Evaluate exact outer radius r(x) in meters at axial coordinate local_x in [0, length]."""
    if length <= 0.0:
        return max(0.0, aft_radius)

    x = max(0.0, min(length, local_x))
    t = x / length
    ctype = component_type.lower()
    s = shape.lower() if shape else ""

    if ctype == "bodytube":
        return max(0.0, aft_radius)

    if ctype == "nosecone":
        return _eval_nosecone_radius(s, shape_parameter, length, aft_radius, x, t)

    if ctype == "transition":
        return _eval_transition_radius(
            s, shape_parameter, length, fore_radius, aft_radius, x, t, shape_clipped
        )

    return max(0.0, fore_radius + (aft_radius - fore_radius) * t)


def _eval_nosecone_radius(
    shape: str,
    param: float,
    length: float,
    base_r: float,
    x: float,
    t: float,
) -> float:
    """Evaluate analytical nose cone radius r(x) for standard OpenRocket shapes."""
    if base_r <= 0.0:
        return 0.0

    if "haack" in shape:
        ratio = max(-1.0, min(1.0, 1.0 - 2.0 * t))
        theta = math.acos(ratio)
        term = theta - 0.5 * math.sin(2.0 * theta) + param * (math.sin(theta) ** 3)
        return base_r * math.sqrt(max(0.0, term) / math.pi)

    if "ogive" in shape:
        # Standard tangent ogive; secant ogive if shape_parameter != 0 and != 1
        rho = (base_r * base_r + length * length) / (2.0 * base_r)
        is_secant = ("secant" in shape) or (abs(param) > 1e-6 and abs(param - 1.0) > 1e-6)
        if is_secant and abs(param) > 1e-6:
            # OpenRocket secant ogive / tangent parameter k uses rho_secant = rho / k
            rho = max(rho / param, 0.5 * math.hypot(length, base_r) + 1e-12)
        if is_secant:
            alpha = math.atan2(base_r, length) - math.acos(
                max(-1.0, min(1.0, math.hypot(length, base_r) / (2.0 * rho)))
            )
            val = rho * rho - (rho * math.cos(alpha) - x) ** 2
            return max(0.0, math.sqrt(max(0.0, val)) + rho * math.sin(alpha))
        val = rho * rho - (length - x) ** 2
        return max(0.0, math.sqrt(max(0.0, val)) + base_r - rho)

    if "ellipsoid" in shape:
        term = 1.0 - ((length - x) / length) ** 2
        return base_r * math.sqrt(max(0.0, term))

    if "parabolic" in shape:
        k = param if param != 0.0 else 0.5
        denom = 2.0 - k if abs(2.0 - k) > 1e-9 else 1.0
        return max(0.0, base_r * ((2.0 * t - k * (t * t)) / denom))

    if "power" in shape:
        n = param if param > 0.0 else 0.5
        return max(0.0, base_r * (t**n))

    # Conical fallback
    return max(0.0, base_r * t)


def _eval_transition_radius(
    shape: str,
    param: float,
    length: float,
    fore_r: float,
    aft_r: float,
    x: float,
    t: float,
    shape_clipped: bool,
) -> float:
    """Evaluate transition radius r(x) between fore_r and aft_r."""
    if abs(aft_r - fore_r) < 1e-12:
        return max(0.0, fore_r)

    r_min = min(fore_r, aft_r)
    r_max = max(fore_r, aft_r)
    u = t if aft_r >= fore_r else (1.0 - t)

    if "conical" in shape or not shape:
        return max(0.0, fore_r + (aft_r - fore_r) * t)

    if "ellipsoid" in shape:
        # Half-elliptical transition curve scaled uniformly across [r_min, r_max]
        frac = math.sqrt(max(0.0, 1.0 - (1.0 - u) ** 2))
        return max(0.0, r_min + (r_max - r_min) * frac)

    if "ogive" in shape:
        dr = r_max - r_min
        rho = (dr * dr + length * length) / (2.0 * dr)
        local = u * length
        val = rho * rho - (length - local) ** 2
        offset = math.sqrt(max(0.0, val)) + dr - rho
        return max(0.0, r_min + max(0.0, offset))

    if "parabolic" in shape:
        k = param if param != 0.0 else 0.5
        denom = 2.0 - k if abs(2.0 - k) > 1e-9 else 1.0
        frac = (2.0 * u - k * u * u) / denom
        return max(0.0, r_min + (r_max - r_min) * frac)

    if "haack" in shape:
        ratio = max(-1.0, min(1.0, 1.0 - 2.0 * u))
        theta = math.acos(ratio)
        term = theta - 0.5 * math.sin(2.0 * theta) + param * (math.sin(theta) ** 3)
        frac = math.sqrt(max(0.0, term) / math.pi)
        return max(0.0, r_min + (r_max - r_min) * frac)

    if "power" in shape:
        n = param if param > 0.0 else 0.5
        return max(0.0, r_min + (r_max - r_min) * (u**n))

    return max(0.0, fore_r + (aft_r - fore_r) * t)
