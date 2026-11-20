"""ISA 1976 atmospheric model and Sutherland viscosity for aerodynamic CFD."""

from __future__ import annotations

import math

from ..models.results import AtmosphereState

_G0 = 9.80665
_R_AIR = 287.05287
_GAMMA = 1.4
_SUTHERLAND_MU0 = 1.716e-5
_SUTHERLAND_T0 = 273.15
_SUTHERLAND_S = 110.4

# Standard atmosphere layer bases: (h_base_m, T_base_K, lapse_K_per_m, P_base_Pa)
_ISA_LAYERS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.0010, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
)


def sutherland_viscosity(temperature_k: float) -> float:
    """Compute air dynamic viscosity (Pa*s) via Sutherland's law."""
    t = max(50.0, temperature_k)
    return _SUTHERLAND_MU0 * ((t / _SUTHERLAND_T0) ** 1.5) * (
        (_SUTHERLAND_T0 + _SUTHERLAND_S) / (t + _SUTHERLAND_S)
    )


def compute_atmosphere(altitude_m: float) -> AtmosphereState:
    """Compute ISA 1976 atmospheric state at geometric altitude (0 to 50,000 m)."""
    h = max(-500.0, min(50000.0, altitude_m))

    layer = _ISA_LAYERS[0]
    for candidate in _ISA_LAYERS:
        if h >= candidate[0]:
            layer = candidate
        else:
            break

    h_b, t_b, lapse, p_b = layer
    dh = h - h_b

    if abs(lapse) < 1e-12:
        temperature = t_b
        pressure = p_b * math.exp(-_G0 * dh / (_R_AIR * t_b))
    else:
        temperature = t_b + lapse * dh
        exponent = -_G0 / (_R_AIR * lapse)
        pressure = p_b * ((temperature / t_b) ** exponent)

    density = pressure / (_R_AIR * temperature)
    viscosity = sutherland_viscosity(temperature)
    speed_of_sound = math.sqrt(_GAMMA * _R_AIR * temperature)

    return AtmosphereState(
        altitude_m=altitude_m,
        temperature_k=temperature,
        pressure_pa=pressure,
        density_kg_m3=density,
        viscosity_pa_s=viscosity,
        speed_of_sound_m_s=speed_of_sound,
    )
