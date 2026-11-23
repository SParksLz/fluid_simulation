"""Concentration-dependent absorption and scattering for SSFR previews.

Coefficients are demonstration values in inverse meters, chosen for the tank's
centimeter-to-decimeter optical paths. Concentration is the volume fraction of
the second medium. The shader integrates homogeneous depth slabs with a
constant incident scattering source approximation.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from examples.mixing.color import colors_numpy


@dataclass(frozen=True)
class OpticalMedium:
    name: str
    absorption: tuple[float, float, float]
    scattering: tuple[float, float, float]

    def __post_init__(self):
        for values in (self.absorption, self.scattering):
            if len(values) != 3 or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("Medium coefficients must be three finite nonnegative values")


COFFEE = OpticalMedium("coffee", (18.0, 45.0, 100.0), (0.1, 0.1, 0.1))
MILK = OpticalMedium("milk", (0.15, 0.25, 0.5), (80.0, 80.0, 80.0))
WATER = OpticalMedium("water", (0.02, 0.008, 0.004), (0.01, 0.01, 0.01))
MEDIUM_PRESETS = {"coffee-milk": (COFFEE, MILK), "water-milk": (WATER, MILK)}
MEDIUM_MODES = ("pigment", *MEDIUM_PRESETS)
MEDIUM_LAYERS = 8
MAX_MEDIUM_LAYERS = 16


def coefficients_numpy(concentration, preset, scale=1.0):
    first, second = MEDIUM_PRESETS[preset]
    c = np.clip(np.asarray(concentration, dtype=np.float64), 0.0, 1.0)[..., None]
    absorption = (1 - c) * np.asarray(first.absorption) + c * np.asarray(second.absorption)
    scattering = (1 - c) * np.asarray(first.scattering) + c * np.asarray(second.scattering)
    return scale * absorption, scale * scattering


def display_colors(concentration, color_model="km", medium="pigment", medium_scale=1.0):
    """Representative colors for markers and particle USD at a 3 cm path.

    A white background and unit incident source define this reference swatch.
    The SSFR shader evaluates its actual thickness and studio illumination.
    """
    if medium == "pigment":
        return colors_numpy(concentration, color_model)
    absorption, scattering = coefficients_numpy(concentration, medium, medium_scale)
    extinction = absorption + scattering
    transmission = np.exp(-0.03 * extinction)
    albedo = np.divide(scattering, extinction, out=np.zeros_like(extinction), where=extinction > 0)
    return (transmission + albedo * (1 - transmission)).astype(np.float32)


def medium_metadata(preset, scale, layers=MEDIUM_LAYERS):
    first, second = MEDIUM_PRESETS[preset]
    return dict(
        preset=preset, concentration="volume fraction of " + second.name,
        coefficient_units="m^-1", coefficient_scale=scale,
        first=asdict(first), second=asdict(second),
        model="front-to-back homogeneous slabs with constant incident scattering source",
        optical_layers=layers, slab_width_particle_radii=4.0,
        final_slab="all remaining visible thickness",
        calibration="demonstration coefficients for tank-scale optical paths",
    )
