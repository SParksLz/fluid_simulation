"""Three-channel Kubelka--Munk approximation for the mixing preview.

Reference: https://vanity-ibex.xyz/blog/kubelka_munk_colormixing/
RGB components act as effective reflectances; scattering S=1 for both
endmembers. This is an opaque-pigment preview, without spectral pigment
data or a finite-thickness optical model. The transported scalar is never
modified here.
"""

import numpy as np
import warp as wp


YELLOW = (0.95, 0.85, 0.05)
BLUE = (0.05, 0.15, 0.95)
COLOR_MODELS = {"rgb": 0, "km": 1}
_YELLOW = wp.constant(wp.vec3(*YELLOW))
_BLUE = wp.constant(wp.vec3(*BLUE))
_MIN_REFLECTANCE = 1.0e-6


@wp.func
def reflectance_to_ks_wp(reflectance: wp.vec3):
    ks = wp.vec3(0.0)
    for channel in range(3):
        r = wp.clamp(reflectance[channel], 1.0e-6, 1.0)
        ks[channel] = (1.0-r) * (1.0-r) / (2.0*r)
    return ks


@wp.func
def ks_to_reflectance_wp(ks: wp.vec3):
    reflectance = wp.vec3(0.0)
    for channel in range(3):
        f = wp.max(ks[channel], 0.0)
        # Rationalized inverse: avoids subtracting nearly equal numbers
        # in 1+f-sqrt(f*(f+2)) for strongly absorbing pigments.
        reflectance[channel] = 1.0 / (1.0 + f + wp.sqrt(f*(f+2.0)))
    return reflectance


@wp.kernel
def concentration_colors(concentration: wp.array(dtype=float),
                         colors: wp.array(dtype=wp.vec3), color_model: int):
    i = wp.tid()
    c = wp.clamp(concentration[i], 0.0, 1.0)
    if color_model == 1:
        ks = (1.0-c)*reflectance_to_ks_wp(_YELLOW) + c*reflectance_to_ks_wp(_BLUE)
        colors[i] = ks_to_reflectance_wp(ks)
    else:
        colors[i] = (1.0-c)*_YELLOW + c*_BLUE


def reflectance_to_ks(reflectance):
    r = np.clip(np.asarray(reflectance, dtype=np.float64), _MIN_REFLECTANCE, 1.0)
    return (1.0-r)**2 / (2.0*r)


def ks_to_reflectance(ks):
    f = np.maximum(np.asarray(ks, dtype=np.float64), 0.0)
    return 1.0 / (1.0 + f + np.sqrt(f*(f+2.0)))


def colors_numpy(concentration, color_model="km"):
    if color_model not in COLOR_MODELS:
        raise ValueError(f"Unknown color model: {color_model}")
    # Only the display mapping is bounded; raw LSMPS values are untouched.
    c = np.clip(np.asarray(concentration, dtype=np.float64), 0.0, 1.0)[..., None]
    if color_model == "km":
        ks = (1.0-c)*reflectance_to_ks(YELLOW) + c*reflectance_to_ks(BLUE)
        return ks_to_reflectance(ks).astype(np.float32)
    return ((1.0-c)*np.asarray(YELLOW) + c*np.asarray(BLUE)).astype(np.float32)
