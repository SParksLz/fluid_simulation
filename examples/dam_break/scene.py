"""Canonical dam-break particle specifications, taken from the APIC demo."""

from dataclasses import asdict, dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class SceneSpec:
    bound_lo: tuple = (-1.20, -0.28, 0.0)
    bound_hi: tuple = (1.20, 0.28, 1.20)
    fill_x_frac: float = 0.30
    fill_z_frac: float = 0.72
    margin: float = 0.03
    density: float = 1000.0
    gravity: tuple = (0.0, 0.0, -10.0)
    fps: float = 60.0
    target_count: int = 400000


DEFAULT_SPEC = SceneSpec()


def spawn_particle_block(lo, hi, res, density: float = 1000.0):
    xs = np.linspace(lo[0], hi[0], res[0])
    ys = np.linspace(lo[1], hi[1], res[1])
    zs = np.linspace(lo[2], hi[2], res[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    spacing = float(
        min(
            (hi[0] - lo[0]) / max(res[0] - 1, 1),
            (hi[1] - lo[1]) / max(res[1] - 1, 1),
            (hi[2] - lo[2]) / max(res[2] - 1, 1),
        )
    )
    volume = spacing**3
    mass = density * volume
    radius = 0.5 * spacing
    return positions, mass, radius, volume, spacing


def spawn_wave_tank(
    tank_lo,
    tank_hi,
    target_count: int = 400000,
    fill_x_frac: float = 0.30,
    fill_z_frac: float = 0.70,
    margin: float = 0.03,
    density: float = 1000.0,
):
    """Preserve the APIC demo's endpoint lattice and resolution rounding."""
    if target_count < 1:
        raise ValueError("Target particle count must be positive")
    tank_lo = np.asarray(tank_lo, dtype=np.float32)
    tank_hi = np.asarray(tank_hi, dtype=np.float32)
    extent = tank_hi - tank_lo
    water_lo = np.array(
        [tank_lo[0] + margin, tank_lo[1] + margin, tank_lo[2] + margin],
        dtype=np.float32,
    )
    water_hi = np.array(
        [
            tank_lo[0] + margin + fill_x_frac * max(extent[0] - 2.0 * margin, 1.0e-3),
            tank_hi[1] - margin,
            tank_lo[2] + margin + fill_z_frac * max(extent[2] - 2.0 * margin, 1.0e-3),
        ],
        dtype=np.float32,
    )
    water_ext = np.maximum(water_hi - water_lo, 1.0e-3)
    water_volume = float(np.prod(water_ext))
    spacing = float((water_volume / max(target_count, 1)) ** (1.0 / 3.0))
    res = tuple(max(2, int(round(e / spacing))) for e in water_ext)
    return spawn_particle_block(tuple(water_lo), tuple(water_hi), res, density=density)


@dataclass
class ParticleScene:
    spec: SceneSpec
    positions: np.ndarray
    mass: float
    radius: float
    volume: float
    spacing: float

    @property
    def count(self):
        return len(self.positions)

    @property
    def rest_volume(self):
        return self.count * self.mass / self.spec.density

    @property
    def ideal_height(self):
        lo, hi = self.spec.bound_lo, self.spec.bound_hi
        return self.rest_volume / ((hi[0] - lo[0]) * (hi[1] - lo[1]))

    def report(self):
        return {
            **asdict(self.spec),
            "actual_count": self.count,
            "lattice_resolution": [len(np.unique(self.positions[:, i])) for i in range(3)],
            "particle_spacing": self.spacing,
            "particle_radius": self.radius,
            "particle_volume": self.volume,
            "particle_mass": self.mass,
            "rest_volume": self.rest_volume,
            "ideal_flat_height": self.ideal_height,
            "initial_position_sha256": hashlib.sha256(self.positions.tobytes()).hexdigest(),
            "water_lo": self.positions.min(axis=0).tolist(),
            "water_hi": self.positions.max(axis=0).tolist(),
            "units": "meters, kilograms, seconds; Z up",
        }


def make_scene(particle_count=DEFAULT_SPEC.target_count):
    spec = SceneSpec(target_count=int(particle_count))
    return ParticleScene(spec, *spawn_wave_tank(
        spec.bound_lo, spec.bound_hi, spec.target_count,
        spec.fill_x_frac, spec.fill_z_frac, spec.margin, spec.density,
    ))
