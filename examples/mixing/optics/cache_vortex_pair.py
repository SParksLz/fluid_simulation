"""Cache a tank replay driven by a localized pair of counter-rotating vortices."""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, grey_opening, maximum_filter
import warp as wp

from examples.mixing.optics.cache_curl_noise import impact_ramp, write_scene, write_snapshot
from examples.mixing.tank import TankMixingSimulation


@wp.func
def vortex_acceleration(offset: wp.vec2, support: float) -> wp.vec2:
    radial = offset / support
    edge = wp.max(0.0, 1.0 - wp.dot(radial, radial))
    # The radial weight keeps the compact azimuthal field divergence-free.
    return 4.2 * edge * edge * edge * wp.vec2(-radial[1], radial[0])


@wp.func
def pool_weight(height: float) -> float:
    lift = wp.clamp((0.22 - height) / 0.05, 0.0, 1.0)
    return lift * lift * lift * (10.0 + lift * (-15.0 + 6.0 * lift))


@wp.kernel
def apply_vortex_pair(
    positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
    center_x: float, support: float, impulse: float,
):
    i = wp.tid()
    p = positions[i]
    if p[2] >= 0.22 or wp.abs(p[0] - center_x) >= support:
        return
    lower = vortex_acceleration(wp.vec2(p[0] - center_x, p[1] - 0.25), support)
    upper = vortex_acceleration(wp.vec2(p[0] - center_x, p[1] - 0.42), support)
    # The two vortices sweep in the same direction between their cores.
    flow = 0.5 * (lower - upper)
    velocities[i] += impulse * pool_weight(p[2]) * wp.vec3(flow[0], flow[1], 0.0)


@wp.kernel
def apply_secondary_vortices(
    positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
    lower_center: wp.vec2, upper_center: wp.vec2,
    lower_impulse: float, upper_impulse: float,
):
    i = wp.tid()
    p = positions[i]
    if p[2] >= 0.22:
        return
    xy = wp.vec2(p[0], p[1])
    lower = vortex_acceleration(xy - lower_center, 0.115)
    upper = vortex_acceleration(xy - upper_center, 0.115)
    # Each secondary vortex counter-rotates beside its primary vortex.
    flow = upper_impulse * upper - lower_impulse * lower
    velocities[i] += pool_weight(p[2]) * wp.vec3(flow[0], flow[1], 0.0)


class MilkFrontTracker:
    """Keep two vortex centers near the flanks of the surface concentration front."""

    spacing = 0.01
    nx, ny = 120, 60

    def __init__(self):
        x = (np.arange(self.nx) + 0.5) * self.spacing
        y = (np.arange(self.ny) + 0.5) * self.spacing
        self.x, self.y = np.meshgrid(x, y)
        # Leave room beside the walls and exclude the footprint of the pure jet.
        region = ((self.x > 0.13) & (self.x < 0.81)
                  & (self.y > 0.095) & (self.y < 0.505)
                  & (np.hypot(self.x - 0.80, self.y - 0.34) > 0.11))
        self.regions = (region & (self.y < 0.32), region & (self.y > 0.35))
        self.centers = np.array([[0.67, 0.20], [0.67, 0.47]], dtype=float)
        self.targets = self.centers.copy()
        self.active = np.zeros(2, dtype=float)
        self.target_active = self.active.copy()

    def update(self, positions, concentration):
        pool = positions[:, 2] < 0.22
        points = positions[pool]
        ix = np.clip((points[:, 0] / self.spacing).astype(int), 0, self.nx - 1)
        iy = np.clip((points[:, 1] / self.spacing).astype(int), 0, self.ny - 1)
        indices = iy * self.nx + ix
        shape = self.ny, self.nx
        top = np.zeros(self.nx * self.ny)
        np.maximum.at(top, indices, points[:, 2])
        top = maximum_filter(top.reshape(shape), size=3)
        top = gaussian_filter(grey_opening(top, size=(5, 5)), 1.0)
        depth = points[:, 2] - (top[iy, ix] - 0.020)
        weight = np.exp(-0.5 * (depth / 0.015) ** 2)
        mass = np.bincount(indices, weight, minlength=top.size).reshape(shape)
        dye = np.bincount(indices, weight * concentration[pool], minlength=top.size).reshape(shape)
        mass = gaussian_filter(mass, 1.35)
        field = gaussian_filter(dye, 1.35) / np.maximum(1.0e-8, mass)
        dy, dx = np.gradient(field, self.spacing)
        gradient = np.hypot(dx, dy)

        # Find a strong edge near 28% milk, within the neighborhood of the last
        # center. Average around its peak so particles cannot snap the target.
        front = (mass > 0.45) & (field > 0.12) & (field < 0.52) & (gradient > 0.9)
        strength = np.minimum(gradient, 12.0) * np.exp(-0.5 * ((field - 0.28) / 0.095) ** 2)
        for i, region in enumerate(self.regions):
            x, y = self.centers[i]
            distance = np.hypot(self.x - x, self.y - y)
            score = np.where(front & region & (distance < 0.19),
                             strength * np.exp(-0.5 * (distance / 0.085) ** 2), 0.0)
            row, col = np.unravel_index(score.argmax(), score.shape)
            if score[row, col] == 0.0:
                self.target_active[i] = 0.0
                continue
            local = np.hypot(self.x - self.x[row, col], self.y - self.y[row, col])
            weights = score * np.exp(-0.5 * (local / 0.028) ** 2)
            # Place the core on the coffee side so the strongest part of the
            # swirl reaches across the milk edge rather than orbiting inside it.
            normal = np.maximum(gradient, 1.0e-8)
            target_x = np.average(self.x - 0.027 * dx / normal, weights=weights)
            target_y = np.average(self.y - 0.027 * dy / normal, weights=weights)
            self.targets[i] = np.clip((target_x, target_y), (0.12, 0.09), (0.78, 0.51))
            self.target_active[i] = 1.0

    def advance(self, dt):
        blend = -math.expm1(-dt / 0.10)
        for i in range(len(self.centers)):
            dx, dy = self.targets[i] - self.centers[i]
            step = min(blend, 0.40 * dt / max(math.hypot(dx, dy), 1.0e-8))
            self.centers[i, 0] += step * dx
            self.centers[i, 1] += step * dy
            self.active[i] += blend * (self.target_active[i] - self.active[i])


def configure_vortices(sim, support, acceleration, secondary_acceleration=0.0, tracker=None):
    advance = sim.solver.step

    def step_with_vortices(state_in, state_out, control, contacts, dt):
        time = sim.sim_time
        strength = impact_ramp(time - 0.15, sim.emitter_duration - 0.15)
        if strength > 0.0:
            center = 0.78 - 0.10 * min(1.65, max(0.0, time - 0.45))
            if acceleration > 0.0:
                wp.launch(apply_vortex_pair, dim=sim.count,
                          inputs=[state_in.particle_q, state_in.particle_qd,
                                  center, support, dt * acceleration * strength],
                          device=sim.model.device)
            if secondary_acceleration > 0.0 and time > 0.65:
                lower_strength = impact_ramp(time - 0.40, sim.emitter_duration - 0.40)
                upper_strength = impact_ramp(time - 0.65, sim.emitter_duration - 0.65)
                if tracker is None:
                    lower_angle = -2.72 + 0.85 * max(0.0, time - 0.65)
                    upper_angle = 2.72 - 0.85 * max(0.0, time - 0.90)
                    lower_center = wp.vec2(center + 0.115 * math.cos(lower_angle),
                                           0.25 + 0.115 * math.sin(lower_angle))
                    upper_center = wp.vec2(center + 0.115 * math.cos(upper_angle),
                                           0.42 + 0.115 * math.sin(upper_angle))
                else:
                    tracker.advance(dt)
                    lower_center, upper_center = (wp.vec2(*p) for p in tracker.centers)
                    lower_strength *= tracker.active[0]
                    upper_strength *= tracker.active[1]
                impulse = dt * secondary_acceleration
                wp.launch(apply_secondary_vortices, dim=sim.count,
                          inputs=[state_in.particle_q, state_in.particle_qd,
                                  lower_center, upper_center,
                                  impulse * lower_strength, impulse * upper_strength],
                          device=sim.model.device)
        advance(state_in, state_out, control, contacts, dt)

    sim.solver.step = step_with_vortices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--vortex-support", type=float, default=0.19,
                        help="outer radius of each vortex in meters")
    parser.add_argument("--vortex-acceleration", type=float, default=1.8,
                        help="acceleration scale in m/s²")
    parser.add_argument("--secondary-acceleration", type=float, default=0.0,
                        help="acceleration scale of each secondary vortex in m/s²")
    parser.add_argument("--track-milk-front", action="store_true",
                        help="steer secondary vortices along the near-surface concentration front")
    args = parser.parse_args()
    if (args.frames < 1 or args.stride < 1
            or not math.isfinite(args.vortex_support) or args.vortex_support <= 0.0
            or not all(math.isfinite(a) and a >= 0.0
                       for a in (args.vortex_acceleration, args.secondary_acceleration))):
        parser.error("use positive frames, stride and vortex support, with finite nonnegative acceleration")
    if args.track_milk_front and args.secondary_acceleration == 0.0:
        parser.error("--track-milk-front needs a positive --secondary-acceleration")

    sim = TankMixingSimulation(initial_coffee_height=0.15, emitter_speed=0.6,
                               emitter_color_period=0.0, diffusivity=0.003)
    tracker = MilkFrontTracker() if args.track_milk_front else None
    configure_vortices(sim, args.vortex_support, args.vortex_acceleration, args.secondary_acceleration, tracker)
    write_scene(sim, args.output)
    write_snapshot(sim, args.output, 0)
    tracks = []
    for frame in range(1, args.frames + 1):
        if tracker is not None and 0.63 <= sim.sim_time < sim.emitter_duration + 0.8 and frame % 2 == 0:
            tracker.update(sim.state_0.particle_q.numpy(), sim.concentration.numpy())
        metrics = sim.step()
        if frame % args.stride == 0 or frame == args.frames:
            write_snapshot(sim, args.output, frame)
            if tracker is not None:
                ramps = [impact_ramp(sim.sim_time - delay, sim.emitter_duration - delay)
                         for delay in (0.40, 0.65)]
                tracks.append([frame, *tracker.centers.flatten(), *(tracker.active * ramps)])
        if frame % 10 == 0 or frame == args.frames:
            print(f"{frame=} {sim.count=} milk_error={metrics['dye_relative_error']:.3g} "
                  f"rms_speed={metrics['rms_speed_m_s']:.3g} m/s", flush=True)
    if tracker is not None:
        np.savetxt(args.output / "secondary-centers.csv", tracks, delimiter=",", fmt="%.6f",
                   header="frame,lower_x,lower_y,upper_x,upper_y,lower_weight,upper_weight", comments="")


if __name__ == "__main__":
    main()
