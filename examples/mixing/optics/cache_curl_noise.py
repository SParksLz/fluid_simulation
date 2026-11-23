"""Cache the coffee-and-milk tank with an optional local 3D curl perturbation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from examples.mixing.tank import TankMixingSimulation


@wp.func
def impact_potential(position: wp.vec3, time: float, scale: float, ceiling: float) -> wp.vec3:
    offset = position - wp.vec3(0.80, 0.34, 0.14)
    local = wp.cw_div(offset, wp.vec3(0.24, 0.20, 0.13))
    window = wp.max(0.0, 1.0 - wp.dot(local, local))
    lift = wp.clamp((ceiling - position[2]) / 0.05, 0.0, 1.0)
    if window <= 0.0 or lift <= 0.0:
        return wp.vec3(0.0)
    p = offset / scale
    sample = wp.vec4(p[0], p[1], p[2], time / 0.8)
    potential = wp.vec3(
        wp.noise(wp.rand_init(71), sample),
        wp.noise(wp.rand_init(231), sample),
        wp.noise(wp.rand_init(478), sample),
    )
    # Localize the vector potential before taking its spatial curl.
    height_fade = lift * lift * lift * (10.0 + lift * (-15.0 + 6.0 * lift))
    return scale * window * window * window * height_fade * potential


@wp.func
def impact_curl(position: wp.vec3, time: float, scale: float, ceiling: float) -> wp.vec3:
    step = 0.01 * scale
    x = wp.vec3(step, 0.0, 0.0)
    y = wp.vec3(0.0, step, 0.0)
    z = wp.vec3(0.0, 0.0, step)
    dx = impact_potential(position + x, time, scale, ceiling) - impact_potential(position - x, time, scale, ceiling)
    dy = impact_potential(position + y, time, scale, ceiling) - impact_potential(position - y, time, scale, ceiling)
    dz = impact_potential(position + z, time, scale, ceiling) - impact_potential(position - z, time, scale, ceiling)
    return wp.vec3(dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]) / (2.0 * step)


@wp.kernel
def apply_curl_acceleration(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    time: float,
    scale: float,
    ceiling: float,
    impulse: float,
):
    i = wp.tid()
    position = positions[i]
    # The enlarged box includes all samples touched by the derivative stencil.
    if wp.abs(position[0] - 0.80) > 0.24 + 0.01 * scale:
        return
    if wp.abs(position[1] - 0.34) > 0.20 + 0.01 * scale:
        return
    if position[2] < 0.01 - 0.01 * scale or position[2] > wp.min(0.27, ceiling) + 0.01 * scale:
        return
    velocities[i] += impulse * impact_curl(position, time, scale, ceiling)


def write_snapshot(sim, output, frame, concentration=None):
    positions = sim.state_0.particle_q.numpy()
    if concentration is None:
        concentration = sim.concentration
    pending = output / f"{frame:03d}.pending.npz"
    np.savez(pending, positions=positions, concentration=concentration.numpy())
    pending.replace(output / f"{frame:03d}.npz")


def impact_ramp(time, duration):
    fade_in = (time - 0.25) / 0.25
    fade_out = (duration + 0.80 - time) / 0.50
    ramp = max(0.0, min(1.0, fade_in, fade_out))
    return ramp * ramp * (3.0 - 2.0 * ramp)


def configure_impact_curl(sim, scale, acceleration, ceiling):
    if not acceleration:
        return
    advance = sim.solver.step

    def step_with_curl(state_in, state_out, control, contacts, dt):
        impulse = dt * acceleration * impact_ramp(sim.sim_time, sim.emitter_duration)
        if impulse > 0.0:
            wp.launch(apply_curl_acceleration, dim=sim.count,
                      inputs=[state_in.particle_q, state_in.particle_qd,
                              sim.sim_time, scale, ceiling, impulse],
                      device=sim.model.device)
        # The existing APIC transfer, wall conditions and pressure projection follow.
        advance(state_in, state_out, control, contacts, dt)

    sim.solver.step = step_with_curl


def write_scene(sim, output):
    output.mkdir(parents=True, exist_ok=True)
    scene = {
        "bound_lo": sim.bound_lo,
        "bound_hi": sim.bound_hi,
        "spacing": sim.spacing,
        "volume": sim.volume,
        "emitter_radius": sim.emitter_radius,
        "emitters": [
            dict(name=e.name, center=e.center, direction=e.direction, concentration=e.concentration)
            for e in sim.emitters
        ],
    }
    (output / "scene.json").write_text(json.dumps(scene))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--diffusivity", type=float, default=3e-3)
    parser.add_argument("--curl-scale", type=float, default=0.14, help="noise scale in meters")
    parser.add_argument("--curl-acceleration", type=float, default=0.64,
                        help="acceleration multiplier in m/s², zero disables the perturbation")
    parser.add_argument("--curl-ceiling", type=float, default=0.23,
                        help="height in meters where the upper fade reaches zero")
    args = parser.parse_args()
    if (args.frames < 1 or args.stride < 1 or
            not math.isfinite(args.curl_scale) or args.curl_scale <= 0.0 or
            not all(math.isfinite(value) and value >= 0.0
                    for value in (args.diffusivity, args.curl_acceleration, args.curl_ceiling))):
        parser.error("expected positive frame count, stride and curl scale, and finite nonnegative forcing and diffusivity")

    sim = TankMixingSimulation(initial_coffee_height=0.15, emitter_speed=0.6,
                               emitter_color_period=0.0, diffusivity=args.diffusivity)
    configure_impact_curl(sim, args.curl_scale, args.curl_acceleration, args.curl_ceiling)
    write_scene(sim, args.output)
    write_snapshot(sim, args.output, 0)
    for frame in range(1, args.frames + 1):
        metrics = sim.step()
        if frame % args.stride == 0 or frame == args.frames:
            write_snapshot(sim, args.output, frame)
        if frame % 10 == 0 or frame == args.frames:
            print(f"{frame=} {sim.count=} milk_error={metrics['dye_relative_error']:.3g} "
                  f"rms_speed={metrics['rms_speed_m_s']:.3g} m/s", flush=True)


if __name__ == "__main__":
    main()
