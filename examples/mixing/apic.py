"""APIC advection plus the cubic particle LSMPS diffusion of Matsunaga (2015)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import newton
import numpy as np
import warp as wp

from solver.apic import SolverAPIC
from solver.apic.lsmps import LSMPSDiffusion
from examples.mixing.color import BLUE, YELLOW, COLOR_MODELS, concentration_colors
from examples.mixing.media import MEDIUM_MODES, display_colors, medium_metadata


@wp.kernel
def stir_water(positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
               acceleration: float, dt: float):
    i = wp.tid()
    x = positions[i]
    X, Y = wp.pi * x[0] / 0.8, wp.pi * x[1] / 0.4
    sx, sy = wp.sin(X), wp.sin(Y)
    ax = acceleration * sx*sx * wp.sin(2.0*Y)
    ay = -0.5 * acceleration * wp.sin(2.0*X) * sy*sy
    velocities[i] += dt * wp.vec3(ax, ay, 0.0)


class MixingSimulation:
    def __init__(self, resolution=24, device="cuda:0", fps=60, substeps=4,
                 diffusivity=1.0e-3, initial_speed=0.5, stirring_acceleration=2.0,
                 color_model="km"):
        if resolution < 12 or resolution % 2 or fps < 1 or substeps < 1:
            raise ValueError("resolution must be even and >=12; fps/substeps must be positive")
        if not all(math.isfinite(v) and v >= 0 for v in (initial_speed, stirring_acceleration)):
            raise ValueError("initial_speed and stirring_acceleration must be finite and nonnegative")
        if color_model not in COLOR_MODELS:
            raise ValueError(f"Unknown color model: {color_model}")
        self.color_model = color_model
        self.stirring_acceleration = stirring_acceleration
        self.resolution, self.fps, self.substeps = resolution, fps, substeps
        self.frame, self.dt = 0, 1.0 / (fps * substeps)
        self.bound_lo, self.bound_hi = (0.0, 0.0, 0.0), (0.8, 0.4, 0.4)
        self.spacing = 0.8 / resolution
        axes = [(np.arange(n) + 0.5) * self.spacing
                for n in (resolution, resolution // 2, resolution // 2)]
        x = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
        u = np.zeros_like(x)
        X, Y = np.pi * x[:, 0] / 0.8, np.pi * x[:, 1] / 0.4
        # Derivatives of a stream function: divergence-free, tangent to all
        # walls. The optional body force below maintains the same swirl.
        u[:, 0] = initial_speed * np.sin(X)**2 * np.sin(2*Y)
        u[:, 1] = -initial_speed * 0.5 * np.sin(2*X) * np.sin(Y)**2
        c = (x[:, 0] >= 0.4).astype(np.float32)
        self.count = len(x)
        self.volume = self.spacing**3
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverAPIC.register_custom_attributes(builder)
        # Use the same attribute enums as the Newton version used by APIC.
        frequency = getattr(newton, "ModelAttributeFrequency", None)
        assignment = getattr(newton, "ModelAttributeAssignment", None)
        if frequency is None:
            frequency, assignment = newton.Model.AttributeFrequency, newton.Model.AttributeAssignment
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name="concentration", frequency=frequency.PARTICLE,
            assignment=assignment.STATE, dtype=wp.float32, default=0.0, namespace="mixing",
        ))
        builder.add_particles(
            pos=x.tolist(), vel=u.tolist(), mass=[1000.0*self.volume] * self.count,
            radius=[0.5*self.spacing] * self.count,
            custom_attributes={"apic:particle_volume": [self.volume] * self.count,
                               "mixing:concentration": c.tolist()},
        )
        self.model = builder.finalize(device=device)
        self.model.set_gravity((0.0, 0.0, 0.0))
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.config = SolverAPIC.Config(
            voxel_size=1.5*self.spacing, bound_lo=self.bound_lo, bound_hi=self.bound_hi,
            wall_margin=0.5*self.spacing, wall_push_strength=0.0, wall_friction=0.0,
            separation_rest_dist=self.spacing, separation_strength=0.15,
            separation_iterations=2, grid_capacity_ratio=16.0,
            projection_tol=1.0e-6, projection_quiet=True,
        )
        with wp.ScopedDevice(self.model.device):
            self.solver = SolverAPIC(self.model, self.config)
        self.diffusion = LSMPSDiffusion(self.count, self.spacing, self.bound_lo, self.bound_hi,
                                        self.model.device, diffusivity=diffusivity)
        self.initial_dye = float(c.astype(np.float64).sum() * self.volume)
        self.initial_variance = float(c.var())
        self.colors = wp.empty(self.count, dtype=wp.vec3, device=self.model.device)
        self.last_diffusion_substeps = 0

    @property
    def concentration(self):
        return self.state_0.mixing.concentration

    def step(self):
        self.last_diffusion_substeps = 0
        for _ in range(self.substeps):
            # Explicit diffusion uses old positions (paper Eq. 26 and Fig. 5).
            self.diffusion.step(self.state_0.particle_q, self.concentration, self.dt)
            self.last_diffusion_substeps += self.diffusion.last_substeps
            if self.stirring_acceleration:
                wp.launch(stir_water, dim=self.count,
                          inputs=[self.state_0.particle_q, self.state_0.particle_qd,
                                  self.stirring_acceleration, self.dt], device=self.model.device)
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            # APIC owns velocity/C; this example owns the passive scalar state.
            wp.copy(self.state_1.mixing.concentration, self.concentration)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.frame += 1
        return self.metrics()

    def update_colors(self):
        wp.launch(concentration_colors, dim=self.count,
                  inputs=[self.concentration, self.colors, COLOR_MODELS[self.color_model]], device=self.model.device)

    def metrics(self):
        x, v, c = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy(), self.concentration.numpy()
        if not (np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(c).all()):
            raise RuntimeError(f"Non-finite mixing state at frame {self.frame}")
        dye = float(c.astype(np.float64).sum() * self.volume)
        variance = float(c.astype(np.float64).var())
        return dict(frame=self.frame, time_s=self.frame/self.fps,
                    concentration_min=float(c.min()), concentration_max=float(c.max()),
                    concentration_mean=float(c.mean()), concentration_variance=variance,
                    dye_integral_m3=dye, dye_relative_error=(dye-self.initial_dye)/self.initial_dye,
                    mixing_index=1.0-math.sqrt(variance/self.initial_variance),
                    maximum_speed_m_s=float(np.linalg.norm(v, axis=1).max()),
                    diffusion_substeps=self.last_diffusion_substeps,
                    minimum_fit_pivot=self.diffusion.last_min_pivot)


class PointUSD:
    def __init__(self, path, sim):
        from pxr import Usd, UsdGeom, Vt, Sdf
        self.Vt = Vt
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(path))
        self.stage.SetTimeCodesPerSecond(sim.fps)
        self.stage.SetFramesPerSecond(sim.fps)
        self.stage.SetStartTimeCode(0.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        is_tank = getattr(sim, "scene_kind", "") == "tank"
        if is_tank:
            root = UsdGeom.Xform.Define(self.stage, "/Mixing")
            self.points = UsdGeom.Points.Define(self.stage, "/Mixing/Particles")
            self.stage.SetDefaultPrim(root.GetPrim())
        else:
            self.points = UsdGeom.Points.Define(self.stage, "/Mixing")
            self.stage.SetDefaultPrim(self.points.GetPrim())
        self.points.CreateWidthsAttr([sim.spacing])
        self.points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self.colors = self.points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        self.scalar = UsdGeom.PrimvarsAPI(self.points).CreatePrimvar(
            "dyeConcentration", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        self.source = (UsdGeom.PrimvarsAPI(self.points).CreatePrimvar(
            "emitterId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex) if is_tank else None)
        self.stage.GetRootLayer().customLayerData = {
            "method": "APIC + cubic LSMPS type-A explicit diffusion",
            "paper_doi": "10.1007/s40571-015-0046-7", "diffusivity_m2_s": sim.diffusion.diffusivity,
            "color_model": sim.color_model,
            "optical_medium": getattr(sim, "optical_medium", "pigment"),
            "optical_medium_scale": getattr(sim, "optical_medium_scale", 1.0),
        }
        self.last_frame = -1
        self.tank = None
        if getattr(sim, "scene_kind", "") == "tank":
            from examples.mixing.tank_guides import TankUSD
            self.tank = TankUSD(self.stage, sim)

    def record(self, sim):
        if self.last_frame == sim.frame:
            return
        x, c = sim.state_0.particle_q.numpy(), sim.concentration.numpy()
        self.points.GetPointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(x), sim.frame)
        colors = display_colors(c, sim.color_model, getattr(sim, "optical_medium", "pigment"),
                                getattr(sim, "optical_medium_scale", 1.0))
        self.colors.Set(self.Vt.Vec3fArray.FromNumpy(colors), sim.frame)
        self.scalar.Set(self.Vt.FloatArray.FromNumpy(c), sim.frame)
        self.points.CreateIdsAttr().Set(self.Vt.Int64Array.FromNumpy(np.arange(sim.count, dtype=np.int64)), sim.frame)
        if self.source is not None:
            self.source.Set(self.Vt.IntArray.FromNumpy(sim.source_ids[:sim.count]), sim.frame)
        if self.tank is not None:
            self.tank.record(sim)
        self.stage.SetEndTimeCode(sim.frame)
        self.last_frame = sim.frame

    def close(self):
        self.stage.GetRootLayer().Save()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("tank", "vortex"), default="tank")
    parser.add_argument("--quality", choices=("standard", "high"), default="standard",
                        help="high uses tank resolution160, 24 substeps, 1080p and 12 optical layers")
    parser.add_argument("--resolution", type=int, help="x resolution: tank default96, vortex default24")
    parser.add_argument("--frames", type=int, help="tank default360; vortex default180")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, help="tank default16; vortex default4")
    parser.add_argument("--emitter-speed", type=float, default=1.2, help="tank nozzle speed in m/s")
    parser.add_argument("--emitter-radius", type=float, default=0.075, help="tank nozzle radius in meters")
    parser.add_argument("--emitter-duration", type=float, default=3.0, help="tank injection duration in seconds")
    parser.add_argument("--initial-coffee-height", type=float, default=0.0,
                        help="prefill the tank with coffee to this height and use only the milk inlet")
    parser.add_argument("--emitter-yaw-offset", type=float, default=20.0,
                        help="front/back nozzle skew in degrees [-45,45]; zero aims directly inward")
    parser.add_argument("--emission-mode", choices=("continuous", "layered"), default="continuous",
                        help="continuous births or the old simultaneous layers")
    parser.add_argument("--emitter-color-period", type=float,
                        help="swap inlet materials every N seconds; pigment default2, optical media default0")
    parser.add_argument("--diffusivity", type=float,
                        help="m^2/s: tank default0.003, vortex default0.001; zero disables diffusion")
    parser.add_argument("--color-model", choices=tuple(COLOR_MODELS), default="km",
                        help="km: three-channel Kubelka-Munk (S=1); rgb: linear RGB baseline")
    parser.add_argument("--initial-speed", type=float, default=0.5, help="initial vortex speed scale, m/s")
    parser.add_argument("--stirring-acceleration", type=float, default=2.0,
                        help="vortex body-force scale, m/s^2; zero disables continued stirring")
    parser.add_argument("--device", default="cuda:0" if wp.get_cuda_device_count() else "cpu")
    parser.add_argument("--viewer", choices=("gl", "rtx", "none"), default="gl")
    parser.add_argument("--headless", action="store_true", help="render GL/RTX without opening a window")
    parser.add_argument("--render-width", type=int, help="viewer image width; high quality default1920")
    parser.add_argument("--render-height", type=int, help="viewer image height; high quality default1080")
    parser.add_argument("--usd", type=Path, help="colored particle USD by default; add --surface for mesh USD")
    parser.add_argument("--usd-every", type=int, default=2)
    surface_mode = parser.add_mutually_exclusive_group()
    surface_mode.add_argument("--surface", dest="no_surface", action="store_false",
                              help="enable surface reconstruction, display and export")
    surface_mode.add_argument("--no-surface", dest="no_surface", action="store_true",
                              help="display/export particles only (default)")
    surface_mode.add_argument("--ssfr", action="store_true",
                              help="screen-space fluid rendering through the RTX SPG pipeline")
    parser.set_defaults(no_surface=True)
    parser.add_argument("--ssfr-filter", choices=("narrow", "curvature"), default="narrow")
    parser.add_argument("--ssfr-iterations", type=int,
                        help="filter iterations; narrow default2, curvature default100")
    parser.add_argument("--ssfr-radius-scale", type=float, default=1.6,
                        help="rendered sphere radius / physical particle radius")
    parser.add_argument("--ssfr-absorption", type=float, default=12.0,
                        help="effective dye absorption strength in 1/m for pigment mode")
    parser.add_argument("--ssfr-medium", choices=MEDIUM_MODES, default="pigment",
                        help="pigment colors or concentration-dependent absorption/scattering presets")
    parser.add_argument("--ssfr-medium-scale", type=float, default=1.0,
                        help="optical coefficient multiplier for coffee-milk/water-milk")
    parser.add_argument("--ssfr-medium-layers", type=int,
                        help="front-to-back optical slabs, 1..16, standard default8, high default12")
    parser.add_argument("--ssfr-ior", type=float, default=1.333)
    parser.add_argument("--ssfr-roughness", type=float, default=0.2,
                        help="water specular roughness in (0,1]; default0.2")
    parser.add_argument("--ssfr-splats", choices=("sphere", "anisotropic"), default="anisotropic",
                        help="particle render kernels; anisotropic uses neighborhood PCA")
    parser.add_argument("--ssfr-view", choices=("color", "depth", "thickness", "normals", "dye", "specular",
                                              "transmittance", "scattering"),
                        default="color", help="SSFR output or a diagnostic view")
    parser.add_argument("--show-particles", action="store_true", help="include particles alongside --surface")
    parser.add_argument("--surface-voxel-size", type=float, help="surface voxel size in meters; default spacing/3")
    parser.add_argument("--surface-color-radius", type=float,
                        help="concentration interpolation support in meters; default 2.5*spacing")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if args.render_width is None:
        args.render_width = 1920 if args.quality == "high" or args.viewer == "gl" else 1280
    if args.render_height is None:
        args.render_height = 1080 if args.quality == "high" or args.viewer == "gl" else 720
    if min(args.render_width, args.render_height) < 1:
        parser.error("render width and height must be positive")
    if args.ssfr_medium_layers is None:
        args.ssfr_medium_layers = 12 if args.quality == "high" else 8
    if args.emitter_color_period is None:
        args.emitter_color_period = 2.0 if args.ssfr_medium == "pigment" else 0.0
    if args.ssfr_medium != "pigment" and not args.ssfr:
        parser.error("--ssfr-medium requires --viewer rtx --ssfr")
    tank_resolution, tank_substeps = (160, 24) if args.quality == "high" else (96, 16)
    if args.resolution is None:
        args.resolution = tank_resolution if args.scene == "tank" else 24
    if args.frames is None:
        args.frames = 360 if args.scene == "tank" else 180
    if args.substeps is None:
        args.substeps = tank_substeps if args.scene == "tank" else 4
    if args.diffusivity is None:
        args.diffusivity = 3.0e-3 if args.scene == "tank" else 1.0e-3
    if args.scene == "tank" and args.resolution < 16:
        parser.error("tank resolution must be >=16")
    if args.scene != "tank" and args.initial_coffee_height:
        parser.error("--initial-coffee-height is available only for the tank scene")
    if not math.isfinite(args.initial_coffee_height) or not 0.0 <= args.initial_coffee_height < 0.65:
        parser.error("initial-coffee-height must be finite and in [0,0.65) m")
    if args.scene == "vortex" and (args.resolution < 12 or args.resolution % 2):
        parser.error("vortex resolution must be even and >=12")
    if args.viewer == "rtx" and not wp.get_device(args.device).is_cuda:
        parser.error("--viewer rtx requires a CUDA --device for GPU attribute uploads")
    if args.ssfr:
        if args.viewer != "rtx":
            parser.error("--ssfr requires --viewer rtx")
        if args.show_particles:
            parser.error("--show-particles is incompatible with --ssfr")
        from examples.mixing.ssfr import SSFRConfig
        if args.ssfr_iterations is None:
            args.ssfr_iterations = 2 if args.ssfr_filter == "narrow" else 100
        try:
            SSFRConfig(args.ssfr_filter, args.ssfr_iterations, args.ssfr_radius_scale,
                       args.ssfr_absorption, args.ssfr_ior, args.ssfr_view, args.ssfr_roughness,
                       splats=args.ssfr_splats, medium=args.ssfr_medium,
                       medium_scale=args.ssfr_medium_scale, medium_layers=args.ssfr_medium_layers)
        except ValueError as error:
            parser.error(str(error))
    if not all(math.isfinite(v) and v > 0.0 for v in
               (args.emitter_speed, args.emitter_radius, args.emitter_duration)):
        parser.error("emitter speed, radius and duration must be finite and positive")
    if not math.isfinite(args.emitter_yaw_offset) or abs(args.emitter_yaw_offset) > 45.0:
        parser.error("emitter-yaw-offset must be finite and within [-45,45] degrees")
    if min(args.frames, args.fps, args.substeps, args.usd_every) < 1:
        parser.error("frames, fps, substeps and usd-every must be positive")
    if not all(math.isfinite(v) and v >= 0 for v in
               (args.diffusivity, args.initial_speed, args.stirring_acceleration, args.emitter_color_period)):
        parser.error("diffusivity, speeds, stirring acceleration and color period must be finite and nonnegative")
    if any(v is not None and (not math.isfinite(v) or v <= 0.0)
           for v in (args.surface_voxel_size, args.surface_color_radius)):
        parser.error("surface-voxel-size and surface-color-radius must be finite and positive")
    return args


def main():
    args = parse_args()
    if args.scene == "tank":
        from examples.mixing.tank import TankMixingSimulation
        sim = TankMixingSimulation(args.resolution, args.device, args.fps, args.substeps,
                                   args.diffusivity, args.color_model, args.emitter_speed,
                                   args.emitter_radius, args.emitter_duration, args.emitter_yaw_offset,
                                   args.emission_mode, args.emitter_color_period,
                                   args.initial_coffee_height)
    else:
        sim = MixingSimulation(args.resolution, args.device, args.fps, args.substeps,
                               args.diffusivity, args.initial_speed, args.stirring_acceleration,
                               args.color_model)
    sim.optical_medium = args.ssfr_medium
    sim.optical_medium_scale = args.ssfr_medium_scale
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viewer, usd, surface, tank = None, None, None, None
    if not args.no_surface and (args.viewer != "none" or args.usd):
        from examples.mixing.surface import MixingSurface, SurfaceUSD
        surface = MixingSurface(sim, args.surface_voxel_size, args.surface_color_radius)
    if args.usd:
        usd = (SurfaceUSD(args.usd, sim, surface, args.show_particles)
               if surface is not None else PointUSD(args.usd, sim))
    if args.viewer != "none":
        if args.viewer == "rtx":
            if args.ssfr:
                from examples.mixing.ssfr import SSFRConfig
                from examples.mixing.viewer_ssfr import ScreenSpaceViewerRTX
                config = SSFRConfig(args.ssfr_filter, args.ssfr_iterations, args.ssfr_radius_scale,
                                    args.ssfr_absorption, args.ssfr_ior, args.ssfr_view, args.ssfr_roughness,
                                    splats=args.ssfr_splats, medium=args.ssfr_medium,
                                    medium_scale=args.ssfr_medium_scale, medium_layers=args.ssfr_medium_layers)
                viewer = ScreenSpaceViewerRTX(
                    particle_radius=0.5*sim.spacing, color_model=COLOR_MODELS[sim.color_model],
                    ssfr_config=config, device=sim.model.device, paused=False,
                    headless=args.headless, fps=sim.fps, up_axis="Z", environment="studio",
                    width=args.render_width, height=args.render_height)
            else:
                from examples.mixing.viewer_rtx import MixingViewerRTX
                viewer = MixingViewerRTX(paused=False, headless=args.headless,
                                         fps=sim.fps, up_axis="Z", environment="studio",
                                         width=args.render_width, height=args.render_height)
        else:
            viewer = newton.viewer.ViewerGL(paused=False, headless=args.headless,
                                          width=args.render_width, height=args.render_height)
        viewer.set_model(sim.model)
        viewer.show_particles = False
        if args.scene == "tank":
            from examples.mixing.tank_guides import TankGL
            tank = TankGL(sim)
            viewer.set_camera(wp.vec3(1.65, -1.5, 1.25), -24.0, 120.0)
        elif surface is not None:
            viewer.set_camera(wp.vec3(1.05, -0.8, 0.95), -32.0, 123.0)
        else:
            viewer.set_camera(wp.vec3(0.4, 0.2, 1.5), -88.0, -90.0)

    def render():
        if viewer is not None:
            viewer.begin_frame(sim.frame/sim.fps)
            if tank is not None:
                tank.draw(viewer, sim, draw_walls=not args.ssfr)
            if surface is not None:
                surface.draw(viewer, sim)
            if args.ssfr:
                viewer.log_fluid_ssfr(sim.state_0.particle_q, sim.model.particle_radius, sim.concentration)
            elif sim.count and (surface is None or args.show_particles):
                sim.update_colors()
                if args.viewer == "rtx":
                    viewer.log_fluid_particles(sim.state_0.particle_q, sim.model.particle_radius, sim.colors)
                else:
                    viewer.log_points("/mixing/particles", sim.state_0.particle_q,
                                      radii=sim.model.particle_radius, colors=sim.colors)
            viewer.end_frame()

    rows = [sim.metrics()]
    print(f"APIC {args.scene} mixing: {sim.count} particles, LSMPS, D={args.diffusivity:g} m^2/s, "
          f"color model={sim.color_model}, optical medium={args.ssfr_medium}", flush=True)
    try:
        render()
        if usd:
            usd.record(sim)
        while sim.frame < args.frames and (viewer is None or viewer.is_running()):
            if viewer is not None and not viewer.should_step():
                render()
                continue
            row = sim.step()
            rows.append(row)
            render()
            if usd and (sim.frame % args.usd_every == 0 or sim.frame == args.frames):
                usd.record(sim)
            if sim.frame % 10 == 0 or sim.frame == args.frames:
                print(f"frame {sim.frame}/{args.frames}: N={sim.count}, c=[{row['concentration_min']:.5f}, "
                      f"{row['concentration_max']:.5f}], MI={row['mixing_index']:.4f}, "
                      f"dye error={row['dye_relative_error']:.3e}", flush=True)
        if usd:
            usd.record(sim)
    finally:
        if usd:
            usd.close()
        if viewer:
            viewer.close()
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(
        scene=args.scene,
        quality=args.quality, render_width=args.render_width, render_height=args.render_height,
        paper_doi="10.1007/s40571-015-0046-7", frames=sim.frame, fps=sim.fps,
        resolution=sim.resolution, substeps_per_frame=sim.substeps, dt_s=sim.dt,
        particles=sim.count, spacing_m=sim.spacing, diffusivity_m2_s=args.diffusivity,
        initial_speed_m_s=args.initial_speed if args.scene == "vortex" else None,
        initial_coffee_height_m=args.initial_coffee_height if args.scene == "tank" else 0.0,
        stirring_acceleration_m_s2=args.stirring_acceleration if args.scene == "vortex" else 0.0,
        color_model=sim.color_model,
        color_endmembers=(dict(yellow=YELLOW, blue=BLUE) if args.ssfr_medium == "pigment" else {}),
        optical_medium=(medium_metadata(args.ssfr_medium, args.ssfr_medium_scale, args.ssfr_medium_layers)
                        if args.ssfr_medium != "pigment" else {}),
        viewer=args.viewer, headless=args.headless, surface_enabled=surface is not None,
        ssfr_enabled=args.ssfr,
        ssfr_parameters=asdict(config) if args.ssfr else {},
        surface_parameters=surface.parameters if surface is not None else {},
        surface_color_radius_m=surface.support if surface is not None else None,
        surface_max_triangles=surface.max_triangles if surface is not None else 0,
        diffusion_scheme=("positive symmetric LSMPS pair flux, explicit Euler, adaptive fit order"
                          if args.scene == "tank" else "3D cubic LSMPS type-A, explicit Euler, box mirror particles"),
        moment_regularization=sim.diffusion.regularization,
        apic_config=asdict(sim.config), final=rows[-1],
        environment={name: version(name) for name in ("newton", "warp-lang", "numpy", "usd-core")},
        python=sys.version, device=str(sim.model.device),
    )
    if hasattr(sim, "report"):
        report.update(sim.report())
    if args.viewer == "rtx":
        report["environment"].update({name: version(name) for name in ("ovrtx", "ovstage")})
        report["rtx_fluid_upload"] = "cuda_device_address" if args.ssfr else "cuda_dlpack"
    (args.output_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    state = dict(positions=sim.state_0.particle_q.numpy(),
                 velocities=sim.state_0.particle_qd.numpy(), concentration=sim.concentration.numpy())
    if hasattr(sim, "source_ids"):
        state["source_ids"] = sim.source_ids[:sim.count]
        state["birth_time_s"] = sim.birth_times[:sim.count]
    np.savez_compressed(args.output_dir / "final_state.npz", **state)
    print(f"Diagnostics: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
