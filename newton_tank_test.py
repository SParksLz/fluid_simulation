import argparse
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
from sph.solver_wcsph import SolverWCSPH
from wcsph_kernel import to_micro_world, to_real_world


@wp.kernel
def scalar_visualize(
    values: wp.array(dtype=float),
    value_scale: float,
    color: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    strength = wp.min(wp.max(values[tid] * value_scale, 0.0), 1.0)
    color[tid] = wp.lerp(wp.vec3(0.2, 0.3, 0.9), wp.vec3(1.0, 0.9, 0.2), strength)


class NewtonTankTest:
    def __init__(
        self,
        device: str = "cuda:0",
        load_from_usd: bool = True,
        usd_path: str | None = None,
        color_field: str = "pressure",
    ) -> None:
        self.device = device
        self.load_from_usd = load_from_usd
        self.color_field = color_field
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / 4.0
        self.sub_step_num = 4
        self.cfl_factor = 0.4
        self.dt_min = 1.0e-4
        self.dt_max = self.frame_dt
        self.bound_size = 100.0
        self.gravity = -10.0
        self.camera_pos = wp.vec3(-0.16, -2.90, 0.69)
        self.camera_pitch = -0.6
        self.camera_yaw = 90.5

        self.colors: wp.array | None = None
        self.render_x: wp.array | None = None
        self.particle_radius = 0.0
        self.particle_count = 0

        if usd_path is None:
            usd_path = (Path(__file__).parent / "temp" / "fluid_particles.usd").as_posix()

        if load_from_usd:
            positions, colors, particle_radius = self.load_particles_from_usd(usd_path, wp.vec3(0.0, 0.0, 0.0))
        else:
            raise NotImplementedError("Only USD-backed setup is implemented for this test script.")

        self.particle_radius = particle_radius
        self.particle_count = positions.shape[0]
        self.colors = wp.array(colors, dtype=wp.vec3, device=self.device)
        self.render_x = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.point_radii = wp.full(
            self.particle_count,
            self.particle_radius * 0.01,
            dtype=wp.float32,
            device=self.device,
        )

        self.model = self.build_model(positions)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.model.set_gravity((0.0, 0.0, self.gravity))

        self.solver = SolverWCSPH(
            self.model,
            SolverWCSPH.SphConfig(
                particle_radius=self.particle_radius,
                particle_length=self.particle_radius * 2.0,
                smoothing_length_coff=1.35,
                bound_width=self.bound_size,
                bound_height=self.bound_size * 0.5,
                bound_length=self.bound_size,
                boundary_damping=-0.1,
            ),
        )

        self.viewer = newton.viewer.ViewerGL()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(self.camera_pos, self.camera_pitch, self.camera_yaw)

    def load_particles_from_usd(self, filename: str, offset: wp.vec3):
        stage = Usd.Stage.Open(filename)
        fluid = stage.GetPrimAtPath("/Fluid/Particles")
        if not fluid.IsValid():
            raise RuntimeError(f"Cannot find /Fluid/Particles in {filename}")

        points = UsdGeom.Points(fluid)
        points_np = np.array(points.GetPointsAttr().Get(), dtype=np.float32)
        colors_np = np.full((len(points_np), 3), (0.0, 0.0, 1.0), dtype=np.float32)
        particle_radius = float(points.GetWidthsAttr().Get()[0] * 0.5 * 100.0)

        points_wp = wp.array(points_np, dtype=wp.vec3, device=self.device)
        wp.launch(kernel=to_micro_world, dim=len(points_np), inputs=[points_wp, 100.0, offset], device=self.device)
        return points_wp.numpy(), colors_np, particle_radius

    def build_model(self, positions: np.ndarray) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_particle_radius = self.particle_radius
        SolverWCSPH.register_custom_attributes(builder)

        rest_density = 1000.0
        surface_tension = 0.05
        viscosity = 0.05
        stiffness = 85000.0
        exponent = 7.0
        mass = 0.8 * (self.particle_radius * 2.0) ** 3 * rest_density

        builder.add_particles(
            pos=positions.tolist(),
            vel=np.zeros_like(positions).tolist(),
            mass=[mass] * len(positions),
            radius=[self.particle_radius] * len(positions),
            custom_attributes={
                "sph:particle_mask": [0] * len(positions),
                "sph:rest_density": [rest_density] * len(positions),
                "sph:surface_tension": [surface_tension] * len(positions),
                "sph:viscosity": [viscosity] * len(positions),
                "sph:stiffness": [stiffness] * len(positions),
                "sph:exponent": [exponent] * len(positions),
            },
        )
        return builder.finalize(device=self.device)

    def compute_cfl_dt(self) -> float:
        particle_spacing = self.particle_radius * 2.0
        velocities = self.state_0.particle_qd.numpy()
        if velocities.size == 0:
            return self.dt_max

        speed_max = float(np.linalg.norm(velocities, axis=1).max(initial=0.0))
        if speed_max < 1.0e-8:
            return self.dt_max

        dt_cfl = self.cfl_factor * particle_spacing / speed_max
        return max(self.dt_min, min(self.dt_max, dt_cfl))

    def step(self):
        frame_time_left = self.frame_dt
        substeps = 0

        while frame_time_left > 0.0 and substeps < self.sub_step_num:
            dt = min(frame_time_left, self.compute_cfl_dt())
            self.sim_dt = dt
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += dt
            frame_time_left -= dt
            substeps += 1

    def render(self):
        wp.launch(
            kernel=to_real_world,
            dim=self.particle_count,
            inputs=[self.state_0.particle_q, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)],
            device=self.device,
        )

        if self.color_field == "rho":
            field = self.state_0.sph.rho
            scale = 1.0 / 1500.0
        else:
            field = self.state_0.sph.pressure
            scale = 1.0 / 50000.0

        wp.launch(
            kernel=scalar_visualize,
            dim=self.particle_count,
            inputs=[field, scale, self.colors],
            device=self.device,
        )

        rho = self.state_0.sph.rho.numpy()
        pressure = self.state_0.sph.pressure.numpy()
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
        print(
            f"t={self.sim_time:.4f} field={self.color_field} "
            f"rho(mean={np.mean(rho):.6e}, min={np.min(rho):.6e}, max={np.max(rho):.6e}) "
            f"p(mean={np.mean(pressure):.6e}, min={np.min(pressure):.6e}, max={np.max(pressure):.6e}) "
            f"speed(max={np.max(speed):.6e})"
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "/wcsph/particles",
            points=self.render_x,
            radii=self.point_radii,
            colors=self.colors,
        )
        self.viewer.end_frame()

    def run(self, frames: int):
        frame = 0
        while self.viewer.is_running() and frame < frames:
            with wp.ScopedTimer("frame", active=True):
                self.step()
                self.render()
            frame += 1
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--usd", default=None)
    parser.add_argument("--color-field", choices=["pressure", "rho"], default="pressure")
    args = parser.parse_args()

    test = NewtonTankTest(
        device=args.device,
        load_from_usd=True,
        usd_path=args.usd,
        color_field=args.color_field,
    )
    test.run(args.frames)


if __name__ == "__main__":
    main()
