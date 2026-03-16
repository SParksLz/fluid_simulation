import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

# Compatibility shim for Newton builds that import modern Warp symbols directly.
if not hasattr(wp, "Device"):
    try:
        from warp.context import Device as _Device
    except Exception:
        _Device = Any
    wp.Device = _Device

if not hasattr(wp, "DeviceLike"):
    try:
        from warp.context import Devicelike as _DeviceLike
    except Exception:
        _DeviceLike = Any
    wp.DeviceLike = _DeviceLike

import newton
from solver.pbf.solver_pbf import SolverPBF


@wp.kernel
def scalar_visualize(
    values: wp.array(dtype=float),
    value_scale: float,
    color: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    strength = wp.min(wp.max(values[tid] * value_scale, 0.0), 1.0)
    color[tid] = wp.lerp(wp.vec3(0.2, 0.3, 0.9), wp.vec3(1.0, 0.9, 0.2), strength)


@wp.kernel
def signed_scalar_visualize(
    values: wp.array(dtype=float),
    value_scale: float,
    color: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    value = values[tid] * value_scale
    strength = wp.min(wp.abs(value), 1.0)
    if value >= 0.0:
        color[tid] = wp.lerp(wp.vec3(1.0, 1.0, 1.0), wp.vec3(1.0, 0.2, 0.2), strength)
    else:
        color[tid] = wp.lerp(wp.vec3(1.0, 1.0, 1.0), wp.vec3(0.2, 0.4, 1.0), strength)


@wp.kernel
def int_scalar_visualize(
    values: wp.array(dtype=wp.int32),
    value_scale: float,
    color: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    strength = wp.min(float(values[tid]) * value_scale, 1.0)
    color[tid] = wp.lerp(wp.vec3(0.15, 0.2, 0.25), wp.vec3(0.2, 1.0, 0.6), strength)


class NewtonPBFTest:
    def __init__(
        self,
        device: str = "cuda:0",
        load_from_usd: bool = True,
        usd_path: str | None = None,
        color_field: str = "rho",
    ) -> None:
        self.device = device
        self.load_from_usd = load_from_usd
        self.color_field = color_field
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sub_step_num = 1
        self.sim_dt = self.frame_dt / float(self.sub_step_num)
        self.fixed_dt = self.sim_dt
        self.length_scale = 0.01
        self.bound_width = 1.0
        self.bound_height = 1.0
        self.bound_length = 1.0
        self.gravity = -10.0 * self.length_scale
        self.camera_pos = wp.vec3(-0.16, -2.90, 0.69)
        self.camera_pitch = -0.6
        self.camera_yaw = 90.5

        self.colors: wp.array | None = None
        self.render_x: wp.array | None = None
        self.particle_radius = 0.0
        self.particle_count = 0
        self.frame_index = 0
        self.last_substeps = 0
        self.last_frame_advance = 0.0
        self.use_graph = str(self.device).startswith("cuda")
        self.graph_dt = self.fixed_dt
        self._solver_graph_forward = None
        self._solver_graph_reverse = None
        self._graph_step_parity = 0
        self._graph_capture_error: str | None = None
        self.log_path = Path(__file__).parent / "temp" / "pbf_frame_metrics.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # self._init_log_file()

        if usd_path is None:
            usd_path = (Path(__file__).parent / "temp" / "fluid_particles.usd").as_posix()

        if load_from_usd:
            positions, colors, particle_radius = self.load_particles_from_usd(usd_path, wp.vec3(0.0, 0.0, 0.0))
        else:
            raise NotImplementedError("Only USD-backed setup is implemented for this test script.")

        self.particle_radius = particle_radius
        self.particle_count = positions.shape[0]
        self._configure_scene_scale(positions)
        self.colors = wp.array(colors, dtype=wp.vec3, device=self.device)
        self.render_x = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.point_radii = wp.full(
            self.particle_count,
            self.particle_radius,
            dtype=wp.float32,
            device=self.device,
        )

        self.model = self.build_model(positions)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.model.set_gravity((0.0, 0.0, self.gravity))

        self.solver = SolverPBF(
            self.model,
            SolverPBF.PbfConfig(
                max_iterations=4,
                lambda_epsilon=1.0e6,
                min_neighbors_for_lambda=8,
                # scorr_k=0.001,
                # scorr_n=4.0,
                # scorr_q=0.3,
                max_delta_position=self.particle_radius * 0.5,
                # xsph_c=0.005,
                particle_radius=self.particle_radius,
                particle_length=self.particle_radius * 2.0,
                smoothing_length_coff=1.35,
                bound_width=self.bound_width,
                bound_height=self.bound_height,
                bound_length=self.bound_length,
                boundary_damping=-0.05,
            ),
        )

        self.viewer = newton.viewer.ViewerGL()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(self.camera_pos, self.camera_pitch, self.camera_yaw)
        self._maybe_capture_solver_graphs()

    def load_particles_from_usd(self, filename: str, offset: wp.vec3):
        stage = Usd.Stage.Open(filename)
        fluid = stage.GetPrimAtPath("/Fluid/Particles")
        if not fluid.IsValid():
            raise RuntimeError(f"Cannot find /Fluid/Particles in {filename}")

        points = UsdGeom.Points(fluid)
        points_np = np.array(points.GetPointsAttr().Get(), dtype=np.float32)
        colors_np = np.full((len(points_np), 3), (0.4, 0.3, 0.75), dtype=np.float32)
        particle_radius = float(points.GetWidthsAttr().Get()[0] * 0.5)

        offset_np = np.array([float(offset[0]), float(offset[1]), float(offset[2])], dtype=np.float32)
        return points_np + offset_np, colors_np, particle_radius

    def _configure_scene_scale(self, positions: np.ndarray) -> None:
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        extent = maxs - mins
        max_dim = float(max(np.max(extent), self.particle_radius * 8.0))
        margin = float(max(self.particle_radius * 8.0, 0.05))

        self.bound_width = float(max(abs(mins[0]), abs(maxs[0])) + margin)
        self.bound_height = float(max(abs(mins[1]), abs(maxs[1])) + margin)
        z_top = float(max(maxs[2] + margin, self.particle_radius * 8.0))
        self.bound_length = 0.5 * (z_top + self.particle_radius)

        center_x = float(0.5 * (mins[0] + maxs[0]))
        center_y = float(0.5 * (mins[1] + maxs[1]))
        self.camera_pos = wp.vec3(
            center_x,
            center_y - 2.4 * max_dim,
            float(maxs[2] + 0.75 * max_dim),
        )
        self.camera_pitch = -0.35
        self.camera_yaw = 90.0

    def build_model(self, positions: np.ndarray) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_particle_radius = self.particle_radius
        SolverPBF.register_custom_attributes(builder)

        rest_density = 1000.0
        surface_tension = 0.12 * (self.length_scale ** 3)
        viscosity = 0.01 * (self.length_scale ** 2)
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
            },
        )
        return builder.finalize(device=self.device)

    def _maybe_capture_solver_graphs(self) -> None:
        if not self.use_graph:
            return
        if not hasattr(wp, "ScopedCapture") or not hasattr(wp, "capture_launch"):
            self._graph_capture_error = "Warp graph capture API is unavailable in this build."
            self.use_graph = False
            return

        try:
            warmup_out = self.model.state()
            self.solver.step(self.state_0, warmup_out, control=None, contacts=None, dt=self.graph_dt)

            with wp.ScopedCapture() as capture_forward:
                self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.graph_dt)
            with wp.ScopedCapture() as capture_reverse:
                self.solver.step(self.state_1, self.state_0, control=None, contacts=None, dt=self.graph_dt)

            self._solver_graph_forward = capture_forward.graph
            self._solver_graph_reverse = capture_reverse.graph
            self._graph_step_parity = 0
            print("PBF CUDA graph capture enabled.")
        except Exception as exc:
            self._graph_capture_error = str(exc)
            self.use_graph = False
            self._solver_graph_forward = None
            self._solver_graph_reverse = None
            print(f"PBF CUDA graph capture disabled: {self._graph_capture_error}")

    def _init_log_file(self) -> None:
        with self.log_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "frame",
                    "sim_time",
                    "field",
                    "rho_mean",
                    "rho_min",
                    "rho_max",
                    "lambda_abs_mean",
                    "lambda_abs_max",
                    "neighbors_mean",
                    "neighbors_max",
                    "delta_p_max",
                    "step_dx_mean",
                    "step_dx_max",
                    "speed_max",
                    "substeps",
                    "frame_advance",
                ]
            )

    def _append_log_row(
        self,
        rho_mean: float,
        rho_min: float,
        rho_max: float,
        lambda_abs_mean: float,
        lambda_abs_max: float,
        neighbors_mean: float,
        neighbors_max: int,
        delta_p_max: float,
        step_dx_mean: float,
        step_dx_max: float,
        speed_max: float,
        substeps: int,
        frame_advance: float,
    ) -> None:
        with self.log_path.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    self.frame_index,
                    f"{self.sim_time:.10f}",
                    self.color_field,
                    f"{rho_mean:.10e}",
                    f"{rho_min:.10e}",
                    f"{rho_max:.10e}",
                    f"{lambda_abs_mean:.10e}",
                    f"{lambda_abs_max:.10e}",
                    f"{neighbors_mean:.10f}",
                    neighbors_max,
                    f"{delta_p_max:.10e}",
                    f"{step_dx_mean:.10e}",
                    f"{step_dx_max:.10e}",
                    f"{speed_max:.10e}",
                    substeps,
                    f"{frame_advance:.10f}",
                ]
            )

    def step(self):
        frame_start_time = self.sim_time
        frame_time_left = self.frame_dt
        substeps = 0

        while frame_time_left > 0.0 and substeps < self.sub_step_num:
            dt = min(self.fixed_dt, frame_time_left)
            if dt <= 0.0:
                raise RuntimeError(f"PBF step produced a non-positive fixed dt: {dt:.6e}")

            self.sim_dt = dt
            if (
                self.use_graph
                and self._solver_graph_forward is not None
                and self._solver_graph_reverse is not None
                and abs(dt - self.graph_dt) < 1.0e-12
            ):
                if self._graph_step_parity == 0:
                    wp.capture_launch(self._solver_graph_forward)
                else:
                    wp.capture_launch(self._solver_graph_reverse)
                self._graph_step_parity ^= 1
            else:
                self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += dt
            frame_time_left -= dt
            substeps += 1

        self.last_substeps = substeps
        self.last_frame_advance = self.sim_time - frame_start_time

    def render(self):
        wp.copy(self.render_x, self.state_0.particle_q)

        # if self.color_field == "lambda":
        #     wp.launch(
        #         kernel=signed_scalar_visualize,
        #         dim=self.particle_count,
        #         inputs=[self.state_0.sph.pbf_lambda, 5000.0, self.colors],
        #         device=self.device,
        #     )
        # elif self.color_field == "neighbor_count":
        #     wp.launch(
        #         kernel=int_scalar_visualize,
        #         dim=self.particle_count,
        #         inputs=[self.state_0.sph.neighbor_count, 1.0 / 80.0, self.colors],
        #         device=self.device,
        #     )
        # else:
        #     wp.launch(
        #         kernel=scalar_visualize,
        #         dim=self.particle_count,
        #         inputs=[self.state_0.sph.rho, 1.0 / 1500.0, self.colors],
        #         device=self.device,
        #     )

        # rho = self.state_0.sph.rho.numpy()
        # pbf_lambda = self.state_0.sph.pbf_lambda.numpy()
        # neighbor_count = self.state_0.sph.neighbor_count.numpy()
        # delta_p = self.state_0.sph.delta_p.numpy()
        # step_displacement = self.state_0.sph.step_displacement.numpy()
        # speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
        # delta_p_norm = np.linalg.norm(delta_p, axis=1)
        # step_dx_norm = np.linalg.norm(step_displacement, axis=1)
        # rho_mean = float(np.mean(rho))
        # rho_min = float(np.min(rho))
        # rho_max = float(np.max(rho))
        # lambda_abs_mean = float(np.mean(np.abs(pbf_lambda)))
        # lambda_abs_max = float(np.max(np.abs(pbf_lambda)))
        # neighbors_mean = float(np.mean(neighbor_count))
        # neighbors_max = int(np.max(neighbor_count))
        # delta_p_max = float(np.max(delta_p_norm))
        # step_dx_mean = float(np.mean(step_dx_norm))
        # step_dx_max = float(np.max(step_dx_norm))
        # speed_max = float(np.max(speed))

        # print(
        #     f"t={self.sim_time:.4f} field={self.color_field} "
        #     f"rho(mean={rho_mean:.6e}, min={rho_min:.6e}, max={rho_max:.6e}) "
        #     f"lambda(abs_mean={lambda_abs_mean:.6e}, abs_max={lambda_abs_max:.6e}) "
        #     f"neighbors(mean={neighbors_mean:.2f}, max={neighbors_max}) "
        #     f"delta_p(max={delta_p_max:.6e}) step_dx(max={step_dx_max:.6e}) "
        #     f"speed(max={speed_max:.6e}) substeps={self.last_substeps}"
        # )
        # self._append_log_row(
        #     rho_mean=rho_mean,
        #     rho_min=rho_min,
        #     rho_max=rho_max,
        #     lambda_abs_mean=lambda_abs_mean,
        #     lambda_abs_max=lambda_abs_max,
        #     neighbors_mean=neighbors_mean,
        #     neighbors_max=neighbors_max,
        #     delta_p_max=delta_p_max,
        #     step_dx_mean=step_dx_mean,
        #     step_dx_max=step_dx_max,
        #     speed_max=speed_max,
        #     substeps=self.last_substeps,
        #     frame_advance=self.last_frame_advance,
        # )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "/pbf/particles",
            points=self.render_x,
            radii=self.point_radii,
            colors=self.colors,
        )
        self.viewer.end_frame()

    def run(self, frames: int):
        frame = 0
        while self.viewer.is_running() and frame < frames:
            with wp.ScopedTimer("frame", active=False):
                self.step()
                self.render()
            self.frame_index += 1
            frame += 1
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--usd", default=None)
    parser.add_argument("--color-field", choices=["rho", "lambda", "neighbor_count"], default="rho")
    args = parser.parse_args()

    test = NewtonPBFTest(
        device=args.device,
        load_from_usd=True,
        usd_path=args.usd,
        color_field=args.color_field,
    )
    test.run(args.frames)


if __name__ == "__main__":
    main()
