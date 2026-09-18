"""Newton ViewerGL demo for SolverAPIC (Nanogrid APIC skeleton, no projection)."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import warp as wp

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
from solver.apic import SolverAPIC


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


class NewtonAPICTest:
    def __init__(
        self,
        device: str = "cuda:0",
        res: int = 20,
        voxel_size: float | None = None,
        show_grid: bool = True,
    ) -> None:
        self.device = device
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sub_step_num = 2
        self.sim_dt = self.frame_dt / float(self.sub_step_num)
        self.gravity = -10.0
        self.show_grid = bool(show_grid)
        self.show_domain_box = True
        self.grid_line_width = 0.0025
        self.grid_color = (0.85, 0.75, 0.25)
        self.enable_projection = True

        self.bound_lo = (-0.5, -0.5, 0.0)
        self.bound_hi = (0.5, 0.5, 1.0)

        positions, mass, radius, volume, spacing = spawn_particle_block(
            lo=(-0.15, -0.15, 0.35),
            hi=(0.15, 0.15, 0.65),
            res=(res, res, res),
        )
        self.particle_radius = radius
        self.particle_count = positions.shape[0]
        if voxel_size is None:
            voxel_size = max(spacing, 0.04)

        self.model = self.build_model(positions, mass, radius, volume)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.model.set_gravity((0.0, 0.0, self.gravity))

        self.solver = SolverAPIC(
            self.model,
            SolverAPIC.Config(
                voxel_size=float(voxel_size),
                bound_lo=self.bound_lo,
                bound_hi=self.bound_hi,
                clamp_eps=1.0e-3,
                grid_capacity_ratio=16.0,
                enable_projection=True,
            ),
        )

        extent = 1.2
        self.camera_pos = wp.vec3(0.0, -2.2 * extent, 0.55)
        self.camera_pitch = -0.25
        self.camera_yaw = 90.0
        self.viewer = None
        self.render_x = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.point_radii = wp.full(self.particle_count, self.particle_radius, dtype=wp.float32, device=self.device)
        self.colors = wp.full(self.particle_count, wp.vec3(0.25, 0.55, 0.95), dtype=wp.vec3, device=self.device)
        self._domain_box_starts, self._domain_box_ends = self._make_domain_box_lines()

    def _make_domain_box_lines(self) -> tuple[wp.array, wp.array]:
        lo = np.array(self.bound_lo, dtype=np.float32)
        hi = np.array(self.bound_hi, dtype=np.float32)
        corners = np.array(
            [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]],
                [lo[0], hi[1], hi[2]],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        starts = np.array([corners[a] for a, _ in edges], dtype=np.float32)
        ends = np.array([corners[b] for _, b in edges], dtype=np.float32)
        return (
            wp.array(starts, dtype=wp.vec3, device=self.device),
            wp.array(ends, dtype=wp.vec3, device=self.device),
        )

    def build_model(self, positions, mass, radius, volume) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_particle_radius = float(radius)
        SolverAPIC.register_custom_attributes(builder)
        n = positions.shape[0]
        builder.add_particles(
            pos=positions.tolist(),
            vel=np.zeros_like(positions).tolist(),
            mass=[float(mass)] * n,
            radius=[float(radius)] * n,
            custom_attributes={"apic:particle_volume": [float(volume)] * n},
        )
        return builder.finalize(device=self.device)

    def step(self) -> None:
        for _ in range(self.sub_step_num):
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def _ui_apic_panel(self, imgui) -> None:
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("APIC Visualizer"):
            _c1, self.show_grid = imgui.checkbox("Show Nanogrid", self.show_grid)
            _c2, self.show_domain_box = imgui.checkbox("Show Domain Box", self.show_domain_box)
            _c3, self.grid_line_width = imgui.slider_float(
                "Grid Line Width", self.grid_line_width, 0.0005, 0.02
            )
            changed_proj, self.enable_projection = imgui.checkbox(
                "Enable Pressure Projection", self.enable_projection
            )
            if changed_proj:
                self.solver.config.enable_projection = bool(self.enable_projection)
            if getattr(self.solver, "_grid", None) is not None:
                try:
                    cell_count = int(self.solver._grid.cell_count())
                except Exception:
                    cell_count = -1
                imgui.text(f"Active cells: {cell_count}")
                imgui.text(f"Voxel size: {self.solver.config.voxel_size:.4f}")
                imgui.text(
                    f"Projection iters: {self.solver.last_projection_iters}  "
                    f"residual: {self.solver.last_projection_residual:.3e}"
                )
                imgui.text(
                    f"C damp={self.solver.config.c_damping:.2f}  "
                    f"|C|max={self.solver.config.c_norm_max:.1f}  "
                    f"vmax={self.solver.config.max_particle_speed:.1f}"
                )

    def render(self) -> None:
        if self.viewer is None:
            return
        wp.copy(self.render_x, self.state_0.particle_q)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "/apic/particles",
            points=self.render_x,
            radii=self.point_radii,
            colors=self.colors,
        )

        if self.show_domain_box:
            self.viewer.log_lines(
                "/apic/domain_box",
                starts=self._domain_box_starts,
                ends=self._domain_box_ends,
                colors=(0.55, 0.55, 0.6),
                width=0.004,
                hidden=False,
            )
        else:
            self.viewer.log_lines(
                "/apic/domain_box",
                starts=None,
                ends=None,
                colors=None,
                hidden=True,
            )

        if self.show_grid:
            starts, ends = self.solver.get_grid_wireframe()
            if starts is not None and ends is not None and starts.shape[0] > 0:
                self.viewer.log_lines(
                    "/apic/nanogrid",
                    starts=starts,
                    ends=ends,
                    colors=self.grid_color,
                    width=float(self.grid_line_width),
                    hidden=False,
                )
            else:
                self.viewer.log_lines(
                    "/apic/nanogrid",
                    starts=None,
                    ends=None,
                    colors=None,
                    hidden=True,
                )
        else:
            self.viewer.log_lines(
                "/apic/nanogrid",
                starts=None,
                ends=None,
                colors=None,
                hidden=True,
            )

        self.viewer.end_frame()

    def setup_viewer(self) -> None:
        self.viewer = newton.viewer.ViewerGL()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(self.camera_pos, self.camera_pitch, self.camera_yaw)
        self.viewer.register_ui_callback(self._ui_apic_panel, position="panel")

    def run(self, frames: int) -> None:
        frame = 0
        while self.viewer.is_running() and frame < frames:
            self.step()
            self.render()
            frame += 1
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Newton APIC skeleton demo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--res", type=int, default=20, help="Particles per axis of the initial block (20^3=8000)")
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--frames", type=int, default=6000, help="Max frames for viewer; used by --no-viewer too")
    parser.add_argument("--no-viewer", action="store_true", help="Headless smoke run")
    parser.add_argument("--hide-grid", action="store_true", help="Disable Nanogrid wireframe by default")
    parser.add_argument("--no-projection", action="store_true", help="Disable pressure projection")
    args = parser.parse_args()

    show_grid = not args.hide_grid
    app = NewtonAPICTest(
        device=args.device,
        res=args.res,
        voxel_size=args.voxel_size,
        show_grid=show_grid,
    )
    if args.no_projection:
        app.enable_projection = False
        app.solver.config.enable_projection = False
    print(
        f"APIC demo: {app.particle_count} particles on {args.device} "
        f"(show_grid={show_grid}, projection={app.solver.config.enable_projection})"
    )

    if args.no_viewer:
        n_frames = args.frames if args.frames > 0 else 30
        if args.frames >= 6000:
            n_frames = 30
        for i in range(n_frames):
            app.step()
            if (i + 1) % 10 == 0:
                q = app.state_0.particle_q.numpy()
                starts, _ends = app.solver.get_grid_wireframe()
                edge_count = 0 if starts is None else int(starts.shape[0])
                print(
                    f"frame {i + 1}/{n_frames}  z_mean={q[:, 2].mean():.4f}  "
                    f"z_min={q[:, 2].min():.4f}  z_span={q[:, 2].max() - q[:, 2].min():.4f}  "
                    f"grid_edges={edge_count}  proj_iters={app.solver.last_projection_iters}  "
                    f"finite={np.isfinite(q).all()}"
                )
        print("Headless APIC run finished.")
        return

    app.setup_viewer()
    app.run(args.frames)


if __name__ == "__main__":
    main()
