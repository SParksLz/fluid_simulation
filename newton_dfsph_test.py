import argparse
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
from solver.sph.solver_dfsph import SolverDFSPH
from wcsph_kernel import to_micro_world, to_real_world


# 目前我加过的 clamp / 限幅 / 防爆 和“抑制数据爆炸”的手段，主要有这些。

# 已加的 Clamp

# factor 分母下限：在 solver_dfsph.py 的 compute_factor_dfsph() 里，原本就有 denom < 1e-6 这一类保护。后来我进一步改成：
# 如果 denom < 1e-6，直接把 factor[i] = 0
# 不再让它走到 1 / 1e-6 = 1e6
# density 侧正误差截断：在 compute_kappa_density() 里保留了
# rho_err = max(rho_star - rest_density, 0.0)
# 也就是只修正压缩，不修正负误差
# divergence 侧压缩截断：在 compute_kappa_divergence() 里改成
# density_change = max(d_rho_dt, 0.0)
# 只对压缩项做修正
# kappa 上限裁剪：在 compute_kappa_density() 里加了
# max_kappa = 1e4
# kappa = min(raw_kappa, max_kappa)
# kappa_v 上限裁剪：在 compute_kappa_divergence() 里加了
# max_kappa_v = 1e6
# kappa_v = min(raw_kappa_v, max_kappa_v)
# 分母除零保护：在速度修正里一直保留了
# rho[i] / rho[j] 用 max(rho, 1e-6) 防止除零
# 位置在 apply_constant_density_correction() 和 apply_divergence_free_correction()
# 已加的防爆方案

# 去掉 CFL 尾步强补：在 newton_dfsph_test.py 里删掉了“最后用剩余时间再补一个大步”的逻辑，避免 CFL 被最后一步破坏。
# 加了 CFL 自适应子步：仍然只在测试脚本里，不在 solver 本体里。
# warm start：给 solver 加过 kappa / kappa_v 的 warm start 开关，后面你又手动关成 False 做对照。
# 收敛判断：
# density 用平均 max(rho_star - rho0, 0)
# divergence 用平均 max(d_rho_dt, 0)
# 达到 eta_density / eta_divergence 就提前退出
# 邻居不足禁用 factor：
# 新增了 neighbor_count
# min_neighbors_for_factor = 15
# 邻居数不足时，factor = 0
# 诊断输出：
# kappa / kappa_v 的 mean/max
# rho(min)
# factor(max)
# neighbors(min/mean)
# 目前从日志看，最主要的爆炸来源

# 不是 rho 崩掉，rho(min) 基本稳定
# 主要是 factor 仍然会偶发非常大
# 然后把 kappa_v 顶到裁剪上限，再把 kappa 拖上去
# 所以现在最像的是“自由表面/退化邻域导致的 factor 尖峰”
# 还没做、但我建议的下一步

# 直接对 factor 源头加上限，比如 max_factor = 50 或 100
# 把 min_neighbors_for_factor 从 15 提高到 30 或 40


@wp.kernel
def divergence_visualize(
    divergence: wp.array(dtype=float),
    color_scale: float,
    color: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    strength = wp.min(wp.abs(divergence[tid]) * color_scale, 1.0)
    color[tid] = wp.lerp(wp.vec3(1.0, 1.0, 1.0), wp.vec3(1.0, 0.0, 0.0), strength)


class NewtonDFSPHTest:
    def __init__(
        self,
        device: str = "cuda:0",
        load_from_usd: bool = True,
        usd_path: str | None = None,
        color_field: str = "kappa",
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
        self.bound_3d_size = wp.vec3(self.bound_size, self.bound_size * 0.5, self.bound_size)
        self.gravity = -10.0
        self.camera_pos = wp.vec3(-0.16, -2.90, 0.69)
        self.camera_pitch = -0.6
        self.camera_yaw = 90.5

        self.colors: wp.array | None = None
        self.render_x: wp.array | None = None
        self.particle_radius = 0.0
        self.particle_count = 0
        self.divergence_color_scale = 0.01

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

        self.solver = SolverDFSPH(
            self.model,
            SolverDFSPH.SphConfig(
                max_rho_iterations=5,
                max_vel_iterations=12,
                particle_radius=self.particle_radius,
                particle_length=self.particle_radius * 2.0,
                smoothing_length_coff=1.35,
                bound_width=self.bound_size,
                bound_height=self.bound_size * 0.5,
                bound_length=self.bound_size,
                warm_start=False,
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
        SolverDFSPH.register_custom_attributes(builder)

        rest_density = 1000.0
        surface_tension = 0.05
        viscosity = 0.05
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

        color_source = self.state_0.sph.kappa if self.color_field == "kappa" else self.state_0.sph.kappa_v
        wp.launch(
            kernel=divergence_visualize,
            dim=self.particle_count,
            inputs=[color_source, self.divergence_color_scale, self.colors],
            device=self.device,
        )

        kappa = self.state_0.sph.kappa.numpy()
        kappa_v = self.state_0.sph.kappa_v.numpy()
        rho = self.state_0.sph.rho.numpy()
        factor = self.state_0.sph.factor.numpy()
        neighbor_count = self.state_0.sph.neighbor_count.numpy()
        print(
            f"t={self.sim_time:.4f} field={self.color_field} "
            f"kappa(mean={np.mean(kappa):.6e}, max={np.max(np.abs(kappa)):.6e}) "
            f"kappa_v(mean={np.mean(kappa_v):.6e}, max={np.max(np.abs(kappa_v)):.6e}) "
            f"rho(min={np.min(rho):.6e}) factor(max={np.max(np.abs(factor)):.6e}) "
            f"neighbors(min={np.min(neighbor_count)}, mean={np.mean(neighbor_count):.2f})"
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "/dfsph/particles",
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
    parser.add_argument("--color-field", choices=["kappa", "kappa_v"], default="kappa")
    args = parser.parse_args()

    test = NewtonDFSPHTest(
        device=args.device,
        load_from_usd=True,
        usd_path=args.usd,
        color_field=args.color_field,
    )
    test.run(args.frames)


if __name__ == "__main__":
    main()
