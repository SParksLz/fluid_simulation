import numpy as np
import warp as wp
# import warp.render
from renderer.render_opengl import CustomOpenGLRenderer
import json
from wcsph_kernel import *
import math
import time
from pxr import Usd, UsdGeom, Vt, Sdf
from pathlib import Path
from collections import defaultdict
from pprint import pprint

@wp.kernel
def velocity_visualize(
    v: wp.array(dtype = wp.vec3),
    color: wp.array(dtype = wp.vec3),
):
    tid = wp.tid()
    v_length = wp.length(v[tid])
    color[tid] = wp.lerp(wp.vec3(0.1, 0.5, 1.0), wp.vec3(1.0, 1.0, 1.0), v_length * 0.05)

    

def _gaussian_1d_weights(radius: int, sigma: float) -> np.ndarray:
    """1D 高斯核权重，长度 2*radius+1，已归一化。"""
    n = 2 * radius + 1
    x = np.arange(n, dtype=np.float64) - radius
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return w.astype(np.float32)


def _build_mesh_adjacency(indices: np.ndarray, n_verts: int):
    """从三角面片索引构建顶点邻接：adj_offsets[i+1]-adj_offsets[i] = 顶点 i 的邻居数，adj_flat 为展平的邻居索引。"""
    indices = np.asarray(indices, dtype=np.int32)
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)
    adj = defaultdict(set)
    for tri in indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    adj_offsets = np.zeros(n_verts + 1, dtype=np.int32)
    for i in range(n_verts):
        adj_offsets[i + 1] = adj_offsets[i] + len(adj.get(i, set()))
    adj_flat = np.zeros(adj_offsets[-1], dtype=np.int32)
    pos = np.zeros(n_verts, dtype=np.int32)
    for i in range(n_verts):
        for j in sorted(adj.get(i, set())):
            adj_flat[adj_offsets[i] + pos[i]] = j
            pos[i] += 1
    return adj_offsets, adj_flat


def _laplacian_smooth_mesh_gpu(verts: np.ndarray, indices: np.ndarray, iterations: int = 2, lambda_factor: float = 0.5, device: str = "cuda:0") -> np.ndarray:
    """在 GPU 上并行做拉普拉斯平滑（Warp kernel）。"""
    n_verts = verts.shape[0]
    if n_verts == 0:
        return verts
    adj_offsets, adj_flat = _build_mesh_adjacency(indices, n_verts)
    verts_wp = wp.array(verts.astype(np.float32), dtype=wp.vec3, device=device)
    new_verts_wp = wp.empty(n_verts, dtype=wp.vec3, device=device)
    adj_offsets_wp = wp.array(adj_offsets, dtype=wp.int32, device=device)
    adj_flat_wp = wp.array(adj_flat, dtype=wp.int32, device=device)
    for _ in range(iterations):
        wp.launch(
            kernel=laplacian_smooth_kernel,
            dim=n_verts,
            inputs=[verts_wp, new_verts_wp, adj_offsets_wp, adj_flat_wp, float(lambda_factor)],
            device=device,
        )
        verts_wp, new_verts_wp = new_verts_wp, verts_wp
    return verts_wp.numpy()


def _laplacian_smooth_mesh(verts: np.ndarray, indices: np.ndarray, iterations: int = 2, lambda_factor: float = 0.5) -> np.ndarray:
    """对三角 mesh 做拉普拉斯平滑（CPU 版本，保留作备用）。"""
    verts = np.asarray(verts, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int32)
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)
    n_verts = verts.shape[0]
    adj = defaultdict(set)
    for tri in indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(n_verts):
            nei = list(adj.get(i, ()))
            if len(nei) == 0:
                continue
            mean_nei = verts[nei].mean(axis=0)
            new_verts[i] = verts[i] + lambda_factor * (mean_nei - verts[i])
        verts = new_verts
    return verts.astype(np.float32)


class sph_material:
    def __init__(self, 
                rho = 1000.0, # rest density
                stiffness = 50000.0,
                exponent = 7.0, 
                mu=0.1,
                tension=0.1) -> None:
        self.rho = rho
        self.stiffness = stiffness
        self.exponent = exponent
        self.mu = mu
        self.tension = tension # tension
 
class sph_model :
    def __init__(self, bound_size, particle_distance) -> None:

        # self.lower_bound: tuple = (-1.0, -1.0, 0.0)
        # self.upper_bound: tuple = (1.0, 1.0, 1.0)
        self.particle_distance = particle_distance

        self.bound_width = bound_size
        self.bound_height = bound_size
        self.bound_length = bound_size

        self.liquid_material = sph_material()

    def build_hash_grid(self):
        grid_cell_size = int(self.bound_height / self.particle_distance)
        self.grid_cell_size = grid_cell_size

        self.grid = wp.HashGrid(grid_cell_size, grid_cell_size, grid_cell_size)
        # self.grid.build(particle_q, self.smoothing_length)

class wcsph:
    def __init__(self, load_from_usd=False) -> None:
        self.verbose = False
        # When True, timer blocks include GPU synchronization for accurate per-kernel timing.
        self.sync_timers = False
        # Neighbor counting is only for diagnostics and is expensive; keep it off in main simulation path.
        self.enable_neighbor_count = False
        # Optional deep diagnostics (copies arrays back to CPU), keep off during normal runs.
        self.profile_substep_stats = False
        self.profile_substep_interval = 1
        # Per-iteration DFSPH diagnostics: prints density error/divergence trend every solver iteration.
        self.log_dfsph_iteration_metrics = True
        # Per-substep summary diagnostics requested during DFSPH tuning.
        self.log_substep_metrics = True
        self.substep_metrics_log_enabled = True
        self.substep_metrics_log_path = Path(__file__).parent / "temp" / "dfsph_substep_metrics.csv"
        self.log_adaptive_dt = True
        self._substep_counter = 0
        self._timing_mode_notified = False
        self.load_from_usd = load_from_usd
        self.sim_time = 0.0
        self.sim_dt = 1.0 / 60.0
        self.device = "cuda:0"

        # self.particle_radius = 0.0125
        self.particle_radius : float = 0.0
        self.bound_size = 100.0
        self.bound_3d_size = wp.vec3(self.bound_size , self.bound_size * 0.5, self.bound_size)

        self.collider: wp.Mesh = None

        if self.load_from_usd:
            current_dir = Path(__file__).parent
            # self.load_particles_from_usd("./temp/particle_test.usd", wp.vec3(0.0, 0.0, 0.0))
            self.load_particles_from_usd((current_dir / "temp" / "fluid_particles.usd").as_posix(), wp.vec3(0.0, 0.0, 0.0))
        else:
            self.n = int(
                self.bound_size * self.bound_size * self.bound_size / (self.smoothing_length**3)
            )  # number particles (small box in corner)
            self.x = wp.empty(self.n, dtype=wp.vec3, device=self.device)


        self.particle_distance = self.particle_radius * 2.0
        self.smoothing_length = self.particle_distance * 1.35
        # self.particle_distance = self.smoothing_length 
        self.sph_model = sph_model(self.bound_size, self.smoothing_length)

        # fluid material
        self.sph_model.liquid_material.tension = 0.02
        self.sph_model.liquid_material.stiffness = 85000.0
        self.sph_model.liquid_material.mu = 0.05

        self.p_volume = 0.8 * (self.particle_distance ** 3)
        self.sub_step_num = 4
        self.gravity = -10.0

        self.camera_pos = (0.0, 8.5, 5.0)
        # self.camera_pos = (0.0, 0.0, 0.175)
        self.camera_pos = (-2.75, 0.5, 0.0)
        self.mc : wp.MarchingCubes | None = None

        # # 体积和 marching cubes 相关参数
        # self.volume_resolution = 512  # 体积网格分辨率
        # self.volume_density = None  # 将在 render 中创建
        # self.volume_threshold = 0.3  # 等值面阈值，可根据密度调整
        margin_scale = 2.0
        voxel_size = self.smoothing_length *  0.65
        margin = self.smoothing_length * margin_scale

        self.world_min = wp.vec3(-self.bound_3d_size[0] - margin, -self.bound_3d_size[1] - margin, 0.0 - margin)
        self.world_max = wp.vec3(self.bound_3d_size[0] + margin, self.bound_3d_size[1] + margin, (self.bound_3d_size[2] * 2) + margin)
        size = self.world_max - self.world_min
        # size[2] = size[2] * 2.0

        self.volume_res_x = int(math.ceil(size[0] / voxel_size))
        self.volume_res_y = int(math.ceil(size[1] / voxel_size))
        self.volume_res_z = int(math.ceil(size[2] / voxel_size))


        index_min = [0, 0, 0]
        index_max = [self.volume_res_x, self.volume_res_y, self.volume_res_z]

        # breakpoint()


        self.volume: wp.Volume = wp.Volume.allocate(
            min = index_min,
            max = index_max,
            voxel_size = voxel_size,
            bg_value = 0.0,
            translation = (self.world_min[0], self.world_min[1], self.world_min[2]),
            points_in_world_space = True,
            device = "cuda:0"
        )

        # 计算体积大小
        # volume_size_x = (self.bound_3d_size[0] + 0.2) * 2.0
        # volume_size_y = (self.bound_3d_size[1] + 0.2) * 2.0
        # volume_size_z = (self.bound_3d_size[2] + 0.2)* 2.0


        # resolution_scale = 0.5
    
        
        # 创建体积密度网格
        # self.volume_density = wp.zeros(
        #     (self.volume_res_x, self.volume_res_y, self.volume_res_z), 
        #     dtype=float, 
        #     device="cuda:0"
        # )
        
        self.volume_threshold = 0.55  # 等值面阈值，可根据密度调整
        self.volume_smooth_iters = 1  # 对 density volume 做 3D 可分离高斯模糊的遍数（每遍 = x+y+z 三向）
        self.volume_gaussian_radius = 1  # 1D 高斯核半径（半宽），核长 2*radius+1
        self.volume_gaussian_sigma = 1.0  # 1D 高斯 sigma
        self.enable_volume_mesh = False
        self.volume_mesh_update_interval = 6
        self._frame_index = 0



        print(f"particle count : {self.n}")
        dev = self.device
        self.mass = wp.full(self.n, self.p_volume * self.sph_model.liquid_material.rho, device=dev)
        self.gamma = wp.full(self.n, self.sph_model.liquid_material.tension, device=dev)
        self.stiffness = wp.full(self.n, self.sph_model.liquid_material.stiffness, device=dev)
        self.exponent = wp.full(self.n, self.sph_model.liquid_material.exponent, device=dev)
        self.mu = wp.full(self.n, self.sph_model.liquid_material.mu, device=dev)
        self.rho_0 = wp.full(self.n, self.sph_model.liquid_material.rho, device=dev)

        self.v = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        # self.colors = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.rho = wp.zeros(self.n, dtype=float, device=dev)
        self.a = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.nei_count = wp.zeros(self.n, dtype=wp.int32, device=dev)
        self.bound_touched = wp.zeros(self.n, dtype=wp.int32, device=dev)
        self.bound_penetration = wp.zeros(self.n, dtype=float, device=dev)
        self.pressure = wp.zeros(self.n, dtype=float, device=dev)
        self.factor = wp.zeros(self.n, dtype=float, device=dev)

        self.use_dfsph = True
        self.v_star = wp.empty(self.n, dtype=wp.vec3, device=dev)
        self.rho_star = wp.zeros(self.n, dtype=float, device=dev)
        self.d_rho_dt = wp.zeros(self.n, dtype=float, device=dev)
        self.kappa = wp.zeros(self.n, dtype=float, device=dev)
        self.dfsph_eta_density = 1e-4 * self.sph_model.liquid_material.rho
        self.dfsph_eta_divergence = 1e-4
        self.dfsph_max_iters_density = 2
        self.dfsph_max_iters_divergence = 1

        self.render_x = wp.empty(self.n, dtype=wp.vec3, device=dev)
        if not hasattr(self, "colors"):
            self.colors = wp.full(self.n, wp.vec3(0.2, 0.3, 0.7), device=dev)

        # set random positions
        # wp.launch(
        #     kernel=initialize_particles,
        #     dim=self.n,
        #     inputs=[
        #         self.x, 
        #         self.smoothing_length, 
        #         self.bound_size, 
        #         self.bound_size, 
        #         self.bound_size,
        #         wp.vec3(-self.bound_size * 0.5, -self.bound_size * 0.5, 0.0),
        #     ],
        # )  # initialize in small area

        # self.save_particle_to_json()

        self.sph_model.build_hash_grid()
        print(
            f"hash grid: {self.sph_model.grid_cell_size}^3, smoothing_length={self.smoothing_length:.6f}, "
            f"neighbor_count_kernel={'on' if self.enable_neighbor_count else 'off'}"
        )

        self.renderer = CustomOpenGLRenderer(
            up_axis="Z",
            camera_pos=self.camera_pos,
            near_plane=0.001,
            far_plane = 1000.0,
            draw_axis=False,
            camera_up=(0.0, 1.0, 0.0),
            camera_front=(1.0, 0.0, 0.0),
        )
        # self.renderer = wp.render.UsdRenderer(
        #     up_axis="Z", 
        #     stage="wcsph_test.usd",
        # )

        # self.preparation()
        self._init_substep_metrics_log()

    def compute_sim_dt(self):
        # Estimate stable substep dt, then convert to frame dt using sub_step_num.
        wp.synchronize_device(self.device)
        a_np = self.a.numpy()
        a_norm = np.linalg.norm(a_np, axis=1)
        a_max = float(np.max(a_norm))
        if not np.isfinite(a_max):
            a_max = 0.0

        eps = 1.0e-12
        c_s = math.sqrt(self.sph_model.liquid_material.stiffness / self.sph_model.liquid_material.rho)

        # CFL-like criteria on substep dt.
        sub_dt_force = 0.25 * math.sqrt(self.smoothing_length / (a_max + eps))
        sub_dt_sound = 0.4 * self.smoothing_length / (c_s * 1.05 + eps)
        sub_dt_target = min(sub_dt_force, sub_dt_sound)
        frame_dt_target = sub_dt_target * float(max(self.sub_step_num, 1))

        prev_dt = float(self.sim_dt)
        dt_min = prev_dt * 0.5
        dt_max = prev_dt * 1.1
        new_dt = max(dt_min, min(dt_max, frame_dt_target))
        self.sim_dt = new_dt

        if self.log_adaptive_dt:
            pprint(
                "adaptive dt: "
                f"prev={prev_dt:.6e}, new={new_dt:.6e}, "
                f"sub_target={sub_dt_target:.6e}, "
                f"sub_force={sub_dt_force:.6e}, sub_sound={sub_dt_sound:.6e}, "
                f"a_max={a_max:.6e}, c_s={c_s:.6e}"
            )


    def _sync_for_timing(self, dev: str | None = None):
        if self.sync_timers:
            wp.synchronize_device(dev or self.device)

    def _init_substep_metrics_log(self):
        if not self.substep_metrics_log_enabled:
            return
        self.substep_metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.substep_metrics_log_path.open("w", encoding="utf-8") as f:
            f.write(
                "substep,max_abs_drho_dt,max_rho_ratio_err,apply_bounds_count,max_penetration\n"
            )

    def _append_substep_metrics_log(
        self,
        max_abs_drho_dt: float,
        max_density_ratio_err: float,
        apply_bounds_count: int,
        max_penetration: float,
    ):
        if not self.substep_metrics_log_enabled:
            return
        with self.substep_metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{self._substep_counter},{max_abs_drho_dt:.10e},{max_density_ratio_err:.10e},"
                f"{apply_bounds_count},{max_penetration:.10e}\n"
            )

    def _timed_kernel(self, name: str, kernel, dim, inputs, device: str):
        if not self.verbose:
            wp.launch(kernel=kernel, dim=dim, inputs=inputs, device=device)
            return
        if self.sync_timers:
            self._sync_for_timing(device)
            t0 = time.perf_counter()
            wp.launch(kernel=kernel, dim=dim, inputs=inputs, device=device)
            self._sync_for_timing(device)
            print(f"        {name} took {(time.perf_counter() - t0) * 1000.0:.2f} ms")
        else:
            with wp.ScopedTimer(name, active=True):
                wp.launch(kernel=kernel, dim=dim, inputs=inputs, device=device)

    def _timed_block(self, name: str, fn, device: str):
        if not self.verbose:
            fn()
            return
        if self.sync_timers:
            self._sync_for_timing(device)
            t0 = time.perf_counter()
            fn()
            self._sync_for_timing(device)
            print(f"        {name} took {(time.perf_counter() - t0) * 1000.0:.2f} ms")
        else:
            with wp.ScopedTimer(name, active=True):
                fn()

    def _apply_bounds_with_stats(self, velocity_array, bounds_stats: dict, device: str):
        wp.launch(
            kernel=apply_bounds,
            dim=self.n,
            inputs=[
                self.x,
                velocity_array,
                self.bound_3d_size,
                -0.1,
                self.bound_touched,
                self.bound_penetration,
            ],
            device=device,
        )
        if not self.log_substep_metrics:
            return
        self._sync_for_timing(device)
        touched_np = self.bound_touched.numpy()
        penetration_np = self.bound_penetration.numpy()
        touched_mask = touched_np != 0
        if bounds_stats["touched_mask"] is None:
            bounds_stats["touched_mask"] = touched_mask
        else:
            bounds_stats["touched_mask"] |= touched_mask
        bounds_stats["max_penetration"] = max(
            bounds_stats["max_penetration"],
            float(np.max(penetration_np)),
        )

    def _print_substep_metrics(self, bounds_stats: dict, device: str):
        if not self.log_substep_metrics:
            return
        self.sph_model.grid.build(self.x, self.smoothing_length)
        wp.launch(
            kernel=compute_density_change_rate,
            dim=self.n,
            inputs=[
                self.sph_model.grid.id,
                self.x,
                self.v,
                self.mass,
                self.smoothing_length,
                self.d_rho_dt,
            ],
            device=device,
        )
        wp.launch(
            kernel=rho_dfsph,
            dim=self.n,
            inputs=[
                self.rho,
                self.x,
                self.rho_0,
                self.p_volume,
                self.smoothing_length,
                self.sph_model.grid.id,
            ],
            device=device,
        )
        self._sync_for_timing(device)
        d_rho_dt_np = self.d_rho_dt.numpy()
        rho_np = self.rho.numpy()
        rho0_np = self.rho_0.numpy()
        max_abs_drho_dt = float(np.max(np.abs(d_rho_dt_np)))
        max_density_ratio_err = float(np.max(rho_np / np.maximum(rho0_np, 1.0e-12) - 1.0))
        bounds_count = (
            0
            if bounds_stats["touched_mask"] is None
            else int(np.count_nonzero(bounds_stats["touched_mask"]))
        )
        print(
            "        substep metrics: "
            f"max_abs_drho_dt={max_abs_drho_dt:.6e}, "
            f"max(rho/rho0-1)={max_density_ratio_err:.6e}, "
            f"apply_bounds_count={bounds_count}, "
            f"max_penetration={bounds_stats['max_penetration']:.6e}"
        )
        self._append_substep_metrics_log(
            max_abs_drho_dt=max_abs_drho_dt,
            max_density_ratio_err=max_density_ratio_err,
            apply_bounds_count=bounds_count,
            max_penetration=bounds_stats["max_penetration"],
        )



    

    def sub_step(self, dt: float | None = None) :
        if dt is None:
            dt = float(self.sim_dt) / float(max(self.sub_step_num, 1))
        else:
            dt = float(dt)
        bounds_stats = {"touched_mask": None, "max_penetration": 0.0}
        def _wcsph_substep() :
            with wp.ScopedTimer("grid build", active=False):
                        # build grid
                self.sph_model.grid.build(self.x, self.smoothing_length)
                if self.enable_neighbor_count:
                    wp.launch(
                        kernel=get_neighbor,
                        dim=self.n,
                        inputs=[
                            self.sph_model.grid.id,
                            self.x,
                            self.smoothing_length,
                            self.nei_count,
                        ]
                    )
            with wp.ScopedTimer("calculate density", active = self.verbose) :

                wp.launch(
                    kernel=rho,
                    dim=self.n,
                    inputs=[
                        self.rho,
                        self.exponent,
                        self.stiffness,
                        self.x,
                        self.rho_0,
                        self.p_volume,
                        self.smoothing_length,
                        self.sph_model.grid.id,
                        self.pressure,
                    ]
                )
            with wp.ScopedTimer("calculate acceleration", active = self.verbose) :
                wp.launch(
                    kernel=acceleration,
                    dim=self.n,
                    inputs=[
                        self.x,
                        self.v,
                        self.rho_0,
                        self.rho,
                        self.a,
                        self.particle_radius,
                        self.stiffness,
                        self.exponent,
                        self.pressure,
                        self.p_volume,
                        self.mass,
                        self.gamma,
                        self.mu,
                        self.gravity,
                        self.smoothing_length,
                        self.sph_model.grid.id,
                    ]
                )
            # # kick
            wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, dt])

            # # drift
            wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, dt])
            # # ground collision
            self._apply_bounds_with_stats(self.v, bounds_stats, self.device)
            # wp.launch(
            #     kernel=update_collider_with_tri_mesh, 
            #     dim=self.n, 
            #     inputs=[
            #         self.x, 
            #         self.v, 
            #         self.collider.id, 
            #         self.particle_radius, 
            #         0.9, 0.1],
            # )
            wp.launch(
                kernel=to_real_world,
                dim=self.n,
                inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)])
        def _dfsph_substep():
            dev = self.device
            # 1) 邻居
            def _build_grid_phase_1():
                self.sph_model.grid.build(self.x, self.smoothing_length)
                if self.enable_neighbor_count:
                    wp.launch(
                        kernel=get_neighbor,
                        dim=self.n,
                        inputs=[
                            self.sph_model.grid.id,
                            self.x,
                            self.smoothing_length,
                            self.nei_count,
                        ],
                        device=dev,
                    )
            self._timed_block("grid build", _build_grid_phase_1, dev)
            # 2) 密度 ρ 与因子 α（论文 Algorithm 1：每步开始时的 ρ、α）
            with wp.ScopedTimer("density and factor", active=self.verbose):
                wp.launch(
                    kernel=rho_dfsph,
                    dim=self.n,
                    inputs=[
                        self.rho,
                        self.x,
                        self.rho_0,
                        self.p_volume,
                        self.smoothing_length,
                        self.sph_model.grid.id,
                    ],
                    device=dev,
                )
                wp.launch(
                    kernel=compute_factor_dfsph,
                    dim=self.n,
                    inputs=[
                        self.sph_model.grid.id,
                        self.x,
                        self.mass,
                        self.smoothing_length,
                        self.factor,
                    ],
                    device=dev,
                )
                self._sync_for_timing(dev)
            # 3) 非压力力 F_adv，预测速度 v* = v + Δt F_adv/m
            with wp.ScopedTimer("non-pressure forces", active=self.verbose):
                wp.launch(
                    kernel=acceleration_non_pressure,
                    dim=self.n,
                    inputs=[
                        self.x,
                        self.v,
                        self.rho_0,
                        self.rho,
                        self.a,
                        self.particle_radius,
                        self.mass,
                        self.gamma,
                        self.mu,
                        self.gravity,
                        self.smoothing_length,
                        self.sph_model.grid.id,
                    ],
                    device=dev,
                )
                self._sync_for_timing(dev)
            self._timed_kernel(
                name="predict_velocity",
                kernel=predict_velocity,
                dim=self.n,
                inputs=[self.v, self.a, dt, self.v_star],
                device=dev,
            )
            # 4) 常数密度修正 (Algorithm 3)：至少 2 次迭代
            with wp.ScopedTimer("density correct iteration", active=self.verbose):
                self._sync_for_timing(dev)
                rho0_np = None
                prev_density_err_max = None
                prev_density_div_max = None
                if self.log_dfsph_iteration_metrics:
                    rho0_np = self.rho_0.numpy()
                for it in range(self.dfsph_max_iters_density):
                    self._timed_kernel(
                        name="compute_density_change_rate",
                        kernel=compute_density_change_rate,
                        dim=self.n,
                        inputs=[
                            self.sph_model.grid.id,
                            self.x,
                            self.v_star,
                            self.mass,
                            self.smoothing_length,
                            self.d_rho_dt,
                        ],
                        device=dev,
                    )
                    self._timed_kernel(
                        name="predict_density_star",
                        kernel=predict_density_star,
                        dim=self.n,
                        inputs=[self.rho, self.d_rho_dt, dt, self.rho_star],
                        device=dev,
                    )
                    if self.log_dfsph_iteration_metrics:
                        rho_star_np = self.rho_star.numpy()
                        d_rho_dt_np = self.d_rho_dt.numpy()
                        rho_err_np = np.maximum(rho_star_np - rho0_np, 0.0)
                        rho_err_max = float(np.max(rho_err_np))
                        rho_err_mean = float(np.mean(rho_err_np))
                        div_abs_np = np.abs(d_rho_dt_np)
                        div_max = float(np.max(div_abs_np))
                        div_mean = float(np.mean(div_abs_np))
                        if prev_density_err_max is None:
                            rho_trend = "n/a"
                        else:
                            rho_trend = "down" if rho_err_max <= prev_density_err_max else "up"
                        if prev_density_div_max is None:
                            div_trend = "n/a"
                        else:
                            div_trend = "down" if div_max <= prev_density_div_max else "up"
                        print(
                            f"        [density iter {it + 1}/{self.dfsph_max_iters_density}] "
                            f"rho_err(max/mean)=({rho_err_max:.6e}/{rho_err_mean:.6e}) [{rho_trend}] "
                            f"| |drho_dt|(max/mean)=({div_max:.6e}/{div_mean:.6e}) [{div_trend}]"
                        )
                        prev_density_err_max = rho_err_max
                        prev_density_div_max = div_max
                    self._timed_kernel(
                        name="compute_kappa_density",
                        kernel=compute_kappa_density,
                        dim=self.n,
                        inputs=[
                            self.rho_star,
                            self.rho_0,
                            self.factor,
                            dt,
                            self.kappa,
                        ],
                        device=dev,
                    )
                    self._timed_kernel(
                        name="apply_constant_density_correction",
                        kernel=apply_constant_density_correction,
                        dim=self.n,
                        inputs=[
                            self.sph_model.grid.id,
                            self.x,
                            self.v_star,
                            self.mass,
                            self.rho,
                            self.kappa,
                            self.smoothing_length,
                            dt,
                        ],
                        device=dev,
                    )
                self._sync_for_timing(dev)
            # 5) 更新位置 x = x + Δt v*
            wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v_star, dt], device=dev)
            # Apply boundary response before rebuilding grid / solving divergence.
            self._apply_bounds_with_stats(self.v_star, bounds_stats, dev)
            # 6) 位置更新后重新算邻居、ρ、α（论文 line 16–20）
            def _build_grid_phase_2():
                self.sph_model.grid.build(self.x, self.smoothing_length)
            self._timed_block("grid rebuild", _build_grid_phase_2, dev)
            wp.launch(
                kernel=rho_dfsph,
                dim=self.n,
                inputs=[
                    self.rho,
                    self.x,
                    self.rho_0,
                    self.p_volume,
                    self.smoothing_length,
                    self.sph_model.grid.id,
                ],
                device=dev,
            )
            wp.launch(
                kernel=compute_factor_dfsph,
                dim=self.n,
                inputs=[
                    self.sph_model.grid.id,
                    self.x,
                    self.mass,
                    self.smoothing_length,
                    self.factor,
                ],
                device=dev,
            )
            # 7) 散度自由修正 (Algorithm 2)：至少 1 次迭代
            with wp.ScopedTimer("divergence correct iteration", active=self.verbose):
                self._sync_for_timing(dev)
                prev_divergence_max = None
                for it in range(self.dfsph_max_iters_divergence):
                    self._timed_kernel(
                        name="compute_density_change_rate(div)",
                        kernel=compute_density_change_rate,
                        dim=self.n,
                        inputs=[
                            self.sph_model.grid.id,
                            self.x,
                            self.v_star,
                            self.mass,
                            self.smoothing_length,
                            self.d_rho_dt,
                        ],
                        device=dev,
                    )
                    if self.log_dfsph_iteration_metrics:
                        d_rho_dt_np = self.d_rho_dt.numpy()
                        div_abs_np = np.abs(d_rho_dt_np)
                        div_max = float(np.max(div_abs_np))
                        div_mean = float(np.mean(div_abs_np))
                        if prev_divergence_max is None:
                            div_trend = "n/a"
                        else:
                            div_trend = "down" if div_max <= prev_divergence_max else "up"
                        print(
                            f"        [div iter {it + 1}/{self.dfsph_max_iters_divergence}] "
                            f"|drho_dt|(max/mean)=({div_max:.6e}/{div_mean:.6e}) [{div_trend}]"
                        )
                        prev_divergence_max = div_max
                    self._timed_kernel(
                        name="compute_kappa_divergence",
                        kernel=compute_kappa_divergence,
                        dim=self.n,
                        inputs=[self.d_rho_dt, self.rho, self.factor, dt, self.kappa],
                        device=dev,
                    )
                    self._timed_kernel(
                        name="apply_divergence_free_correction",
                        kernel=apply_divergence_free_correction,
                        dim=self.n,
                        inputs=[
                            self.sph_model.grid.id,
                            self.x,
                            self.v_star,
                            self.mass,
                            self.rho,
                            self.kappa,
                            self.smoothing_length,
                            dt,
                        ],
                        device=dev,
                    )
                self._sync_for_timing(dev)
            # 8) v = v*，边界与渲染坐标
            wp.launch(kernel=copy_velocity, dim=self.n, inputs=[self.v_star, self.v], device=dev)
            # self._apply_bounds_with_stats(self.v, bounds_stats, dev)
            wp.launch(
                kernel=to_real_world,
                dim=self.n,
                inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)],
                device=dev,
            )
            if self.profile_substep_stats and (self._substep_counter % max(self.profile_substep_interval, 1) == 0):
                wp.launch(
                    kernel=get_neighbor,
                    dim=self.n,
                    inputs=[
                        self.sph_model.grid.id,
                        self.x,
                        self.smoothing_length,
                        self.nei_count,
                    ],
                    device=dev,
                )
                self._sync_for_timing(dev)
                nei_np = self.nei_count.numpy()
                v_np = self.v.numpy()
                v_norm = np.linalg.norm(v_np, axis=1)
                print(
                    "        stats: "
                    f"nei(mean/p95/max)=({nei_np.mean():.1f}/{np.percentile(nei_np, 95):.1f}/{int(nei_np.max())}), "
                    f"vmax={v_norm.max():.4f}, vmean={v_norm.mean():.4f}"
                )
            self._substep_counter += 1
        if self.use_dfsph:
            _dfsph_substep()
        else:
            _wcsph_substep()
        self._print_substep_metrics(bounds_stats, self.device)

    def preparation(self) :
        with wp.ScopedTimer("preparation", active=True):
            dev = self.device
            self.sph_model.grid.build(self.x, self.smoothing_length)
            if self.enable_neighbor_count:
                wp.launch(
                    kernel=get_neighbor,
                    dim=self.n,
                    inputs=[
                        self.sph_model.grid.id,
                        self.x,
                        self.smoothing_length,
                        self.nei_count,
                    ],
                    device=dev,
                )
            # 初始密度：写入 self.rho，并传入 grid_id
            wp.launch(
                kernel=rho_dfsph,
                dim=self.n,
                inputs=[
                    self.rho,
                    self.x,
                    self.rho_0,
                    self.p_volume,
                    self.smoothing_length,
                    self.sph_model.grid.id,
                ],
                device=dev,
            )
            if self.use_dfsph:
                wp.launch(
                    kernel=compute_factor_dfsph,
                    dim=self.n,
                    inputs=[
                        self.sph_model.grid.id,
                        self.x,
                        self.mass,
                        self.smoothing_length,
                        self.factor,
                    ],
                    device=dev,
                )
            # else:
            #     wp.launch(
            #         kernel=compute_factor,
            #         dim=self.n,
            #         inputs=[
            #             self.sph_model.grid.id,
            #             self.x,
            #             self.smoothing_length,
            #             self.p_volume,
            #             self.factor,
            #         ],
            #     )
        # print("----rho-----")
        # print(self.rho)
        # print("-----factor-----")
        # print(self.factor)
        # print("----nei_count----")
        # print(self.nei_count)



    def step(self) :
        # pass
        if self.verbose and not self.sync_timers and not self._timing_mode_notified:
            print("timing note: verbose timers are async launch times; set sync_timers=True for accurate GPU timings")
            self._timing_mode_notified = True
        with wp.ScopedTimer("step", active=True):
            self.compute_sim_dt()
            sub_dt = float(self.sim_dt) / float(max(self.sub_step_num, 1))
            for _ in range(self.sub_step_num):
                with wp.ScopedTimer("sub_step", active=False):
                    self.sub_step(sub_dt)
                    # print(self.render_x)
                    self.sim_time += sub_dt

    def render_test(self):
        # if self.renderer is None:
        #     return

        with wp.ScopedTimer("render", active=False):
            # 在渲染之前更新 render_x，确保使用最新的位置
            wp.launch(
                kernel=to_real_world,
                dim=self.n,
                inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)])
            
            self.renderer.begin_frame(self.sim_time)


            render_x_np = self.render_x.numpy()

            
            self.renderer.render_points(
                points=self.render_x, 
                radius=self.particle_radius * 0.01, 
                name="points", 
                colors=(0.2, 0.3, 0.7)
            )
            self.renderer.end_frame()

    def render(self):
        with wp.ScopedTimer("render", active=True):
            # 在渲染之前更新 render_x，确保使用最新的位置
            wp.launch(
                kernel=to_real_world,
                dim=self.n,
                inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)],
                device=self.device,
            )
            
            # self.renderer.begin_frame(self.sim_time)
            # if self.enable_volume_mesh and (self._frame_index % self.volume_mesh_update_interval == 0):
            #     density_buffer = wp.zeros(
            #         (
            #             self.volume_res_x,
            #             self.volume_res_y,
            #             self.volume_res_z,
            #         ),
            #         dtype=float,
            #         device=self.device,
            #     )
            #     debug_info = wp.zeros(3, dtype=wp.vec3, device=self.device)

            #     wp.launch(
            #         kernel=rasterize_particles_to_nvdb_volume,
            #         dim=self.n,
            #         inputs=[
            #             self.x,
            #             self.volume.id,
            #             density_buffer,
            #             self.smoothing_length,
            #             self.p_volume,
            #             self.volume_res_x,
            #             self.volume_res_y,
            #             self.volume_res_z,
            #             debug_info,
            #         ],
            #         device=self.device,
            #     )

            #     if self.mc is None:
            #         self.mc = wp.MarchingCubes(
            #             nx=self.volume_res_x,
            #             ny=self.volume_res_y,
            #             nz=self.volume_res_z,
            #             domain_bounds_lower_corner=wp.vec3(self.world_min[0], self.world_min[1], self.world_min[2]),
            #             domain_bounds_upper_corner=wp.vec3(self.world_max[0], self.world_max[1], self.world_max[2]),
            #             device=self.device,
            #         )
            #     else:
            #         self.mc.domain_bounds_lower_corner = wp.vec3(self.world_min[0], self.world_min[1], self.world_min[2])
            #         self.mc.domain_bounds_upper_corner = wp.vec3(self.world_max[0], self.world_max[1], self.world_max[2])

            #     volume_smooth_iters = self.volume_smooth_iters
            #     radius = self.volume_gaussian_radius
            #     sigma = self.volume_gaussian_sigma
            #     dim_vol = (self.volume_res_x, self.volume_res_y, self.volume_res_z)
            #     density_smoothed = wp.zeros(dim_vol, dtype=float, device=self.device)
            #     buf_a = wp.zeros(dim_vol, dtype=float, device=self.device)
            #     buf_b = wp.zeros(dim_vol, dtype=float, device=self.device)
            #     weights_np = _gaussian_1d_weights(radius, sigma)
            #     weights_wp = wp.array(weights_np, dtype=float, device=self.device)
            #     for _ in range(volume_smooth_iters):
            #         src = density_buffer if _ == 0 else density_smoothed
            #         wp.launch(kernel=smooth_volume_1d_x, dim=dim_vol, inputs=[src, buf_a, self.volume_res_x, self.volume_res_y, self.volume_res_z, weights_wp, radius], device=self.device)
            #         wp.launch(kernel=smooth_volume_1d_y, dim=dim_vol, inputs=[buf_a, buf_b, self.volume_res_x, self.volume_res_y, self.volume_res_z, weights_wp, radius], device=self.device)
            #         wp.launch(kernel=smooth_volume_1d_z, dim=dim_vol, inputs=[buf_b, density_smoothed, self.volume_res_x, self.volume_res_y, self.volume_res_z, weights_wp, radius], device=self.device)
            #     field_for_mc = density_smoothed if volume_smooth_iters > 0 else density_buffer

            #     self.mc.surface(
            #         field=field_for_mc,
            #         threshold=self.volume_threshold,
            #     )

            # 可选：同时渲染粒子点（用于调试）
            # wp.launch(
            #     kernel=velocity_visualize,
            #     dim=self.n,
            #     inputs=[self.v, self.colors],
            #     device=self.device,
            # )
            self.renderer.render_points(
                points=self.render_x,
                radius=self.particle_radius * 0.01,
                name="points",
                # colors=self.colors.numpy(),
            )

            # grid_pts = wp.array(self.get_voxel_positions(), dtype=wp.vec3)
            # debug_grid = wp.empty(grid_pts.shape[0], dtype=wp.vec3)


            # wp.launch(
            #     kernel=to_real_world,
            #     dim=grid_pts.shape[0],
            #     inputs=[grid_pts, debug_grid, 0.01, wp.vec3(0.0, 0.0, 0.0)])

            # self.renderer.render_points(
            #     points=self.render_x, 
            #     radius=self.particle_radius * 0.01, 
            #     name="grid", 
            #     colors=(0.8, 0.8, 0.8)
            # )

            
            self.renderer.end_frame()
            self._frame_index += 1

    def get_voxel_positions(self, return_centers=True, return_indices=False, filter_active=False):
        """
        获取体素位置信息，用于调试
        
        Args:
            return_centers: 如果为 True，返回体素中心的世界坐标位置
            return_indices: 如果为 True，返回体素索引 (i, j, k)
            filter_active: 如果为 True，只返回有密度值的体素（密度 > 0）
        
        Returns:
            如果 return_centers=True 且 return_indices=False:
                ndarray of shape (N, 3) - 体素中心的世界坐标
            如果 return_centers=True 且 return_indices=True:
                tuple: (centers, indices)
                - centers: ndarray of shape (N, 3) - 体素中心坐标
                - indices: ndarray of shape (N, 3) - 体素索引 (i, j, k)
            如果 return_centers=False 且 return_indices=True:
                ndarray of shape (N, 3) - 体素索引
        """
        # 获取密度值（用于过滤）
        if filter_active:
            density_np = self.volume_density.numpy()
            active_mask = density_np > 0
            active_indices = np.argwhere(active_mask)
        else:
            # 生成所有体素索引
            i_indices = np.arange(self.volume_resolution_x)
            j_indices = np.arange(self.volume_resolution_y)
            k_indices = np.arange(self.volume_resolution_z)
            active_indices = np.array(np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')).T.reshape(-1, 3)
        
        if return_centers:
            # 计算体素中心的世界坐标
            # 体素中心 = volume_origin + (索引 + 0.5) * voxel_world_size
            centers = np.zeros((len(active_indices), 3))
            centers[:, 0] = self.volume_origin[0] + (active_indices[:, 0] + 0.5) * self.voxel_world_size_x
            centers[:, 1] = self.volume_origin[1] + (active_indices[:, 1] + 0.5) * self.voxel_world_size_y
            centers[:, 2] = self.volume_origin[2] + (active_indices[:, 2] + 0.5) * self.voxel_world_size_z
            
            if return_indices:
                return centers, active_indices
            else:
                return centers
        else:
            if return_indices:
                return active_indices
            else:
                return None
    
    def get_voxel_info(self):
        """
        获取体素网格的完整信息，用于调试
        
        Returns:
            dict: 包含体素网格信息的字典
        """
        density_np = self.volume_density.numpy()
        active_count = np.sum(density_np > 0)
        total_count = density_np.size
        
        info = {
            'resolution': (self.volume_resolution_x, self.volume_resolution_y, self.volume_resolution_z),
            'volume_origin': self.volume_origin.copy(),
            'volume_size': self.volume_size.copy(),
            'voxel_world_size': np.array([
                self.voxel_world_size_x,
                self.voxel_world_size_y,
                self.voxel_world_size_z
            ]),
            'total_voxels': total_count,
            'active_voxels': active_count,
            'density_min': float(np.min(density_np)),
            'density_max': float(np.max(density_np)),
            'density_mean': float(np.mean(density_np)),
            'density_sum': float(np.sum(density_np))
        }
        
        return info
    
    def get_voxel_corners(self, i, j, k):
        """
        获取指定体素的8个角点坐标
        
        Args:
            i, j, k: 体素索引
        
        Returns:
            ndarray of shape (8, 3) - 体素的8个角点坐标
        """
        if not (0 <= i < self.volume_resolution_x and 
                0 <= j < self.volume_resolution_y and 
                0 <= k < self.volume_resolution_z):
            raise ValueError(f"Voxel index ({i}, {j}, {k}) out of range")
        
        # 体素的边界
        x_min = self.volume_origin[0] + i * self.voxel_world_size_x
        x_max = x_min + self.voxel_world_size_x
        y_min = self.volume_origin[1] + j * self.voxel_world_size_y
        y_max = y_min + self.voxel_world_size_y
        z_min = self.volume_origin[2] + k * self.voxel_world_size_z
        z_max = z_min + self.voxel_world_size_z
        
        # 8个角点
        corners = np.array([
            [x_min, y_min, z_min],  # 0: 左下后
            [x_max, y_min, z_min],  # 1: 右下后
            [x_max, y_max, z_min],  # 2: 右上后
            [x_min, y_max, z_min],  # 3: 左上后
            [x_min, y_min, z_max],  # 4: 左下前
            [x_max, y_min, z_max],  # 5: 右下前
            [x_max, y_max, z_max],  # 6: 右上前
            [x_min, y_max, z_max],  # 7: 左上前
        ])
        
        return corners
    def save_particle_to_json(self, filename="particles.json"):
        """
            将粒子位置数据保存到 JSON 文件
        """
        # 将 Warp 数组转换为 numpy 数组，然后转换为列表
        positions = self.x.numpy()
        
        # 转换为列表格式：[[x1, y1, z1], [x2, y2, z2], ...]
        particles_data = positions.tolist()
        
        # 保存到 JSON 文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(particles_data, f, indent=2)
        
        print(f"粒子数据已保存到 {filename}，共 {self.n} 个粒子")
    def load_particles_from_usd(self, filename, offset: wp.vec3):
        stage = Usd.Stage.Open(filename)
        fluid = stage.GetPrimAtPath("/Fluid/Particles")
        if fluid.IsValid():
            points = UsdGeom.Points(fluid)
            points_np = np.array(points.GetPointsAttr().Get())
            # 复制一份并将 x 取反后拼接到原粒子
            points_copy = np.array(points_np, copy=True)
            # points_copy[:, 0] = points_copy[:, 0] + 0.5

            blue_color = np.full((len(points_copy), 3), (0.0, 0.0, 1.0))
            # yellow_color = np.full((len(points_copy), 3), (1.0, 1.0, 0.0))

            # points_np = np.concatenate([points_np, points_copy], axis=0)
            colors_np = np.concatenate(
                [
                    blue_color, 
                    # yellow_color,
                ], axis=0)
            self.particle_radius = points.GetWidthsAttr().Get()[0] * 0.5 * 100.0
            self.x = wp.array(points_np, dtype=wp.vec3, device=self.device)
            self.colors = wp.array(colors_np, dtype=wp.vec3, device=self.device)
            self.n = len(points_np)
            wp.launch(
                kernel=to_micro_world, 
                dim=self.n, 
                inputs=[self.x, 100.0, offset])
            print(f"粒子数据已加载到 {filename}，共 {self.n} 个粒子")
        container = stage.GetPrimAtPath("/ParticleTest/Container")
        if container.IsValid() :
            mesh = UsdGeom.Mesh(container)
            np_vtx = np.array(mesh.GetPointsAttr().Get())
            vtx = wp.array(np_vtx, dtype=wp.vec3)
            wp.launch(
                kernel=to_micro_world, 
                dim=len(np_vtx), 
                inputs=[vtx, 100.0, offset])
            idx = wp.array(np.array(mesh.GetFaceVertexIndicesAttr().Get()))

            self.collider = wp.Mesh(vtx, idx)
            print(self.collider.points)
            



if __name__ == "__main__" :
    test = wcsph(True)
    test.use_dfsph = True
    test.sync_timers = True
    test.profile_substep_stats = True
    test.profile_substep_interval = 1
    test.enable_neighbor_count = False
    test.preparation()
    test.sub_step_num = 4
    test.sim_dt = 1.0 / 60
    pt_array = test.x
    pt_nei_count = test.nei_count
    # print(pt_nei_count)
    # print(wp.__version__)


    for i in range(6000) :
        with wp.ScopedTimer("frame", active=True):
            test.step()
        with wp.ScopedTimer("render", active=True):
            test.render()
            # test.render_test()
    # test.renderer.save()
    # print(pt_array)
