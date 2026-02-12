import numpy as np
import warp as wp
import warp.render
import json
from wcsph_kernel import *
import math
from pxr import Usd, UsdGeom, Vt, Sdf
from pathlib import Path

class sph_material:
    def __init__(self, 
                rho = 1000.0, # rest density
                stiffness = 50000.0,
                exponent = 7.0, 
                mu=0.005,
                tension=0.01) -> None:
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

        self.grid = wp.HashGrid(grid_cell_size, grid_cell_size, grid_cell_size)
        # self.grid.build(particle_q, self.smoothing_length)

class wcsph:
    def __init__(self, load_from_usd=False) -> None:
        self.verbose = False
        self.load_from_usd = load_from_usd
        self.sim_time = 0.0
        self.sim_dt = 0.005
        

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
            self.x = wp.empty(self.n, dtype=wp.vec3)


        self.particle_distance = self.particle_radius * 2.0
        self.smoothing_length = self.particle_distance * 1.35
        # self.particle_distance = self.smoothing_length 
        self.sph_model = sph_model(self.bound_size, self.smoothing_length)

        # fluid material
        self.sph_model.liquid_material.tension = 0.01
        self.sph_model.liquid_material.stiffness = 85000.0
        self.sph_model.liquid_material.mu = 3.0

        self.p_volume = 0.8 * (self.particle_distance ** 3)
        self.sub_step_num = 6
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
        voxel_size = self.smoothing_length *  0.5
        margin = self.smoothing_length * margin_scale

        self.world_min = wp.vec3(-self.bound_3d_size[0] - margin, -self.bound_3d_size[1] - margin, 0.0 - margin)
        self.world_max = wp.vec3(self.bound_3d_size[0] + margin, self.bound_3d_size[1] + margin, self.bound_3d_size[2] + margin)
        size = self.world_max - self.world_min

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

        self.target_voxel_size = self.smoothing_length *  0.5  # 可以根据需要调整这个系数

        # resolution_scale = 0.5
    
        
        # 创建体积密度网格
        # self.volume_density = wp.zeros(
        #     (self.volume_res_x, self.volume_res_y, self.volume_res_z), 
        #     dtype=float, 
        #     device="cuda:0"
        # )
        
        self.volume_threshold = 0.5  # 等值面阈值，可根据密度调整



        print(f"particle count : {self.n}")
        self.mass = wp.full(self.n, self.p_volume * self.sph_model.liquid_material.rho)
        self.gamma = wp.full(self.n, self.sph_model.liquid_material.tension)
        self.stiffness = wp.full(self.n, self.sph_model.liquid_material.stiffness)
        self.exponent = wp.full(self.n, self.sph_model.liquid_material.exponent)
        self.mu = wp.full(self.n, self.sph_model.liquid_material.mu)
        self.rho_0 = wp.full(self.n, self.sph_model.liquid_material.rho)





        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.nei_count = wp.zeros(self.n, dtype=wp.int32)
        self.pressure = wp.zeros(self.n, dtype=float)
        self.factor = wp.zeros(self.n, dtype=float)

        self.render_x = wp.empty(self.n, dtype=wp.vec3)

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

        self.renderer = wp.render.OpenGLRenderer(
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
    def compute_sim_dt(
            self,
            dt_min, 
            dt_max, 
        ) :

        eps = 1e-12
        a_max = max(np.linalg.norm(a_i) for a_i in self.a.numpy())

        c_s= math.sqrt((self.sph_model.liquid_material.stiffness / self.sph_model.liquid_material.rho))

        dt_force = 0.25 * self.smoothing_length / (a_max + eps)
        dt_sound = 0.4 * self.smoothing_length / (c_s * 1.05 + eps)
        dt = min(dt_force, dt_sound)
        dt = max(dt_min, min(dt_max, dt))
        dt = min(dt, self.sim_dt * 1.1)
        self.sim_dt = dt
        print(self.sim_dt)




    

    def sub_step(self) :
        # self.compute_sim_dt(self.sim_dt * 0.9, self.sim_dt * 1.1)
        with wp.ScopedTimer("grid build", active=False):
                    # build grid
            self.sph_model.grid.build(self.x, self.smoothing_length)
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
        wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.sim_dt])

        # # drift
        wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.sim_dt])
        # # ground collision
        wp.launch(kernel=apply_bounds, dim=self.n, inputs=[self.x, self.v ,self.bound_3d_size ,-0.1])
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

    def preparation(self) :
        with wp.ScopedTimer("preparation", active=True):
                    # build grid
            self.sph_model.grid.build(self.x, self.smoothing_length)

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

            wp.launch(
                kernel=compute_factor,
                dim=self.n,
                inputs=[
                    self.sph_model.grid.id,
                    self.x,
                    self.smoothing_length,
                    self.p_volume,
                    self.factor,
                ]
            )
        # print("----rho-----")
        # print(self.rho)
        # print("-----factor-----")
        # print(self.factor)
        # print("----nei_count----")
        # print(self.nei_count)



    def step(self) :
        # pass
        with wp.ScopedTimer("step", active=True):
            for _ in range(self.sub_step_num):
                with wp.ScopedTimer("sub_step", active=False):
                    self.sub_step()
                    # print(self.render_x)
                    self.sim_time += self.sim_dt

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
        with wp.ScopedTimer("render", active=False):
            # 在渲染之前更新 render_x，确保使用最新的位置
            wp.launch(
                kernel=to_real_world,
                dim=self.n,
                inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)])
            
            self.renderer.begin_frame(self.sim_time)
            
            # render_x_np = self.render_x.numpy()

            density_buffer = wp.zeros(
                (
                    self.volume_res_x, 
                    self.volume_res_y, 
                    self.volume_res_z
                ), 
                dtype=float, 
                device="cuda:0"
            )
            
            # 创建调试信息数组
            debug_info = wp.zeros(3, dtype=wp.vec3, device="cuda:0")

            wp.launch(
                kernel=rasterize_particles_to_nvdb_volume,
                dim=self.n,
                inputs=[
                    self.x,
                    self.volume.id,
                    density_buffer,  # 添加 density_buffer
                    self.smoothing_length,  # 注意单位转换
                    self.p_volume,
                    self.volume_res_x,  # 添加分辨率参数
                    self.volume_res_y,
                    self.volume_res_z,
                    debug_info  # 添加调试信息
                ],
                device="cuda:0"
            )

            # wp.launch(
            #     kernel=write_sphere,
            #     dim=(self.volume_res_x, self.volume_res_y, self.volume_res_z),
            #     inputs=[
            #         self.volume.id,
            #         wp.vec3(0.0, 0.0, 0.0),
            #         self.smoothing_length,
            #     ],
            # )

            # wp.launch(
            #     kernel=copy_density_to_volume,
            #     dim = (self.volume_res_x, self.volume_res_y, self.volume_res_z),
            #     inputs=[
            #         density_buffer,
            #         self.volume.id,
            #         self.volume_res_x,
            #         self.volume_res_y,
            #         self.volume_res_z,
            #     ],
            #     device="cuda:0",
            # )
            
            # 打印调试信息
            # debug_np = debug_info.numpy()
            # print("=== 调试信息 ===")
            # print(f"第一个粒子的位置: {debug_np[1]}")
            # print(f"第一个粒子的体素索引: {debug_np[0]}")
            # print(f"体素大小向量: {debug_np[2]}")
            # print(f"体素大小标量: {np.linalg.norm(debug_np[2])}")
            # print(f"粒子数量: {self.n}")
            # print(f"Volume 分辨率: ({self.volume_res_x}, {self.volume_res_y}, {self.volume_res_z})")
            # print(f"smoothing_length: {self.smoothing_length * 0.01}")
            # print(f"p_volume: {self.p_volume}")
            
            # # 检查 density_buffer 是否有非零值
            # density_np = density_buffer.numpy()
            # non_zero_count = np.count_nonzero(density_np)
            # print(f"density_buffer 非零值数量: {non_zero_count}")
            # if non_zero_count > 0:
            #     print(f"density_buffer 最大值: {np.max(density_np)}")
            #     print(f"density_buffer 最小值（非零）: {np.min(density_np[density_np > 0])}")
            # else:
            #     print("警告: density_buffer 全为 0！")
            
            # # 检查粒子位置是否在 Volume 范围内
            # if self.n > 0:
            #     first_particle_pos = debug_np[1]
            #     voxel_idx = debug_np[0]
            #     print(f"第一个粒子体素索引范围检查:")
            #     print(f"  vx: {voxel_idx[0]} (范围: 0-{self.volume_res_x-1})")
            #     print(f"  vy: {voxel_idx[1]} (范围: 0-{self.volume_res_y-1})")
            #     print(f"  vz: {voxel_idx[2]} (范围: 0-{self.volume_res_z-1})")
            
            # print(density_buffer.numpy())
            # breakpoint()


            # wp.Volume.save_to_nvdb(self.volume, "volume.nvdb")

            # 使用 kernel 将粒子栅格化到体积网格
            # wp.launch(
            #     kernel=rasterize_particles_to_nvdb_volume,
            #     dim=self.n,
            #     inputs=[

            #     ],
            #     device="cuda:0"
            # )

            # wp.Volume.save_to_numpy(self.volume_density, "volume_density.npy")

            
            # 初始化或更新 MarchingCubes 上下文
            if self.mc is None:
                self.mc = wp.MarchingCubes(
                    nx=self.volume_res_x,
                    ny=self.volume_res_y,
                    nz=self.volume_res_z,
                    domain_bounds_lower_corner=wp.vec3(self.world_min[0], self.world_min[1], self.world_min[2]),
                    domain_bounds_upper_corner=wp.vec3(self.world_max[0], self.world_max[1], self.world_max[2]),
                    device="cuda:0"
                )
            else:
                # 更新边界（分辨率在初始化时已确定，通常不需要改变）
                self.mc.domain_bounds_lower_corner = wp.vec3(self.world_min[0], self.world_min[1], self.world_min[2])
                self.mc.domain_bounds_upper_corner = wp.vec3(self.world_max[0], self.world_max[1], self.world_max[2])

            # 5. 使用 marching cubes 提取表面 mesh
            # 将 volume_density 转换为 float32 类型（marching cubes 需要）
            # volume_density_f32 = wp.array(
            #     self.volume_density.numpy().astype(np.float32),
            #     dtype=wp.float32,
            #     device="cuda:0"
            # )
            # breakpoint()
            
            self.mc.surface(
                field=density_buffer,
                threshold=self.volume_threshold
            )
            
            # 6. 渲染 mesh（如果有顶点）
            if self.mc.verts is not None and self.mc.verts.shape[0] > 0:
                # verts_np = self.mc.verts.numpy()
                indices_np = self.mc.indices.numpy()
                
                if self.mc.verts.shape[0] > 0 and self.mc.indices.shape[0] > 0:
                    real_verts = wp.empty(self.mc.verts.shape[0], dtype=wp.vec3)
                    wp.launch(
                            kernel=to_real_world,
                            dim=self.mc.verts.shape[0],
                            inputs=[self.mc.verts, real_verts, 0.01, wp.vec3(0.0, 0.0, 0.0)]
                        )
                    verts_np = real_verts.numpy()
                    self.renderer.render_mesh(
                        name="fluid_mesh",
                        points=verts_np,
                        indices=indices_np,
                        update_topology=True,

                    )
            
            # 可选：同时渲染粒子点（用于调试）
            # self.renderer.render_points(
            #     points=self.render_x.numpy(), 
            #     radius=self.particle_radius * 0.01, 
            #     name="points", 
            #     colors=(0.2, 0.3, 0.7)
            # )

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
            self.particle_radius = points.GetWidthsAttr().Get()[0] * 0.5 * 100.0
            self.x = wp.array(points_np, dtype=wp.vec3)
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
    pt_array = test.x
    pt_nei_count = test.nei_count
    # print(pt_nei_count)
    # print(wp.__version__)


    for i in range(6000) :
        with wp.ScopedTimer("frame", active=True):
            test.step()
            test.render()
            # test.render_test()
    # test.renderer.save()
    # print(pt_array)