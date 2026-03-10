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
        self.force_grid = wp.HashGrid(grid_cell_size, grid_cell_size, grid_cell_size)
        # self.grid.build(particle_q, self.smoothing_length)

class wcsph:
    def __init__(self, load_from_usd=False) -> None:
        self.step_counter = 0
        self.verbose = False
        self.load_from_usd = load_from_usd
        self.sim_time = 0.0
        self.sim_dt = 0.0005

        self.force_weight = 0.8


        self.drive_start_time = 0.5
        self.drive_during_time = 5.0
        self.drive_end_time = self.drive_start_time + self.drive_during_time
        self.ghost_particle_drive_speed = 0.0016
        self.ghost_particle_drive_dir = wp.vec3(0.0, 0.0, -1.0)

        self.dropper_mesh: wp.Mesh = None
        self.container_mesh: wp.Mesh = None

        

        # self.particle_radius = 0.0125
        self.particle_radius : float = 0.0
        self.bound_size = 100.0
        self.bound_3d_size = wp.vec3(self.bound_size , self.bound_size * 0.5, self.bound_size)
        self.collider: wp.Mesh = None
        self.ghost_density = 1.0
        self.ghost_wall_density = 1000.0
        self.ghost_mass = 0.0

        if self.load_from_usd:
            current_dir = Path(__file__).parent
            # self.load_particles_from_usd("./temp/particle_test.usd", wp.vec3(0.0, 0.0, 0.0))
            self.load_particles_from_usd((current_dir / "temp" / "fluid_suction_scene.usd").as_posix(), wp.vec3(0.0, 0.0, 0.0))
        else:
            self.n = int(
                self.bound_size * self.bound_size * self.bound_size / (self.smoothing_length**3)
            )  # number particles (small box in corner)
            self.x = wp.empty(self.n, dtype=wp.vec3)


        self.particle_distance = self.particle_radius * 2.0
        self.smoothing_length = self.particle_distance * 1.8
        # self.particle_distance = self.smoothing_length 
        self.sph_model = sph_model(120.0, self.smoothing_length)

        # fluid material
        self.sph_model.liquid_material.tension = 0.11
        self.sph_model.liquid_material.stiffness = 50000.0
        self.sph_model.liquid_material.mu = 0.25

        self.p_volume = 0.8 * (self.particle_distance ** 3)
        self.sub_step_num = 20
        self.gravity = -10.0

        self.camera_pos = (-0.25, 0.095, 0.0)
        # self.camera_pos = (0.0, 0.0, 0.175)

        self.suction_start = wp.vec3(0.0, 0.0, 0.22)
        self.suction_end=  wp.vec3(0.0, 0.0, 0.81)



        print(f"particle count : {self.n}")
        self.mass = wp.full(self.n, self.p_volume * self.sph_model.liquid_material.rho)
        self.gamma = wp.full(self.n, self.sph_model.liquid_material.tension)
        self.stiffness = wp.full(self.n, self.sph_model.liquid_material.stiffness)
        self.exponent = wp.full(self.n, self.sph_model.liquid_material.exponent)
        self.mu = wp.full(self.n, self.sph_model.liquid_material.mu)
        self.rho_0 = wp.full(self.n, self.sph_model.liquid_material.rho)
        self.suction_test = wp.zeros(self.n, dtype=wp.vec3)
        self.suction_direction = wp.zeros(self.n, dtype=wp.vec3)
        self.suction_distance = wp.zeros(self.n, dtype=float)
        self.weight_test = wp.zeros(self.n, dtype=float)





        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.nei_count = wp.zeros(self.n, dtype=wp.int32)
        self.bound_touched = wp.zeros(self.n, dtype=wp.int32)
        self.bound_penetration = wp.zeros(self.n, dtype=float)
        self.pressure = wp.zeros(self.n, dtype=float)
        self.factor = wp.zeros(self.n, dtype=float)

        # self.force_field = wp.zeros(self.n, dtype=wp.vec3)


        self.render_x = wp.empty(self.n, dtype=wp.vec3)
        self.sph_model.build_hash_grid()


        # self.renderer = wp.render.OpenGLRenderer(
        #     up_axis="Z",
        #     camera_pos=self.camera_pos,
        #     near_plane=0.001,
        #     far_plane = 1000.0,
        #     draw_axis=False,
        #     camera_up=(0.0, 1.0, 0.0),
        #     camera_front=(1.0, 0.0, 0.0),
        # )
        
        self.renderer = wp.render.UsdRenderer(
            up_axis="Z", 
            stage="wcsph_test.usd",
        )

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
            self.sph_model.force_grid.build(self.force_field_p, self.smoothing_length)

            
            # wp.launch(
            #     kernel=get_neighbor,
            #     dim=self.n,
            #     inputs=[
            #         self.sph_model.grid.id,
            #         self.x,
            #         self.smoothing_length,
            #         self.nei_count,
            #     ]
            # )
        wp.launch(
            kernel=drive_collider,
            dim=self.combine_vtx.shape[0],
            inputs=[
                self.combine_vtx,
                self.combine_vel,
                self.collider_mask,
                1,
                self.ghost_particle_drive_dir,
                self.ghost_particle_drive_speed,
                self.sim_time,
                self.drive_start_time,
                self.drive_end_time,
            ]
        )
        self.collider.refit()
        # self.collider = wp.Mesh(
        #     self.combine_vtx,
        #     self.combine_idx,
        # )
        wp.launch(
            kernel=drive_ghost_particle,
            dim=self.force_field_p_count,
            inputs=[
                self.force_field_p,
                self.ghost_mask,
                self.ghost_particle_drive_dir,
                self.ghost_particle_drive_speed,
                self.sim_time,
                self.drive_start_time,
                self.drive_end_time,
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
                    self.ghost_mask,
                    self.rho_0,
                    self.ghost_density,
                    self.ghost_wall_density,
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
                    self.ghost_mask,
                    self.ghost_density,
                    self.ghost_mass,
                    self.ghost_wall_density,
                    self.force_field_p,
                    self.force_field,
                    self.force_weight,
                    self.p_volume,
                    self.mass,
                    self.gamma,
                    self.mu,
                    self.gravity,
                    self.smoothing_length,
                    self.sph_model.grid.id,
                    self.sph_model.force_grid.id,
                ]
            )
        # print(f"weight: {[i for i in self.weight_test.numpy().tolist() if i > 0.1]}")
        # print(f"distance: {[i for i in self.suction_distance.numpy().tolist() if i < 10.0]}")
        # print(f"smoothing_length: {self.smoothing_length}")
        # # kick
        wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.sim_dt])

        # # drift
        wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.sim_dt])
        # # ground collision
        # wp.launch(
        #     kernel=apply_bounds,
        #     dim=self.n,
        #     inputs=[self.x, self.v, self.bound_3d_size, -0.1, self.bound_touched, self.bound_penetration],
        # )
        # wp.launch(
        #     kernel=update_collider_with_tri_mesh, 
        #     dim=self.n, 
        #     inputs=[
        #         self.x, 
        #         self.v, 
        #         self.collider.id, 
        #         self.particle_radius, 
        #         1.0, 0.1],
        # )
        wp.launch(
            kernel=to_real_world,
            dim=self.n,
            inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)])
        # wp.launch(
        #     kernel=to_real_world,
        #     dim=self.combine_vtx.shape[0],
        #     inputs=[self.combine_vtx, self.collider_render_vtx, 0.01, wp.vec3(0.0, 0.0, 0.0)])

    # def preparation(self) :
    #     with wp.ScopedTimer("preparation", active=True):
    #                 # build grid
    #         self.sph_model.grid.build(self.x, self.smoothing_length)

    #         wp.launch(
    #             kernel=get_neighbor,
    #             dim=self.n,
    #             inputs=[
    #                 self.sph_model.grid.id,
    #                 self.x,
    #                 self.smoothing_length,
    #                 self.nei_count,
    #             ]
    #         )

    #         wp.launch(
    #             kernel=rho,
    #             dim=self.n,
    #             inputs=[
    #                 self.rho,
    #                 self.exponent,
    #                 self.stiffness,
    #                 self.x,
    #                 self.rho_0,
    #                 self.ghost_mask,
    #                 self.ghost_density,
    #                 self.ghost_wall_density,
    #                 self.p_volume,
    #                 self.smoothing_length,
    #                 self.sph_model.grid.id,
    #                 self.pressure,
    #             ]
    #         )

    #         wp.launch(
    #             kernel=compute_factor,
    #             dim=self.n,
    #             inputs=[
    #                 self.sph_model.grid.id,
    #                 self.x,
    #                 self.smoothing_length,
    #                 self.p_volume,
    #                 self.factor,
    #             ]
    #         )

    def step(self) :
        # pass
        if self.step_counter > 285:
            self.ghost_particle_drive_speed = 0.0
        with wp.ScopedTimer("step", active=True):
            for _ in range(self.sub_step_num):
                # pass
                with wp.ScopedTimer("sub_step", active=False):
                    self.sub_step()
                    # print(self.render_x)
                    self.sim_time += self.sim_dt

            self.step_counter += 1  
    def render(self):
        # if self.renderer is None:
        #     return

        with wp.ScopedTimer("render", active=False):
            # 在渲染之前更新 render_x，确保使用最新的位置
            wp.launch(
                kernel=to_real_world,
                dim=self.n,
                inputs=[self.x, self.render_x, 0.01, wp.vec3(0.0, 0.0, 0.0)])
            
            self.renderer.begin_frame(self.sim_time)

            # 使用 mask 过滤出 fluid particles (mask == 0)
            render_x_np = self.render_x.numpy()
            ghost_mask_np = self.ghost_mask.numpy()
            
            # 只选择 fluid particles (mask == 0)
            fluid_indices = ghost_mask_np == 0
            ghost_indices = ghost_mask_np == 1
            ghost_wall_indices = ghost_mask_np == 2
            # force_indices = ghost_mask_np == 3

            collider_mask_np = self.collider_mask.numpy()
            collider_container_indices = collider_mask_np == 0
            collider_dropper_indices = collider_mask_np == 1

            fluid_particles = render_x_np[fluid_indices]
            ghost_particles = render_x_np[ghost_indices]
            ghost_wall_particles = render_x_np[ghost_wall_indices]
            # force_particles = render_x_np[force_indices]

            # collider_container_particles = self.collider_render_vtx.numpy()[collider_container_indices]
            # collider_dropper_particles = self.collider_render_vtx.numpy()[collider_dropper_indices]



            
            self.renderer.render_points(
                points=fluid_particles, 
                radius=self.particle_radius * 0.01, 
                name="points", 
                colors=(0.2, 0.3, 0.7)
            )
            self.renderer.render_mesh(
                points = self.collider_render_vtx.numpy(),
                indices = self.collider.indices.numpy(),
                name = "collider",
                colors = (0.33, 0.3, 0.7)
            )
            self.renderer.render_points(
                points=ghost_particles, 
                radius=self.particle_radius * 0.0025, 
                name="ghost_particle", 
                colors=(0.7, 0.7, 0.7)
            )

            self.renderer.render_points(
                points=ghost_wall_particles, 
                radius=self.particle_radius * 0.0025, 
                name="ghost_wall", 
                colors=(0.9, 0.3, 0.7)
            )
            # breakpoint()
            self.renderer.render_points(
                points=self.force_field_p.numpy() * 0.01, 
                radius=self.particle_radius * 0.0025, 
                name="force", 
                colors=(0.5, 0.8, 0.9)
            )

            self.renderer.end_frame()

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
        fluid = stage.GetPrimAtPath("/Fluid/FluidParticles")
        dropper = stage.GetPrimAtPath("/Fluid/Dropper")
        container = stage.GetPrimAtPath("/Fluid/Container")
        ghost_particle = stage.GetPrimAtPath("/Fluid/Dropper/Ghost_Particles")
        ghost_wall_particle = stage.GetPrimAtPath("/Fluid/Container/Ghost_Wall_Particles")
        # force_particle = stage.GetPrimAtPath("/Fluid/a_pos")

        
        # 收集所有粒子位置
        all_particles = []
        ghost_mask_list = []
        # force_field_list = None
        # 加载 fluid particles (标记为 0)
        if fluid.IsValid():
            points = UsdGeom.Points(fluid)
            points_np = np.array(points.GetPointsAttr().Get())
            self.particle_radius = points.GetWidthsAttr().Get()[0] * 0.5 * 100.0
            all_particles.append(points_np)
            # fluid particles 标记为 0
            ghost_mask_list.append(np.zeros(len(points_np), dtype=np.int32))
        
        # 加载 ghost particles (标记为 1)
        if ghost_particle.IsValid():
            ghost_points = UsdGeom.Points(ghost_particle)
            force_field_list = np.array(ghost_particle.GetAttribute("dir").Get())
            force_points = np.array(ghost_points.GetPointsAttr().Get())
            self.force_field = wp.array(force_field_list, dtype=wp.vec3)
            self.force_field_p = wp.array(force_points, dtype=wp.vec3)
            self.force_field_p_count = len(force_points)
            # ghost_points_np = np.array(ghost_points.GetPointsAttr().Get())
            # all_particles.append(ghost_points_np)
            # # ghost particles 标记为 1
            # ghost_mask_list.append(np.ones(len(ghost_points_np), dtype=np.int32))
        # if dropper.IsValid():
        #     ghost_points = UsdGeom.Points(ghost_particle)
        #     force_field_list = np.array(ghost_particle.GetAttribute("dir").Get())
        #     force_points = np.array(ghost_points.GetPointsAttr().Get())

        if len(all_particles) > 0:
            combined_particles = np.vstack(all_particles)
            combined_mask = np.concatenate(ghost_mask_list)
            
            # 创建 warp 数组
            self.x = wp.array(combined_particles, dtype=wp.vec3)
            # self.force_field = wp.array(force_field_list, dtype=wp.vec3)
            self.ghost_mask = wp.array(combined_mask, dtype=wp.int32)
            self.n = len(combined_particles)

            
            # 应用坐标转换（如果需要）
            wp.launch(
                kernel=to_micro_world, 
                dim=self.n, 
                inputs=[self.x, 100.0, offset]
            )
            wp.launch(
                kernel=to_micro_world,
                dim=self.force_field_p_count,
                inputs=[self.force_field_p, 100.0, offset]
            )


        # if fluid.IsValid():
        #     points = UsdGeom.Points(fluid)
        #     points_np = np.array(points.GetPointsAttr().Get())
        #     self.particle_radius = points.GetWidthsAttr().Get()[0] * 0.5 * 100.0
        #     self.x = wp.array(points_np, dtype=wp.vec3)
        #     self.n = len(points_np)
        #     wp.launch(
        #         kernel=to_micro_world, 
        #         dim=self.n, 
        #         inputs=[self.x, 100.0, offset])
        #     print(f"粒子数据已加载到 {filename}，共 {self.n} 个粒子")
        # if ghost_particle.IsValid():
        #     points = UsdGeom.Points(ghost_particle)
        #     points_np = np.array(points.GetPointsAttr().Get())
        #     self.ghost_particle_radius = points.GetWidthsAttr().Get()[0] * 0.5 * 100.0
        #     self.ghost_particle_x = wp.array(points_np, dtype=wp.vec3)
        #     self.ghost_particle_n = len(points_np)
        #     wp.launch(
        #         kernel=to_micro_world, 
        #         dim=self.ghost_particle_n, 
        #         inputs=[self.ghost_particle_x, 100.0, offset])
        #     print(f"粒子数据已加载到 ghost_particles，共 {self.ghost_particle_n} 个粒子")
        # container = stage.GetPrimAtPath("/ParticleTest/Container")
        prim_list = [container, dropper]
        self.collect_collision(prim_list, offset)
        # breakpoint()
        # if container.IsValid() :
        #     mesh = UsdGeom.Mesh(container)
        #     np_vtx = np.array(mesh.GetPointsAttr().Get())
        #     vtx = wp.array(np_vtx, dtype=wp.vec3)
        #     wp.launch(
        #         kernel=to_micro_world, 
        #         dim=len(np_vtx), 
        #         inputs=[vtx, 100.0, offset])
        #     idx = wp.array(np.array(mesh.GetFaceVertexIndicesAttr().Get()))

        #     self.collider = wp.Mesh(vtx, idx)
        #     print(self.collider.points)

        # if dropper.IsValid() :
        #     mesh = UsdGeom.Mesh(dropper)
        #     np_vtx = np.array(mesh.GetPointsAttr().Get())
        #     self.dropper_vtx = wp.array(np_vtx, dtype=wp.vec3)
        #     self.dropper_render_x = wp.empty(len(np_vtx), dtype=wp.vec3)
        #     wp.launch(
        #         kernel=to_micro_world, 
        #         dim=len(np_vtx), 
        #         inputs=[self.dropper_vtx, 100.0, offset])
        #     idx = wp.array(np.array(mesh.GetFaceVertexIndicesAttr().Get()))

        #     self.dropper_mesh = wp.Mesh(self.dropper_vtx, idx)
        #     # print(self.collider.points)

    def collect_collision(self, prim_list, offset: wp.vec3) :
        idx_offset = 0
        all_vtx = [] 
        all_idx = []
        # combined_particles = np.vstack(all_particles)
        # combined_mask = np.concatenate(ghost_mask_list)
        collider_mask = []
        collider_idx = 0

        for prim in prim_list :
            if prim.IsValid() :
                mesh = UsdGeom.Mesh(prim)
                np_vtx = np.array(mesh.GetPointsAttr().Get())
                all_vtx.append(np_vtx)
                collider_mask.append(np.full(len(np_vtx), collider_idx, dtype=np.int32))

                # vtx = wp.array(np_vtx, dtype=wp.vec3)
                # wp.launch(
                #     kernel=to_micro_world, 
                #     dim=len(np_vtx), 
                #     inputs=[vtx, 100.0, offset])
                idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
                idx += idx_offset
                all_idx.append(idx)

                idx_offset += len(np_vtx)
                collider_idx += 1
                # breakpoint()

                # idx = wp.array(idx, dtype=wp.int32)
        if len(np_vtx) > 0 :
            combine_vtx = np.vstack(all_vtx)
            combine_idx = np.concatenate(all_idx)
            combine_mask = np.concatenate(collider_mask)


            self.combine_vtx = wp.array(combine_vtx, dtype=wp.vec3)
            self.combine_idx = wp.array(combine_idx, dtype=wp.int32)
            self.combine_vel = wp.empty(len(combine_vtx), dtype=wp.vec3)


            self.collider_render_vtx = wp.empty(len(combine_vtx), dtype=wp.vec3)
            wp.launch(
                kernel=to_micro_world, 
                dim=self.combine_vtx.shape[0], 
                inputs=[self.combine_vtx, 100.0, offset]
            )

            self.collider = wp.Mesh(
                # wp.array(self.combine_vtx, dtype=wp.vec3),
                # wp.array(self.combine_idx, dtype=wp.int32),
                self.combine_vtx,
                self.combine_idx,
                self.combine_vel,
            )
            self.collider_mask = wp.array(combine_mask, dtype=wp.int32)
            # wp.launch(
            #     kernel=to_micro_world,
            #     dim=self.combine_vtx.shape[0],
            #     inputs=[self.combine_vtx, self.collider_render_vtx, 0.01, wp.vec3(0.0, 0.0, 0.0)])



if __name__ == "__main__" :
    test = wcsph(True)
    pt_array = test.x
    pt_nei_count = test.nei_count
    # print(pt_nei_count)
    # print(wp.__version__)


    for i in range(900) :
        with wp.ScopedTimer("frame", active=True):
            test.render()
            test.step()
            print(test.step_counter)
    test.renderer.save()
    # print(pt_array)
