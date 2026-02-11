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
        self.sph_model.liquid_material.stiffness = 50000.0
        self.sph_model.liquid_material.mu = 0.2

        self.p_volume = 0.8 * (self.particle_distance ** 3)
        self.sub_step_num = 6
        self.gravity = -10.0

        self.camera_pos = (0.0, 8.5, 5.0)
        # self.camera_pos = (0.0, 0.0, 0.175)
        self.camera_pos = (-2.75, 0.5, 0.0)



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
            # print("self.x (前几个值):", self.x.numpy()[:5])
            # print("self.render_x (前几个值):", self.render_x.numpy()[:5])
            # self.renderer.render_points(
            #     points=self.render_x.numpy(), radius=0.0003, name="points", colors=(0.2, 0.3, 0.7)
            # )
            # breakpoint()
            
            self.renderer.render_points(
                points=self.render_x.numpy(), 
                radius=self.particle_radius * 0.01, 
                name="points", 
                colors=(0.2, 0.3, 0.7)
            )
            # self.renderer.render_mesh(
            #     name = "container",
            #     points=self.collider.points.numpy(),
            #     indices=self.collider.indices.numpy(),
            # )
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
            test.render()
            test.step()
    # test.renderer.save()
    # print(pt_array)