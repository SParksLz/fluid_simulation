from dataclasses import dataclass

import warp as wp

import newton
from newton import ParticleFlags
from newton.solvers import SolverBase


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def get_cubic(r_norm: float, radius: float):
    res = float(0.0)
    h = radius
    k = 8.0 / (wp.pi * cube(h))
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = square(q)
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            res = 2.0 * k * cube(1.0 - q)
    return res


@wp.func
def get_cubic_derivative(r: wp.vec3, smoothing_length: float):
    res = wp.vec3(0.0)
    r_norm = wp.length(r)
    h = smoothing_length
    k = 8.0 / (wp.pi * cube(h))
    q = r_norm / h
    if r_norm > 1.0e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
        else:
            q_term = 1.0 - q
            res = -6.0 * k * q_term * q_term * grad_q
    return res


@wp.func
def is_dynamic_particle(particle_flags: wp.int32, particle_mask: int):
    return (particle_flags & ParticleFlags.ACTIVE) != 0 and particle_mask == 0


@wp.func
def is_neighbor_particle_active(particle_flags: wp.int32):
    return (particle_flags & ParticleFlags.ACTIVE) != 0


@wp.func
def cal_acc_with_non_pressure(
    a: wp.vec3,
    rho_nei: float,
    vel: wp.vec3,
    vel_nei: wp.vec3,
    mass: float,
    mass_nei: float,
    mu: float,
    gamma: float,
    d_current_nei: wp.vec3,
    e_dist: float,
    distance: float,
    smoothing_length: float,
):
    a -= gamma / mass * mass_nei * d_current_nei * get_cubic(e_dist, smoothing_length)
    v_current_nei = wp.dot(vel - vel_nei, d_current_nei)
    d = 10.0
    a += (
        d
        * mu
        * (mass_nei / wp.max(rho_nei, 1.0e-6))
        * v_current_nei
        / (distance * distance + 0.01 * (smoothing_length * smoothing_length))
        * get_cubic_derivative(d_current_nei, smoothing_length)
    )
    return a


@wp.func
def cal_acc_with_pressure(
    a: wp.vec3,
    d_current_nei: wp.vec3,
    pressure: float,
    pressure_nei: float,
    rho: float,
    rho_nei: float,
    mass_nei: float,
    smoothing_length: float,
):
    dp_i = pressure / wp.max(rho * rho, 1.0e-6)
    dp_nei = pressure_nei / wp.max(rho_nei * rho_nei, 1.0e-6)
    a += -mass_nei * (dp_i + dp_nei) * get_cubic_derivative(d_current_nei, smoothing_length)
    return a


@wp.kernel
def compute_particle_volume(
    particle_mass: wp.array(dtype=float),
    rest_density: wp.array(dtype=float),
    particle_volume: wp.array(dtype=float),
):
    tid = wp.tid()
    particle_volume[tid] = particle_mass[tid] / wp.max(rest_density[tid], 1.0e-6)


@wp.kernel
def compute_density_pressure(
    particle_flags: wp.array(dtype=wp.int32),
    particle_x: wp.array(dtype=wp.vec3),
    rest_density: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    exponent: wp.array(dtype=float),
    particle_volume: wp.array(dtype=float),
    smoothing_length: float,
    grid_id: wp.uint64,
    rho: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_neighbor_particle_active(particle_flags[i]):
        rho[i] = 0.0
        pressure[i] = 0.0
        return

    x = particle_x[i]
    rho_temp = float(0.0)
    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)
    for index in neighbors:
        if not is_neighbor_particle_active(particle_flags[index]):
            continue
        distance = x - particle_x[index]
        mass_nei = rest_density[index] * particle_volume[index]
        rho_temp += mass_nei * get_cubic(wp.length(distance), smoothing_length)

    rho_i = rho_temp
    rho[i] = rho_i
    pressure[i] = stiffness[i] * (wp.pow(rho_i / wp.max(rest_density[i], 1.0e-6), exponent[i]) - 1.0)


@wp.kernel
def acceleration_wcsph(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_pressure: wp.array(dtype=float),
    particle_radius: float,
    mass: wp.array(dtype=float),
    surface_tension: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3),
    particle_world: wp.array(dtype=wp.int32),
    smoothing_length: float,
    grid_id: wp.uint64,
    particle_a: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_dynamic_particle(particle_flags[i], particle_mask[i]):
        particle_a[i] = wp.vec3(0.0)
        return

    x = particle_x[i]
    rho = particle_rho[i]
    pressure = particle_pressure[i]
    acc = wp.vec3(0.0)
    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)
    for index in neighbors:
        if index == i or not is_neighbor_particle_active(particle_flags[index]):
            continue

        direction = x - particle_x[index]
        distance = wp.length(direction)
        mass_nei = mass[index]
        rho_nei = particle_rho[index]
        pressure_nei = particle_pressure[index]

        acc = cal_acc_with_non_pressure(
            acc,
            rho_nei,
            particle_v[i],
            particle_v[index],
            mass[i],
            mass_nei,
            viscosity[i],
            surface_tension[i],
            direction,
            wp.max(distance, particle_radius),
            distance,
            smoothing_length,
        )
        acc = cal_acc_with_pressure(
            acc,
            direction,
            pressure,
            pressure_nei,
            rho,
            rho_nei,
            mass_nei,
            smoothing_length,
        )

    world_idx = particle_world[i]
    world_g = gravity[wp.max(world_idx, 0)]
    particle_a[i] = acc + world_g


@wp.kernel
def integrate_velocity(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    v_in: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    dt: float,
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        v_out[tid] = v_in[tid]
        return
    v_out[tid] = v_in[tid] + dt * a[tid]


@wp.kernel
def integrate_positions(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_in: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
    dt: float,
    x_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        x_out[tid] = x_in[tid]
        return
    x_out[tid] = x_in[tid] + v_out[tid] * dt


@wp.kernel
def apply_bounds(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    size: wp.vec3,
    damping_coef: float,
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        return

    x = particle_x[tid]
    v = particle_v[tid]

    if x[0] < -size[0]:
        x = wp.vec3(-size[0], x[1], x[2])
        if v[0] < 0.0:
            v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[0] > size[0]:
        x = wp.vec3(size[0], x[1], x[2])
        if v[0] > 0.0:
            v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[1] < -size[1]:
        x = wp.vec3(x[0], -size[1], x[2])
        if v[1] < 0.0:
            v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[1] > size[1]:
        x = wp.vec3(x[0], size[1], x[2])
        if v[1] > 0.0:
            v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        if v[2] < 0.0:
            v = wp.vec3(v[0], v[1], v[2] * damping_coef)
    if x[2] > size[2] * 2.0:
        x = wp.vec3(x[0], x[1], size[2] * 2.0)
        if v[2] > 0.0:
            v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    particle_x[tid] = x
    particle_v[tid] = v


class SolverWCSPH(SolverBase):
    @dataclass
    class SphConfig:
        particle_radius: float = 0.1
        particle_length: float = 0.2
        smoothing_length_coff: float = 1.35
        bound_width: float = 10.0
        bound_height: float = 10.0
        bound_length: float = 10.0
        boundary_damping: float = -0.1

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_mask",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=int,
                default=0,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="rest_density",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1000.0,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="surface_tension",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.01,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="viscosity",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.005,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="stiffness",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=85000.0,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="exponent",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=7.0,
                namespace="sph",
            )
        )
        for name in ("rho", "pressure"):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    assignment=newton.Model.AttributeAssignment.STATE,
                    dtype=wp.float32,
                    default=0.0,
                    namespace="sph",
                )
            )

    def __init__(self, model: newton.Model, config: SphConfig):
        super().__init__(model=model)
        self.config = config

        if model.particle_count > 1:
            if model.particle_grid is None:
                model.particle_grid = wp.HashGrid(128, 128, 128)
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        self._particle_accel = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self._particle_volume = wp.empty(model.particle_count, dtype=wp.float32, device=model.device)
        self._bounds = wp.vec3(config.bound_width, config.bound_height, config.bound_length)

    @property
    def smoothing_length(self) -> float:
        return self.config.particle_length * self.config.smoothing_length_coff

    def _copy_passthrough_state(self, state_in: newton.State, state_out: newton.State) -> None:
        if state_in.body_count:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
            wp.copy(state_out.body_f, state_in.body_f)
        if state_in.joint_q is not None and state_out.joint_q is not None:
            wp.copy(state_out.joint_q, state_in.joint_q)
        if state_in.joint_qd is not None and state_out.joint_qd is not None:
            wp.copy(state_out.joint_qd, state_in.joint_qd)

    def _update_particle_volume(self) -> None:
        wp.launch(
            kernel=compute_particle_volume,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, self.model.sph.rest_density],
            outputs=[self._particle_volume],
            device=self.model.device,
        )

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        del control, contacts

        self._copy_passthrough_state(state_in, state_out)

        if self.model.particle_count == 0:
            return

        if self.model.particle_grid is None:
            raise RuntimeError("SolverWCSPH requires model.particle_grid to be available for particle neighborhoods.")

        model = self.model
        smoothing_length = float(self.smoothing_length)
        particle_radius = float(
            self.config.particle_radius if self.config.particle_radius > 0.0 else model.particle_max_radius
        )

        self._update_particle_volume()
        model.particle_grid.build(state_in.particle_q, radius=smoothing_length)

        wp.launch(
            kernel=compute_density_pressure,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                state_in.particle_q,
                model.sph.rest_density,
                model.sph.stiffness,
                model.sph.exponent,
                self._particle_volume,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[state_out.sph.rho, state_out.sph.pressure],
            device=model.device,
        )
        wp.launch(
            kernel=acceleration_wcsph,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.sph.rho,
                state_out.sph.pressure,
                particle_radius,
                model.particle_mass,
                model.sph.surface_tension,
                model.sph.viscosity,
                model.gravity,
                model.particle_world,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[self._particle_accel],
            device=model.device,
        )
        wp.launch(
            kernel=integrate_velocity,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_qd,
                self._particle_accel,
                dt,
            ],
            outputs=[state_out.particle_qd],
            device=model.device,
        )
        wp.launch(
            kernel=integrate_positions,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_q,
                state_out.particle_qd,
                dt,
            ],
            outputs=[state_out.particle_q],
            device=model.device,
        )
        wp.launch(
            kernel=apply_bounds,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_out.particle_q,
                state_out.particle_qd,
                self._bounds,
                float(self.config.boundary_damping),
            ],
            device=model.device,
        )

    def update_contacts(self, contacts: newton.Contacts) -> None:
        del contacts
