# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
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


@wp.kernel
def compute_particle_volume(
    particle_mass: wp.array(dtype=float),
    rest_density: wp.array(dtype=float),
    particle_volume: wp.array(dtype=float),
):
    tid = wp.tid()
    particle_volume[tid] = particle_mass[tid] / wp.max(rest_density[tid], 1.0e-6)


@wp.kernel
def compute_density(
    particle_flags: wp.array(dtype=wp.int32),
    particle_x: wp.array(dtype=wp.vec3),
    rest_density: wp.array(dtype=float),
    particle_volume: wp.array(dtype=float),
    smoothing_length: float,
    grid_id: wp.uint64,
    rho: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_neighbor_particle_active(particle_flags[i]):
        rho[i] = 0.0
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

    rho[i] = rho_temp


@wp.kernel
def compute_factor_dfsph(
    particle_flags: wp.array(dtype=wp.int32),
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    smoothing_length: float,
    min_neighbors_for_factor: int,
    factor: wp.array(dtype=float),
    neighbor_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_neighbor_particle_active(particle_flags[i]):
        factor[i] = 0.0
        neighbor_count[i] = 0
        return

    xi = x[i]
    sum_grad = wp.vec3(0.0)
    sum_grad_sq = float(0.0)
    count = int(0)
    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if not is_neighbor_particle_active(particle_flags[j]):
            continue
        if j != i:
            count += 1
        grad = mass[j] * get_cubic_derivative(xi - x[j], smoothing_length)
        sum_grad += grad
        sum_grad_sq += wp.dot(grad, grad)

    neighbor_count[i] = count
    denom = wp.dot(sum_grad, sum_grad) + sum_grad_sq
    if count < min_neighbors_for_factor or denom < 1.0e-6:
        factor[i] = 0.0
    else:
        factor[i] = 1.0 / denom


@wp.kernel
def acceleration_non_pressure(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
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
    acc = wp.vec3(0.0)
    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)
    for index in neighbors:
        if index == i or not is_neighbor_particle_active(particle_flags[index]):
            continue
        direction = x - particle_x[index]
        distance = wp.length(direction)
        acc = cal_acc_with_non_pressure(
            acc,
            particle_rho[index],
            particle_v[i],
            particle_v[index],
            mass[i],
            mass[index],
            viscosity[i],
            surface_tension[i],
            direction,
            wp.max(distance, particle_radius),
            distance,
            smoothing_length,
        )

    world_idx = particle_world[i]
    world_g = gravity[wp.max(world_idx, 0)]
    particle_a[i] = acc + world_g


@wp.kernel
def predict_velocity(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    v: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    dt: float,
    v_star: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        v_star[tid] = wp.vec3(0.0)
        return
    v_star[tid] = v[tid] + dt * a[tid]


@wp.kernel
def compute_density_change_rate(
    particle_flags: wp.array(dtype=wp.int32),
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    smoothing_length: float,
    d_rho_dt: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_neighbor_particle_active(particle_flags[i]):
        d_rho_dt[i] = 0.0
        return

    xi = x[i]
    vi = v[i]
    div_val = float(0.0)
    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if not is_neighbor_particle_active(particle_flags[j]):
            continue
        grad_w = get_cubic_derivative(xi - x[j], smoothing_length)
        div_val += mass[j] * wp.dot(vi - v[j], grad_w)

    d_rho_dt[i] = div_val


@wp.kernel
def predict_density_star(
    particle_flags: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    d_rho_dt: wp.array(dtype=float),
    dt: float,
    rho_star: wp.array(dtype=float),
):
    tid = wp.tid()
    if not is_neighbor_particle_active(particle_flags[tid]):
        rho_star[tid] = 0.0
        return
    rho_star[tid] = rho[tid] + dt * d_rho_dt[tid]


@wp.kernel
def compute_kappa_density(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    rho_star: wp.array(dtype=float),
    rest_density: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    dt: float,
    max_kappa: float,
    kappa: wp.array(dtype=float),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        kappa[tid] = 0.0
        return
    rho_err = wp.max(rho_star[tid] - rest_density[tid], 0.0)
    kappa[tid] = wp.min((rho_err / (dt * dt)) * factor[tid], max_kappa)


@wp.kernel
def apply_constant_density_correction(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v_star: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_dynamic_particle(particle_flags[i], particle_mask[i]):
        return

    xi = x[i]
    dv = wp.vec3(0.0)
    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if not is_neighbor_particle_active(particle_flags[j]):
            continue
        coeff = mass[j] * (kappa[i] / wp.max(rho[i], 1.0e-6) + kappa[j] / wp.max(rho[j], 1.0e-6))
        dv += coeff * get_cubic_derivative(xi - x[j], smoothing_length)

    v_star[i] -= dt * dv


@wp.kernel
def integrate_positions(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_in: wp.array(dtype=wp.vec3),
    v_star: wp.array(dtype=wp.vec3),
    dt: float,
    x_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        x_out[tid] = x_in[tid]
        return
    x_out[tid] = x_in[tid] + v_star[tid] * dt


@wp.kernel
def compute_kappa_divergence(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    d_rho_dt: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    factor: wp.array(dtype=float),
    dt: float,
    max_kappa_v: float,
    kappa_v: wp.array(dtype=float),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        kappa_v[tid] = 0.0
        return
    density_change = wp.max(d_rho_dt[tid], 0.0)
    kappa_v[tid] = wp.min((1.0 / dt) * density_change * rho[tid] * factor[tid], max_kappa_v)


@wp.kernel
def apply_divergence_free_correction(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v_star: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    kappa_v: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_dynamic_particle(particle_flags[i], particle_mask[i]):
        return

    xi = x[i]
    dv = wp.vec3(0.0)
    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if not is_neighbor_particle_active(particle_flags[j]):
            continue
        coeff = mass[j] * (kappa_v[i] / wp.max(rho[i], 1.0e-6) + kappa_v[j] / wp.max(rho[j], 1.0e-6))
        dv += coeff * get_cubic_derivative(xi - x[j], smoothing_length)

    v_star[i] -= dt * dv


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
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[0] > size[0]:
        x = wp.vec3(size[0], x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[1] < -size[1]:
        x = wp.vec3(x[0], -size[1], x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[1] > size[1]:
        x = wp.vec3(x[0], size[1], x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)
    if x[2] > size[2] * 2.0:
        x = wp.vec3(x[0], x[1], size[2] * 2.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    particle_x[tid] = x
    particle_v[tid] = v


class SolverDFSPH(SolverBase):
    @dataclass
    class SphConfig:
        max_rho_iterations: int = 2
        max_vel_iterations: int = 1
        warm_start: bool = True
        eta_density: float = 1.0e-1
        eta_divergence: float = 1.0e-4
        max_kappa: float = 1.0e4
        max_kappa_v: float = 1.0e6
        min_neighbors_for_factor: int = 15
        particle_radius: float = 0.1
        particle_length: float = 0.2
        smoothing_length_coff: float = 1.35
        bound_width: float = 10.0
        bound_height: float = 10.0
        bound_length: float = 10.0
        boundary_damping: float = -0.1
        voxel_size: float = 0.3

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
        for name, dtype in (
            ("rho", wp.float32),
            ("factor", wp.float32),
            ("rho_star", wp.float32),
            ("d_rho_dt", wp.float32),
            ("kappa", wp.float32),
            ("kappa_v", wp.float32),
            ("neighbor_count", wp.int32),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    assignment=newton.Model.AttributeAssignment.STATE,
                    dtype=dtype,
                    default=0 if dtype == wp.int32 else 0.0,
                    namespace="sph",
                )
            )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="v_star",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
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

    def _dynamic_particle_mask(self) -> np.ndarray:
        particle_flags = self.model.particle_flags.numpy()
        particle_mask = self.model.sph.particle_mask.numpy()
        return ((particle_flags & int(ParticleFlags.ACTIVE)) != 0) & (particle_mask == 0)

    def _density_residual(self, rho_star: wp.array, rest_density: wp.array, dynamic_mask: np.ndarray) -> float:
        if not np.any(dynamic_mask):
            return 0.0
        rho_star_np = rho_star.numpy()
        rest_density_np = rest_density.numpy()
        rho_err = np.maximum(rho_star_np[dynamic_mask] - rest_density_np[dynamic_mask], 0.0)
        if rho_err.size == 0:
            return 0.0
        return float(np.mean(rho_err))

    def _divergence_residual(self, d_rho_dt: wp.array, dynamic_mask: np.ndarray) -> float:
        if not np.any(dynamic_mask):
            return 0.0
        d_rho_dt_np = d_rho_dt.numpy()
        density_change = np.maximum(d_rho_dt_np[dynamic_mask], 0.0)
        if density_change.size == 0:
            return 0.0
        return float(np.mean(density_change))

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
            raise RuntimeError("SolverDFSPH requires model.particle_grid to be available for particle neighborhoods.")

        self._update_particle_volume()

        model = self.model
        smoothing_length = float(self.smoothing_length)
        particle_radius = float(
            self.config.particle_radius if self.config.particle_radius > 0.0 else model.particle_max_radius
        )

        model.particle_grid.build(state_in.particle_q, radius=smoothing_length)

        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                state_in.particle_q,
                model.sph.rest_density,
                self._particle_volume,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[state_out.sph.rho],
            device=model.device,
        )
        wp.launch(
            kernel=compute_factor_dfsph,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.particle_grid.id,
                state_in.particle_q,
                model.particle_mass,
                smoothing_length,
                int(self.config.min_neighbors_for_factor),
            ],
            outputs=[state_out.sph.factor, state_out.sph.neighbor_count],
            device=model.device,
        )
        wp.launch(
            kernel=acceleration_non_pressure,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.sph.rho,
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
            kernel=predict_velocity,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_qd,
                self._particle_accel,
                dt,
            ],
            outputs=[state_out.sph.v_star],
            device=model.device,
        )

        if self.config.warm_start:
            wp.copy(state_out.sph.kappa, state_in.sph.kappa)
            wp.launch(
                kernel=apply_constant_density_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_out.sph.v_star,
                    model.particle_mass,
                    state_out.sph.rho,
                    state_out.sph.kappa,
                    smoothing_length,
                    dt,
                ],
                device=model.device,
            )

        dynamic_mask = self._dynamic_particle_mask()

        for density_iter in range(self.config.max_rho_iterations):
            wp.launch(
                kernel=compute_density_change_rate,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_out.sph.v_star,
                    model.particle_mass,
                    smoothing_length,
                ],
                outputs=[state_out.sph.d_rho_dt],
                device=model.device,
            )
            wp.launch(
                kernel=predict_density_star,
                dim=model.particle_count,
                inputs=[model.particle_flags, state_out.sph.rho, state_out.sph.d_rho_dt, dt],
                outputs=[state_out.sph.rho_star],
                device=model.device,
            )
            density_residual = self._density_residual(state_out.sph.rho_star, model.sph.rest_density, dynamic_mask)
            if density_iter > 0 and density_residual <= self.config.eta_density:
                break
            wp.launch(
                kernel=compute_kappa_density,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    state_out.sph.rho_star,
                    model.sph.rest_density,
                    state_out.sph.factor,
                    dt,
                    float(self.config.max_kappa),
                ],
                outputs=[state_out.sph.kappa],
                device=model.device,
            )
            wp.launch(
                kernel=apply_constant_density_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_out.sph.v_star,
                    model.particle_mass,
                    state_out.sph.rho,
                    state_out.sph.kappa,
                    smoothing_length,
                    dt,
                ],
                device=model.device,
            )

        wp.launch(
            kernel=integrate_positions,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_q,
                state_out.sph.v_star,
                dt,
            ],
            outputs=[state_out.particle_q],
            device=model.device,
        )

        model.particle_grid.build(state_out.particle_q, radius=smoothing_length)

        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                state_out.particle_q,
                model.sph.rest_density,
                self._particle_volume,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[state_out.sph.rho],
            device=model.device,
        )
        wp.launch(
            kernel=compute_factor_dfsph,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.particle_grid.id,
                state_out.particle_q,
                model.particle_mass,
                smoothing_length,
                int(self.config.min_neighbors_for_factor),
            ],
            outputs=[state_out.sph.factor, state_out.sph.neighbor_count],
            device=model.device,
        )

        if self.config.warm_start:
            wp.copy(state_out.sph.kappa_v, state_in.sph.kappa_v)
            wp.launch(
                kernel=apply_divergence_free_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    state_out.particle_q,
                    state_out.sph.v_star,
                    model.particle_mass,
                    state_out.sph.rho,
                    state_out.sph.kappa_v,
                    smoothing_length,
                    dt,
                ],
                device=model.device,
            )

        for divergence_iter in range(self.config.max_vel_iterations):
            wp.launch(
                kernel=compute_density_change_rate,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.particle_grid.id,
                    state_out.particle_q,
                    state_out.sph.v_star,
                    model.particle_mass,
                    smoothing_length,
                ],
                outputs=[state_out.sph.d_rho_dt],
                device=model.device,
            )
            divergence_residual = self._divergence_residual(state_out.sph.d_rho_dt, dynamic_mask)
            if divergence_iter > 0 and divergence_residual <= self.config.eta_divergence:
                break
            wp.launch(
                kernel=compute_kappa_divergence,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    state_out.sph.d_rho_dt,
                    state_out.sph.rho,
                    state_out.sph.factor,
                    dt,
                    float(self.config.max_kappa_v),
                ],
                outputs=[state_out.sph.kappa_v],
                device=model.device,
            )
            wp.launch(
                kernel=apply_divergence_free_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    state_out.particle_q,
                    state_out.sph.v_star,
                    model.particle_mass,
                    state_out.sph.rho,
                    state_out.sph.kappa_v,
                    smoothing_length,
                    dt,
                ],
                device=model.device,
            )

        wp.copy(state_out.particle_qd, state_out.sph.v_star)
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

