# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

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
    if h > 0.0 and r_norm < h:
        h2 = h * h
        rhs = h2 - r_norm * r_norm
        k = 315.0 / (64.0 * wp.pi * cube(cube(h)))
        res = k * cube(rhs)
    return res


@wp.func
def get_cubic_derivative(r: wp.vec3, smoothing_length: float):
    res = wp.vec3(0.0)
    r_norm = wp.length(r)
    h = smoothing_length
    if h > 0.0 and r_norm > 1.0e-6 and r_norm < h:
        h6 = cube(h) * cube(h)
        coeff = -45.0 / (wp.pi * h6)
        term = (h - r_norm) * (h - r_norm) / r_norm
        res = coeff * term * r
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
def compute_scorr(
    r_norm: float,
    smoothing_length: float,
    scorr_k: float,
    scorr_n: float,
    w_delta_q: float,
):
    if w_delta_q <= 1.0e-6:
        return 0.0
    ratio = get_cubic(r_norm, smoothing_length) / w_delta_q
    return -scorr_k * wp.pow(ratio, scorr_n)


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
    for j in neighbors:
        if not is_neighbor_particle_active(particle_flags[j]):
            continue
        r = x - particle_x[j]
        mass_j = rest_density[j] * particle_volume[j]
        rho_temp += mass_j * get_cubic(wp.length(r), smoothing_length)

    rho[i] = rho_temp


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
    for j in neighbors:
        if j == i or not is_neighbor_particle_active(particle_flags[j]):
            continue
        d = x - particle_x[j]
        dist = wp.length(d)
        acc = cal_acc_with_non_pressure(
            acc,
            particle_rho[j],
            particle_v[i],
            particle_v[j],
            mass[i],
            mass[j],
            viscosity[i],
            surface_tension[i],
            d,
            wp.max(dist, particle_radius),
            dist,
            smoothing_length,
        )

    particle_a[i] = acc


@wp.kernel
def predict_velocity_with_gravity(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    v_in: wp.array(dtype=wp.vec3),
    gravity: wp.array(dtype=wp.vec3),
    particle_world: wp.array(dtype=wp.int32),
    dt: float,
    v_pred: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        v_pred[tid] = v_in[tid]
        return
    world_idx = particle_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]
    v_pred[tid] = v_in[tid] + dt * world_g


@wp.kernel
def predict_positions(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_in: wp.array(dtype=wp.vec3),
    v_pred: wp.array(dtype=wp.vec3),
    dt: float,
    x_pred: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        x_pred[tid] = x_in[tid]
        return
    x_pred[tid] = x_in[tid] + dt * v_pred[tid]


@wp.kernel
def apply_non_pressure_prediction(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    particle_a: wp.array(dtype=wp.vec3),
    dt: float,
    v_pred: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        return

    delta_v = dt * particle_a[tid]
    v_pred[tid] = v_pred[tid] + delta_v
    x_pred[tid] = x_pred[tid] + dt * delta_v


@wp.kernel
def compute_pbf_lambda(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rest_density: wp.array(dtype=float),
    smoothing_length: float,
    lambda_epsilon: float,
    min_neighbors_for_lambda: int,
    pbf_lambda: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    neighbor_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_dynamic_particle(particle_flags[i], particle_mask[i]):
        pbf_lambda[i] = 0.0
        rho[i] = 0.0
        neighbor_count[i] = 0
        return

    rho0_i = wp.max(rest_density[i], 1.0e-6)
    xi = x_pred[i]

    rho_i = float(0.0)
    grad_i = wp.vec3(0.0)
    sum_grad_sq = float(0.0)
    count = int(0)

    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if not is_neighbor_particle_active(particle_flags[j]):
            continue
        xij = xi - x_pred[j]
        r = wp.length(xij)
        if j != i and r < smoothing_length:
            count += 1
        mass_j = mass[j]
        rho_i += mass_j * get_cubic(r, smoothing_length)
        grad_j = -(mass_j / rho0_i) * get_cubic_derivative(xij, smoothing_length)
        sum_grad_sq += wp.dot(grad_j, grad_j)
        grad_i -= grad_j

    sum_grad_sq += wp.dot(grad_i, grad_i)
    # Enforce incompressibility as a unilateral constraint.  Allowing a
    # negative density error produces a positive lambda that attracts
    # under-dense particles and leads to tensile clumping at free surfaces.
    c_i = wp.max(rho_i / rho0_i - 1.0, 0.0)

    rho[i] = rho_i
    neighbor_count[i] = count
    if min_neighbors_for_lambda > 0 and count < min_neighbors_for_lambda:
        pbf_lambda[i] = 0.0
    else:
        pbf_lambda[i] = -c_i / (sum_grad_sq + lambda_epsilon)


@wp.kernel
def compute_delta_position(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rest_density: wp.array(dtype=float),
    pbf_lambda: wp.array(dtype=float),
    smoothing_length: float,
    scorr_k: float,
    scorr_n: float,
    scorr_w_delta_q: float,
    max_delta_position: float,
    delta_p: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)

    if not is_dynamic_particle(particle_flags[i], particle_mask[i]):
        delta_p[i] = wp.vec3(0.0)
        return

    xi = x_pred[i]
    rho0_i = wp.max(rest_density[i], 1.0e-6)
    dp = wp.vec3(0.0)

    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if j == i or not is_neighbor_particle_active(particle_flags[j]):
            continue
        xij = xi - x_pred[j]
        grad = get_cubic_derivative(xij, smoothing_length)
        scorr = compute_scorr(wp.length(xij), smoothing_length, scorr_k, scorr_n, scorr_w_delta_q)
        dp += (pbf_lambda[i] + pbf_lambda[j] + scorr) * mass[j] * grad

    dp = dp / rho0_i

    if max_delta_position > 0.0:
        dp_len = wp.length(dp)
        if dp_len > max_delta_position:
            dp = dp * (max_delta_position / wp.max(dp_len, 1.0e-12))

    delta_p[i] = dp


@wp.kernel
def apply_delta_position(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_pred: wp.array(dtype=wp.vec3),
    delta_p: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        return
    x_pred[tid] = x_pred[tid] + delta_p[tid]


@wp.kernel
def apply_position_bounds(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_pred: wp.array(dtype=wp.vec3),
    size: wp.vec3,
    particle_radius: float,
    boundary_epsilon: float,
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        return

    x = x_pred[tid]
    x_min = -size[0] + particle_radius + boundary_epsilon
    x_max = size[0] - particle_radius - boundary_epsilon
    y_min = -size[1] + particle_radius + boundary_epsilon
    y_max = size[1] - particle_radius - boundary_epsilon
    z_min = particle_radius + boundary_epsilon
    z_max = size[2] * 2.0 - particle_radius - boundary_epsilon
    if x[0] < x_min:
        x[0] = x_min
    if x[0] > x_max:
        x[0] = x_max
    if x[1] < y_min:
        x[1] = y_min
    if x[1] > y_max:
        x[1] = y_max
    if x[2] < z_min:
        x[2] = z_min
    if x[2] > z_max:
        x[2] = z_max

    x_pred[tid] = x


@wp.kernel
def update_velocity_from_positions(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_in: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    v_in: wp.array(dtype=wp.vec3),
    dt: float,
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        v_out[tid] = v_in[tid]
        return
    v_out[tid] = (x_pred[tid] - x_in[tid]) / wp.max(dt, 1.0e-8)


@wp.kernel
def compute_step_displacement(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    x_in: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    step_displacement: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        step_displacement[tid] = wp.vec3(0.0)
        return
    step_displacement[tid] = x_pred[tid] - x_in[tid]


@wp.kernel
def apply_xsph_viscosity(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v_in: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    smoothing_length: float,
    xsph_c: float,
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)
    if not is_dynamic_particle(particle_flags[i], particle_mask[i]):
        v_out[i] = v_in[i]
        return

    vi = v_in[i]
    xi = x[i]
    corr = wp.vec3(0.0)
    neighbors = wp.hash_grid_query(grid_id, xi, smoothing_length)
    for j in neighbors:
        if j == i or not is_neighbor_particle_active(particle_flags[j]):
            continue
        w = get_cubic(wp.length(xi - x[j]), smoothing_length)
        corr += (mass[j] / wp.max(rho[j], 1.0e-6)) * (v_in[j] - vi) * w

    v_out[i] = vi + xsph_c * corr


@wp.kernel
def apply_bounds(
    particle_flags: wp.array(dtype=wp.int32),
    particle_mask: wp.array(dtype=int),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    size: wp.vec3,
    particle_radius: float,
    boundary_epsilon: float,
    damping_coef: float,
):
    tid = wp.tid()
    if not is_dynamic_particle(particle_flags[tid], particle_mask[tid]):
        return

    x = particle_x[tid]
    v = particle_v[tid]

    x_min = -size[0] + particle_radius + boundary_epsilon
    x_max = size[0] - particle_radius - boundary_epsilon
    y_min = -size[1] + particle_radius + boundary_epsilon
    y_max = size[1] - particle_radius - boundary_epsilon
    z_min = particle_radius + boundary_epsilon
    z_max = size[2] * 2.0 - particle_radius - boundary_epsilon

    if x[0] < x_min:
        x = wp.vec3(x_min, x[1], x[2])
        if v[0] < 0.0:
            v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[0] > x_max:
        x = wp.vec3(x_max, x[1], x[2])
        if v[0] > 0.0:
            v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[1] < y_min:
        x = wp.vec3(x[0], y_min, x[2])
        if v[1] < 0.0:
            v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[1] > y_max:
        x = wp.vec3(x[0], y_max, x[2])
        if v[1] > 0.0:
            v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[2] < z_min:
        x = wp.vec3(x[0], x[1], z_min)
        if v[2] < 0.0:
            v = wp.vec3(v[0], v[1], v[2] * damping_coef)
    if x[2] > z_max:
        x = wp.vec3(x[0], x[1], z_max)
        if v[2] > 0.0:
            v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    particle_x[tid] = x
    particle_v[tid] = v


class SolverPBF(SolverBase):
    @dataclass
    class PbfConfig:
        max_iterations: int = 10
        lambda_epsilon: float = 100.0
        min_neighbors_for_lambda: int = 0
        scorr_k: float = 1.0e-4
        scorr_n: float = 4.0
        scorr_q: float = 0.3
        max_delta_position: float = 0.0
        xsph_c: float = 0.01
        particle_radius: float = 0.1
        particle_length: float = 0.2
        smoothing_length_coff: float = 1.35
        bound_width: float = 10.0
        bound_height: float = 10.0
        bound_length: float = 10.0
        boundary_damping: float = 0.0
        boundary_epsilon: float = 0.0
        rebuild_grid_each_iteration: bool = False

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        if hasattr(newton, "ModelAttributeFrequency") and hasattr(newton.ModelAttributeFrequency, "PARTICLE"):
            frequency = newton.ModelAttributeFrequency
            assignment = newton.ModelAttributeAssignment
        elif hasattr(newton.Model, "AttributeFrequency") and hasattr(newton.Model.AttributeFrequency, "PARTICLE"):
            frequency = newton.Model.AttributeFrequency
            assignment = newton.Model.AttributeAssignment
        else:
            raise RuntimeError(
                "SolverPBF requires a Newton build with PARTICLE custom-attribute frequency support."
            )

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_mask",
                frequency=frequency.PARTICLE,
                assignment=assignment.MODEL,
                dtype=int,
                default=0,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="rest_density",
                frequency=frequency.PARTICLE,
                assignment=assignment.MODEL,
                dtype=wp.float32,
                default=1000.0,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="surface_tension",
                frequency=frequency.PARTICLE,
                assignment=assignment.MODEL,
                dtype=wp.float32,
                default=0.01,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="viscosity",
                frequency=frequency.PARTICLE,
                assignment=assignment.MODEL,
                dtype=wp.float32,
                default=0.005,
                namespace="sph",
            )
        )
        for name, dtype in (
            ("rho", wp.float32),
            ("pbf_lambda", wp.float32),
            ("neighbor_count", wp.int32),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=frequency.PARTICLE,
                    assignment=assignment.STATE,
                    dtype=dtype,
                    default=0 if dtype == wp.int32 else 0.0,
                    namespace="sph",
                )
            )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="delta_p",
                frequency=frequency.PARTICLE,
                assignment=assignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="step_displacement",
                frequency=frequency.PARTICLE,
                assignment=assignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="sph",
            )
        )

    def __init__(self, model: newton.Model, config: PbfConfig):
        super().__init__(model=model)
        self.config = config

        if model.particle_count > 1:
            if model.particle_grid is None:
                model.particle_grid = wp.HashGrid(128, 128, 128)
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        self._particle_accel = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self._particle_volume = wp.empty(model.particle_count, dtype=wp.float32, device=model.device)
        self._v_pred = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self._v_tmp = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self._x_pred = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self._bounds = wp.vec3(config.bound_width, config.bound_height, config.bound_length)
        self._boundary_epsilon = float(config.boundary_epsilon)

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

    def _w_delta_q(self, smoothing_length: float) -> float:
        q = float(self.config.scorr_q)
        r = q * smoothing_length
        return float(get_cubic(r, smoothing_length))

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
            raise RuntimeError("SolverPBF requires model.particle_grid to be available for particle neighborhoods.")

        model = self.model
        smoothing_length = float(self.smoothing_length)
        particle_radius = float(
            self.config.particle_radius if self.config.particle_radius > 0.0 else model.particle_max_radius
        )

        self._update_particle_volume()

        wp.launch(
            kernel=predict_velocity_with_gravity,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_qd,
                model.gravity,
                model.particle_world,
                dt,
            ],
            outputs=[self._v_pred],
            device=model.device,
        )
        wp.launch(
            kernel=predict_positions,
            dim=model.particle_count,
            inputs=[model.particle_flags, model.sph.particle_mask, state_in.particle_q, self._v_pred, dt],
            outputs=[self._x_pred],
            device=model.device,
        )
        wp.launch(
            kernel=apply_position_bounds,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                self._x_pred,
                self._bounds,
                particle_radius,
                self._boundary_epsilon,
            ],
            device=model.device,
        )

        model.particle_grid.build(self._x_pred, radius=smoothing_length)
        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                self._x_pred,
                model.sph.rest_density,
                self._particle_volume,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[state_out.sph.rho],
            device=model.device,
        )
        wp.launch(
            kernel=acceleration_non_pressure,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                self._x_pred,
                self._v_pred,
                state_out.sph.rho,
                particle_radius,
                model.particle_mass,
                model.sph.surface_tension,
                model.sph.viscosity,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[self._particle_accel],
            device=model.device,
        )
        wp.launch(
            kernel=apply_non_pressure_prediction,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                self._particle_accel,
                dt,
                self._v_pred,
                self._x_pred,
            ],
            device=model.device,
        )
        wp.launch(
            kernel=apply_position_bounds,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                self._x_pred,
                self._bounds,
                particle_radius,
                self._boundary_epsilon,
            ],
            device=model.device,
        )

        w_delta_q = float(self._w_delta_q(smoothing_length))

        for _ in range(self.config.max_iterations):
            wp.launch(
                kernel=compute_pbf_lambda,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    self._x_pred,
                    model.particle_mass,
                    model.sph.rest_density,
                    smoothing_length,
                    float(self.config.lambda_epsilon),
                    int(self.config.min_neighbors_for_lambda),
                ],
                outputs=[state_out.sph.pbf_lambda, state_out.sph.rho, state_out.sph.neighbor_count],
                device=model.device,
            )
            wp.launch(
                kernel=compute_delta_position,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    self._x_pred,
                    model.particle_mass,
                    model.sph.rest_density,
                    state_out.sph.pbf_lambda,
                    smoothing_length,
                    float(self.config.scorr_k),
                    float(self.config.scorr_n),
                    w_delta_q,
                    float(self.config.max_delta_position),
                ],
                outputs=[state_out.sph.delta_p],
                device=model.device,
            )
            wp.launch(
                kernel=apply_delta_position,
                dim=model.particle_count,
                inputs=[model.particle_flags, model.sph.particle_mask, self._x_pred, state_out.sph.delta_p],
                device=model.device,
            )
            if self.config.rebuild_grid_each_iteration:
                wp.launch(
                    kernel=apply_position_bounds,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_flags,
                        model.sph.particle_mask,
                        self._x_pred,
                        self._bounds,
                        particle_radius,
                        self._boundary_epsilon,
                    ],
                    device=model.device,
                )
                model.particle_grid.build(self._x_pred, radius=smoothing_length)

        wp.launch(
            kernel=apply_position_bounds,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                self._x_pred,
                self._bounds,
                particle_radius,
                self._boundary_epsilon,
            ],
            device=model.device,
        )

        # Reuse the single grid built from the gravity-predicted positions for
        # the pressure solve, final density, and XSPH passes.  With per-iteration
        # rebuilds disabled, all neighbor kernels intentionally share this
        # fixed candidate set.
        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                self._x_pred,
                model.sph.rest_density,
                self._particle_volume,
                smoothing_length,
                model.particle_grid.id,
            ],
            outputs=[state_out.sph.rho],
            device=model.device,
        )
        # wp.launch(
        #     kernel=compute_step_displacement,
        #     dim=model.particle_count,
        #     inputs=[
        #         model.particle_flags,
        #         model.sph.particle_mask,
        #         state_in.particle_q,
        #         self._x_pred,
        #     ],
        #     outputs=[state_out.sph.step_displacement],
        #     device=model.device,
        # )
        wp.launch(
            kernel=update_velocity_from_positions,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_in.particle_q,
                self._x_pred,
                state_in.particle_qd,
                dt,
            ],
            outputs=[state_out.particle_qd],
            device=model.device,
        )

        if self.config.xsph_c > 0.0:
            wp.launch(
                kernel=apply_xsph_viscosity,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    model.sph.particle_mask,
                    model.particle_grid.id,
                    self._x_pred,
                    state_out.particle_qd,
                    model.particle_mass,
                    state_out.sph.rho,
                    smoothing_length,
                    float(self.config.xsph_c),
                ],
                outputs=[self._v_tmp],
                device=model.device,
            )
            wp.copy(state_out.particle_qd, self._v_tmp)

        wp.copy(state_out.particle_q, self._x_pred)
        wp.launch(
            kernel=apply_bounds,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                model.sph.particle_mask,
                state_out.particle_q,
                state_out.particle_qd,
                self._bounds,
                particle_radius,
                self._boundary_epsilon,
                float(self.config.boundary_damping),
            ],
            device=model.device,
        )

    def update_contacts(self, contacts: newton.Contacts) -> None:
        del contacts
