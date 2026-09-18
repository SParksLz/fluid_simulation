# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""APIC fluid on warp.fem.Nanogrid with optional pressure projection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.fem import Domain, Field, Sample, at_node, div, grad, integrand
from warp.sparse import BsrMatrix, bsr_mm, bsr_mv, bsr_transposed

import newton
from newton import ParticleFlags
from newton.solvers import SolverBase


@wp.func
def apply_box_free_slip(
    x: wp.vec3,
    vel: wp.vec3,
    lo: wp.vec3,
    hi: wp.vec3,
    band: float,
    friction: float,
):
    """Axis-separated free-slip for AABB, including edges/corners.

    Removes the into-wall velocity component for every nearby face, and optionally
    damps tangential speed to reduce corner accumulation.
    """
    v = vel
    contacted = int(0)

    if x[0] <= lo[0] + band:
        if v[0] < 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        contacted = 1
    if x[0] >= hi[0] - band:
        if v[0] > 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        contacted = 1
    if x[1] <= lo[1] + band:
        if v[1] < 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        contacted = 1
    if x[1] >= hi[1] - band:
        if v[1] > 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        contacted = 1
    if x[2] <= lo[2] + band:
        if v[2] < 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
        contacted = 1
    if x[2] >= hi[2] - band:
        if v[2] > 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
        contacted = 1

    if contacted != 0 and friction > 0.0:
        keep = wp.max(0.0, 1.0 - friction)
        v = v * keep
    return v


@integrand
def integrate_fraction(s: Sample, phi: Field):
    return phi(s)


@integrand
def integrate_velocity(
    s: Sample,
    domain: Domain,
    u: Field,
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    particle_world: wp.array(dtype=wp.int32),
    bound_lo: wp.vec3,
    bound_hi: wp.vec3,
    wall_band: float,
    wall_friction: float,
    dt: float,
):
    """APIC P2G with gravity and AABB free-slip."""
    pid = s.qp_index
    if (particle_flags[pid] & ParticleFlags.ACTIVE) == 0:
        return 0.0

    node_offset = domain(at_node(u, s)) - domain(s)
    vel_apic = velocities[pid] + velocity_gradients[pid] * node_offset

    world_idx = wp.max(particle_world[pid], 0)
    vel_adv = vel_apic + dt * gravity[world_idx]
    x = domain(s)
    vel_adv = apply_box_free_slip(x, vel_adv, bound_lo, bound_hi, wall_band, wall_friction)
    return wp.dot(u(s), vel_adv)


@integrand
def velocity_boundary_projector_form(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    bound_lo: wp.vec3,
    bound_hi: wp.vec3,
    wall_band: float,
):
    """Separable AABB free-slip projector (valid on faces, edges, and corners)."""
    x = domain(s)
    uv = u(s)
    vv = v(s)
    out = float(0.0)
    if x[0] <= bound_lo[0] + wall_band or x[0] >= bound_hi[0] - wall_band:
        out += uv[0] * vv[0]
    if x[1] <= bound_lo[1] + wall_band or x[1] >= bound_hi[1] - wall_band:
        out += uv[1] * vv[1]
    if x[2] <= bound_lo[2] + wall_band or x[2] >= bound_hi[2] - wall_band:
        out += uv[2] * vv[2]
    return out


@integrand
def divergence_form(s: Sample, domain: Domain, u: Field, psi: Field):
    return div(u, s) * psi(s)


@wp.kernel
def scale_transposed_divergence_mat(
    tr_divergence_mat_offsets: wp.array(dtype=int),
    tr_divergence_mat_values: wp.array(dtype=Any),
    inv_fraction_int: wp.array(dtype=float),
):
    u_i = wp.tid()
    block_beg = tr_divergence_mat_offsets[u_i]
    block_end = tr_divergence_mat_offsets[u_i + 1]
    for b in range(block_beg, block_end):
        tr_divergence_mat_values[b] = tr_divergence_mat_values[b] * inv_fraction_int[u_i]


def solve_incompressibility(
    divergence_mat: BsrMatrix,
    dirichlet_projector: BsrMatrix,
    inv_volume: wp.array,
    pressure: wp.array,
    velocity: wp.array,
    quiet: bool = True,
    tol: float = 1.0e-5,
    max_iters: int = 200,
) -> tuple[float, int]:
    """Project grid velocity to a (approximately) divergence-free field."""
    rhs = wp.empty_like(pressure)
    bsr_mv(A=divergence_mat, x=velocity, y=rhs, alpha=-1.0)

    bsr_mm(alpha=-1.0, x=divergence_mat, y=dirichlet_projector, z=divergence_mat, beta=1.0)

    transposed_divergence_mat = bsr_transposed(divergence_mat)
    wp.launch(
        kernel=scale_transposed_divergence_mat,
        dim=inv_volume.shape[0],
        inputs=[
            transposed_divergence_mat.offsets,
            transposed_divergence_mat.values,
            inv_volume,
        ],
        device=inv_volume.device,
    )

    schur = bsr_mm(divergence_mat, transposed_divergence_mat)
    residual, iters = fem_example_utils.bsr_cg(
        schur,
        b=rhs,
        x=pressure,
        quiet=quiet,
        tol=tol,
        method="cr",
        max_iters=max_iters,
    )

    bsr_mv(A=transposed_divergence_mat, x=pressure, y=velocity, alpha=1.0, beta=1.0)
    return float(residual), int(iters)


@integrand
def update_particles(
    s: Sample,
    domain: Domain,
    grid_vel: Field,
    dt: float,
    bound_lo: wp.vec3,
    bound_hi: wp.vec3,
    clamp_eps: float,
    wall_friction: float,
    particle_flags: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    vel_grad: wp.array(dtype=wp.mat33),
):
    """G2P + advect + axis-separated wall response (no single-normal corner bug)."""
    pid = s.qp_index
    if (particle_flags[pid] & ParticleFlags.ACTIVE) == 0:
        pos[pid] = pos_prev[pid]
        return

    p_vel = grid_vel(s)
    vel_grad[pid] = grad(grid_vel, s)

    pos_adv = pos_prev[pid] + dt * p_vel
    lo = bound_lo + wp.vec3(clamp_eps, clamp_eps, clamp_eps)
    hi = bound_hi - wp.vec3(clamp_eps, clamp_eps, clamp_eps)

    # Component-wise clamp + kill into-wall velocity for every violated face.
    vx = p_vel[0]
    vy = p_vel[1]
    vz = p_vel[2]
    contacted = int(0)

    if pos_adv[0] < lo[0]:
        pos_adv = wp.vec3(lo[0], pos_adv[1], pos_adv[2])
        if vx < 0.0:
            vx = 0.0
        contacted = 1
    elif pos_adv[0] > hi[0]:
        pos_adv = wp.vec3(hi[0], pos_adv[1], pos_adv[2])
        if vx > 0.0:
            vx = 0.0
        contacted = 1

    if pos_adv[1] < lo[1]:
        pos_adv = wp.vec3(pos_adv[0], lo[1], pos_adv[2])
        if vy < 0.0:
            vy = 0.0
        contacted = 1
    elif pos_adv[1] > hi[1]:
        pos_adv = wp.vec3(pos_adv[0], hi[1], pos_adv[2])
        if vy > 0.0:
            vy = 0.0
        contacted = 1

    if pos_adv[2] < lo[2]:
        pos_adv = wp.vec3(pos_adv[0], pos_adv[1], lo[2])
        if vz < 0.0:
            vz = 0.0
        contacted = 1
    elif pos_adv[2] > hi[2]:
        pos_adv = wp.vec3(pos_adv[0], pos_adv[1], hi[2])
        if vz > 0.0:
            vz = 0.0
        contacted = 1

    p_vel = wp.vec3(vx, vy, vz)
    if contacted != 0 and wall_friction > 0.0:
        keep = wp.max(0.0, 1.0 - wall_friction)
        p_vel = p_vel * keep
        vel_grad[pid] = vel_grad[pid] * keep

    pos[pid] = pos_adv
    vel[pid] = p_vel


@wp.kernel
def stabilize_affine_C(
    flags: wp.array(dtype=wp.int32),
    C: wp.array(dtype=wp.mat33),
    damping: float,
    max_norm: float,
):
    """Damp and Frobenius-clamp APIC affine matrices to limit splash blow-up."""
    i = wp.tid()
    if (flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    c = C[i] * damping
    n2 = float(0.0)
    for r in range(3):
        for col in range(3):
            v = c[r, col]
            n2 += v * v
    n = wp.sqrt(n2)
    if max_norm > 0.0 and n > max_norm and n > 0.0:
        c = c * (max_norm / n)
    C[i] = c


@wp.kernel
def clamp_particle_speed(
    flags: wp.array(dtype=wp.int32),
    qd: wp.array(dtype=wp.vec3),
    max_speed: float,
):
    i = wp.tid()
    if (flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if max_speed <= 0.0:
        return
    v = qd[i]
    speed = wp.length(v)
    if speed > max_speed:
        qd[i] = v * (max_speed / speed)


@wp.kernel
def invert_volume_kernel(values: wp.array(dtype=float)):
    i = wp.tid()
    m = values[i]
    values[i] = wp.where(m == 0.0, 0.0, 1.0 / m)


@wp.kernel
def scalar_vector_multiply(
    alpha: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    y[i] = alpha[i] * x[i]


@wp.kernel
def copy_particle_state(
    flags: wp.array(dtype=wp.int32),
    q_in: wp.array(dtype=wp.vec3),
    qd_in: wp.array(dtype=wp.vec3),
    c_in: wp.array(dtype=wp.mat33),
    q_out: wp.array(dtype=wp.vec3),
    qd_out: wp.array(dtype=wp.vec3),
    c_out: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    q_out[i] = q_in[i]
    qd_out[i] = qd_in[i]
    c_out[i] = c_in[i]
    if (flags[i] & ParticleFlags.ACTIVE) == 0:
        pass


@wp.func
def _voxel_corner(volume: wp.uint64, ijk: wp.vec3i, ox: int, oy: int, oz: int):
    return wp.volume_index_to_world(
        volume,
        wp.vec3(float(ijk[0] + ox), float(ijk[1] + oy), float(ijk[2] + oz)),
    )


@wp.kernel
def build_voxel_wireframe_edges(
    volume: wp.uint64,
    voxels: wp.array(dtype=wp.vec3i),
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    """Emit 12 cube edges per allocated Nanogrid voxel."""
    tid = wp.tid()
    ijk = voxels[tid]
    c000 = _voxel_corner(volume, ijk, 0, 0, 0)
    c100 = _voxel_corner(volume, ijk, 1, 0, 0)
    c010 = _voxel_corner(volume, ijk, 0, 1, 0)
    c110 = _voxel_corner(volume, ijk, 1, 1, 0)
    c001 = _voxel_corner(volume, ijk, 0, 0, 1)
    c101 = _voxel_corner(volume, ijk, 1, 0, 1)
    c011 = _voxel_corner(volume, ijk, 0, 1, 1)
    c111 = _voxel_corner(volume, ijk, 1, 1, 1)

    base = tid * 12
    # bottom face
    starts[base + 0] = c000
    ends[base + 0] = c100
    starts[base + 1] = c100
    ends[base + 1] = c110
    starts[base + 2] = c110
    ends[base + 2] = c010
    starts[base + 3] = c010
    ends[base + 3] = c000
    # top face
    starts[base + 4] = c001
    ends[base + 4] = c101
    starts[base + 5] = c101
    ends[base + 5] = c111
    starts[base + 6] = c111
    ends[base + 6] = c011
    starts[base + 7] = c011
    ends[base + 7] = c001
    # vertical edges
    starts[base + 8] = c000
    ends[base + 8] = c001
    starts[base + 9] = c100
    ends[base + 9] = c101
    starts[base + 10] = c110
    ends[base + 10] = c111
    starts[base + 11] = c010
    ends[base + 11] = c011


class SolverAPIC(SolverBase):
    @dataclass
    class Config:
        voxel_size: float = 0.05
        grid_padding_voxels: int = 2
        bound_lo: tuple[float, float, float] = (-1.0, -1.0, 0.0)
        bound_hi: tuple[float, float, float] = (1.0, 1.0, 2.0)
        clamp_eps: float = 1.0e-4
        wall_band: float = -1.0  # <0 => 0.75 * voxel_size
        wall_friction: float = 0.15
        mass_epsilon: float = 1.0e-8
        grid_capacity_ratio: float = 16.0
        enable_projection: bool = True
        projection_tol: float = 1.0e-5
        projection_max_iters: int = 200
        projection_quiet: bool = True
        # Stabilize APIC affine blow-up after impacts / free-surface contact.
        c_damping: float = 0.92
        c_norm_max: float = 25.0
        max_particle_speed: float = 8.0

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
                "SolverAPIC requires a Newton build with PARTICLE custom-attribute frequency support."
            )

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_volume",
                frequency=frequency.PARTICLE,
                assignment=assignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="apic",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="C",
                frequency=frequency.PARTICLE,
                assignment=assignment.STATE,
                dtype=wp.mat33,
                default=wp.mat33(0.0),
                namespace="apic",
            )
        )

    def __init__(self, model: newton.Model, config: Config):
        super().__init__(model=model)
        self.config = config
        self.temporary_store = fem.TemporaryStore()
        self._grid_status = wp.zeros(1, dtype=wp.uint32, device=model.device)
        self._volume = None
        self._grid = None
        self._linear_basis_space = None
        self._velocity_space = None
        self._fraction_space = None
        self._strain_space = None
        self._bound_lo = wp.vec3(*config.bound_lo)
        self._bound_hi = wp.vec3(*config.bound_hi)
        if float(config.wall_band) >= 0.0:
            self._wall_band = float(config.wall_band)
        else:
            self._wall_band = 0.75 * float(config.voxel_size)
        self._wall_friction = float(config.wall_friction)
        self._bsr_options = {"construction": "row_compress", "capacity": "auto"}
        self._initialized = False
        self._viz_edge_starts: wp.array | None = None
        self._viz_edge_ends: wp.array | None = None
        self._viz_edge_capacity = 0
        self.last_projection_residual = 0.0
        self.last_projection_iters = 0

        if model.particle_count > 0:
            self._ensure_grid(model.state().particle_q)

    def get_grid_wireframe(self) -> tuple[wp.array, wp.array] | tuple[None, None]:
        """Return line-segment endpoints for the current Nanogrid voxels.

        Each allocated voxel contributes 12 cube edges. Arrays are on the model device.
        """
        if self._volume is None:
            return None, None

        voxels_raw = self._volume.get_voxels()
        if voxels_raw is None or int(voxels_raw.shape[0]) == 0:
            return None, None

        voxel_count = int(voxels_raw.shape[0])
        if voxels_raw.dtype == wp.vec3i:
            voxels = voxels_raw
        else:
            voxels_np = voxels_raw.numpy().reshape((-1, 3)).astype(np.int32, copy=False)
            voxels = wp.array(voxels_np, dtype=wp.vec3i, device=self.model.device)

        edge_count = voxel_count * 12
        if self._viz_edge_capacity != edge_count:
            self._viz_edge_starts = wp.empty(edge_count, dtype=wp.vec3, device=self.model.device)
            self._viz_edge_ends = wp.empty(edge_count, dtype=wp.vec3, device=self.model.device)
            self._viz_edge_capacity = edge_count

        wp.launch(
            kernel=build_voxel_wireframe_edges,
            dim=voxel_count,
            inputs=[self._volume.id, voxels, self._viz_edge_starts, self._viz_edge_ends],
            device=self.model.device,
        )
        return self._viz_edge_starts, self._viz_edge_ends

    def _estimate_grid_capacity(self, positions: wp.array) -> dict[str, int]:
        ratio = float(self.config.grid_capacity_ratio)
        if ratio <= 0.0:
            raise ValueError("grid_capacity_ratio must be positive")

        initial_volume = wp.Volume.allocate_by_voxels(
            voxel_points=positions,
            voxel_size=float(self.config.voxel_size),
            device=positions.device,
        )
        first_counts = initial_volume.get_active_stats()
        if first_counts.voxel_count == 0:
            first_counts = wp.Volume.ActiveStats(1, 1, 1, 1)

        def scale(count: int, limit: int) -> int:
            return max(1, min(limit, math.ceil(count * ratio)))

        active_capacity = scale(first_counts.voxel_count, max(positions.shape[0], 1))
        leaf_capacity = scale(first_counts.leaf_node_count, active_capacity)
        lower_capacity = scale(first_counts.lower_node_count, leaf_capacity)
        upper_capacity = scale(first_counts.upper_node_count, lower_capacity)
        return {
            "max_active_voxels": active_capacity,
            "max_leaf_nodes": leaf_capacity,
            "max_lower_nodes": lower_capacity,
            "max_upper_nodes": upper_capacity,
        }

    def _ensure_grid(self, positions: wp.array) -> None:
        if self._initialized:
            return
        capacity = self._estimate_grid_capacity(positions)
        self._volume = wp.Volume.allocate_by_voxels(
            voxel_points=positions,
            voxel_size=float(self.config.voxel_size),
            rebuildable=True,
            status=self._grid_status,
            device=positions.device,
            **capacity,
        )
        self._grid = fem.Nanogrid(self._volume, rebuildable=True, temporary_store=self.temporary_store)
        self._linear_basis_space = fem.make_polynomial_basis_space(self._grid, degree=1)
        self._velocity_space = fem.make_collocated_function_space(self._linear_basis_space, dtype=wp.vec3)
        self._fraction_space = fem.make_collocated_function_space(self._linear_basis_space, dtype=float)
        self._strain_space = fem.make_polynomial_space(
            self._grid,
            dtype=float,
            degree=0,
            discontinuous=True,
        )
        self._initialized = True

    def _copy_passthrough_state(self, state_in: newton.State, state_out: newton.State) -> None:
        if state_in.body_count:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
            wp.copy(state_out.body_f, state_in.body_f)
        if state_in.joint_q is not None and state_out.joint_q is not None:
            wp.copy(state_out.joint_q, state_in.joint_q)
        if state_in.joint_qd is not None and state_out.joint_qd is not None:
            wp.copy(state_out.joint_qd, state_in.joint_qd)

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

        model = self.model
        if model.particle_count == 0:
            return

        fem.set_default_temporary_store(self.temporary_store)
        try:
            self._ensure_grid(state_in.particle_q)

            # Rebuild sparse grid around current particles.
            self._grid.rebuild(state_in.particle_q, status=self._grid_status)
            self._linear_basis_space.topology.rebuild()
            self._strain_space.topology.rebuild()

            whole_domain = fem.Cells(self._grid)
            pic = fem.PicQuadrature(
                domain=whole_domain,
                positions=state_in.particle_q,
                measures=model.apic.particle_volume,
                temporary_store=self.temporary_store,
            )

            cell_mask = wp.empty(shape=self._grid.cell_count(), dtype=int, device=model.device)
            pic.fill_element_mask(cell_mask)
            geo_partition = fem.ExplicitGeometryPartition(
                self._grid,
                cell_mask,
                max_cell_count=self._grid.cell_count(),
                max_side_count=0,
                temporary_store=self.temporary_store,
            )
            domain = fem.Cells(geo_partition)
            pic.domain = domain

            velocity_partition = fem.make_space_partition(
                self._velocity_space.topology,
                geometry_partition=geo_partition,
                with_halo=False,
                max_node_count=self._grid.vertex_count(),
                temporary_store=self.temporary_store,
            )
            strain_partition = fem.make_space_partition(
                self._strain_space.topology,
                geometry_partition=geo_partition,
                with_halo=False,
                max_node_count=self._grid.cell_count(),
                temporary_store=self.temporary_store,
            )
            velocity_restriction = fem.make_space_restriction(
                space_partition=velocity_partition,
                domain=domain,
                temporary_store=self.temporary_store,
            )
            strain_restriction = fem.make_space_restriction(
                space_partition=strain_partition,
                domain=domain,
                temporary_store=self.temporary_store,
            )

            velocity_test = fem.make_test(self._velocity_space, space_restriction=velocity_restriction)
            velocity_trial = fem.make_trial(self._velocity_space, space_restriction=velocity_restriction)
            fraction_test = fem.make_test(self._fraction_space, space_restriction=velocity_restriction)
            strain_test = fem.make_test(self._strain_space, space_restriction=strain_restriction)
            velocity_field = self._velocity_space.make_field(velocity_partition)
            pressure_field = self._strain_space.make_field(strain_partition)

            vel_projector = fem.integrate(
                velocity_boundary_projector_form,
                fields={"u": velocity_trial, "v": velocity_test},
                values={
                    "bound_lo": self._bound_lo,
                    "bound_hi": self._bound_hi,
                    "wall_band": float(self._wall_band),
                },
                assembly="nodal",
                output_dtype=float,
                bsr_options=self._bsr_options,
                temporary_store=self.temporary_store,
            )
            fem.normalize_dirichlet_projector(vel_projector)

            inv_volume = fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": fraction_test},
                output_dtype=float,
                temporary_store=self.temporary_store,
            )
            wp.launch(kernel=invert_volume_kernel, dim=inv_volume.shape, inputs=[inv_volume], device=model.device)

            particle_world = model.particle_world
            velocity_int = fem.integrate(
                integrate_velocity,
                quadrature=pic,
                fields={"u": velocity_test},
                values={
                    "velocities": state_in.particle_qd,
                    "velocity_gradients": state_in.apic.C,
                    "particle_flags": model.particle_flags,
                    "gravity": model.gravity,
                    "particle_world": particle_world,
                    "bound_lo": self._bound_lo,
                    "bound_hi": self._bound_hi,
                    "wall_band": float(self._wall_band),
                    "wall_friction": float(self._wall_friction),
                    "dt": float(dt),
                },
                output_dtype=wp.vec3,
                temporary_store=self.temporary_store,
            )

            wp.launch(
                kernel=scalar_vector_multiply,
                dim=inv_volume.shape[0],
                inputs=[inv_volume, velocity_int, velocity_field.dof_values],
                device=model.device,
            )

            bsr_mv(
                A=vel_projector,
                x=velocity_field.dof_values,
                y=velocity_field.dof_values,
                alpha=-1.0,
                beta=1.0,
            )

            self.last_projection_residual = 0.0
            self.last_projection_iters = 0
            if self.config.enable_projection:
                divergence_matrix = fem.integrate(
                    divergence_form,
                    quadrature=pic,
                    fields={"u": velocity_trial, "psi": strain_test},
                    output_dtype=float,
                    bsr_options=self._bsr_options,
                    temporary_store=self.temporary_store,
                )
                residual, iters = solve_incompressibility(
                    divergence_matrix,
                    vel_projector,
                    inv_volume,
                    pressure_field.dof_values,
                    velocity_field.dof_values,
                    quiet=bool(self.config.projection_quiet),
                    tol=float(self.config.projection_tol),
                    max_iters=int(self.config.projection_max_iters),
                )
                self.last_projection_residual = residual
                self.last_projection_iters = iters

            # Write outputs: start from inputs then overwrite active particles via G2P.
            wp.launch(
                kernel=copy_particle_state,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.apic.C,
                    state_out.particle_q,
                    state_out.particle_qd,
                    state_out.apic.C,
                ],
                device=model.device,
            )

            fem.interpolate(
                update_particles,
                at=pic,
                values={
                    "pos": state_out.particle_q,
                    "pos_prev": state_in.particle_q,
                    "vel": state_out.particle_qd,
                    "vel_grad": state_out.apic.C,
                    "dt": float(dt),
                    "bound_lo": self._bound_lo,
                    "bound_hi": self._bound_hi,
                    "clamp_eps": float(self.config.clamp_eps),
                    "wall_friction": float(self._wall_friction),
                    "particle_flags": model.particle_flags,
                },
                fields={"grid_vel": velocity_field},
                temporary_store=self.temporary_store,
            )

            wp.launch(
                kernel=stabilize_affine_C,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    state_out.apic.C,
                    float(self.config.c_damping),
                    float(self.config.c_norm_max),
                ],
                device=model.device,
            )
            if float(self.config.max_particle_speed) > 0.0:
                wp.launch(
                    kernel=clamp_particle_speed,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_flags,
                        state_out.particle_qd,
                        float(self.config.max_particle_speed),
                    ],
                    device=model.device,
                )
        finally:
            fem.set_default_temporary_store(None)
