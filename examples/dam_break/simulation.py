"""Solver adapters for the common scene, using meters throughout."""

from dataclasses import asdict
import math

import newton
import numpy as np
import warp as wp

from solver.apic import SolverAPIC
from solver.pbf import SolverPBF
from solver.sph.solver_dfsph import SolverDFSPH
from solver.sph.solver_wcsph import SolverWCSPH


SOLVERS = {"apic": SolverAPIC, "wcsph": SolverWCSPH, "dfsph": SolverDFSPH, "pbf": SolverPBF}


@wp.kernel
def particle_norm_maxima(
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    maxima: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    v = velocities[i]
    v2 = wp.dot(v, v)
    if not (wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2]) and wp.isfinite(v2)):
        v2 = wp.inf
    wp.atomic_max(maxima, 0, v2)
    if i < accelerations.shape[0]:
        a = accelerations[i]
        a2 = wp.dot(a, a)
        if not (wp.isfinite(a[0]) and wp.isfinite(a[1]) and wp.isfinite(a[2]) and wp.isfinite(a2)):
            a2 = wp.inf
        wp.atomic_max(maxima, 1, a2)


@wp.kernel
def check_particle_finite(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=wp.float32),
    invalid: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    x, v = positions[i], velocities[i]
    bad = not (wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2])
               and wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2]))
    if i < density.shape[0]:
        bad = bad or not wp.isfinite(density[i])
    if bad:
        wp.atomic_max(invalid, 0, 1)


class DamBreakSimulation:
    def __init__(self, scene, method, device="cuda:0", substeps=4, voxel_size=None,
                 enable_projection=True, *, wcsph_stiffness=250000.0,
                 wcsph_viscosity=1.0e-6, wcsph_smoothing_length_coff=2.0,
                 wcsph_clamp_negative_pressure=True):
        if method == "wcsph":
            for name, value in (("stiffness", wcsph_stiffness),
                                ("smoothing_length_coff", wcsph_smoothing_length_coff)):
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"WCSPH {name} must be finite and positive")
            if not math.isfinite(wcsph_viscosity) or wcsph_viscosity < 0.0:
                raise ValueError("WCSPH viscosity must be finite and non-negative")
        self.scene = scene
        self.method = method
        self.frame_dt = 1.0 / scene.spec.fps
        self.substeps = substeps
        self.frame = 0
        self.sim_time = 0.0
        self.last_substeps = 0
        self.total_substeps = 0
        solver_cls = SOLVERS[method]
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_particle_radius = scene.radius
        solver_cls.register_custom_attributes(builder)
        self.material = {}
        if method == "apic":
            attributes = {"apic:particle_volume": [scene.volume] * scene.count}
        else:
            # These SPH kernels use kinematic viscosity and a kernel cohesion coefficient.
            # No legacy x100 coordinate conversion or 0.8 mass multiplier is used here.
            self.material = {"rest_density": scene.spec.density, "surface_tension": 0.0,
                             "viscosity": 1.0e-6}
            if method == "wcsph":
                self.material.update(stiffness=wcsph_stiffness, exponent=7.0,
                                     viscosity=wcsph_viscosity)
            attributes = {f"sph:{key}": [value] * scene.count for key, value in self.material.items()}
            attributes["sph:particle_mask"] = [0] * scene.count
        builder.add_particles(
            pos=scene.positions.tolist(), vel=np.zeros_like(scene.positions).tolist(),
            mass=[scene.mass] * scene.count, radius=[scene.radius] * scene.count,
            custom_attributes=attributes,
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0))
        self.model = builder.finalize(device=device)
        self.model.set_gravity(scene.spec.gravity)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        if method == "apic":
            config = SolverAPIC.Config(
                voxel_size=voxel_size if voxel_size is not None else max(scene.spacing * 1.5, 0.010),
                bound_lo=scene.spec.bound_lo, bound_hi=scene.spec.bound_hi,
                clamp_eps=1.0e-3, grid_capacity_ratio=16.0,
                enable_projection=enable_projection, wall_friction=0.0,
                projection_tol=1.0e-6, projection_max_iters=1000,
            )
        else:
            bounds = dict(bound_width=scene.spec.bound_hi[0], bound_height=scene.spec.bound_hi[1],
                          bound_length=scene.spec.bound_hi[2] * 0.5)
            # The kernels' support radius is h, not 2h. h=2d accommodates the shared
            # APIC mass (rho0*d^3) and gives DFSPH enough interior neighbors.
            common = dict(particle_radius=scene.radius, particle_length=scene.spacing,
                          smoothing_length_coff=2.0, boundary_damping=0.0, **bounds)
            if method == "wcsph":
                common["smoothing_length_coff"] = wcsph_smoothing_length_coff
                config = SolverWCSPH.SphConfig(
                    clamp_negative_pressure=wcsph_clamp_negative_pressure, **common)
            elif method == "dfsph":
                # Kappa has a length-squared factor; convert the old x100 demo's
                # safety caps for the meter-scale example without changing the solver.
                config = SolverDFSPH.SphConfig(max_rho_iterations=5, max_vel_iterations=12,
                                               warm_start=False, max_kappa=1.0,
                                               max_kappa_v=100.0, **common)
            else:
                config = SolverPBF.PbfConfig(
                    max_iterations=4, lambda_epsilon=100.0, min_neighbors_for_lambda=8,
                    scorr_k=0.0, max_delta_position=scene.radius * 0.5,
                    xsph_c=0.05, **common,
                )
        with wp.ScopedDevice(self.model.device):
            self.solver = solver_cls(self.model, config)
        self._invalid_state = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._empty_density = wp.empty(0, dtype=wp.float32, device=self.model.device)
        self._empty_acceleration = wp.empty(0, dtype=wp.vec3, device=self.model.device)
        self._norm_maxima = wp.zeros(2, dtype=wp.float32, device=self.model.device)

    def _cfl_maxima(self):
        """Reduce on the device; each CFL substep reads back only two scalars."""
        accel = (self.solver._particle_accel
                 if self.method == "wcsph" and self.total_substeps else self._empty_acceleration)
        self._norm_maxima.zero_()
        wp.launch(particle_norm_maxima, dim=self.scene.count,
                  inputs=[self.state_0.particle_qd, accel, self._norm_maxima],
                  device=self.model.device)
        speed2, accel2 = self._norm_maxima.numpy()
        if not math.isfinite(speed2):
            raise RuntimeError(f"Non-finite velocity at frame {self.frame}")
        if not math.isfinite(accel2):
            raise RuntimeError("Non-finite WCSPH acceleration")
        return math.sqrt(speed2), math.sqrt(accel2)

    def step(self):
        """Advance a complete display frame; never discard unused CFL time."""
        if self.method in ("apic", "pbf"):
            steps = self.substeps
            dt = self.frame_dt / steps
            for _ in range(steps):
                self._substep(dt)
        else:
            remaining = self.frame_dt
            steps = 0
            while remaining > 1.0e-12:
                speed, accel_max = self._cfl_maxima()
                limit = self.frame_dt / self.substeps
                if self.method == "wcsph":
                    sound_speed = math.sqrt(self.material["stiffness"] * self.material["exponent"]
                                            / self.scene.spec.density)
                    limit = min(limit, 0.25 * self.solver.smoothing_length / (sound_speed + speed))
                    # The first acceleration buffer is uninitialized and is excluded above.
                    if accel_max > 0.0:
                        limit = min(limit, 0.25 * math.sqrt(self.solver.smoothing_length / accel_max))
                    # Explicit viscosity also needs a diffusion timestep limit when
                    # users increase damping. This is inactive for the water baseline.
                    viscosity = self.material["viscosity"]
                    if viscosity > 0.0:
                        limit = min(limit, 0.025 * self.solver.smoothing_length**2 / viscosity)
                elif speed > 0.0:
                    limit = min(limit, 0.4 * self.scene.spacing / speed)
                dt = min(remaining, limit)
                if dt < 1.0e-10 or steps >= 4096:
                    raise RuntimeError("CFL step cannot safely complete this frame")
                self._substep(dt)
                remaining -= dt
                steps += 1
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        self.last_substeps = steps

    def _substep(self, dt):
        with wp.ScopedDevice(self.model.device):
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.total_substeps += 1

    def metrics(self):
        x, v = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        finite = bool(np.isfinite(x).all() and np.isfinite(v).all())
        if not finite:
            raise RuntimeError(f"Non-finite particle state at frame {self.frame}")
        result = dict(frame=self.frame, simulation_time=self.sim_time, substeps=self.last_substeps,
                      total_substeps=self.total_substeps, finite=finite,
                      speed_max=float(np.linalg.norm(v, axis=1).max(initial=0.0)),
                      height_mean=float(x[:, 2].mean()), height_min=float(x[:, 2].min()),
                      height_max=float(x[:, 2].max()))
        if self.method == "apic":
            result.update(projection_iterations=self.solver.last_projection_iters,
                          projection_residual=float(self.solver.last_projection_residual))
        elif self.frame:
            rho = self.state_0.sph.rho.numpy()
            if not np.isfinite(rho).all():
                raise RuntimeError("Non-finite density")
            result.update(density_mean=float(rho.mean()), density_min=float(rho.min()),
                          density_max=float(rho.max()))
            if self.method == "wcsph":
                pressure = self.state_0.sph.pressure.numpy()
                if not np.isfinite(pressure).all():
                    raise RuntimeError("Non-finite pressure")
                density_p99 = float(np.percentile(rho, 99))
                result.update(density_p99=density_p99,
                              density_compression_p99=max(0.0, density_p99 / self.scene.spec.density - 1.0),
                              pressure_min=float(pressure.min()), pressure_max=float(pressure.max()))
        return result

    def check_finite(self):
        """Validate every frame on the device and read back a single integer."""
        self._invalid_state.zero_()
        density = self._empty_density if self.method == "apic" else self.state_0.sph.rho
        wp.launch(check_particle_finite, dim=self.scene.count,
                  inputs=[self.state_0.particle_q, self.state_0.particle_qd,
                          density, self._invalid_state], device=self.model.device)
        if self._invalid_state.numpy()[0]:
            raise RuntimeError(f"Non-finite particle state at frame {self.frame}")

    def report(self):
        return dict(method=self.method, config=asdict(self.solver.config), material=self.material,
                    frame_dt=self.frame_dt, base_substeps=self.substeps,
                    time_step="fixed" if self.method in ("apic", "pbf") else "CFL, complete frame",
                    solver_execution="plain launches")
