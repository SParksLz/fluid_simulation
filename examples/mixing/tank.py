"""A tank mixing scene with optional prefilled coffee and a milk inlet."""

from dataclasses import dataclass
import math

import newton
import numpy as np
import warp as wp

from examples.mixing.color import BLUE, YELLOW, COLOR_MODELS, concentration_colors
from solver.apic import SolverAPIC
from solver.apic.lsmps import LSMPSDiffusion


@dataclass
class Emitter:
    name: str
    center: tuple
    direction: tuple
    concentration: float
    color: tuple

    def tangent_axes(self):
        """Orthonormal nozzle plane, including emitters with a y component."""
        direction = np.asarray(self.direction, dtype=np.float32)
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = reference - np.dot(reference, direction)*direction
        u /= np.linalg.norm(u)
        return u, np.cross(direction, u)


class TankMixingSimulation:
    """Fixed storage with growing active array views, never particle recycling."""

    scene_kind = "tank"

    def __init__(self, resolution=96, device="cuda:0", fps=60, substeps=16,
                 diffusivity=3.0e-3, color_model="km", emitter_speed=1.2,
                 emitter_radius=0.075, emitter_duration=3.0, emitter_yaw_offset=20.0,
                 emission_mode="continuous", emitter_color_period=2.0,
                 initial_coffee_height=0.0):
        if resolution < 16 or fps < 1 or substeps < 1:
            raise ValueError("Tank resolution >=16, fps/substeps >=1 are required")
        if not all(math.isfinite(v) and v > 0 for v in
                   (emitter_speed, emitter_radius, emitter_duration)):
            raise ValueError("Emitter speed, radius and duration must be finite and positive")
        if emitter_radius > 0.12:
            raise ValueError("Emitter radius must be <=0.12 m inside this tank")
        if not math.isfinite(emitter_yaw_offset) or abs(emitter_yaw_offset) > 45.0:
            raise ValueError("emitter_yaw_offset must be finite and within [-45,45] degrees")
        if color_model not in COLOR_MODELS:
            raise ValueError(f"Unknown color model: {color_model}")
        if emission_mode not in ("continuous", "layered"):
            raise ValueError("emission_mode must be continuous or layered")
        if not math.isfinite(emitter_color_period) or emitter_color_period < 0.0:
            raise ValueError("emitter_color_period must be finite and nonnegative")
        if not math.isfinite(initial_coffee_height) or not 0.0 <= initial_coffee_height < 0.65:
            raise ValueError("initial_coffee_height must be finite and in [0,0.65) m")
        self.emission_mode = emission_mode
        self.emitter_color_period = emitter_color_period
        self.initial_coffee_height = initial_coffee_height
        self.resolution, self.fps, self.substeps = resolution, fps, substeps
        self.color_model, self.frame = color_model, 0
        self.bound_lo, self.bound_hi = (0.0, 0.0, 0.0), (1.2, 0.6, 0.65)
        self.spacing, self.dt = 1.2/resolution, 1.0/(fps*substeps)
        if 0.0 < emitter_color_period < self.dt:
            raise ValueError("Emitter color period must be at least one substep, or zero to disable")
        if emitter_radius < self.spacing:
            raise ValueError("Emitter radius must span at least one particle spacing; increase resolution")
        if self.dt*emitter_speed > 0.5*self.spacing:
            raise ValueError("Emitter moves more than half a particle spacing per substep; increase --substeps")
        self.volume = self.spacing**3
        self.emitter_speed, self.emitter_radius = emitter_speed, emitter_radius
        self.emitter_duration = emitter_duration
        self.emitter_yaw_offset = emitter_yaw_offset
        direction = 1.0/math.sqrt(2.0)
        yaw = math.radians(emitter_yaw_offset)
        inward, tangent = direction*math.cos(yaw), direction*math.sin(yaw)
        # Front/back skew injects angular momentum of the same sign about
        # the tank's vertical axis while retaining the 45-degree downward tilt.
        self.emitters = [Emitter("Yellow", (0.28, 0.3, 0.52), (inward, -tangent, -direction), 0.0, YELLOW),
                         Emitter("Blue", (0.92, 0.3, 0.52), (-inward, tangent, -direction), 1.0, BLUE)]
        self.emitter_enabled = np.array([initial_coffee_height <= 0.0, True], dtype=bool)
        axis = np.arange(-math.ceil(emitter_radius/self.spacing),
                         math.ceil(emitter_radius/self.spacing)+1)*self.spacing
        disc = np.array([(u, v) for u in axis for v in axis
                         if u*u+v*v <= emitter_radius**2+1.0e-12], dtype=np.float32)
        self.disc = disc
        self.particles_per_row = len(disc)
        self.period = self.spacing/emitter_speed
        self.total_rows = max(1, math.ceil(emitter_duration/self.period-1.0e-10))
        if emission_mode == "continuous":
            # Spread each layer's columns over a full period: emit a few
            # particle pairs each substep rather than a whole coherent disc.
            interval = self.period/self.particles_per_row
            times = (np.arange(math.ceil(emitter_duration/interval))+0.5)*interval
            self.emission_times = times[times < emitter_duration]
            order = np.random.default_rng(42).permutation(self.particles_per_row)
            self.emission_columns = order[np.arange(len(self.emission_times)) % self.particles_per_row]
        else:
            self.emission_times = np.repeat(np.arange(self.total_rows)*self.period, self.particles_per_row)
            self.emission_columns = np.tile(np.arange(self.particles_per_row), self.total_rows)
        if len(self.emission_times) == 0:
            raise ValueError("Injection duration is shorter than half a particle emission interval")
        initial_x = np.arange(0.5*self.spacing, self.bound_hi[0], self.spacing, dtype=np.float32)
        initial_y = np.arange(0.5*self.spacing, self.bound_hi[1], self.spacing, dtype=np.float32)
        initial_z = np.arange(0.5*self.spacing, initial_coffee_height, self.spacing, dtype=np.float32)
        if initial_coffee_height > 0.0 and len(initial_z):
            initial_positions = np.stack(np.meshgrid(initial_x, initial_y, initial_z, indexing="ij"), axis=-1)
            initial_positions = initial_positions.reshape(-1, 3).astype(np.float32)
        else:
            initial_positions = np.empty((0, 3), dtype=np.float32)
        self.initial_particle_count = len(initial_positions)
        self.capacity = self.initial_particle_count + int(self.emitter_enabled.sum())*len(self.emission_times)
        tank_volume = float(np.prod(np.subtract(self.bound_hi, self.bound_lo)))
        if self.capacity*self.volume > 0.8*tank_volume:
            raise ValueError("Requested inflow exceeds 80% of tank volume; shorten --emitter-duration")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverAPIC.register_custom_attributes(builder)
        frequency = getattr(newton, "ModelAttributeFrequency", None)
        assignment = getattr(newton, "ModelAttributeAssignment", None)
        if frequency is None:
            frequency, assignment = newton.Model.AttributeFrequency, newton.Model.AttributeAssignment
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name="concentration", frequency=frequency.PARTICLE, assignment=assignment.STATE,
            dtype=wp.float32, default=0.0, namespace="mixing"))
        builder.add_particles(pos=[[0.0, 0.0, 0.0]]*self.capacity,
                              vel=[[0.0, 0.0, 0.0]]*self.capacity,
                              mass=[1000.0*self.volume]*self.capacity,
                              radius=[0.5*self.spacing]*self.capacity,
                              custom_attributes={"apic:particle_volume": [self.volume]*self.capacity,
                                                 "mixing:concentration": [0.0]*self.capacity})
        self.model = builder.finalize(device=device)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self._states = (self.state_0, self.state_1)
        self._model_storage = {name: getattr(self.model, name) for name in (
            "particle_q", "particle_qd", "particle_mass", "particle_inv_mass", "particle_radius",
            "particle_flags", "particle_world")}
        self._volume_storage = self.model.apic.particle_volume
        self._state_storage = {id(s): (s.particle_q, s.particle_qd, s.particle_f,
                                      s.apic.C, s.mixing.concentration) for s in self._states}
        self._color_storage = wp.empty(self.capacity, dtype=wp.vec3, device=self.model.device)
        self.source_ids = np.empty(self.capacity, dtype=np.int32)
        self.birth_times = np.empty(self.capacity, dtype=np.float32)
        self.config = SolverAPIC.Config(
            voxel_size=1.5*self.spacing, bound_lo=self.bound_lo, bound_hi=self.bound_hi,
            wall_margin=0.5*self.spacing, wall_push_strength=0.0, wall_friction=0.0,
            # Preserve affine motion through the many small emitter substeps.
            # Keep a safety cap for violent impacts and a mild spacing correction.
            c_damping=1.0, c_norm_max=160.0,
            separation_rest_dist=self.spacing, separation_strength=0.10,
            separation_iterations=1, grid_capacity_ratio=2.0,
            projection_tol=1.0e-6, projection_quiet=True,
        )
        with wp.ScopedDevice(self.model.device):
            # Reserve the entire future tank domain, rather than sizing a
            # sparse grid from the first tiny emitter layer.
            h = self.config.voxel_size
            axes = [np.arange(math.floor(a/h)-1, math.ceil(b/h)+2)*h
                    for a, b in zip(self.bound_lo, self.bound_hi)]
            seeds = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
            self.solver = SolverAPIC(self.model, self.config,
                                     grid_capacity_positions=wp.array(seeds, dtype=wp.vec3, device=self.model.device))
        self.diffusion = LSMPSDiffusion(self.capacity, self.spacing, self.bound_lo, self.bound_hi,
                                        self.model.device, diffusivity=diffusivity,
                                        allow_degenerate=True, conservative=True)
        self.events_emitted, self.rows_emitted, self.sim_time, self.count = 0, 0.0, 0.0, 0
        self.initial_dye = 0.0
        self.emitted_dye_volume = 0.0
        self.last_diffusion_substeps = 0
        if self.initial_particle_count:
            initial = wp.array(initial_positions, dtype=wp.vec3, device=self.model.device)
            zero_velocity = wp.zeros(self.initial_particle_count, dtype=wp.vec3, device=self.model.device)
            zero_concentration = wp.zeros(self.initial_particle_count, dtype=wp.float32, device=self.model.device)
            for state in self._states:
                wp.copy(state.particle_q[:self.initial_particle_count], initial)
                wp.copy(state.particle_qd[:self.initial_particle_count], zero_velocity)
                state.particle_f[:self.initial_particle_count].zero_()
                state.apic.C[:self.initial_particle_count].zero_()
                wp.copy(state.mixing.concentration[:self.initial_particle_count], zero_concentration)
            self.source_ids[:self.initial_particle_count] = -1
            self.birth_times[:self.initial_particle_count] = 0.0
        self._select_active(self.initial_particle_count)

    @property
    def concentration(self):
        return self.state_0.mixing.concentration

    def emitter_concentration(self, emitter, time_s=None):
        """Swap inlet dyes periodically; existing particle dye is untouched."""
        time_s = self.sim_time if time_s is None else time_s
        phase = (int(math.floor((time_s+1.0e-10)/self.emitter_color_period)) % 2
                 if self.emitter_color_period else 0)
        return 1.0-emitter.concentration if phase else emitter.concentration

    def _select_active(self, count):
        self.count = count
        self.model.particle_count = count
        for name, storage in self._model_storage.items():
            setattr(self.model, name, storage[:count])
        self.model.apic.particle_volume = self._volume_storage[:count]
        for state in self._states:
            q, qd, f, C, c = self._state_storage[id(state)]
            state.particle_q, state.particle_qd, state.particle_f = q[:count], qd[:count], f[:count]
            state.apic.C, state.mixing.concentration = C[:count], c[:count]
        self.colors = self._color_storage[:count]
        self.diffusion.resize(count)

    def _emit_due(self):
        due = int(np.searchsorted(self.emission_times, self.sim_time+1.0e-10, side="right"))
        if due <= self.events_emitted:
            return
        positions, velocities, concentrations, sources, births = [], [], [], [], []
        columns = self.disc[self.emission_columns[self.events_emitted:due]]
        number = len(columns)
        for source, emitter in enumerate(self.emitters):
            if not self.emitter_enabled[source]:
                continue
            d = np.asarray(emitter.direction, dtype=np.float32)
            u, v = emitter.tangent_axes()
            position = np.asarray(emitter.center, dtype=np.float32) + columns[:, :1]*u + columns[:, 1:]*v
            positions.extend(position)
            velocities.extend(np.tile(d*self.emitter_speed, (number, 1)))
            concentrations.extend([self.emitter_concentration(emitter)]*number)
            sources.extend([source]*number)
            births.extend([self.sim_time]*number)
        start, stop = self.count, self.count+len(positions)
        if stop > self.capacity:
            raise RuntimeError("Emitter exhausted reserved particle storage")
        p = wp.array(np.asarray(positions), dtype=wp.vec3, device=self.model.device)
        vel = wp.array(np.asarray(velocities), dtype=wp.vec3, device=self.model.device)
        c = wp.array(np.asarray(concentrations, dtype=np.float32), device=self.model.device)
        for state in self._states:
            q, qd, f, C, scalar = self._state_storage[id(state)]
            wp.copy(q[start:stop], p)
            wp.copy(qd[start:stop], vel)
            f[start:stop].zero_()
            C[start:stop].zero_()
            wp.copy(scalar[start:stop], c)
        self.source_ids[start:stop] = sources
        self.birth_times[start:stop] = births
        self.events_emitted = due
        self.rows_emitted = due/self.particles_per_row
        self.emitted_dye_volume += float(np.sum(concentrations, dtype=np.float64)*self.volume)
        self._select_active(stop)

    def step(self):
        self.last_diffusion_substeps = 0
        for _ in range(self.substeps):
            self._emit_due()
            self.diffusion.step(self.state_0.particle_q, self.concentration, self.dt)
            self.last_diffusion_substeps += self.diffusion.last_substeps
            if self.count:
                self.solver.step(self.state_0, self.state_1, None, None, self.dt)
                wp.copy(self.state_1.mixing.concentration, self.concentration)
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.dt
        self.frame += 1
        return self.metrics()

    def update_colors(self):
        if self.count:
            wp.launch(concentration_colors, dim=self.count,
                      inputs=[self.concentration, self.colors, COLOR_MODELS[self.color_model]], device=self.model.device)

    def metrics(self):
        x, v, c = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy(), self.concentration.numpy()
        if not all(np.isfinite(a).all() for a in (x, v, c)):
            raise RuntimeError(f"Non-finite emitter tank state at frame {self.frame}")
        dye = float(c.astype(np.float64).sum()*self.volume)
        expected = (0.5*self.count*self.volume if self.initial_particle_count == 0
                    else self.initial_dye + self.emitted_dye_volume)
        variance = float(c.astype(np.float64).var()) if self.count else 0.0
        speed2 = np.sum(v*v, axis=1)
        angular = (x[:, 0]-0.6)*v[:, 1]-(x[:, 1]-0.3)*v[:, 0]
        height = self.count*self.volume/(1.2*0.6)
        pool = x[:, 2] < min(0.3, height+2.0*self.spacing)
        return dict(frame=self.frame, time_s=self.frame/self.fps, active_particles=self.count,
                    emitted_particles=self.count, emitted_rows_per_source=self.rows_emitted,
                    concentration_min=float(c.min()) if self.count else 0.0,
                    concentration_max=float(c.max()) if self.count else 0.0,
                    concentration_mean=float(c.mean()) if self.count else 0.0,
                    concentration_variance=variance, dye_integral_m3=dye,
                    expected_dye_integral_m3=expected,
                    dye_relative_error=(dye-expected)/expected if expected else 0.0,
                    mixing_index=1.0-math.sqrt(variance/0.25) if self.count else 0.0,
                    maximum_speed_m_s=float(np.linalg.norm(v, axis=1).max()) if self.count else 0.0,
                    rms_speed_m_s=float(np.sqrt(speed2.mean())) if self.count else 0.0,
                    kinetic_energy_j=float(0.5*1000.0*self.volume*speed2.astype(float).sum()),
                    specific_angular_momentum_z_m2_s=float(angular.mean()) if self.count else 0.0,
                    pool_specific_angular_momentum_z_m2_s=float(angular[pool].mean()) if np.any(pool) else 0.0,
                    diffusion_substeps=self.last_diffusion_substeps,
                    minimum_fit_pivot=self.diffusion.last_min_pivot,
                    diffusion_quadratic_particles=self.diffusion.last_quadratic,
                    diffusion_skipped_particles=self.diffusion.last_skipped,
                    emitted_rest_volume_m3=self.count*self.volume,
                    expected_settled_height_m=self.count*self.volume/(1.2*0.6))

    def report(self):
        return dict(scene="tank", gravity_m_s2=[0.0, 0.0, -9.81], particle_capacity=self.capacity,
                    emitter_speed_m_s=self.emitter_speed, emitter_radius_m=self.emitter_radius,
                    emitter_duration_s=self.emitter_duration, emitter_period_s=self.period,
                    emitter_yaw_offset_deg=self.emitter_yaw_offset,
                    emission_mode=self.emission_mode,
                    emitter_color_period_s=self.emitter_color_period,
                    initial_coffee_height_m=self.initial_coffee_height,
                    initial_particle_count=self.initial_particle_count,
                    emitter_enabled=self.emitter_enabled.tolist(),
                    particles_per_emitter_row=self.particles_per_row,
                    emitters=[vars(e) for e in self.emitters],
                    scalar_stabilization="positive symmetric LSMPS pair flux; quadratic fallback; skip unresolved fits")
