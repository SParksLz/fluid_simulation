"""Newton ViewerGL demo for SolverAPIC wave-tank with particle surface reconstruction."""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

import numpy as np
import warp as wp

# Compatibility shim for Newton builds that import modern Warp symbols directly.
if not hasattr(wp, "Device"):
    try:
        from warp.context import Device as _Device
    except Exception:
        _Device = Any
    wp.Device = _Device

if not hasattr(wp, "DeviceLike"):
    try:
        from warp.context import Devicelike as _DeviceLike
    except Exception:
        _DeviceLike = Any
    wp.DeviceLike = _DeviceLike

import newton
from newton.geometry import ParticleSurface
from newton.viewer import ViewerRTX
from solver.apic import SolverAPIC


def spawn_particle_block(lo, hi, res, density: float = 1000.0):
    xs = np.linspace(lo[0], hi[0], res[0])
    ys = np.linspace(lo[1], hi[1], res[1])
    zs = np.linspace(lo[2], hi[2], res[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    spacing = float(
        min(
            (hi[0] - lo[0]) / max(res[0] - 1, 1),
            (hi[1] - lo[1]) / max(res[1] - 1, 1),
            (hi[2] - lo[2]) / max(res[2] - 1, 1),
        )
    )
    volume = spacing**3
    mass = density * volume
    radius = 0.5 * spacing
    return positions, mass, radius, volume, spacing


def spawn_wave_tank(
    tank_lo,
    tank_hi,
    target_count: int = 400000,
    fill_x_frac: float = 0.30,
    fill_z_frac: float = 0.70,
    margin: float = 0.03,
    density: float = 1000.0,
):
    """Rectangular tank with a water column stacked on the -X side (dam-break / wave)."""
    tank_lo = np.asarray(tank_lo, dtype=np.float32)
    tank_hi = np.asarray(tank_hi, dtype=np.float32)
    extent = tank_hi - tank_lo

    water_lo = np.array(
        [
            tank_lo[0] + margin,
            tank_lo[1] + margin,
            tank_lo[2] + margin,
        ],
        dtype=np.float32,
    )
    water_hi = np.array(
        [
            tank_lo[0] + margin + fill_x_frac * max(extent[0] - 2.0 * margin, 1.0e-3),
            tank_hi[1] - margin,
            tank_lo[2] + margin + fill_z_frac * max(extent[2] - 2.0 * margin, 1.0e-3),
        ],
        dtype=np.float32,
    )
    water_ext = np.maximum(water_hi - water_lo, 1.0e-3)
    water_volume = float(np.prod(water_ext))
    spacing = float((water_volume / max(target_count, 1)) ** (1.0 / 3.0))

    res = (
        max(2, int(round(water_ext[0] / spacing))),
        max(2, int(round(water_ext[1] / spacing))),
        max(2, int(round(water_ext[2] / spacing))),
    )
    return spawn_particle_block(tuple(water_lo), tuple(water_hi), res, density=density)


class NewtonAPICTest:
    def __init__(
        self,
        device: str = "cuda:0",
        particle_count: int = 400000,
        voxel_size: float | None = None,
        show_grid: bool = False,
        show_surface: bool = True,
        show_particles: bool = False,
        surface_voxel_size: float | None = None,
        anisotropic: bool = True,
    ) -> None:
        self.device = device
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sub_step_num = 4
        self.sim_dt = self.frame_dt / float(self.sub_step_num)
        self.gravity = -10.0
        self.show_grid = bool(show_grid)
        self.show_domain_box = True
        self.show_surface = bool(show_surface)
        self.show_particles = bool(show_particles)
        self.grid_line_width = 0.0025
        self.grid_color = (0.85, 0.75, 0.25)
        self.enable_projection = True
        self.surface_triangle_count = 0

        # Long rectangular wave tank (X long, Y narrow, Z up).
        self.bound_lo = (-1.20, -0.28, 0.0)
        self.bound_hi = (1.20, 0.28, 1.20)

        positions, mass, radius, volume, spacing = spawn_wave_tank(
            tank_lo=self.bound_lo,
            tank_hi=self.bound_hi,
            target_count=int(particle_count),
            fill_x_frac=0.30,
            fill_z_frac=0.72,
            margin=0.03,
        )
        self.particle_radius = radius
        self.particle_spacing = float(spacing)
        self.particle_count = positions.shape[0]
        if voxel_size is None:
            # Keep ~1.5 particles per axis; avoid the old 0.025 floor that
            # made 100k–400k runs project on an overly coarse grid.
            voxel_size = max(spacing * 1.5, 0.010)
        self.voxel_size = float(voxel_size)

        self.model = self.build_model(positions, mass, radius, volume)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.model.set_gravity((0.0, 0.0, self.gravity))

        self.solver = SolverAPIC(
            self.model,
            SolverAPIC.Config(
                voxel_size=self.voxel_size,
                bound_lo=self.bound_lo,
                bound_hi=self.bound_hi,
                clamp_eps=1.0e-3,
                grid_capacity_ratio=16.0,
                enable_projection=True,
                wall_friction=0.0,
                projection_tol=1.0e-6,
                projection_max_iters=1000,
            ),
        )

        # Surface params (ParticleSurface created lazily; skipped with --no-surface).
        # Tunable live from the UI — changing them rebuilds ParticleSurface + CUDA graph.
        # Defaults tuned for this wave-tank look (anisotropic Yu-Turk).
        if surface_voxel_size is None:
            surface_voxel_size = 0.007
        self.surface_voxel_size = float(surface_voxel_size)
        self.surface_kernel_radius = 0.007
        self.surface_threshold = 0.441
        self.surface_kernel_scale = 0.801
        self.surface_anisotropic = bool(anisotropic)
        self.surface_anisotropy_ratio = 12.768
        self.surface_anisotropy_scale = 1.338
        self.surface_anisotropy_strength = 0.871
        self.surface_anisotropy_min_neighbors = 4
        self.surface_field_smooth_iterations = 1
        self.surface_field_smooth_radius = 2
        self.surface_mesh_smooth_iterations = 1
        self.surface = None
        self.surface_max_grid_cells = 0
        self.surface_path = "/apic/water_surface"
        self.surface_mesh = None
        self.surface_graph = None
        self._surface_logged = False
        self._last_surface_verts = None
        self._last_surface_indices = None
        self._last_surface_normals = None
        self._rtx_water_material_bound = False
        self.water_opacity = 0.35
        if self.show_surface:
            self._ensure_surface()

        # Side view of the long tank so the wave travels left -> right.
        self.camera_pos = wp.vec3(0.0, -2.6, 0.70)
        self.camera_pitch = -0.18
        self.camera_yaw = 90.0
        self.viewer = None
        self.render_x = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.point_radii = wp.full(self.particle_count, self.particle_radius, dtype=wp.float32, device=self.device)
        self.colors = wp.full(self.particle_count, wp.vec3(0.25, 0.55, 0.95), dtype=wp.vec3, device=self.device)
        self.show_ideal_height = True
        tank_area = float(
            (self.bound_hi[0] - self.bound_lo[0]) * (self.bound_hi[1] - self.bound_lo[1])
        )
        total_fluid_volume = float(self.particle_count) * float(volume)
        self.ideal_height = total_fluid_volume / max(tank_area, 1.0e-8)
        self._domain_box_starts, self._domain_box_ends = self._make_domain_box_lines()
        self._ideal_height_starts, self._ideal_height_ends = self._make_ideal_height_lines()

    def _make_domain_box_lines(self) -> tuple[wp.array, wp.array]:
        lo = np.array(self.bound_lo, dtype=np.float32)
        hi = np.array(self.bound_hi, dtype=np.float32)
        corners = np.array(
            [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]],
                [lo[0], hi[1], hi[2]],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        starts = np.array([corners[a] for a, _ in edges], dtype=np.float32)
        ends = np.array([corners[b] for _, b in edges], dtype=np.float32)
        return (
            wp.array(starts, dtype=wp.vec3, device=self.device),
            wp.array(ends, dtype=wp.vec3, device=self.device),
        )

    def _make_ideal_height_lines(self) -> tuple[wp.array, wp.array]:
        """Rectangle at z = ideal flat-fill height (volume / tank floor area)."""
        lo = np.array(self.bound_lo, dtype=np.float32)
        hi = np.array(self.bound_hi, dtype=np.float32)
        z = float(np.clip(self.ideal_height, lo[2], hi[2]))
        corners = np.array(
            [
                [lo[0], lo[1], z],
                [hi[0], lo[1], z],
                [hi[0], hi[1], z],
                [lo[0], hi[1], z],
            ],
            dtype=np.float32,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        starts = np.array([corners[a] for a, _ in edges], dtype=np.float32)
        ends = np.array([corners[b] for _, b in edges], dtype=np.float32)
        return (
            wp.array(starts, dtype=wp.vec3, device=self.device),
            wp.array(ends, dtype=wp.vec3, device=self.device),
        )

    def build_model(self, positions, mass, radius, volume, velocities=None) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_particle_radius = float(radius)
        SolverAPIC.register_custom_attributes(builder)
        n = positions.shape[0]
        if velocities is None:
            velocities = np.zeros_like(positions)
        builder.add_particles(
            pos=positions.tolist(),
            vel=np.asarray(velocities, dtype=np.float32).tolist(),
            mass=[float(mass)] * n,
            radius=[float(radius)] * n,
            custom_attributes={"apic:particle_volume": [float(volume)] * n},
        )
        # Visual ground for RTX/GL lighting (solver still uses AABB walls).
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0),
        )
        return builder.finalize(device=self.device)

    def _rebuild_surface(self) -> None:
        """Drop and recreate ParticleSurface from current UI params."""
        self.surface = None
        self.surface_graph = None
        self.surface_mesh = None
        self._ensure_surface()

    def _ensure_surface(self) -> None:
        """Create ParticleSurface on first use (same knobs as MPM water dam break)."""
        if self.surface is not None:
            return
        surface_voxel_size = max(float(self.surface_voxel_size), 1.0e-4)
        self.surface_voxel_size = surface_voxel_size
        kernel_radius = max(float(self.surface_kernel_radius), surface_voxel_size)
        self.surface_kernel_radius = kernel_radius
        pad = 4.0 * surface_voxel_size + kernel_radius
        tank_ext = np.asarray(self.bound_hi, dtype=np.float64) - np.asarray(self.bound_lo, dtype=np.float64)
        est_cells = int(np.prod(np.ceil((tank_ext + 2.0 * pad) / surface_voxel_size)))
        surface_max_grid_cells = max(8_000_000, int(1.5 * est_cells))
        anisotropic = bool(self.surface_anisotropic)
        self.surface = ParticleSurface(
            voxel_size=surface_voxel_size,
            max_grid_cells=surface_max_grid_cells,
            kernel_radius=kernel_radius,
            threshold=float(self.surface_threshold),
            smooth_lambda=0.0,
            anisotropic=anisotropic,
            kernel_scale=float(self.surface_kernel_scale),
            anisotropy_ratio=float(self.surface_anisotropy_ratio),
            anisotropy_scale=float(self.surface_anisotropy_scale),
            anisotropy_min_neighbors=int(self.surface_anisotropy_min_neighbors),
            anisotropy_binning=anisotropic,
            anisotropy_strength=float(self.surface_anisotropy_strength),
            field_smooth_iterations=int(self.surface_field_smooth_iterations),
            field_smooth_radius=int(self.surface_field_smooth_radius),
            field_mode="density",
            redistance_iterations=0,
            mesh_smooth_iterations=int(self.surface_mesh_smooth_iterations),
            device=self.model.device,
        )
        self.surface_max_grid_cells = surface_max_grid_cells
        self._capture_surface_extraction()

    def _extract_surface(self) -> ParticleSurface.ExtractionMesh:
        self._ensure_surface()
        return self.surface.extract(
            self.state_0.particle_q,
            self.model.particle_radius,
            compute_normals=True,
            particle_flags=self.model.particle_flags,
        )

    def _capture_surface_extraction(self) -> None:
        self.surface_graph = None
        self.surface_mesh = None
        if self.surface is None:
            return
        if not self.model.device.is_cuda:
            return
        if self.sub_step_num % 2 != 0:
            warnings.warn("Sim substeps must be even for graph capture of surface extraction", stacklevel=2)
            return
        self.surface_mesh = self._extract_surface()
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.surface_mesh = self._extract_surface()
        self.surface_graph = capture.graph

    def _bind_rtx_water_material(self) -> None:
        """Transparent water UsdPreviewSurface for ViewerRTX (same as MPM dam-break)."""
        if self._rtx_water_material_bound or not isinstance(self.viewer, ViewerRTX):
            return
        try:
            from pxr import Sdf, UsdShade
        except ImportError:
            return

        mesh_prim = self.viewer.stage.GetPrimAtPath(f"/root{self.surface_path}")
        if not mesh_prim:
            return

        material_path = "/root/Materials/Water"
        material = UsdShade.Material.Define(self.viewer.stage, material_path)
        shader = UsdShade.Shader.Define(self.viewer.stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.12, 0.42, 0.65))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(self.water_opacity))
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.03)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
        self._rtx_water_material_bound = True

    def _log_surface_mesh(self, verts, indices, normals, *, hidden: bool) -> None:
        # ViewerGL crashes on 0-vertex dynamic mesh updates — never upload empty buffers.
        mesh_kwargs = dict(
            hidden=True,
            dynamic=True,
            color=(0.12, 0.42, 0.8),
            roughness=0.05,
            metallic=0.0,
            opacity=float(self.water_opacity),
        )
        if verts is None or indices is None or int(verts.shape[0]) == 0 or int(indices.shape[0]) == 0:
            if self._surface_logged and self._last_surface_verts is not None:
                self.viewer.log_mesh(
                    self.surface_path,
                    self._last_surface_verts,
                    self._last_surface_indices,
                    self._last_surface_normals,
                    **mesh_kwargs,
                )
            self.surface_triangle_count = 0
            return
        self._last_surface_verts = verts
        self._last_surface_indices = indices
        self._last_surface_normals = normals
        self._surface_logged = True
        self.surface_triangle_count = 0 if hidden else int(indices.shape[0]) // 3
        mesh_kwargs["hidden"] = hidden
        self.viewer.log_mesh(
            self.surface_path,
            verts,
            indices,
            normals,
            **mesh_kwargs,
        )
        self._bind_rtx_water_material()

    def step(self) -> None:
        for _ in range(self.sub_step_num):
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def _ui_apic_panel(self, imgui) -> None:
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("APIC Visualizer"):
            changed_surf, self.show_surface = imgui.checkbox("Show Water Surface", self.show_surface)
            if changed_surf and self.show_surface:
                self._ensure_surface()
            _c0b, self.show_particles = imgui.checkbox("Show Particles", self.show_particles)
            _c1, self.show_grid = imgui.checkbox("Show Nanogrid", self.show_grid)
            _c2, self.show_domain_box = imgui.checkbox("Show Domain Box", self.show_domain_box)
            _c2b, self.show_ideal_height = imgui.checkbox("Show Ideal Height", self.show_ideal_height)
            imgui.text(f"Ideal flat height: {self.ideal_height:.4f} m")
            _c3, self.grid_line_width = imgui.slider_float(
                "Grid Line Width", self.grid_line_width, 0.0005, 0.02
            )
            changed_proj, self.enable_projection = imgui.checkbox(
                "Enable Pressure Projection", self.enable_projection
            )
            if changed_proj:
                self.solver.config.enable_projection = bool(self.enable_projection)
            imgui.text(f"Surface tris: {self.surface_triangle_count}")
            if getattr(self.solver, "_grid", None) is not None:
                try:
                    cell_count = int(self.solver._grid.cell_count())
                except Exception:
                    cell_count = -1
                imgui.text(f"Active cells: {cell_count}")
                imgui.text(f"Sim voxel: {self.solver.config.voxel_size:.4f}")
                imgui.text(
                    f"Projection iters: {self.solver.last_projection_iters}  "
                    f"residual: {self.solver.last_projection_residual:.3e}"
                )
                imgui.text(
                    f"C damp={self.solver.config.c_damping:.3f}  "
                    f"|C|max={self.solver.config.c_norm_max:.1f}  "
                    f"vmax={self.solver.config.max_particle_speed:.1f}"
                )
                try:
                    c_np = self.state_0.apic.C.numpy().reshape(self.particle_count, -1)
                    c_mean = float(np.linalg.norm(c_np, axis=1).mean())
                    imgui.text(f"mean |C|={c_mean:.3f}")
                except Exception:
                    pass

        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Surface Reconstruction"):
            rebuild = False
            c, self.surface_voxel_size = imgui.slider_float(
                "Surf Voxel", self.surface_voxel_size, 0.004, 0.04
            )
            rebuild = rebuild or c
            c, self.surface_kernel_radius = imgui.slider_float(
                "Kernel Radius", self.surface_kernel_radius, 0.01, 0.12
            )
            rebuild = rebuild or c
            c, self.surface_threshold = imgui.slider_float(
                "Iso Threshold", self.surface_threshold, 0.05, 0.8
            )
            rebuild = rebuild or c
            c, self.surface_kernel_scale = imgui.slider_float(
                "Kernel Scale", self.surface_kernel_scale, 0.2, 1.5
            )
            rebuild = rebuild or c
            c, self.surface_anisotropic = imgui.checkbox("Anisotropic", self.surface_anisotropic)
            rebuild = rebuild or c
            if self.surface_anisotropic:
                c, self.surface_anisotropy_ratio = imgui.slider_float(
                    "Aniso Ratio", self.surface_anisotropy_ratio, 1.0, 32.0
                )
                rebuild = rebuild or c
                c, self.surface_anisotropy_scale = imgui.slider_float(
                    "Aniso Scale", self.surface_anisotropy_scale, 0.5, 4.0
                )
                rebuild = rebuild or c
                c, self.surface_anisotropy_strength = imgui.slider_float(
                    "Aniso Strength", self.surface_anisotropy_strength, 0.0, 1.0
                )
                rebuild = rebuild or c
                c, self.surface_anisotropy_min_neighbors = imgui.slider_int(
                    "Aniso Min Neighbors", int(self.surface_anisotropy_min_neighbors), 1, 32
                )
                rebuild = rebuild or c
            c, self.surface_field_smooth_iterations = imgui.slider_int(
                "Field Smooth Iters", int(self.surface_field_smooth_iterations), 0, 8
            )
            rebuild = rebuild or c
            c, self.surface_field_smooth_radius = imgui.slider_int(
                "Field Smooth Radius", int(self.surface_field_smooth_radius), 1, 4
            )
            rebuild = rebuild or c
            c, self.surface_mesh_smooth_iterations = imgui.slider_int(
                "Mesh Smooth Iters", int(self.surface_mesh_smooth_iterations), 0, 8
            )
            rebuild = rebuild or c
            if imgui.button("Reset Surface Defaults"):
                self.surface_voxel_size = 0.007
                self.surface_kernel_radius = 0.007
                self.surface_threshold = 0.441
                self.surface_kernel_scale = 0.801
                self.surface_anisotropic = True
                self.surface_anisotropy_ratio = 12.768
                self.surface_anisotropy_scale = 1.338
                self.surface_anisotropy_strength = 0.871
                self.surface_anisotropy_min_neighbors = 4
                self.surface_field_smooth_iterations = 1
                self.surface_field_smooth_radius = 2
                self.surface_mesh_smooth_iterations = 1
                rebuild = True
            if rebuild and self.show_surface:
                self._rebuild_surface()
            imgui.text(f"max_cells={self.surface_max_grid_cells}  tris={self.surface_triangle_count}")

    def _is_usd_viewer(self) -> bool:
        return isinstance(self.viewer, newton.viewer.ViewerUSD)

    def render(self) -> None:
        if self.viewer is None:
            return
        is_usd = self._is_usd_viewer()
        if self.show_particles or not is_usd:
            wp.copy(self.render_x, self.state_0.particle_q)
        self.viewer.begin_frame(self.sim_time)

        # USD: never call log_state — ViewerBase._log_particles would dump every
        # particle into a PointInstancer each frame (GBs / hang at 400k × thousands).
        if not is_usd and hasattr(self.viewer, "log_state"):
            try:
                self.viewer.log_state(self.state_0)
            except Exception:
                pass

        # Skip log_points entirely when hidden: ViewerUSD still time-samples
        # positions/scales even with hidden=True.
        if self.show_particles:
            self.viewer.log_points(
                "/apic/particles",
                points=self.render_x,
                radii=self.point_radii,
                colors=self.colors,
                hidden=False,
            )

        if self.show_surface:
            self._ensure_surface()
            if self.surface_graph is None:
                self.surface_mesh = self._extract_surface()
            else:
                wp.capture_launch(self.surface_graph)
            verts, indices, normals = self.surface_mesh.to_arrays()
            self._log_surface_mesh(verts, indices, normals, hidden=False)
        elif not is_usd:
            self._log_surface_mesh(None, None, None, hidden=True)

        if self.show_domain_box:
            self.viewer.log_lines(
                "/apic/domain_box",
                starts=self._domain_box_starts,
                ends=self._domain_box_ends,
                colors=(0.55, 0.55, 0.6),
                width=0.004,
                hidden=False,
            )
        elif not is_usd:
            self.viewer.log_lines(
                "/apic/domain_box",
                starts=None,
                ends=None,
                colors=None,
                hidden=True,
            )

        # Ideal-height / nanogrid overlays: skip on USD (line PointInstancers bloat).
        if not is_usd:
            if self.show_ideal_height:
                self.viewer.log_lines(
                    "/apic/ideal_height",
                    starts=self._ideal_height_starts,
                    ends=self._ideal_height_ends,
                    colors=(1.0, 0.35, 0.15),
                    width=0.006,
                    hidden=False,
                )
            else:
                self.viewer.log_lines(
                    "/apic/ideal_height",
                    starts=None,
                    ends=None,
                    colors=None,
                    hidden=True,
                )

            if self.show_grid:
                starts, ends = self.solver.get_grid_wireframe()
                if starts is not None and ends is not None and starts.shape[0] > 0:
                    self.viewer.log_lines(
                        "/apic/nanogrid",
                        starts=starts,
                        ends=ends,
                        colors=self.grid_color,
                        width=float(self.grid_line_width),
                        hidden=False,
                    )
                else:
                    self.viewer.log_lines(
                        "/apic/nanogrid",
                        starts=None,
                        ends=None,
                        colors=None,
                        hidden=True,
                    )
            else:
                self.viewer.log_lines(
                    "/apic/nanogrid",
                    starts=None,
                    ends=None,
                    colors=None,
                    hidden=True,
                )

        self.viewer.end_frame()

    def setup_viewer(self, viewer: str = "gl", output_path: str | None = None, num_frames: int | None = None) -> None:
        if viewer == "usd":
            if not output_path:
                raise ValueError("--output-path / --output is required when using --viewer usd")
            # Prefer binary crate for large dynamic-mesh recordings.
            if output_path.endswith(".usda"):
                output_path = output_path[:-5] + ".usdc"
                print(f"Note: writing binary USD ({output_path}) instead of ASCII .usda")
            # Cap long recordings: dynamic water mesh topology is time-sampled every frame.
            nframes = int(num_frames) if num_frames is not None and num_frames > 0 else 300
            if nframes >= 6000:
                print(f"Warning: --frames {nframes} too large for USD; clamping to 300")
                nframes = 300
            elif nframes > 600:
                print(
                    f"Warning: {nframes} USD frames with dynamic surface is heavy "
                    f"(~tens of MB/frame). Prefer <=300 for a quick export."
                )
            self.viewer = newton.viewer.ViewerUSD(
                output_path=output_path,
                fps=60,
                up_axis="Z",
                num_frames=nframes,
            )
            # Lean export: surface mesh only (no model particles via log_state).
            self.show_particles = False
            self.show_surface = True
            self.show_grid = False
            self.show_ideal_height = False
            if hasattr(self.viewer, "show_particles"):
                self.viewer.show_particles = False
            self._ensure_surface()
            print(f"Recording USD to {output_path} ({nframes} frames, surface-only)")
        elif viewer == "rtx":
            # studio = brighter key/fill lights; default dome alone looks nearly black
            # with only a dark translucent water mesh and no opaque scene props.
            self.viewer = ViewerRTX(paused=False, fps=60, up_axis="Z", environment="studio")
            if hasattr(self.viewer, "show_visual"):
                self.viewer.show_visual = True
            if hasattr(self.viewer, "show_ground"):
                self.viewer.show_ground = True
            if hasattr(self.viewer, "show_particles"):
                self.viewer.show_particles = bool(self.show_particles)
            self.show_particles = False
            self.show_surface = True
            self.show_domain_box = True
            self._ensure_surface()
            self._rtx_water_material_bound = False
            # Slightly more opaque so water reads against the lit ground.
            if float(self.water_opacity) < 0.45:
                self.water_opacity = 0.45
        elif viewer == "gl":
            self.viewer = newton.viewer.ViewerGL()
        else:
            raise ValueError(f"Unsupported viewer: {viewer}")

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(self.camera_pos, self.camera_pitch, self.camera_yaw)
        if viewer in ("gl", "rtx") and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._ui_apic_panel, position="panel")

    def run(self, frames: int) -> None:
        frame = 0
        # USD viewer stops via is_running()/num_frames; GL uses frames as a soft cap.
        max_frames = frames if frames > 0 else 10**9
        if self._is_usd_viewer() and getattr(self.viewer, "num_frames", None):
            max_frames = min(max_frames, int(self.viewer.num_frames))
        report_every = 10 if self._is_usd_viewer() else 0
        while self.viewer.is_running() and frame < max_frames:
            # Honor ViewerGL Pause / Space / single-step (.) like newton.examples.run.
            should = True
            if hasattr(self.viewer, "should_step"):
                should = bool(self.viewer.should_step())
            if should:
                self.step()
                frame += 1
            self.render()
            if report_every and should and frame % report_every == 0:
                tris = int(self.surface_triangle_count)
                print(f"USD frame {frame}/{max_frames}  surf_tris={tris}", flush=True)
        self.viewer.close()
        if hasattr(self.viewer, "output_path") and self.viewer.output_path:
            path = self.viewer.output_path
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"USD written: {path} ({size_mb:.1f} MB)")
            except OSError:
                print(f"USD written: {path}")


def main():
    parser = argparse.ArgumentParser(description="Newton APIC wave-tank demo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--particles",
        type=int,
        default=400000,
        help="Target particle count for the side water column",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=None,
        help="Deprecated alias: if set, uses res^3 as --particles",
    )
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--frames", type=int, default=6000, help="Max frames for viewer; used by --no-viewer too")
    parser.add_argument("--no-viewer", action="store_true", help="Headless smoke run")
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "usd", "rtx"],
        help="gl = ViewerGL; usd = record USD; rtx = ViewerRTX with transparent water material",
    )
    parser.add_argument(
        "--water-opacity",
        type=float,
        default=0.35,
        help="Water mesh opacity for RTX / USD PreviewSurface [0,1]",
    )
    parser.add_argument(
        "--output-path",
        "--output",
        type=str,
        default="apic_wave_tank.usdc",
        dest="output_path",
        help="USD output path when --viewer usd (alias: --output). Prefer .usdc",
    )
    parser.add_argument("--show-grid", action="store_true", help="Enable Nanogrid wireframe by default")
    parser.add_argument("--no-projection", action="store_true", help="Disable pressure projection")
    parser.add_argument("--no-surface", action="store_true", help="Disable particle surface reconstruction")
    parser.add_argument("--show-particles", action="store_true", help="Show particle points (surface on by default)")
    parser.add_argument("--surface-voxel-size", type=float, default=None, help="Surface grid voxel size [m]")
    parser.add_argument("--no-anisotropic", action="store_true", help="Disable anisotropic Yu-Turk kernels")
    args = parser.parse_args()

    show_grid = bool(args.show_grid)
    particle_count = int(args.res) ** 3 if args.res is not None else int(args.particles)
    app = NewtonAPICTest(
        device=args.device,
        particle_count=particle_count,
        voxel_size=args.voxel_size,
        show_grid=show_grid,
        show_surface=not args.no_surface,
        show_particles=bool(args.show_particles) or bool(args.no_surface),
        surface_voxel_size=args.surface_voxel_size,
        anisotropic=not args.no_anisotropic,
    )
    if args.no_projection:
        app.enable_projection = False
        app.solver.config.enable_projection = False
    app.water_opacity = float(np.clip(args.water_opacity, 0.0, 1.0))
    q0 = app.state_0.particle_q.numpy()
    print(
        f"APIC wave-tank: {app.particle_count} particles on {args.device} "
        f"(show_grid={show_grid}, projection={app.solver.config.enable_projection}, "
        f"surface={app.show_surface})"
    )
    print(
        f"  tank=[{app.bound_lo} .. {app.bound_hi}]  "
        f"water x=[{q0[:, 0].min():.3f}, {q0[:, 0].max():.3f}]  "
        f"z=[{q0[:, 2].min():.3f}, {q0[:, 2].max():.3f}]"
    )
    print(f"  ideal_flat_height={app.ideal_height:.4f} m  (volume / tank floor area)")
    print(
        f"  voxel={app.solver.config.voxel_size:.4f}  "
        f"substeps={app.sub_step_num}  "
        f"proj_tol={app.solver.config.projection_tol:g}  "
        f"proj_max_iters={app.solver.config.projection_max_iters}"
    )
    if app.surface is None:
        print("  surface=disabled (lazy-init if toggled on in UI)")
    else:
        print(
            f"  surface_voxel={app.surface.voxel_size:.4f}  "
            f"max_cells={app.surface_max_grid_cells}  "
            f"anisotropic={app.surface_anisotropic}  "
            f"graph={app.surface_graph is not None}"
        )

    if args.no_viewer:
        n_frames = args.frames if args.frames > 0 else 30
        if args.frames >= 6000:
            n_frames = 30
        for i in range(n_frames):
            app.step()
            if (i + 1) % 10 == 0:
                q = app.state_0.particle_q.numpy()
                tris = 0
                if app.show_surface:
                    app._ensure_surface()
                    if app.surface_graph is None:
                        app.surface_mesh = app._extract_surface()
                    else:
                        wp.capture_launch(app.surface_graph)
                    _v, indices, _n = app.surface_mesh.to_arrays()
                    tris = 0 if indices is None else int(indices.shape[0]) // 3
                print(
                    f"frame {i + 1}/{n_frames}  z_mean={q[:, 2].mean():.4f}  "
                    f"z_min={q[:, 2].min():.4f}  z_span={q[:, 2].max() - q[:, 2].min():.4f}  "
                    f"surf_tris={tris}  proj_iters={app.solver.last_projection_iters}  "
                    f"proj_res={app.solver.last_projection_residual:.2e}  "
                    f"finite={np.isfinite(q).all()}"
                )
        print("Headless APIC run finished.")
        return

    n_frames = args.frames
    # USD: default --frames 6000 would hang; clamp unless user picked a finite clip.
    if args.viewer == "usd" and n_frames >= 6000:
        print("USD: default --frames is huge; using 300 (pass --frames N explicitly for more)")
        n_frames = 300
    app.setup_viewer(viewer=args.viewer, output_path=args.output_path, num_frames=n_frames)
    app.run(n_frames)


if __name__ == "__main__":
    main()
