"""OVRTX viewer with a CUDA screen-space fluid Sensor Processing Graph."""

import math

import numpy as np
import warp as wp

from examples.mixing.ssfr import FluidBuffers, SSFRConfig, build_graph
from examples.mixing.viewer_rtx import MixingViewerRTX


class ScreenSpaceViewerRTX(MixingViewerRTX):
    def __init__(self, *, particle_radius, color_model=1, ssfr_config=None, device="cuda:0", **kwargs):
        super().__init__(**kwargs)
        self.ssfr_config = ssfr_config or SSFRConfig()
        self.fluid_buffers = FluidBuffers(self.camera.width, self.camera.height, device,
                                         self.ssfr_config, particle_radius=particle_radius)
        self._ssfr_radius = particle_radius
        self._ssfr_color_model = color_model
        self._ssfr_graph = None
        self._ssfr_particles = None
        self._ssfr_uniforms = {}

    def log_fluid_ssfr(self, points, radii, concentration):
        if not points.device.is_cuda:
            raise ValueError("SSFR requires CUDA particle arrays")
        event = wp.get_stream(points.device).record_event()
        self._ssfr_particles = ((points, radii, concentration), event)

    def _add_camera_lights_and_render_product(self):
        # SPG resolves native AOV resources inside the consuming RenderProduct.
        self._render_var_path = self._render_product_path + "/LdrColor"
        super()._add_camera_lights_and_render_product()
        self._ssfr_graph = build_graph(self, self.ssfr_config, self._ssfr_radius, self._ssfr_color_model)

    def _update_ovrtx_mesh_points(self):
        super()._update_ovrtx_mesh_points()
        if self._ssfr_particles is None:
            raise RuntimeError("Call log_fluid_ssfr before rendering the SSFR viewer")
        particles, event = self._ssfr_particles
        wp.get_stream(self.fluid_buffers.device).wait_event(event)
        matrix = self._compute_camera_matrix()
        tan_half_fov = math.tan(math.radians(self.camera.fov)/2)
        self.fluid_buffers.prepare(*particles, matrix, self.camera.near,
                                   self.camera.far, tan_half_fov)
        raster = self._ssfr_graph["raster"]
        uniforms = {(raster, key): np.asarray([value], dtype=np.uint64)
                    for key, value in self.fluid_buffers.pointers.items()}
        for path in (raster, *self._ssfr_graph["filters"], self._ssfr_graph["shade"]):
            uniforms[path, "tanHalfFov"] = np.asarray([tan_half_fov], dtype=np.float32)
        for key, value in (("right", matrix[0, :3]), ("up", matrix[1, :3]), ("forward", -matrix[2, :3])):
            uniforms[self._ssfr_graph["shade"], key] = np.asarray([value], dtype=np.float32)
        for key, value in (("near", self.camera.near), ("far", self.camera.far)):
            uniforms[raster, key] = np.asarray([value], dtype=np.float32)
        for (path, key), value in uniforms.items():
            previous = self._ssfr_uniforms.get((path, key))
            if previous is None or not np.array_equal(previous, value):
                self._write_runtime_attribute([path], "inputs:"+key, value)
                self._ssfr_uniforms[path, key] = value.copy()

    def _get_ldr_color_render_var(self, frame):
        if self._ssfr_graph is not None:
            for name in (self._ssfr_graph["output"], "FluidColor"):
                if name in frame.render_vars:
                    return frame.render_vars[name]
        raise RuntimeError("OVRTX did not produce the SSFR FluidColor AOV")
