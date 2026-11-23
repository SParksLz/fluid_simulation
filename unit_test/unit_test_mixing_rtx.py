"""Real ovstage writes with readback blocked in the RTX fluid upload path."""

from importlib.util import find_spec
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from examples.mixing.surface import SurfaceFrame, palette_texture
from examples.mixing.viewer_rtx import MixingViewerRTX


RTX_AVAILABLE = (wp.get_cuda_device_count() > 0 and find_spec("ovstage") is not None
                 and find_spec("ovrtx") is not None)


@unittest.skipUnless(RTX_AVAILABLE, "CUDA, ovstage and ovrtx are required")
class TestRTXGPUUploads(unittest.TestCase):
    def setUp(self):
        import ovstage

        self.viewer = MixingViewerRTX(headless=True)
        self.viewer._ovstage = ovstage.Stage("mixing upload test")
        self.viewer._ovstage_paths = ovstage.PathDictionary(self.viewer._ovstage)

    def tearDown(self):
        self.viewer._release_ovstage()
        self.viewer.close()

    def read_attribute(self, path, attribute):
        import ovstage

        viewer = self.viewer
        query = viewer._get_ovstage_query([path])
        token = viewer._ovstage_paths.intern_token(attribute)
        with viewer._ovstage.read_attributes(
                query, [token], ovstage.OrdinalRange.latest(viewer._ovstage_ordinal)) as read:
            read.wait()
            group = read.fetch_next()
            self.assertIsNotNone(group)
            try:
                return np.array(group.array(0))
            finally:
                viewer._ovstage.release_group(group)

    def upload(self, log):
        viewer = self.viewer
        viewer._ovstage_ordinal += 1

        write_attributes = viewer._ovstage.write_attributes

        def check_cuda_payload(query, writes, **kwargs):
            pointers = {
                array.ptr
                for pending in (viewer._pending_gpu_points, viewer._pending_gpu_surfaces)
                for attributes, _, _ in pending.values()
                for array in attributes.values()
            }
            for write in writes:
                tensor = write.tensors[0]
                self.assertEqual(tensor.device.device_type.value, 2)
                self.assertIn(int(tensor.data), pointers)
                self.assertIsNotNone(write.cuda_stream)
            return write_attributes(query, writes, **kwargs)

        with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected fluid upload readback")):
            log()
            with patch.object(viewer._ovstage, "write_attributes", side_effect=check_cuda_payload):
                viewer._flush_gpu_attributes(viewer._pending_gpu_points, viewer._point_batch_paths)
                viewer._flush_gpu_attributes(viewer._pending_gpu_surfaces, viewer._mesh_prim_paths)
        viewer._ovstage.advance_write_floor(viewer._ovstage_ordinal).wait()

    def test_particle_growth_shrink_and_colors_use_cuda_payloads(self):
        viewer = self.viewer
        device = wp.get_device("cuda:0")
        producer = wp.Stream(device)
        previous_scales = None
        for count in (0, 2, 5, 3, 0, 4):
            positions = np.arange(3*count, dtype=np.float32).reshape(count, 3)/100.0
            colors = np.tile(np.array([count/10.0, 0.3, 0.8], dtype=np.float32), (count, 1))
            with wp.ScopedStream(producer):
                points = wp.array(positions, dtype=wp.vec3, device=device)
                radii = wp.full(count, 0.025, dtype=float, device=device)
                rgb = wp.array(colors, dtype=wp.vec3, device=device)
                # Leave the producer stream before consuming on the viewer's stream.
                with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected log readback")):
                    viewer.log_fluid_particles(points, radii, rgb)
            self.upload(lambda: None)
            if not viewer._point_batch_paths:
                self.assertEqual(count, 0)
                continue
            path = viewer._point_batch_paths["/mixing/particles"]
            visibility = viewer._ovstage_paths.intern_token("inherited" if count else "invisible")
            np.testing.assert_array_equal(self.read_attribute(path, "visibility"), [visibility])
            if count:
                np.testing.assert_array_equal(self.read_attribute(path, "positions").reshape(count, 3), positions)
                np.testing.assert_array_equal(self.read_attribute(path, "primvars:displayColor").reshape(count, 3), colors)
                np.testing.assert_allclose(self.read_attribute(path, "scales"), 0.025)
                np.testing.assert_array_equal(self.read_attribute(path, "ids"), np.arange(count))
                np.testing.assert_array_equal(self.read_attribute(path, "protoIndices"), np.zeros(count))
                np.testing.assert_array_equal(self.read_attribute(path, "primvars:displayColor:indices"), np.arange(count))
            if count == 5:
                previous_scales = viewer._particle_attributes["scales"]
            elif count == 3:
                self.assertIs(viewer._particle_attributes["scales"], previous_scales)

    def test_surface_topology_uv_and_empty_frames_use_cuda_payloads(self):
        viewer = self.viewer
        palette = palette_texture("km")
        for triangles in (0, 1, 2, 0, 1):
            count = 3*triangles
            positions = np.arange(3*count, dtype=np.float32).reshape(count, 3)/100.0
            uv = np.full((count, 2), triangles/4.0, dtype=np.float32)
            mesh = SurfaceFrame(
                vertices=wp.array(positions, dtype=wp.vec3, device="cuda:0"),
                indices=wp.array(np.arange(count, dtype=np.int32), device="cuda:0"),
                normals=wp.full(count, wp.vec3(0, 0, 1), dtype=wp.vec3, device="cuda:0"),
                concentration=wp.zeros(count, dtype=float, device="cuda:0"),
                uvs=wp.array(uv, dtype=wp.vec2, device="cuda:0"),
            )
            self.upload(lambda: viewer.log_fluid_surface(mesh, palette))
            if not viewer._mesh_prim_paths:
                self.assertEqual(triangles, 0)
                continue
            path = viewer._mesh_prim_paths["/mixing/surface"]
            visibility = viewer._ovstage_paths.intern_token("inherited" if triangles else "invisible")
            np.testing.assert_array_equal(self.read_attribute(path, "visibility"), [visibility])
            if triangles:
                np.testing.assert_array_equal(self.read_attribute(path, "points").reshape(count, 3), positions)
                np.testing.assert_array_equal(self.read_attribute(path, "primvars:st").reshape(count, 2), uv)
                np.testing.assert_array_equal(self.read_attribute(path, "faceVertexCounts"), np.full(triangles, 3))
                np.testing.assert_array_equal(self.read_attribute(path, "faceVertexIndices"), np.arange(count))
                np.testing.assert_array_equal(self.read_attribute(path, "normals").reshape(count, 3),
                                               np.tile([0, 0, 1], (count, 1)))


if __name__ == "__main__":
    unittest.main()
