"""Numerical regression tests for the CUDA/SPG screen-space fluid renderer."""

from importlib.util import find_spec
import math
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from examples.mixing.ssfr import FluidBuffers, SSFRConfig
from examples.mixing.media import MEDIUM_PRESETS, coefficients_numpy, display_colors


CUDA_AVAILABLE = wp.get_cuda_device_count() > 0
RTX_AVAILABLE = CUDA_AVAILABLE and find_spec("ovrtx") is not None and find_spec("ovstage") is not None


class TestSSFRConfig(unittest.TestCase):
    def test_invalid_configuration(self):
        for kwargs in (dict(filter="bilateral"), dict(iterations=-1), dict(radius_scale=0),
                       dict(absorption=math.nan), dict(ior=.9), dict(debug="unknown"),
                       dict(roughness=0), dict(roughness=math.nan), dict(roughness=1.1),
                       dict(splats="unknown"), dict(medium="unknown"),
                       dict(medium_scale=-1), dict(medium_scale=math.nan),
                       dict(medium_scale=math.inf), dict(medium_scale=1.e39),
                       dict(medium_layers=0), dict(medium_layers=17), dict(medium_layers=2.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SSFRConfig(**kwargs)

    def test_medium_coefficients_and_reference_swatches(self):
        for preset, (first, second) in MEDIUM_PRESETS.items():
            absorption, scattering = coefficients_numpy([0, .5, 1], preset)
            np.testing.assert_allclose(absorption[[0, 2]], [first.absorption, second.absorption])
            np.testing.assert_allclose(scattering[[0, 2]], [first.scattering, second.scattering])
            np.testing.assert_allclose(absorption[1], absorption[[0, 2]].mean(axis=0))
            np.testing.assert_allclose(scattering[1], scattering[[0, 2]].mean(axis=0))
            np.testing.assert_array_equal(display_colors([0, .5, 1], medium=preset, medium_scale=0), 1)
            swatches = display_colors([0, .5, 1], medium=preset)
            self.assertTrue(np.isfinite(swatches).all())
            self.assertTrue(np.all((swatches >= 0) & (swatches <= 1)))
        absorption, scattering = coefficients_numpy([0, .5, 1], "water-milk")
        transmission = np.exp(-.1*(absorption+scattering))
        self.assertTrue(np.all(np.diff(transmission, axis=0) < 0))
        np.testing.assert_array_equal(coefficients_numpy([-1, 2], "water-milk")[0], absorption[[0, 2]])

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
    def test_medium_cli_requires_ssfr_and_preserves_explicit_switching(self):
        from examples.mixing.apic import parse_args

        with patch("sys.argv", ["apic.py", "--viewer", "none", "--device", "cpu"]):
            self.assertEqual(parse_args().emitter_color_period, 2)
        args = ["apic.py", "--viewer", "rtx", "--ssfr", "--ssfr-medium", "coffee-milk"]
        with patch("sys.argv", args):
            self.assertEqual(parse_args().emitter_color_period, 0)
        with patch("sys.argv", args+["--emitter-color-period", "2"]):
            self.assertEqual(parse_args().emitter_color_period, 2)
        with patch("sys.argv", ["apic.py", "--ssfr-medium", "coffee-milk"]), \
                patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
class TestSSFRTiles(unittest.TestCase):
    def test_offscreen_and_invalid_particles_are_culled(self):
        buffers = FluidBuffers(64, 48, "cuda:0", SSFRConfig())
        points = wp.array([[0, 0, -2], [1.e6, 0, -2], [0, 0, 2],
                           [math.nan, 0, -2], [0, 0, -2], [0, 0, -200]],
                          dtype=wp.vec3, device="cuda:0")
        radii = wp.array([.05, .05, .05, .05, -.05, .05], dtype=float, device="cuda:0")
        dye = wp.zeros(6, dtype=float, device="cuda:0")
        buffers.prepare(points, radii, dye, np.eye(4), .01, 100, .5)
        self.assertGreater(buffers.index_count, 0)
        np.testing.assert_array_equal(buffers.indices.numpy()[:buffers.index_count], 0)

    def test_growth_empty_near_plane_and_gpu_snapshots(self):
        buffers = FluidBuffers(64, 48, "cuda:0", SSFRConfig())
        matrix = np.eye(4)
        # Identity USD camera looks down -Z. A near-plane-intersecting sphere
        # covers all tiles, exercising allocation beyond a small fixed budget.
        for count, z in ((0, -2.0), (1, -.03), (64, -2.0), (2, -2.0), (0, -2.0)):
            x = np.tile(np.array([0, 0, z], dtype=np.float32), (count, 1))
            points = wp.array(x, dtype=wp.vec3, device="cuda:0")
            radii = wp.full(count, .05, dtype=float, device="cuda:0")
            dye = wp.full(count, .25, dtype=float, device="cuda:0")
            original_numpy = wp.array.numpy
            reads = []

            def only_capacity_scalar(array):
                reads.append(array.shape)
                self.assertEqual(array.shape, (1,))
                self.assertEqual(array.dtype, wp.int32)
                return original_numpy(array)

            with patch.object(wp.array, "numpy", only_capacity_scalar):
                buffers.prepare(points, radii, dye, matrix, .01, 100, .5)
            self.assertEqual(reads, [(1,)])
            offsets = buffers.offsets.numpy()
            indices = buffers.indices.numpy()[:buffers.index_count]
            self.assertEqual(offsets[0], 0)
            self.assertEqual(offsets[-1], len(indices))
            self.assertTrue(np.all(np.diff(offsets) >= 0))
            if count:
                self.assertTrue(np.all(indices < count))
                self.assertEqual(set(indices), set(range(count)))
                if z == -.03:
                    self.assertEqual(buffers.index_count, buffers.tile_count)
                points.zero_()
                dye.zero_()
                np.testing.assert_allclose(buffers.projected.numpy()[:count, 2], -z)
                np.testing.assert_allclose(buffers.concentration.numpy()[:count], .25)
            else:
                self.assertEqual(buffers.index_count, 0)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
class TestAnisotropicSplats(unittest.TestCase):
    def prepare(self, positions):
        buffers = FluidBuffers(128, 96, "cuda:0", SSFRConfig(), particle_radius=.05)
        points = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        radii = wp.full(len(positions), .05, dtype=float, device="cuda:0")
        dye = wp.full(len(positions), .3, dtype=float, device="cuda:0")
        original_numpy = wp.array.numpy

        def capacity_scalar_only(array):
            self.assertEqual(array.shape, (1,))
            self.assertEqual(array.dtype, wp.int32)
            return original_numpy(array)

        with patch.object(wp.array, "numpy", capacity_scalar_only):
            buffers.prepare(points, radii, dye, np.eye(4), .01, 100, .5)
        np.testing.assert_array_equal(points.numpy(), positions)
        return buffers

    def test_flat_sheet_kernels_preserve_volume_and_limit_center_motion(self):
        axis = np.arange(-4, 5, dtype=np.float32)*.1
        points = np.array([[x, y, -2.] for x in axis for y in axis], dtype=np.float32)
        buffers = self.prepare(points)
        kernels = buffers.world_kernels.numpy()[:len(points)]
        values = np.linalg.eigvalsh(kernels)
        self.assertTrue(np.isfinite(kernels).all())
        self.assertTrue(np.all(values > 0))
        self.assertLess(float(np.max(values[:, 2]/values[:, 0])), 4.001)
        np.testing.assert_allclose(np.linalg.det(kernels)*.08**3, 1., atol=1.e-4)
        center = len(points)//2
        self.assertGreater(kernels[center, 2, 2], 2*kernels[center, 0, 0])
        shift = np.linalg.norm(buffers.render_positions.numpy()[:len(points)]-points, axis=1)
        self.assertLessEqual(float(shift.max()), .25*.05+1.e-6)

    def test_rotation_equivariance_and_sparse_droplets(self):
        points = np.array([[x*.1, y*.1, -2.] for x in range(-3, 4) for y in range(-3, 4)],
                          dtype=np.float32)
        angle = .63
        rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                             [-math.sin(angle), 0, math.cos(angle)]], dtype=np.float32)
        before = self.prepare(points)
        rotated = self.prepare(points@rotation.T)
        expected = rotation@before.world_kernels.numpy()[:len(points)]@rotation.T
        np.testing.assert_allclose(rotated.world_kernels.numpy()[:len(points)], expected,
                                   atol=1.e-3, rtol=2.e-4)
        droplets = np.array([[-.4, 0, -2], [.4, 0, -2]], dtype=np.float32)
        sparse = self.prepare(droplets)
        np.testing.assert_allclose(sparse.world_kernels.numpy()[:2],
                                   np.broadcast_to(np.eye(3)/.08, (2, 3, 3)), atol=1.e-5)
        np.testing.assert_array_equal(sparse.render_positions.numpy()[:2], droplets)


@unittest.skipUnless(RTX_AVAILABLE, "CUDA, ovstage and ovrtx are required")
class TestSSFRRenderer(unittest.TestCase):
    def make_viewer(self, config=None):
        from examples.mixing.viewer_ssfr import ScreenSpaceViewerRTX

        viewer = ScreenSpaceViewerRTX(particle_radius=.1, width=512, height=384,
                                      headless=True, async_rendering=True, environment="studio",
                                      ssfr_config=config or SSFRConfig())
        viewer.set_camera(wp.vec3(0, 0, 0), 0, 90)
        self.addCleanup(viewer.close)
        return viewer

    def render(self, viewer, centers, dyes):
        viewer.begin_frame(0)
        producer = wp.Stream("cuda:0")
        with wp.ScopedStream(producer):
            positions = wp.array(np.asarray(centers, dtype=np.float32).reshape(-1, 3),
                                 dtype=wp.vec3, device="cuda:0")
            radii = wp.full(len(centers), .1, dtype=float, device="cuda:0")
            concentration = wp.array(dyes, dtype=float, device="cuda:0")
            viewer.log_fluid_ssfr(positions, radii, concentration)
        viewer.end_frame()
        # Force capture of this asynchronous frame, then use the published AOVs.
        image = viewer.get_frame().numpy()
        return image

    def aov(self, viewer, key):
        import ovrtx

        frame = next(iter(viewer._render_products.values())).frames[0]
        render_var = frame.render_vars[viewer._ssfr_graph[key]]
        with render_var.map(device=ovrtx.Device.CPU) as mapping:
            return np.array(np.from_dlpack(mapping), copy=True)

    def test_medium_gpu_transmittance_matches_filtered_optical_thickness(self):
        for preset in MEDIUM_PRESETS:
            viewer = self.make_viewer(SSFRConfig(medium=preset, debug="transmittance"))
            for c in (0., .5, 1.):
                image = self.render(viewer, [[0, 2, 0]], [c])
                thickness = self.aov(viewer, "optical_smooth")[192, 256, 0]
                absorption, scattering = coefficients_numpy(c, preset)
                linear = np.exp(-(absorption+scattering)*thickness)
                expected = np.where(linear <= .0031308, 12.92*linear,
                                    1.055*linear**(1/2.4)-.055)*255
                np.testing.assert_allclose(image[192, 256], expected, atol=1)
            viewer.close()

    def test_medium_gpu_scattering_and_zero_density(self):
        viewer = self.make_viewer(SSFRConfig(medium="coffee-milk", debug="scattering"))
        image = self.render(viewer, [[0, 2, 0]], [.5])
        thickness = self.aov(viewer, "optical_smooth")[192, 256, 0]
        absorption, scattering = coefficients_numpy(.5, "coffee-milk")
        extinction = absorption+scattering
        shader = viewer.stage.GetPrimAtPath(viewer._ssfr_graph["shade"])
        key = np.array(shader.GetAttribute("inputs:keyColor").Get())
        dome = np.array(shader.GetAttribute("inputs:domeColor").Get())
        linear = scattering/extinction*(1-np.exp(-extinction*thickness))*(2.5*dome+.5*key)
        expected = np.where(linear <= .0031308, 12.92*linear,
                            1.055*linear**(1/2.4)-.055)*255
        np.testing.assert_allclose(image[192, 256], expected, atol=1)
        viewer.close()
        for debug, expected in (("transmittance", 255), ("scattering", 0)):
            viewer = self.make_viewer(SSFRConfig(medium="water-milk", medium_scale=0, debug=debug))
            image = self.render(viewer, [[0, 2, 0]], [.5])
            np.testing.assert_array_equal(image[192, 256], expected)
            viewer.close()

    def test_medium_reference_swatch_matches_gpu(self):
        viewer = self.make_viewer(SSFRConfig(medium="coffee-milk", debug="dye"))
        image = self.render(viewer, [[0, 2, 0]], [.5])
        linear = display_colors([.5], medium="coffee-milk")[0]
        expected = np.where(linear <= .0031308, 12.92*linear,
                            1.055*linear**(1/2.4)-.055)*255
        np.testing.assert_allclose(image[192, 256], expected, atol=1)

    def test_medium_depth_slabs_conserve_thickness_and_preserve_order(self):
        viewer = self.make_viewer(SSFRConfig(medium="coffee-milk", debug="scattering", medium_layers=12))
        colors = []
        for dyes in ([0., 1.], [1., 0.]):
            image = self.render(viewer, [[0, 2, 0], [0, 3, 0]], dyes)
            colors.append(image[192, 256].astype(float))
            for layers_key, optical_key in (("layers", "optical"), ("layers_smooth", "optical_smooth")):
                layers = self.aov(viewer, layers_key).reshape(12, 384, 512, 2)
                optical = self.aov(viewer, optical_key)
                self.assertTrue(np.isfinite(layers).all())
                self.assertTrue(np.all(layers >= 0))
                np.testing.assert_allclose(layers.sum(axis=0), optical, atol=2.e-6, rtol=1.e-5)
            # The nearest slab must contain the front liquid's concentration.
            self.assertAlmostEqual(float(layers[0, 192, 256, 1]/layers[0, 192, 256, 0]), dyes[0], places=5)
        # An absorbing coffee foreground suppresses milk scattered behind it.
        self.assertTrue(np.all(colors[1] > colors[0]+30))
        self.render(viewer, [], [])
        self.assertEqual(float(self.aov(viewer, "layers_smooth").max()), 0)

    def test_sphere_depth_thickness_dye_growth_and_clear(self):
        viewer = self.make_viewer()
        for centers, dyes in (([], []), ([[0, 2, 0]], [.25]),
                              ([[0, 2, 0], [0, 3, 0]], [0., 1.]), ([], [])):
            image = self.render(viewer, centers, dyes)
            raw = self.aov(viewer, "raw_depth").squeeze()
            depth = self.aov(viewer, "depth").squeeze()
            optical = self.aov(viewer, "optical")
            self.assertEqual(image.shape, (384, 512, 3))
            self.assertTrue(np.isfinite(depth).all())
            self.assertTrue(np.all(depth[raw > 0] > 0))
            if centers:
                self.assertAlmostEqual(float(raw[192, 256]), 1.84, delta=.001)
                expected = len(centers)*.32*6/(math.pi*1.6**3)
                self.assertAlmostEqual(float(optical[192, 256, 0]), expected, delta=.001)
                self.assertAlmostEqual(float(optical[192, 256, 1]/optical[192, 256, 0]),
                                       float(np.mean(dyes)), delta=.001)
            else:
                self.assertEqual(float(optical.max()), 0)
                self.assertEqual(float(depth.max()), 0)

    def test_opaque_scene_clips_chords_and_hides_particles(self):
        viewer = self.make_viewer()
        # An opaque plane cuts the first sphere through its center and fully
        # hides the second sphere. Thickness must include only the front half.
        vertices = wp.array([[-5, 2, -5], [5, 2, -5], [5, 2, 5], [-5, 2, 5]],
                            dtype=wp.vec3, device="cuda:0")
        triangles = wp.array([0, 1, 2, 0, 2, 3], dtype=int, device="cuda:0")
        viewer.log_mesh("/occluder", vertices, triangles, color=(.2, .2, .2), backface_culling=False)
        self.render(viewer, [[0, 2, 0], [0, 3, 0]], [0., 1.])
        optical = self.aov(viewer, "optical")
        expected = .16*6/(math.pi*1.6**3)
        self.assertAlmostEqual(float(optical[192, 256, 0]), expected, delta=.002)
        self.assertEqual(float(optical[192, 256, 1]), 0)
        self.render(viewer, [[0, 3, 0]], [1.])
        self.assertEqual(float(self.aov(viewer, "optical").max()), 0)

    def test_curvature_and_camera_updates(self):
        viewer = self.make_viewer(SSFRConfig(filter="curvature", iterations=8, debug="normals"))
        self.render(viewer, [[0, 2, 0]], [.5])
        before = self.aov(viewer, "depth").squeeze()
        self.assertTrue(np.isfinite(before).all())
        self.assertGreater(float(before.max()), 0)
        viewer.set_camera(wp.vec3(.3, 0, 0), 0, 90)
        self.render(viewer, [[0, 2, 0]], [.5])
        after = self.aov(viewer, "depth").squeeze()
        self.assertTrue(np.isfinite(after).all())
        self.assertLess(np.argwhere(after > 0)[:, 1].mean(), np.argwhere(before > 0)[:, 1].mean())

    def test_zero_iterations_and_dye_debug_view(self):
        from examples.mixing.color import colors_numpy

        viewer = self.make_viewer(SSFRConfig(iterations=0, debug="dye"))
        image = self.render(viewer, [[0, 2, 0]], [.5])
        np.testing.assert_array_equal(self.aov(viewer, "raw_depth"), self.aov(viewer, "depth"))
        linear = colors_numpy([.5])[0]
        expected = np.where(linear <= .0031308, 12.92*linear,
                            1.055*linear**(1/2.4)-.055)*255
        np.testing.assert_allclose(image[192, 256], expected, atol=1)

    def test_smoothed_optical_preserves_constant_concentration_and_clears(self):
        viewer = self.make_viewer()
        self.render(viewer, [[-.08, 2, 0], [.08, 2, 0]], [.3, .3])
        raw = self.aov(viewer, "optical")
        smooth = self.aov(viewer, "optical_smooth")
        valid = smooth[:, :, 0] > 1.e-6
        self.assertTrue(np.isfinite(smooth).all())
        self.assertTrue(np.all(smooth >= 0))
        np.testing.assert_allclose(smooth[:, :, 1][valid]/smooth[:, :, 0][valid], .3, atol=1.e-6)
        # A derivative jump at the overlap seam should be reduced by filtering.
        row = 192
        raw_roughness = np.abs(np.diff(raw[row, :, 0], n=2)).sum()
        smooth_roughness = np.abs(np.diff(smooth[row, :, 0], n=2)).sum()
        self.assertLess(smooth_roughness, raw_roughness)
        self.render(viewer, [], [])
        self.assertEqual(float(self.aov(viewer, "optical_smooth").max()), 0)

    def test_static_highlights_are_repeatable_and_ior_one_has_zero_reflection(self):
        viewer = self.make_viewer(SSFRConfig(debug="specular"))
        first = self.render(viewer, [[0, 2, 0]], [.5])
        valid = self.aov(viewer, "depth").squeeze() > 0
        self.assertGreater(int(first[valid].max()), 0)
        self.assertTrue(np.all(first[192, 256] > 0))
        for _ in range(3):
            current = self.render(viewer, [[0, 2, 0]], [.5])
            np.testing.assert_array_equal(first[valid], current[valid])
        viewer.close()
        matched = self.make_viewer(SSFRConfig(debug="specular", ior=1.0))
        image = self.render(matched, [[0, 2, 0]], [.5])
        valid = self.aov(matched, "depth").squeeze() > 0
        self.assertEqual(int(image[valid].max()), 0)

    def test_ellipsoid_ray_depth_and_thickness_match_reference(self):
        viewer = self.make_viewer(SSFRConfig(debug="specular"))
        centers = np.array([[x*.2, 2., z*.2] for x in range(-3, 4) for z in range(-3, 4)],
                           dtype=np.float32)
        first = self.render(viewer, centers, [.3]*len(centers))
        kernels = viewer.fluid_buffers.view_kernels.numpy()[:len(centers)].astype(np.float64)
        positions = viewer.fluid_buffers.projected.numpy()[:len(centers), :3].astype(np.float64)
        scale = 2*math.tan(math.radians(viewer.camera.fov)/2)/384
        ray = np.array([.5*scale, -.5*scale, 1.])
        ray /= np.linalg.norm(ray)
        kernel_ray = kernels@ray
        kernel_center = np.einsum("nij,nj->ni", kernels, positions)
        length2 = np.sum(kernel_ray**2, axis=1)
        nearest = np.sum(kernel_ray*kernel_center, axis=1)/length2
        radial2 = 1-np.sum((kernel_center-kernel_ray*nearest[:, None])**2, axis=1)
        valid = radial2 > 0
        half_chord = np.sqrt(radial2[valid]/length2[valid])
        depth = np.min(nearest[valid]-half_chord)*ray[2]
        thickness = 2*half_chord.sum()*6/(math.pi*1.6**3)
        self.assertAlmostEqual(float(self.aov(viewer, "raw_depth").squeeze()[192, 256]), depth, delta=.001)
        optical = self.aov(viewer, "optical")[192, 256]
        self.assertAlmostEqual(float(optical[0]), thickness, delta=.001)
        self.assertAlmostEqual(float(optical[1]/optical[0]), .3, delta=1.e-5)
        current = self.render(viewer, centers, [.3]*len(centers))
        valid = self.aov(viewer, "raw_depth").squeeze() > 0
        # Neighborhood fitting must keep static glints stable as buffers and
        # the GPU neighbor index are rebuilt for the next frame.
        difference = np.abs(first.astype(np.int16)-current.astype(np.int16))
        self.assertLessEqual(int(difference[valid].max()), 1)

    def test_anisotropic_cylinder_has_smoother_silhouette(self):
        centers = [[x*.2, 4.+y*.2, z*.2] for z in range(-9, 10)
                   for x in range(-3, 4) for y in range(-3, 4) if x*x+y*y <= 9]
        roughness = []
        for mode in ("sphere", "anisotropic"):
            viewer = self.make_viewer(SSFRConfig(splats=mode))
            self.render(viewer, centers, [.5]*len(centers))
            mask = self.aov(viewer, "raw_depth").squeeze() > 0
            self.assertTrue(np.all(mask[130:254].any(axis=1)))
            boundary = mask[130:254].argmax(axis=1).astype(float)
            smooth = np.convolve(boundary, np.ones(15)/15, mode="valid")
            roughness.append(float(np.std(boundary[7:-7]-smooth)))
            viewer.close()
        self.assertLess(roughness[1], .7*roughness[0])


if __name__ == "__main__":
    unittest.main()
