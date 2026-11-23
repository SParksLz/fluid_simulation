"""Consistency, diffusion, passive advection and animated color checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import warp as wp

from examples.mixing.apic import MixingSimulation, PointUSD
from examples.mixing.color import (BLUE, YELLOW, COLOR_MODELS, colors_numpy,
                                   concentration_colors, reflectance_to_ks, ks_to_reflectance)
from solver.apic.lsmps import LSMPSDiffusion


DEVICE = "cuda:0" if wp.get_cuda_device_count() else "cpu"


def box_particles(n=8):
    axis = (np.arange(n) + 0.5) / n
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)


class TestLSMPS(unittest.TestCase):
    def test_irregular_particle_polynomial_consistency(self):
        x = box_particles()
        x += np.random.default_rng(123).uniform(-0.01, 0.01, x.shape).astype(np.float32)
        p = wp.array(x, dtype=wp.vec3, device=DEVICE)
        diffusion = LSMPSDiffusion(len(x), 0.125, (0, 0, 0), (1, 1, 1), DEVICE)
        diffusion.prepare(p)
        c = wp.array((x*x).sum(axis=1), device=DEVICE)
        laplacian = diffusion.laplacian(c).numpy()
        interior = ((x > 0.3125) & (x < 0.6875)).all(axis=1)
        np.testing.assert_allclose(laplacian[interior], 6.0, atol=2.0e-3)
        constant = wp.full(len(x), 0.37, dtype=float, device=DEVICE)
        np.testing.assert_array_equal(diffusion.laplacian(constant).numpy(), 0.0)

    def test_neumann_cosine_diffusion_matches_analytic_decay(self):
        x = box_particles()
        p = wp.array(x, dtype=wp.vec3, device=DEVICE)
        initial = (0.5 + 0.4*np.cos(np.pi*x[:, 0])).astype(np.float32)
        c = wp.array(initial, device=DEVICE)
        diffusion = LSMPSDiffusion(len(x), 0.125, (0, 0, 0), (1, 1, 1), DEVICE, diffusivity=0.05)
        for _ in range(20):
            diffusion.step(p, c, 0.01)
        expected = 0.5 + 0.4*np.cos(np.pi*x[:, 0])*np.exp(-0.05*np.pi**2*0.2)
        np.testing.assert_allclose(c.numpy(), expected, atol=2.0e-3)
        self.assertAlmostEqual(float(c.numpy().mean()), float(initial.mean()), delta=2.0e-5)

    def test_isolated_particle_reports_bad_neighborhood(self):
        p = wp.array([[0.5, 0.5, 0.5]], dtype=wp.vec3, device=DEVICE)
        diffusion = LSMPSDiffusion(1, 0.1, (0, 0, 0), (1, 1, 1), DEVICE)
        with self.assertRaisesRegex(RuntimeError, "Singular LSMPS"):
            diffusion.prepare(p)


class TestMixingExample(unittest.TestCase):
    def test_diffusion_is_passive_and_zero_diffusion_preserves_labels(self):
        flowing = MixingSimulation(12, DEVICE, diffusivity=1.0e-3)
        control = MixingSimulation(12, DEVICE, diffusivity=0.0, color_model="rgb")
        initial_c = control.concentration.numpy().copy()
        initial_x = control.state_0.particle_q.numpy().copy()
        for _ in range(4):
            row = flowing.step()
            control.step()
        np.testing.assert_array_equal(control.concentration.numpy(), initial_c)
        np.testing.assert_allclose(flowing.state_0.particle_q.numpy(),
                                   control.state_0.particle_q.numpy(), atol=1.0e-4)
        self.assertGreater(float(np.max(np.abs(initial_x-control.state_0.particle_q.numpy()))), 1.0e-3)
        self.assertGreater(row["mixing_index"], 0.0)
        self.assertLess(abs(row["dye_relative_error"]), 1.0e-4)

    def test_usd_has_animated_vertex_colors_and_raw_concentration(self):
        from pxr import Usd, UsdGeom
        sim = MixingSimulation(12, DEVICE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixing.usdc"
            output = PointUSD(path, sim)
            output.record(sim)
            sim.step()
            output.record(sim)
            output.close()
            stage = Usd.Stage.Open(str(path))
            points = UsdGeom.Points(stage.GetPrimAtPath("/Mixing"))
            color = points.GetDisplayColorPrimvar()
            scalar = UsdGeom.PrimvarsAPI(points).GetPrimvar("dyeConcentration")
            self.assertEqual(color.GetInterpolation(), UsdGeom.Tokens.vertex)
            self.assertEqual(color.GetAttr().GetTimeSamples(), [0.0, 1.0])
            self.assertEqual(len(points.GetPointsAttr().Get(1)), sim.count)
            np.testing.assert_allclose(np.asarray(scalar.Get(1)), sim.concentration.numpy())
            np.testing.assert_allclose(np.asarray(color.Get(1)),
                                       colors_numpy(sim.concentration.numpy(), sim.color_model))
            self.assertEqual(stage.GetRootLayer().customLayerData["color_model"], "km")
            self.assertGreater(float(np.max(np.abs(np.asarray(color.Get(0))-np.asarray(color.Get(1))))), 1.0e-5)


class TestPigmentColor(unittest.TestCase):
    def test_reflectance_roundtrip_including_black_and_white(self):
        reflectance = np.array([0.0, 1.0e-8, 1.0e-6, 0.05, 0.5, 1.0])
        reconstructed = ks_to_reflectance(reflectance_to_ks(reflectance))
        np.testing.assert_allclose(reconstructed, np.clip(reflectance, 1.0e-6, 1.0), rtol=1.0e-6)
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_yellow_and_blue_keep_endpoints_and_mix_green(self):
        colors = colors_numpy(np.array([0.0, 0.5, 1.0]), "km")
        np.testing.assert_allclose(colors[[0, 2]], [YELLOW, BLUE], atol=1.0e-6)
        self.assertGreater(colors[1, 1], 2.0*colors[1, 0])
        self.assertGreater(colors[1, 1], 2.0*colors[1, 2])
        np.testing.assert_allclose(colors_numpy(np.array([0.5]), "rgb"), [[0.5, 0.5, 0.5]])

    def test_viewer_kernel_agrees_with_export_for_both_models(self):
        values = np.linspace(-0.1, 1.1, 33).astype(np.float32)
        concentration = wp.array(values, device=DEVICE)
        colors = wp.empty(len(values), dtype=wp.vec3, device=DEVICE)
        for model, index in COLOR_MODELS.items():
            wp.launch(concentration_colors, dim=len(values),
                      inputs=[concentration, colors, index], device=DEVICE)
            np.testing.assert_allclose(colors.numpy(), colors_numpy(values, model), atol=1.0e-6)
        np.testing.assert_array_equal(concentration.numpy(), values)


if __name__ == "__main__":
    unittest.main()
