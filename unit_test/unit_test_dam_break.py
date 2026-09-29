"""Shared-scene and physical-time checks for the dam-break examples."""

import unittest
from dataclasses import replace
import contextlib
import io
import sys
from unittest.mock import patch

import numpy as np

from examples.dam_break.scene import DEFAULT_SPEC, ParticleScene, make_scene
from examples.dam_break.simulation import DamBreakSimulation, SOLVERS
from examples.dam_break.guides import height_guides
from examples.dam_break.run import parse_args


class TestDamBreak(unittest.TestCase):
    def test_default_apic_specifications(self):
        scene = make_scene()
        self.assertEqual(scene.count, 401856)
        self.assertEqual(scene.report()["lattice_resolution"], [78, 56, 92])
        np.testing.assert_allclose(scene.positions.min(axis=0), [-1.17, -0.25, 0.03], atol=1e-6)
        np.testing.assert_allclose(scene.positions.max(axis=0), [-0.468, 0.25, 0.8508], atol=1e-6)
        self.assertEqual(scene.radius * 2.0, scene.spacing)
        self.assertEqual(scene.mass, scene.spec.density * scene.volume)
        self.assertAlmostEqual(scene.ideal_height, scene.count * scene.mass / 1000.0 / 1.344)

    def test_volume_ruler_heights(self):
        scene = make_scene()
        levels = [guide for guide in height_guides(scene) if guide.volume_fraction is not None]
        self.assertEqual([g.volume_fraction for g in levels], [1.0, 0.9, 0.8])
        for guide in levels:
            # Labels contain extra strokes; the first segment marks the physical water level.
            self.assertAlmostEqual(float(guide.starts[0, 2]), scene.ideal_height * guide.volume_fraction,
                                   places=6)
            self.assertTrue(np.isfinite(guide.starts).all() and np.isfinite(guide.ends).all())

    def test_finite_check_detects_invalid_state_between_metrics(self):
        sim = DamBreakSimulation(make_scene(512), "pbf", device="cpu")
        sim.check_finite()
        positions = sim.state_0.particle_q.numpy()
        positions[0, 0] = np.nan
        sim.state_0.particle_q.assign(positions)
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            sim.check_finite()

    def test_solvers_share_initial_particles_and_mass(self):
        scene = make_scene(512)
        for method in SOLVERS:
            with self.subTest(method=method):
                sim = DamBreakSimulation(scene, method, device="cpu")
                np.testing.assert_array_equal(sim.state_0.particle_q.numpy(), scene.positions)
                np.testing.assert_array_equal(sim.state_0.particle_qd.numpy(), np.zeros_like(scene.positions))
                np.testing.assert_allclose(sim.model.particle_radius.numpy(), scene.radius)
                np.testing.assert_allclose(sim.model.particle_mass.numpy(), scene.mass)
                np.testing.assert_allclose(sim.model.gravity.numpy()[0], scene.spec.gravity)

    def test_every_method_advances_complete_frames(self):
        scene = make_scene(512)
        for method in SOLVERS:
            with self.subTest(method=method):
                sim = DamBreakSimulation(scene, method, device="cpu", substeps=4)
                for frame in range(1, 3):
                    sim.step()
                    self.assertEqual(sim.sim_time, frame / 60.0)
                    self.assertGreaterEqual(sim.last_substeps, 4)
                    metrics = sim.metrics()
                    self.assertTrue(metrics["finite"])
                    x = sim.state_0.particle_q.numpy()
                    self.assertTrue(np.all(x >= np.asarray(scene.spec.bound_lo) - 1e-6))
                    self.assertTrue(np.all(x <= np.asarray(scene.spec.bound_hi) + 1e-6))
                # WCSPH's acoustic limit must refine beyond the base timestep.
                if method == "wcsph":
                    self.assertGreater(sim.total_substeps, 8)


class TestWcsphTuning(unittest.TestCase):
    def test_underdense_block_falls_without_pressure_attraction(self):
        sim = DamBreakSimulation(make_scene(512), "wcsph", device="cpu")
        sim.step()
        self.assertTrue(np.all(sim.state_0.sph.rho.numpy() < 1000.0))
        np.testing.assert_array_equal(sim.state_0.sph.pressure.numpy(), 0.0)
        expected = np.tile(np.asarray(sim.scene.spec.gravity) * sim.frame_dt, (sim.scene.count, 1))
        np.testing.assert_allclose(sim.state_0.particle_qd.numpy(), expected, atol=1e-7)

        signed = DamBreakSimulation(make_scene(512), "wcsph", device="cpu",
                                   wcsph_clamp_negative_pressure=False)
        signed.step()
        self.assertTrue(np.any(signed.state_0.sph.pressure.numpy() < 0.0))
        self.assertGreater(signed.metrics()["speed_max"], sim.metrics()["speed_max"])

    def test_compressed_particles_repel_and_conserve_momentum(self):
        # A compact tetrahedron exercises positive pressure without gravity or walls.
        d = 0.01
        center = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
        offsets = np.asarray([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                             dtype=np.float32) * (0.05 * d)
        scene = ParticleScene(replace(DEFAULT_SPEC, gravity=(0.0, 0.0, 0.0), target_count=4),
                              center + offsets, 1000.0 * d**3, d * 0.5, d**3, d)
        sim = DamBreakSimulation(scene, "wcsph", device="cpu", wcsph_viscosity=0.0)
        sim._substep(1e-6)
        self.assertTrue(np.all(sim.state_0.sph.rho.numpy() > 1000.0))
        self.assertTrue(np.all(sim.state_0.sph.pressure.numpy() > 0.0))
        velocity = sim.state_0.particle_qd.numpy()
        self.assertTrue(np.all(np.sum(velocity * offsets, axis=1) > 0.0))
        np.testing.assert_allclose(velocity.sum(axis=0), 0.0, atol=1e-6)

    def test_cfl_reduction_preserves_maxima_and_rejects_nonfinite_values(self):
        sim = DamBreakSimulation(make_scene(512), "wcsph", device="cpu")
        rng = np.random.default_rng(42)
        velocity = rng.normal(size=(sim.scene.count, 3)).astype(np.float32)
        acceleration = (rng.normal(size=velocity.shape) * 20).astype(np.float32)
        sim.state_0.particle_qd.assign(velocity)
        sim.solver._particle_accel.assign(acceleration)
        sim.total_substeps = 1
        speed, accel = sim._cfl_maxima()
        np.testing.assert_allclose([speed, accel],
                                   [np.linalg.norm(velocity, axis=1).max(),
                                    np.linalg.norm(acceleration, axis=1).max()], rtol=1e-6)
        velocity[0, 0] = np.nan
        sim.state_0.particle_qd.assign(velocity)
        with self.assertRaisesRegex(RuntimeError, "Non-finite velocity"):
            sim._cfl_maxima()
        velocity[0, 0] = 0.0
        sim.state_0.particle_qd.assign(velocity)
        acceleration[0, 0] = np.inf
        sim.solver._particle_accel.assign(acceleration)
        with self.assertRaisesRegex(RuntimeError, "Non-finite WCSPH acceleration"):
            sim._cfl_maxima()

    def test_viscous_timestep_damps_velocity_and_completes_frame(self):
        sim = DamBreakSimulation(make_scene(512), "wcsph", device="cpu", wcsph_viscosity=10.0)
        velocity = np.zeros((sim.scene.count, 3), dtype=np.float32)
        velocity[:, 0] = np.where(np.arange(sim.scene.count) % 2, -0.01, 0.01)
        sim.state_0.particle_qd.assign(velocity)
        sim.model.set_gravity((0.0, 0.0, 0.0))
        dts = []
        advance = sim._substep

        def record_and_advance(dt):
            dts.append(dt)
            advance(dt)

        sim._substep = record_and_advance
        sim.step()
        sim.check_finite()
        self.assertLessEqual(max(dts), 0.025 * sim.solver.smoothing_length**2 / 10.0)
        self.assertAlmostEqual(sum(dts), sim.frame_dt)
        final_energy = np.sum(sim.state_0.particle_qd.numpy()**2)
        self.assertLess(final_energy, np.sum(velocity**2))

    def test_wcsph_cli_options_validate_values_and_solver(self):
        argv = ["wcsph.py", "--wcsph-stiffness", "150000", "--wcsph-viscosity", "0",
                "--wcsph-smoothing-length-coff", "2.2", "--wcsph-allow-negative-pressure"]
        with patch.object(sys, "argv", argv):
            args = parse_args("wcsph")
        self.assertEqual(args.wcsph_stiffness, 150000.0)
        self.assertEqual(args.wcsph_viscosity, 0.0)
        self.assertEqual(args.wcsph_smoothing_length_coff, 2.2)
        self.assertTrue(args.wcsph_allow_negative_pressure)
        for method, option, value in [("pbf", "--wcsph-stiffness", "250000"),
                                      ("wcsph", "--wcsph-viscosity", "nan"),
                                      ("wcsph", "--wcsph-stiffness", "0")]:
            with self.subTest(method=method, option=option, value=value):
                with patch.object(sys, "argv", ["run.py", option, value]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_args(method)


if __name__ == "__main__":
    unittest.main()
