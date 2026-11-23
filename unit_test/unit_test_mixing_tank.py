"""Growing emitter state, injection accounting, robust scalar diffusion."""

from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from examples.mixing.apic import PointUSD
from examples.mixing.color import BLUE, YELLOW
from examples.mixing.tank import TankMixingSimulation
from solver.apic.lsmps import LSMPSDiffusion


DEVICE = "cuda:0" if wp.get_cuda_device_count() else "cpu"


class TestTankEmitters(unittest.TestCase):
    def test_prefilled_coffee_uses_only_the_milk_inlet(self):
        sim = TankMixingSimulation(24, DEVICE, fps=30, substeps=2,
                                    emitter_speed=.6, emitter_duration=.12,
                                    emitter_color_period=0, initial_coffee_height=.1,
                                    diffusivity=0.0)
        self.assertGreater(sim.initial_particle_count, 0)
        self.assertEqual(sim.count, sim.initial_particle_count)
        self.assertEqual(sim.emitter_enabled.tolist(), [False, True])
        np.testing.assert_array_equal(sim.source_ids[:sim.count], -1)
        np.testing.assert_array_equal(sim.concentration.numpy()[:sim.count], 0.0)
        initial_count = sim.count
        for _ in range(8):
            sim.step()
        self.assertGreater(sim.count, initial_count)
        self.assertEqual(np.count_nonzero(sim.source_ids[:sim.count] == 0), 0)
        self.assertGreater(np.count_nonzero(sim.source_ids[:sim.count] == 1), 0)
        self.assertGreaterEqual(float(sim.concentration.numpy()[:sim.count].min()), 0.0)
        self.assertLessEqual(float(sim.concentration.numpy()[:sim.count].max()), 1.0)
        self.assertAlmostEqual(sim.metrics()["expected_dye_integral_m3"],
                               sim.emitted_dye_volume, places=10)

    def test_high_quality_cli_and_explicit_overrides(self):
        from examples.mixing.apic import parse_args

        command = ["apic.py", "--viewer", "none", "--device", "cpu", "--quality", "high"]
        with patch("sys.argv", command):
            args = parse_args()
        self.assertEqual((args.resolution, args.substeps), (160, 24))
        self.assertEqual((args.render_width, args.render_height), (1920, 1080))
        self.assertEqual(args.ssfr_medium_layers, 12)
        with patch("sys.argv", command+["--resolution", "128", "--substeps", "20",
                                      "--render-width", "960", "--render-height", "540",
                                      "--ssfr-medium-layers", "8"]):
            args = parse_args()
        self.assertEqual((args.resolution, args.substeps), (128, 20))
        self.assertEqual((args.render_width, args.render_height), (960, 540))
        self.assertEqual(args.ssfr_medium_layers, 8)
        with patch("sys.argv", command+["--render-width", "0"]), \
                patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args()

    def test_optical_medium_point_usd_uses_reference_colors_and_metadata(self):
        from pxr import Usd, UsdGeom
        from examples.mixing.media import display_colors

        sim = TankMixingSimulation(24, DEVICE, emitter_duration=.02, diffusivity=0,
                                   emitter_color_period=0)
        sim.optical_medium = "coffee-milk"
        sim.optical_medium_scale = 1.
        sim.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"medium.usdc"
            output = PointUSD(path, sim)
            output.record(sim)
            output.close()
            stage = Usd.Stage.Open(str(path))
            self.assertEqual(stage.GetRootLayer().customLayerData["optical_medium"], "coffee-milk")
            points = UsdGeom.Points(stage.GetPrimAtPath("/Mixing/Particles"))
            colors = np.array(points.GetDisplayColorPrimvar().Get(sim.frame))
            expected = display_colors(sim.concentration.numpy(), medium="coffee-milk")
            np.testing.assert_allclose(colors, expected, atol=1.e-6)

    def test_dense_default_specifications_match_cli(self):
        from examples.mixing.apic import parse_args

        with patch("sys.argv", ["apic.py", "--viewer", "none", "--device", "cpu"]):
            args = parse_args()
        sim = TankMixingSimulation(device=DEVICE, diffusivity=0.0)
        self.assertEqual(args.resolution, 96)
        self.assertEqual(sim.resolution, args.resolution)
        self.assertAlmostEqual(sim.spacing, .0125)
        self.assertAlmostEqual(.5*sim.spacing, .00625)
        self.assertEqual(sim.particles_per_row, 113)
        self.assertEqual(sim.capacity, 65088)

    def test_skewed_nozzles_stay_circular_and_inject_a_common_rotation(self):
        for angle in (20.0, -20.0, 0.0):
            sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.01,
                                       diffusivity=0.0, emitter_yaw_offset=angle,
                                       emission_mode="layered")
            sim._emit_due()
            x, velocity = sim.state_0.particle_q.numpy(), sim.state_0.particle_qd.numpy()
            for source, emitter in enumerate(sim.emitters):
                selected = sim.source_ids[:sim.count] == source
                offset = x[selected]-np.asarray(emitter.center)
                direction = np.asarray(emitter.direction)
                np.testing.assert_allclose(offset@direction, 0.0, atol=1.0e-6)
                np.testing.assert_allclose(np.linalg.norm(velocity[selected], axis=1), sim.emitter_speed, atol=1.0e-6)
                self.assertLessEqual(float(np.linalg.norm(offset, axis=1).max()), sim.emitter_radius+1.0e-6)
                angular = ((x[selected, 0]-0.6)*velocity[selected, 1]
                           -(x[selected, 1]-0.3)*velocity[selected, 0]).mean()
                if angle:
                    self.assertGreater(float(angular)*np.sign(angle), 0.01)
                else:
                    self.assertAlmostEqual(float(angular), 0.0, delta=1.0e-6)

    def test_new_particles_initialize_both_states_without_resetting_old_dye(self):
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.1, diffusivity=0.0,
                                   emission_mode="layered")
        self.assertEqual(sim.count, 0)
        self.assertEqual(sim.state_0.particle_count, 0)
        sim._emit_due()
        first = sim.count
        self.assertEqual(first, 2*sim.particles_per_row)
        sim.concentration.fill_(0.25)
        for state in sim._states:
            _, qd, force, C, _ = sim._state_storage[id(state)]
            qd[first:].fill_(wp.vec3(float("nan")))
            force[first:].fill_(wp.vec3(float("nan")))
            C[first:].fill_(wp.mat33(float("nan")))
        sim.sim_time = sim.period
        sim._emit_due()
        self.assertEqual(sim.count, 2*first)
        np.testing.assert_array_equal(sim.concentration.numpy()[:first], 0.25)
        for state in sim._states:
            new = state.mixing.concentration.numpy()[first:]
            np.testing.assert_array_equal(new[:sim.particles_per_row], 0.0)
            np.testing.assert_array_equal(new[sim.particles_per_row:], 1.0)
            self.assertTrue(np.isfinite(state.particle_qd.numpy()).all())
            np.testing.assert_array_equal(state.apic.C.numpy()[first:], 0.0)
            np.testing.assert_array_equal(state.particle_f.numpy()[first:], 0.0)

    def test_sources_stop_and_dye_accounting_includes_inflow(self):
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.06, diffusivity=0.0)
        for _ in range(8):
            row = sim.step()
        self.assertEqual(sim.count, sim.capacity)
        self.assertEqual(sim.model.particle_count, sim.state_0.particle_count)
        expected = 2*sim.emitter_duration*sim.particles_per_row/sim.period
        self.assertLessEqual(abs(sim.count-expected), 2.0)
        self.assertEqual(np.count_nonzero(sim.source_ids[:sim.count] == 0), sim.count//2)
        np.testing.assert_array_equal(np.sort(np.unique(sim.concentration.numpy())), [0.0, 1.0])
        self.assertEqual(row["dye_relative_error"], 0.0)
        self.assertAlmostEqual(row["emitted_rest_volume_m3"], sim.count*sim.volume)
        before = sim.count
        sim.step()
        self.assertEqual(sim.count, before)
        x = sim.state_0.particle_q.numpy()
        self.assertGreaterEqual(float(x[:, 2].min()), sim.spacing/2-1.0e-5)

    def test_continuous_emission_spreads_births_and_keeps_source_flux_balanced(self):
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.12, diffusivity=0.0)
        counts = []
        old_count = 0
        for time in np.arange(0.0, 0.13, sim.dt):
            sim.sim_time = float(time)
            sim._emit_due()
            counts.append(sim.count-old_count)
            old_count = sim.count
            sources = sim.source_ids[:sim.count]
            self.assertEqual(np.count_nonzero(sources == 0), np.count_nonzero(sources == 1))
        self.assertEqual(sim.count, sim.capacity)
        maximum_per_step = 2*math.ceil(sim.dt/(sim.period/sim.particles_per_row))
        self.assertLessEqual(max(counts), maximum_per_step)
        self.assertLess(max(counts), 2*sim.particles_per_row)
        self.assertGreater(len(np.unique(sim.birth_times[:sim.count])), 10)

    def test_point_usd_starts_empty_and_contains_tank_and_emitters(self):
        from pxr import Usd, UsdGeom, UsdUtils
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.06)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tank.usdc"
            output = PointUSD(path, sim)
            output.record(sim)
            sim.step()
            output.record(sim)
            output.close()
            stage = Usd.Stage.Open(str(path))
            points = UsdGeom.Points(stage.GetPrimAtPath("/Mixing/Particles"))
            self.assertEqual(len(points.GetPointsAttr().Get(0)), 0)
            self.assertEqual(len(points.GetPointsAttr().Get(1)), sim.count)
            self.assertTrue(stage.GetPrimAtPath("/Mixing/Tank/Walls"))
            for name, c in [("Yellow", 0.0), ("Blue", 1.0)]:
                emitter = stage.GetPrimAtPath("/Mixing/Emitters/"+name)
                self.assertEqual(emitter.GetCustomDataByKey("initialDyeConcentration"), c)
                self.assertEqual(emitter.GetAttribute("dyeConcentration").Get(0), c)
            checker = UsdUtils.ComplianceChecker()
            checker.CheckCompliance(str(path))
            self.assertEqual(checker.GetErrors(), [])
            self.assertEqual(checker.GetFailedChecks(), [])

    def test_separation_storage_grows_geometrically_for_continuous_inflow(self):
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.1, diffusivity=0.0)
        interval = sim.period/sim.particles_per_row
        buffers = []
        for events in (2, 3, 4):
            sim.sim_time = events*interval
            sim._emit_due()
            sim.solver._apply_particle_separation(sim.state_0)
            buffers.append(sim.solver._sep_delta)
            self.assertGreaterEqual(len(buffers[-1]), sim.count)
            self.assertTrue(np.isfinite(sim.state_0.particle_q.numpy()).all())
        self.assertIsNot(buffers[0], buffers[1])
        self.assertIs(buffers[1], buffers[2])

    def test_inlet_dyes_swap_every_two_seconds_without_recoloring_existing_dye(self):
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=4.1, diffusivity=0.0,
                                   emission_mode="layered")
        sim._emit_due()
        sim.concentration.fill_(0.25)
        previous = sim.concentration.numpy().copy()
        for time, expected_yellow in ((2.0, 1.0), (4.0, 0.0)):
            self.assertEqual(sim.emitter_concentration(sim.emitters[0], time-1.0e-5), 1.0-expected_yellow)
            start = sim.count
            sim.sim_time = time
            sim._emit_due()
            result = sim.concentration.numpy()
            np.testing.assert_array_equal(result[:start], previous)
            sources = sim.source_ids[start:sim.count]
            np.testing.assert_array_equal(result[start:][sources == 0], expected_yellow)
            np.testing.assert_array_equal(result[start:][sources == 1], 1.0-expected_yellow)
            previous = result.copy()
        sim.emitter_color_period = 0.0
        self.assertEqual(sim.emitter_concentration(sim.emitters[0], 2.0), 0.0)
        self.assertEqual(sim.emitter_concentration(sim.emitters[1], 2.0), 1.0)

    def test_usd_nozzle_switch_stays_sharp_across_sparse_samples(self):
        from pxr import Usd, UsdGeom
        sim = TankMixingSimulation(24, DEVICE, emitter_duration=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"switch.usdc"
            output = PointUSD(path, sim)
            for frame in (0, 126, 246):
                sim.frame, sim.sim_time = frame, frame/sim.fps
                output.record(sim)
            output.close()
            stage = Usd.Stage.Open(str(path))
            for name, initial in (("Yellow", 0.0), ("Blue", 1.0)):
                nozzle = UsdGeom.BasisCurves(stage.GetPrimAtPath("/Mixing/Emitters/"+name))
                concentration = nozzle.GetPrim().GetAttribute("dyeConcentration")
                self.assertEqual(concentration.Get(119.9), initial)
                self.assertEqual(concentration.Get(120), 1.0-initial)
                self.assertEqual(concentration.Get(239.9), 1.0-initial)
                self.assertEqual(concentration.Get(240), initial)
                color = nozzle.GetDisplayColorAttr()
                np.testing.assert_allclose(np.asarray(color.Get(120))[0],
                                           BLUE if initial == 0.0 else YELLOW, atol=1.0e-6)


class TestFreeSurfaceDiffusion(unittest.TestCase):
    def test_irregular_pair_flux_is_bounded_and_preserves_total_dye(self):
        axis = (np.arange(8)+0.5)/8
        x = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
        x += np.random.default_rng(7).uniform(-0.01, 0.01, x.shape).astype(np.float32)
        p = wp.array(x, dtype=wp.vec3, device=DEVICE)
        initial = (x[:, 0] > 0.5).astype(np.float32)
        c = wp.array(initial, device=DEVICE)
        diffusion = LSMPSDiffusion(len(x), 0.125, (0, 0, 0), (1, 1, 1), DEVICE,
                                   diffusivity=0.05, allow_degenerate=True, conservative=True)
        for _ in range(8):
            diffusion.step(p, c, 0.01)
        result = c.numpy()
        self.assertGreaterEqual(float(result.min()), -1.0e-6)
        self.assertLessEqual(float(result.max()), 1.0+1.0e-6)
        self.assertAlmostEqual(float(result.astype(float).sum()), float(initial.sum()), delta=1.0e-3)
        self.assertLess(float(result.var()), float(initial.var()))

    def test_sparse_planar_stream_skips_unresolved_diffusion(self):
        axis = (np.arange(6)+0.5)/6
        xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        x = np.column_stack([xy, np.full(len(xy), 0.5)]).astype(np.float32)
        p = wp.array(x, dtype=wp.vec3, device=DEVICE)
        initial = (x[:, 0] > 0.5).astype(np.float32)
        c = wp.array(initial, device=DEVICE)
        diffusion = LSMPSDiffusion(100, 1/6, (0, 0, 0), (1, 1, 1), DEVICE,
                                   allow_degenerate=True, conservative=True)
        diffusion.resize(len(x))
        diffusion.step(p, c, 0.01)
        np.testing.assert_array_equal(c.numpy(), initial)
        self.assertEqual(diffusion.last_skipped, len(x))


if __name__ == "__main__":
    unittest.main()
