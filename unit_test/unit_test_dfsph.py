import sys
from pathlib import Path
import unittest

import numpy as np

try:
    import warp as wp
except Exception:
    wp = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if wp is not None:
    from wcsph_kernel import (  # noqa: E402
        acceleration_non_pressure,
        apply_divergence_free_correction,
        apply_constant_density_correction,
        compute_density_change_rate,
        compute_factor_dfsph,
        compute_kappa_density,
        compute_kappa_divergence,
        copy_velocity,
        drift,
        initialize_particles,
        predict_density_star,
        predict_velocity,
        rho_dfsph,
    )


def _select_device() -> str:
    for candidate in ("cuda:0", "cpu"):
        try:
            _ = wp.empty(1, dtype=float, device=candidate)
            return candidate
        except Exception:
            continue
    raise RuntimeError("No available Warp device found (tried cuda:0 and cpu).")


def _setup_5x5x5_case(device: str):
    particle_distance = 1.0
    smoothing_length = particle_distance * 1.35
    box_size = 5.0
    n = 5 * 5 * 5

    x = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel=initialize_particles,
        dim=n,
        inputs=[x, particle_distance, box_size, box_size, box_size, wp.vec3(0.0, 0.0, 0.0)],
        device=device,
    )

    rest_density_val = 1000.0
    p_volume = 0.8 * (particle_distance**3)
    mass_val = rest_density_val * p_volume

    state = {
        "n": n,
        "x": x,
        "v": wp.zeros(n, dtype=wp.vec3, device=device),
        "a": wp.zeros(n, dtype=wp.vec3, device=device),
        "v_star": wp.empty(n, dtype=wp.vec3, device=device),
        "rho": wp.zeros(n, dtype=float, device=device),
        "rho_star": wp.zeros(n, dtype=float, device=device),
        "d_rho_dt": wp.zeros(n, dtype=float, device=device),
        "factor": wp.zeros(n, dtype=float, device=device),
        "kappa": wp.zeros(n, dtype=float, device=device),
        "mass": wp.full(n, mass_val, device=device),
        "rho_0": wp.full(n, rest_density_val, device=device),
        "gamma": wp.full(n, 0.0, device=device),
        "mu": wp.full(n, 0.01, device=device),
        "p_volume": p_volume,
        "smoothing_length": smoothing_length,
        "particle_radius": particle_distance * 0.5,
        "gravity": -9.8,
        "dt": 1.0 / 240.0,
        "grid": wp.HashGrid(16, 16, 16),
    }
    return state


def _run_minimal_dfsph_step(state, device: str, density_iters: int = 2, divergence_iters: int = 1):
    n = state["n"]
    grid = state["grid"]
    x = state["x"]
    smoothing_length = state["smoothing_length"]
    dt = state["dt"]

    grid.build(x, smoothing_length)

    wp.launch(
        kernel=rho_dfsph,
        dim=n,
        inputs=[state["rho"], x, state["rho_0"], state["p_volume"], smoothing_length, grid.id],
        device=device,
    )
    wp.launch(
        kernel=compute_factor_dfsph,
        dim=n,
        inputs=[grid.id, x, state["mass"], smoothing_length, state["factor"]],
        device=device,
    )
    wp.launch(
        kernel=acceleration_non_pressure,
        dim=n,
        inputs=[
            x,
            state["v"],
            state["rho_0"],
            state["rho"],
            state["a"],
            state["particle_radius"],
            state["mass"],
            state["gamma"],
            state["mu"],
            state["gravity"],
            smoothing_length,
            grid.id,
        ],
        device=device,
    )
    wp.launch(kernel=predict_velocity, dim=n, inputs=[state["v"], state["a"], dt, state["v_star"]], device=device)

    for _ in range(density_iters):
        wp.launch(
            kernel=compute_density_change_rate,
            dim=n,
            inputs=[grid.id, x, state["v_star"], state["mass"], smoothing_length, state["d_rho_dt"]],
            device=device,
        )
        wp.launch(
            kernel=predict_density_star,
            dim=n,
            inputs=[state["rho"], state["d_rho_dt"], dt, state["rho_star"]],
            device=device,
        )
        wp.launch(
            kernel=compute_kappa_density,
            dim=n,
            inputs=[state["rho_star"], state["rho_0"], state["factor"], dt, state["kappa"]],
            device=device,
        )
        wp.launch(
            kernel=apply_constant_density_correction,
            dim=n,
            inputs=[grid.id, x, state["v_star"], state["mass"], state["rho"], state["kappa"], smoothing_length, dt],
            device=device,
        )

    wp.launch(kernel=drift, dim=n, inputs=[x, state["v_star"], dt], device=device)

    grid.build(x, smoothing_length)
    wp.launch(
        kernel=rho_dfsph,
        dim=n,
        inputs=[state["rho"], x, state["rho_0"], state["p_volume"], smoothing_length, grid.id],
        device=device,
    )
    wp.launch(
        kernel=compute_factor_dfsph,
        dim=n,
        inputs=[grid.id, x, state["mass"], smoothing_length, state["factor"]],
        device=device,
    )

    for _ in range(divergence_iters):
        wp.launch(
            kernel=compute_density_change_rate,
            dim=n,
            inputs=[grid.id, x, state["v_star"], state["mass"], smoothing_length, state["d_rho_dt"]],
            device=device,
        )
        wp.launch(
            kernel=compute_kappa_divergence,
            dim=n,
            inputs=[state["d_rho_dt"], state["rho"], state["factor"], dt, state["kappa"]],
            device=device,
        )
        wp.launch(
            kernel=apply_divergence_free_correction,
            dim=n,
            inputs=[grid.id, x, state["v_star"], state["mass"], state["rho"], state["kappa"], smoothing_length, dt],
            device=device,
        )

    wp.launch(kernel=copy_velocity, dim=n, inputs=[state["v_star"], state["v"]], device=device)


@unittest.skipIf(wp is None, "warp is not installed")
class TestDFSphMinimal(unittest.TestCase):
    def test_sample_particles_in_5x5x5_box(self):
        device = _select_device()
        state = _setup_5x5x5_case(device)

        x_np = state["x"].numpy()
        self.assertEqual(x_np.shape, (125, 3))

        # initialize_particles adds tiny jitter (0.001 * spacing), keep a small tolerance.
        self.assertGreaterEqual(float(x_np[:, 0].min()), -0.01)
        self.assertGreaterEqual(float(x_np[:, 1].min()), -0.01)
        self.assertGreaterEqual(float(x_np[:, 2].min()), -0.01)
        self.assertLessEqual(float(x_np[:, 0].max()), 4.01)
        self.assertLessEqual(float(x_np[:, 1].max()), 4.01)
        self.assertLessEqual(float(x_np[:, 2].max()), 4.01)

    def test_minimal_dfsph_step_runs_without_nan(self):
        device = _select_device()
        state = _setup_5x5x5_case(device)

        _run_minimal_dfsph_step(state, device=device, density_iters=2, divergence_iters=1)

        x_np = state["x"].numpy()
        v_np = state["v"].numpy()
        rho_np = state["rho"].numpy()

        self.assertTrue(np.isfinite(x_np).all())
        self.assertTrue(np.isfinite(v_np).all())
        self.assertTrue(np.isfinite(rho_np).all())
        self.assertTrue((rho_np > 0.0).all())


if __name__ == "__main__":
    unittest.main()
