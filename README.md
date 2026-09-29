# Fluid Simulation

An experimental fluid simulation repository built on NVIDIA Warp, with Newton-integrated WCSPH, DFSPH, PBF, and APIC solvers. It includes particle simulations, interactive viewers, USD scene loading, and an APIC wave-tank demo with surface reconstruction.

**APIC and DFSPH are still being tuned and validated. They are not final implementations; solver parameters, stability controls, and numerical behavior may change.**

## Demos

### APIC Demo

[![APIC Demo](resources/apic_preview.gif)](resources/apic.mp4)

[Full APIC video (MP4)](resources/apic.mp4)

### WCSPH Demo

![WCSPH Demo](resources/wsph_demo_02.gif)

## Solvers

| Method | Implementation | Description |
| --- | --- | --- |
| WCSPH | `solver/sph/solver_wcsph.py` | Weakly compressible SPH, with pressure computed from a density-based equation of state. |
| DFSPH | `solver/sph/solver_dfsph.py` | SPH with iterative velocity corrections for density error and density change rate. |
| PBF | `solver/pbf/solver_pbf.py` | Position-based density constraints, artificial pressure, and XSPH velocity smoothing. |
| APIC | `solver/apic/solver_apic.py` | Particle–grid simulation on a sparse `warp.fem.Nanogrid`, with affine velocity transfer and optional pressure projection. |

The Newton solvers follow `SolverBase.step(state_in, state_out, control, contacts, dt)`. Particle positions and velocities use Newton state arrays; additional quantities such as SPH density, PBF multipliers, and APIC affine matrices are registered as custom attributes.

APIC transfers particle velocities to the grid, applies gravity and box boundary conditions, projects grid velocities toward a divergence-free field, and transfers velocities and gradients back to particles. Its current configuration also includes particle separation and affine/speed limits. Surface reconstruction is a visualization step, separate from the pressure solve. These implementations are intended for experiments and learning; the included smoke tests cover a small set of cases.

## Environment

The documented setup uses **Python 3.13.5, Newton 1.6.0, and Warp 1.17.0**. Newton/Warp were checked in an isolated installation on **2026-09-28**; the remaining package versions and hardware below come from the development machine. This is a reproducibility reference, rather than a claim that every simulation has been fully validated.

The shared dam-break examples were also checked with **Python 3.12.13, Newton 1.6.0, and Warp 1.17.0** on 2026-09-29. See the [example guide](examples/dam_break/README.md) for the checks and parameter settings.

| Component | Version / configuration |
| --- | --- |
| OS | Ubuntu 24.04.1 LTS, x86_64 |
| Python | **3.13.5** |
| Newton (`newton`) | **1.6.0** |
| NVIDIA Warp (`warp-lang`) | **1.17.0** |
| NumPy | 2.4.3 |
| OpenUSD Python bindings (`usd-core`) | 26.8 |
| Pyglet | 2.1.16 |
| ImGui Bundle (`imgui-bundle`) | 1.92.801 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| NVIDIA driver | 580.173.02 |
| CUDA version reported by Warp | Toolkit 12.9; driver CUDA compatibility 13.0 |

CUDA values above describe the Warp runtime and driver, not a required local `nvcc` installation. The default demo device is `cuda:0`; interactive OpenGL viewers also require a working desktop/display environment. Small solver tests can use CPU.

### APIC demo compatibility

Surface reconstruction requires `newton.geometry.ParticleSurface`, introduced in **Newton 1.6.0**. Newton 1.6.0 also requires **`warp-lang>=1.17.0`**; the pinned setup uses Warp 1.17.0. See the [Newton 1.6.0 release notes](https://github.com/newton-physics/newton/releases/tag/v1.6.0) and [ParticleSurface API](https://newton-physics.github.io/newton/1.6.0/api/_generated/newton.geometry.ParticleSurface.html).

The four shared examples and archived Newton entry points use the documented Newton 1.6.0 / Warp 1.17.0 environment. The [archived APIC viewer](backup/examples/newton_apic_test.py) imports `ParticleSurface` at startup, so it also requires Newton 1.6.0 when surface display is disabled.

## Installation

For the dependency versions listed above, create a Python 3.13.5 environment and install the pinned packages:

```bash
git clone https://github.com/SParksLz/fluid_simulation.git
cd fluid_simulation

conda create -n fluid-simulation python=3.13.5 -y
conda activate fluid-simulation
python -m pip install -r requirements.txt
```

`requirements.txt` covers the solvers, APIC surface reconstruction, OpenGL viewers, and USD support. The optional RTX viewer additionally requires `ovrtx`; it is not installed or validated in the environment snapshot above.

To inspect the installed core versions:

```bash
python --version
python -c "from importlib.metadata import version; print('Newton:', version('newton')); print('Warp:', version('warp-lang'))"
```

## Running the examples

Run commands from the repository root. `--device` selects the Warp device.

### Shared dam-break examples

The examples in [`examples/dam_break/`](examples/dam_break/README.md) use the same APIC water column for **APIC, WCSPH, DFSPH, and PBF**: tank bounds `(-1.2, -0.28, 0)` to `(1.2, 0.28, 1.2)` meters, target 400,000 particles (401,856 actual), identical particle mass/radius/positions, zero initial velocity, and gravity `(0, 0, -10)`. All use meters directly, advance complete 1/60-second frames, and share the APIC surface reconstruction and camera settings. Solver-specific numerical settings remain distinct.

```bash
python examples/dam_break/apic.py
python examples/dam_break/wcsph.py
python examples/dam_break/dfsph.py
python examples/dam_break/pbf.py

# Headless surface USD animation, with the same options for each solver
python examples/dam_break/apic.py --frames 300 --no-viewer --usd-every 2 \
  --usd examples/dam_break/output/apic/surface.usdc

# Small headless particle simulation
python examples/dam_break/dfsph.py --particles 10000 --frames 60 --no-viewer --no-surface

# WCSPH particles with extra numerical damping
python examples/dam_break/wcsph.py --no-surface --wcsph-viscosity 5e-4
```

Default duration is 300 frames (5 seconds). Each run records scene specifications, an initial-position hash, solver settings, environment, and metrics under `examples/dam_break/output/<solver>/`. See the [dam-break guide](examples/dam_break/README.md) for particle-only USD export and surface options. **APIC and DFSPH remain under development.**

### Archived examples

The earlier root-level examples are now in [`backup/examples/`](backup/examples/README.md), including the four Newton entry points, the standalone tank/suction experiments, and their `wcsph_kernel.py`. Their default input paths still resolve to the repository's `temp/` directory. See the [archive guide](backup/examples/README.md) for the original commands and coordinate conventions.

For example, the earlier APIC viewer is available with `python backup/examples/newton_apic_test.py --help`.

## Tests

Run the APIC, original DFSPH kernel, and shared dam-break tests without installing pytest:

```bash
python -m unit_test.unit_test_apic
python -m unittest discover -s unit_test -p 'unit_test_dfsph.py' -v
python -m unittest unit_test.unit_test_dam_break -v
```

The APIC and original DFSPH suites select CUDA when available; the shared dam-break checks use CPU. On Linux, force CPU for small smoke tests with:

```bash
CUDA_VISIBLE_DEVICES="" python -m unit_test.unit_test_apic
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s unit_test -p 'unit_test_dfsph.py' -v
```

The APIC suite covers custom attributes, falling particles within a box, and a projection/thickness check. The DFSPH suite exercises the kernels in `backup/examples/wcsph_kernel.py`. The shared dam-break suite checks common particle specifications, complete frame advancement, theoretical height guides, and WCSPH pressure/timestep behavior.

On 2026-09-28, the documented Python/Newton/Warp combination passed all three APIC CPU tests, both original DFSPH CPU tests, and the `--help` import checks for all four Newton demos. These checks do not establish accuracy or stability for full-size GPU simulations.

After archiving the earlier entry points on 2026-09-29, all **15 CPU checks** and the `--help` checks for the four shared and four archived Newton entry points passed in the Python 3.12.13 / Newton 1.6.0 / Warp 1.17.0 environment.

## Repository layout

```text
solver/
  apic/solver_apic.py       Sparse FEM APIC solver
  sph/solver_wcsph.py       Newton WCSPH solver
  sph/solver_dfsph.py       Newton DFSPH solver
  pbf/solver_pbf.py         Newton PBF solver
examples/dam_break/         Shared scene, four solver demos, surface/USD output
backup/examples/           Earlier Newton and standalone Warp examples and kernels
renderer/render_opengl.py   Custom Warp OpenGL renderer
unit_test/                 APIC, original DFSPH, and shared dam-break checks
temp/                      Input USD scenes and local experiment outputs
resources/                 Demo videos and GIFs
docs/superpowers/          Historical APIC design and implementation notes
```

The APIC design notes describe the original skeleton before pressure projection was added; the current solver code is the source of truth for implemented features.
