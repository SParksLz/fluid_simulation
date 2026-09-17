# APIC Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Newton-integrated classic APIC solver using `warp.fem.Nanogrid`, without incompressible projection.

**Architecture:** Adapt Warp’s `example_apic_fluid.py` pipeline (Nanogrid rebuild → PicQuadrature → APIC P2G → box BC → G2P/advect), wrap it as `SolverAPIC(SolverBase)`, and drive it from `newton_apic_test.py`. Skip `solve_incompressibility` entirely in v1.

**Tech Stack:** Python, Warp 1.16+, `warp.fem`, Newton 1.5.x (`env_newton_1_5`), ViewerGL.

## Global Constraints

- Environment: conda `env_newton_1_5`
- Grid: `fem.Nanogrid` (sparse), not `Grid3D`
- No pressure / Poisson / EOS / SPH comparison in v1
- Future projection hook remains after grid BC and before G2P
- Follow existing Newton solver patterns (`register_custom_attributes`, `step(state_in, state_out, ...)`)
- Reference implementation: `/home/zhenliu/miniconda3/envs/env_newton_1_5/lib/python3.12/site-packages/warp/examples/fem/example_apic_fluid.py`

## File Structure

| File | Responsibility |
|---|---|
| `solver/apic/__init__.py` | Export `SolverAPIC` |
| `solver/apic/solver_apic.py` | Config, FEM integrands, Nanogrid lifecycle, `SolverAPIC.step` |
| `newton_apic_test.py` | Spawn particle block, ViewerGL loop, smoke demo |
| `unit_test/unit_test_apic.py` | Headless multi-step smoke (no GUI): finite positions, particles stay in box |

---

### Task 1: Scaffold SolverAPIC + custom attributes

**Files:**
- Create: `solver/apic/__init__.py`
- Create: `solver/apic/solver_apic.py`
- Test: `unit_test/unit_test_apic.py` (import + attribute registration only in this task)

**Interfaces:**
- Produces: `SolverAPIC`, `SolverAPIC.Config`, `SolverAPIC.register_custom_attributes(builder)`
- Custom attrs: model `apic:particle_volume` (`float`), state `apic:C` (`wp.mat33`)

- [ ] **Step 1: Create package export**

```python
# solver/apic/__init__.py
from .solver_apic import SolverAPIC

__all__ = ["SolverAPIC"]
```

- [ ] **Step 2: Scaffold `SolverAPIC` with Config and attribute registration**

Implement in `solver/apic/solver_apic.py`:

```python
@dataclass
class Config:
    voxel_size: float = 0.05
    grid_padding_voxels: int = 2
    bound_lo: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    bound_hi: tuple[float, float, float] = (1.0, 1.0, 2.0)
    mass_epsilon: float = 1.0e-8
    grid_capacity_ratio: float = 16.0

@classmethod
def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
    # register apic:particle_volume (MODEL, float, default 1.0)
    # register apic:C (STATE, wp.mat33, default identity/zero — use zeros)
```

Also implement `__init__` storing `config`, `fem.TemporaryStore()`, and placeholders for volume/grid/spaces (created lazily on first `step`).

- [ ] **Step 3: Write failing import/registration test**

```python
# unit_test/unit_test_apic.py
import newton
from solver.apic import SolverAPIC

def test_register_custom_attributes():
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverAPIC.register_custom_attributes(builder)
    builder.add_particles(
        pos=[[0.0, 0.0, 0.5]],
        vel=[[0.0, 0.0, 0.0]],
        mass=[1.0],
        radius=[0.02],
        custom_attributes={"apic:particle_volume": [0.000125]},
    )
    model = builder.finalize(device="cpu")
    state = model.state()
    assert hasattr(model, "apic")
    assert hasattr(state, "apic")
    assert state.apic.C.shape[0] == 1
```

- [ ] **Step 4: Run test**

```bash
conda run -n env_newton_1_5 python -m pytest unit_test/unit_test_apic.py::test_register_custom_attributes -v
```

Expected: PASS after Step 2.

- [ ] **Step 5: Commit**

```bash
git add solver/apic/__init__.py solver/apic/solver_apic.py unit_test/unit_test_apic.py
git commit -m "Scaffold SolverAPIC with Nanogrid-oriented config and attributes."
```

---

### Task 2: FEM integrands + Nanogrid APIC step (no projection)

**Files:**
- Modify: `solver/apic/solver_apic.py`
- Test: `unit_test/unit_test_apic.py`

**Interfaces:**
- Consumes: Task 1 scaffold
- Produces: `SolverAPIC.step(state_in, state_out, control, contacts, dt)` performing:
  1. rebuild/allocate Nanogrid from `state_in.particle_q`
  2. `PicQuadrature` with `model.apic.particle_volume` as measures
  3. APIC velocity gather (`integrate_velocity` style from Warp example)
  4. nodal inverse-mass normalize
  5. box free-slip projector via AABB SDF (`bound_lo`/`bound_hi`)
  6. **skip** `solve_incompressibility`
  7. G2P + advect into `state_out.particle_q/qd` and `state_out.apic.C`
  8. particle clamp into box (epsilon inset)

Reference snippets to adapt (do not import the example module as a hard dependency for runtime; copy the needed integrands):

- `integrate_fraction`, `integrate_velocity`, `update_particles`, `invert_volume_kernel`, `scalar_vector_multiply`
- Replace half-ball `collision_sdf` with an **inverted box** SDF (inside domain = positive or follow example’s convention: `sdf <= 0` means collider; for a container, outside walls / below floor should constrain).

Recommended box convention for this repo (Z-up, matching PBF tests):

- Domain AABB `[bound_lo, bound_hi]`
- Outside or on wall → apply free-slip (remove outward normal velocity)
- Particles clamped to `[bound_lo+eps, bound_hi-eps]`

Nanogrid lifecycle (follow Warp example):

```python
volume = wp.Volume.allocate_by_voxels(
    voxel_points=positions,
    voxel_size=config.voxel_size,
    rebuildable=True,
    status=self._grid_status,
    # capacity estimates similar to example_apic_fluid
)
grid = fem.Nanogrid(volume, rebuildable=True)
# later steps:
grid.rebuild(positions, status=self._grid_status)
linear_basis_space.topology.rebuild()
```

Velocity/fraction spaces: degree-1 collocated `vec3` / `float` on the Nanogrid.  
Active-cell partition via `pic.fill_element_mask` + `ExplicitGeometryPartition` as in the example.

- [ ] **Step 1: Implement integrands + `step` without projection**

Leave a clearly commented hook:

```python
# Future: pressure projection goes here (after BC, before G2P).
```

- [ ] **Step 2: Add headless multi-step smoke test**

```python
def test_apic_falls_and_stays_in_box():
    # Build a small particle block above z=0.3 inside a box on cuda:0 if available else cpu
    # Run ~60 substeps of SolverAPIC.step
    # Assert all positions finite
    # Assert all positions within bound_lo/hi (+ small tol)
    # Assert mean z decreased vs initial (gravity along -Z or model gravity)
```

Use `model.set_gravity((0,0,-10))` and `up_axis=Z`.

- [ ] **Step 3: Run tests**

```bash
conda run -n env_newton_1_5 python -m pytest unit_test/unit_test_apic.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add solver/apic/solver_apic.py unit_test/unit_test_apic.py
git commit -m "Implement Nanogrid APIC transfer step without pressure projection."
```

---

### Task 3: Viewer demo script

**Files:**
- Create: `newton_apic_test.py`

**Interfaces:**
- Consumes: `SolverAPIC` from Task 2
- Produces: runnable ViewerGL demo

- [ ] **Step 1: Create `newton_apic_test.py`**

Mirror the structure of `newton_pbf_test.py` enough to:

- spawn a particle cube (procedural; USD optional)
- register attributes, build model, create `SolverAPIC`
- substep each frame
- `newton.viewer.ViewerGL`

CLI: `--device cuda:0`, `--steps N` for headless quit after N frames (optional flag `--headless` that runs N steps then exits without opening GL if easy; otherwise document Ctrl+C).

Minimum procedural spawn: ~16³ or smaller block, `voxel_size≈2*particle_spacing`.

- [ ] **Step 2: Smoke-run headless path or short scripted steps**

Prefer driving the same setup as the unit test, or:

```bash
conda run -n env_newton_1_5 python newton_apic_test.py --device cuda:0 --frames 30 --no-viewer
```

If ViewerGL cannot be disabled cleanly, rely on unit test for CI and manually note viewer command:

```bash
conda run -n env_newton_1_5 python newton_apic_test.py --device cuda:0
```

- [ ] **Step 3: Commit**

```bash
git add newton_apic_test.py
git commit -m "Add Newton ViewerGL demo for SolverAPIC skeleton."
```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| `solver/apic/` + `SolverAPIC` | Task 1–2 |
| `fem.Nanogrid` | Task 2 |
| APIC `C` / velocity gradient | Task 1 attr + Task 2 G2P |
| No incompressibility | Task 2 skips projection |
| Gravity + box BC + clamp | Task 2 |
| Newton `step` integration | Task 2 |
| Demo script | Task 3 |
| `env_newton_1_5` verification | Task 2–3 |

## Notes for implementer

- Prefer copying/adapting Warp example integrands over subclassing `SolverImplicitMPM`.
- Guard zero nodal mass with `mass_epsilon` / `invert_volume_kernel` pattern (`0 → 0`, else `1/m`).
- Keep particle_volume as a model attribute set at spawn (`spacing³` or packing fraction × cell volume).
