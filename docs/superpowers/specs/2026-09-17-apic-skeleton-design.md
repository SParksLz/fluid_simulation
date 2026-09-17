# APIC Skeleton Design (No Incompressibility)

Date: 2026-09-17  
Status: Draft for implementation planning  
Environment: conda `env_newton_1_5` (Newton 1.5.x + Warp with `warp.fem`)

## Goal

Add a **classic APIC framework** to this repository so we can later implement incompressible fluid simulation on a particle–grid pipeline.

v1 scope is intentionally narrow:

- Implement APIC particle ↔ grid transfer and time integration.
- Do **not** implement incompressible projection, pressure Poisson, EOS, viscosity models, or SPH comparisons.

Success for v1: particles fall under gravity inside a box, using APIC on a FEM sparse grid, visible in Newton ViewerGL.

## Context

Existing solvers in this repo (`SolverWCSPH`, `SolverDFSPH`, `SolverPBF`) are Lagrangian SPH-style and integrate with Newton via `SolverBase.step(state_in, state_out, control, contacts, dt)`.

Newton 1.5 already ships `SolverImplicitMPM`, which:

- defaults to `transfer_scheme="apic"`;
- uses `warp.fem` geometry, with default `grid_type="sparse"` → `fem.Nanogrid`;
- targets granular / elasto-plastic rheology, **not** classic incompressible liquids.

We will **borrow FEM + APIC transfer patterns** from ImplicitMPM, but **not** reuse its rheology / stress / yield pipeline as the fluid core.

## Non-Goals (v1)

- Pressure projection / divergence-free velocity correction
- Weakly compressible EOS pressure
- Constitutive models (elasticity, plasticity, viscosity beyond optional later hooks)
- Coupling to rigid bodies beyond simple domain walls
- Performance parity with ImplicitMPM or CUDA-graph hardening (nice-to-have only)
- Porting or wrapping `SolverImplicitMPM` as the fluid solver

## Approach

**Self-contained `SolverAPIC(SolverBase)` using `warp.fem.Nanogrid` + `PicQuadrature`.**

Rationale:

- Matches the user’s preference to use FEM as the grid representation.
- Aligns with ImplicitMPM’s default sparse grid path (`Nanogrid`), without dragging in rheology.
- Keeps a clean insertion point for a future FEM pressure projection between grid velocity update and G2P.

## Architecture

### Layout

```
solver/apic/
  __init__.py
  solver_apic.py          # SolverAPIC + kernels / integrands
newton_apic_test.py       # minimal Newton + ViewerGL demo
```

### Data ownership

| Layer | Representation |
|---|---|
| Particles | Newton `Model` / `State`: `particle_q`, `particle_qd`, `particle_mass`, flags |
| APIC affine | Custom state attribute `apic:C` (`wp.mat33`) per particle |
| Grid geometry | `warp.fem.Nanogrid` built from particle-occupied voxels (+ padding) |
| Particle sampling | `fem.PicQuadrature` over the Nanogrid |
| Grid velocity | FEM `Q1` vector field (nodal velocities) |
| Grid mass | Nodal mass from particle mass scatter |

Optional v1 config knobs:

- `voxel_size`
- `grid_padding` (voxel padding around particles)
- gravity from `model.gravity`
- simple box bounds for wall velocity constraints / particle clamp

### Time step pipeline

1. **Build / refresh sparse grid** from particle positions → `fem.Nanogrid`.
2. **Build `PicQuadrature`** from particle positions (and masses as needed).
3. **P2G (APIC)**  
   Scatter particle mass and APIC momentum  
   \(v_i \leftarrow v_p + C_p (x_i - x_p)\)  
   into nodal mass / momentum; form nodal velocity \(v_i = q_i / m_i\).
4. **Grid forces / BCs only**  
   Apply gravity to nodal velocities.  
   Apply simple domain-wall constraints on boundary-adjacent DOFs (zero or reflect normal component).  
   **No pressure / projection.**
5. **G2P (APIC)**  
   Interpolate particle velocity from the grid field; update affine `C` from nodal velocities (APIC formula).
6. **Advection**  
   \(x \leftarrow x + dt\, v\).  
   Optional particle clamp against the box to avoid permanent escape while the grid BC is still minimal.

### Explicitly deferred insertion point

Future incompressible step belongs **after grid gravity/BC and before G2P**:

`P2G → gravity/BC → [pressure projection] → G2P → advect`

v1 leaves that bracket empty.

## Newton integration pattern

Follow existing solvers:

- `SolverAPIC.register_custom_attributes(builder)` before `builder.finalize`
- `SolverAPIC(model, SolverAPIC.Config(...))`
- `step(state_in, state_out, control, contacts, dt)` writes particle positions/velocities (and `apic:C`) into `state_out`
- Pass through body/joint state if present (same as PBF passthrough)

Demo script `newton_apic_test.py` mirrors `newton_pbf_test.py` structure enough to:

- create a particle block / load a simple distribution
- construct model with APIC attributes
- substep with fixed `dt`
- render with `newton.viewer.ViewerGL`

## Relationship to ImplicitMPM

| Borrow | Do not borrow |
|---|---|
| `Nanogrid` sparse grid idea | Rheology / yield / hardening |
| `PicQuadrature` particle sampling | Implicit stress / strain solve |
| APIC transfer formulas / integrand style | Volume-fraction packing constraints as the “incompressibility” |
| Newton `SolverBase` + custom attributes pattern | Treating high bulk modulus solid as a substitute for liquid projection |

## Testing / acceptance

Manual:

- Run `newton_apic_test.py` under `env_newton_1_5`
- Particles fall and remain roughly inside the box
- Simulation remains numerically stable for at least several hundred frames at a modest timestep

Optional sanity (if cheap):

- Compare PIC (`C=0` always) vs APIC on the same scene; APIC should preserve local shear/rotation better (qualitative)

## Risks

- **Nanogrid rebuild cost / API details**: follow ImplicitMPM’s sparse construction patterns carefully; start with a conservative padding and fixed `voxel_size`.
- **Empty nodes / tiny mass**: guard nodal velocity with mass threshold to avoid blow-ups.
- **Boundary treatment too weak**: particle clamp + nodal wall BC both recommended in v1 to keep the demo usable.
- **Over-coupling to ImplicitMPM internals**: prefer re-implementing a minimal APIC transfer on FEM primitives over subclassing `SolverImplicitMPM`.

## Implementation order (for the later plan)

1. Scaffold `solver/apic` + Config + custom attributes
2. Nanogrid + PicQuadrature build from particles
3. P2G mass/momentum + nodal velocity
4. Gravity + box BC
5. G2P velocity + `C` update + advect
6. `newton_apic_test.py` demo
7. Smoke-run in `env_newton_1_5`
