import numpy as np
import warp as wp

import newton
from solver.apic import SolverAPIC


def _attr_namespaces_ok():
    return hasattr(newton, "ModelAttributeFrequency") or (
        hasattr(newton.Model, "AttributeFrequency") and hasattr(newton.Model.AttributeFrequency, "PARTICLE")
    )


def test_register_custom_attributes():
    if not _attr_namespaces_ok():
        print("skip: particle custom attributes unavailable")
        return
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


def _spawn_block(builder: newton.ModelBuilder, lo, hi, res, density=1000.0):
    xs = np.linspace(lo[0], hi[0], res[0])
    ys = np.linspace(lo[1], hi[1], res[1])
    zs = np.linspace(lo[2], hi[2], res[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    spacing = float(
        min(
            (hi[0] - lo[0]) / max(res[0] - 1, 1),
            (hi[1] - lo[1]) / max(res[1] - 1, 1),
            (hi[2] - lo[2]) / max(res[2] - 1, 1),
        )
    )
    volume = spacing**3
    mass = density * volume
    radius = 0.5 * spacing
    n = positions.shape[0]
    builder.add_particles(
        pos=positions.tolist(),
        vel=np.zeros_like(positions).tolist(),
        mass=[mass] * n,
        radius=[radius] * n,
        custom_attributes={"apic:particle_volume": [volume] * n},
    )
    return spacing, positions


def test_apic_falls_and_stays_in_box():
    if not _attr_namespaces_ok():
        print("skip: particle custom attributes unavailable")
        return
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    bound_lo = (-0.4, -0.4, 0.0)
    bound_hi = (0.4, 0.4, 0.8)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverAPIC.register_custom_attributes(builder)
    spacing, _positions0 = _spawn_block(
        builder,
        lo=(-0.1, -0.1, 0.35),
        hi=(0.1, 0.1, 0.55),
        res=(6, 6, 6),
    )
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -10.0))

    config = SolverAPIC.Config(
        voxel_size=max(spacing, 0.04),
        bound_lo=bound_lo,
        bound_hi=bound_hi,
        clamp_eps=1.0e-3,
        grid_capacity_ratio=8.0,
        enable_projection=True,
        projection_quiet=True,
    )
    solver = SolverAPIC(model, config)
    state_0 = model.state()
    state_1 = model.state()

    z0 = float(np.mean(state_0.particle_q.numpy()[:, 2]))
    dt = 1.0 / 120.0
    cur = state_0
    for i in range(60):
        if i % 2 == 0:
            solver.step(state_0, state_1, None, None, dt)
            cur = state_1
        else:
            solver.step(state_1, state_0, None, None, dt)
            cur = state_0

    q = cur.particle_q.numpy()
    assert np.isfinite(q).all(), "non-finite particle positions"
    assert q[:, 0].min() >= bound_lo[0] - 1.0e-3
    assert q[:, 0].max() <= bound_hi[0] + 1.0e-3
    assert q[:, 1].min() >= bound_lo[1] - 1.0e-3
    assert q[:, 1].max() <= bound_hi[1] + 1.0e-3
    assert q[:, 2].min() >= bound_lo[2] - 1.0e-3
    assert q[:, 2].max() <= bound_hi[2] + 1.0e-3
    z1 = float(np.mean(q[:, 2]))
    assert z1 < z0 - 1.0e-3, f"expected fall under gravity, z0={z0}, z1={z1}"
    assert solver.last_projection_iters >= 0


def test_projection_keeps_thickness_after_impact():
    """After hitting the floor, projected APIC should keep nonzero fluid height."""
    if not _attr_namespaces_ok():
        print("skip: particle custom attributes unavailable")
        return
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    bound_lo = (-0.5, -0.5, 0.0)
    bound_hi = (0.5, 0.5, 1.0)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverAPIC.register_custom_attributes(builder)
    spacing, _ = _spawn_block(
        builder,
        lo=(-0.1, -0.1, 0.35),
        hi=(0.1, 0.1, 0.55),
        res=(6, 6, 6),
    )
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -10.0))
    solver = SolverAPIC(
        model,
        SolverAPIC.Config(
            voxel_size=max(spacing, 0.04),
            bound_lo=bound_lo,
            bound_hi=bound_hi,
            clamp_eps=1.0e-3,
            grid_capacity_ratio=8.0,
            enable_projection=True,
            projection_quiet=True,
        ),
    )
    state_0 = model.state()
    state_1 = model.state()
    dt = 1.0 / 120.0
    cur = state_0
    saw_projection = False
    max_span_after_proj = 0.0
    for i in range(50):
        if i % 2 == 0:
            solver.step(state_0, state_1, None, None, dt)
            cur = state_1
        else:
            solver.step(state_1, state_0, None, None, dt)
            cur = state_0
        q = cur.particle_q.numpy()
        z_span = float(q[:, 2].max() - q[:, 2].min())
        if solver.last_projection_iters > 0:
            saw_projection = True
            max_span_after_proj = max(max_span_after_proj, z_span)

    print(
        f"max_span_after_proj={max_span_after_proj:.4f} saw_projection={saw_projection} "
        f"last_iters={solver.last_projection_iters}"
    )
    assert np.isfinite(q).all()
    assert saw_projection, "expected pressure solve to run after impact"
    assert max_span_after_proj > 0.05, (
        f"expected fluid thickness after projection starts, got {max_span_after_proj}"
    )

if __name__ == "__main__":
    test_register_custom_attributes()
    print("test_register_custom_attributes OK")
    test_apic_falls_and_stays_in_box()
    print("test_apic_falls_and_stays_in_box OK")
    test_projection_keeps_thickness_after_impact()
    print("test_projection_keeps_thickness_after_impact OK")
