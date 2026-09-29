"""Run any fluid solver on the same APIC dam-break scene."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import newton
import warp as wp

from examples.dam_break.scene import DEFAULT_SPEC, make_scene
from examples.dam_break.simulation import DamBreakSimulation, SOLVERS
from examples.dam_break.surface import SurfaceOutput, tank_edges
from examples.dam_break.guides import height_guides


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def positive_float(value):
    value = float(value)
    if not 0.0 < value < float("inf"):
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return value


def nonnegative_float(value):
    value = float(value)
    if not 0.0 <= value < float("inf"):
        raise argparse.ArgumentTypeError("Must be a finite non-negative number")
    return value


def parse_args(method=None):
    parser = argparse.ArgumentParser(description="Shared APIC-specification dam-break example")
    if method is None:
        parser.add_argument("--solver", choices=SOLVERS, default="apic")
    else:
        parser.set_defaults(solver=method)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--particles", type=positive_int, default=DEFAULT_SPEC.target_count,
                        help="Shared target count; the APIC lattice determines the actual count")
    parser.add_argument("--frames", type=positive_int, default=300)
    parser.add_argument("--substeps", type=positive_int, default=4,
                        help="Base substeps per 1/60 s frame; SPH may use more for CFL")
    parser.add_argument("--viewer", choices=["gl", "none"], default="gl")
    parser.add_argument("--no-viewer", dest="viewer", action="store_const", const="none")
    parser.add_argument("--no-surface", action="store_true")
    parser.add_argument("--show-particles", action="store_true")
    parser.add_argument("--usd", help="Output surface animation (.usd or .usdc); particles with --no-surface")
    parser.add_argument("--usd-every", type=positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None, help="Metrics and run metadata directory")
    parser.add_argument("--surface-voxel-size", type=positive_float, default=0.007)
    parser.add_argument("--surface-max-grid-cells", type=positive_int, default=None)
    parser.add_argument("--no-anisotropic", action="store_true")
    parser.add_argument("--water-opacity", type=float, default=0.35)
    parser.add_argument("--voxel-size", type=positive_float, default=None, help="APIC simulation voxel size")
    parser.add_argument("--no-projection", action="store_true", help="Disable APIC pressure projection")
    parser.add_argument("--wcsph-stiffness", type=positive_float, default=None,
                        help="WCSPH Tait pressure constant B in Pa (default: 250000)")
    parser.add_argument("--wcsph-viscosity", type=nonnegative_float, default=None,
                        help="WCSPH kinematic viscosity in m^2/s (default: 1e-6)")
    parser.add_argument("--wcsph-smoothing-length-coff", type=positive_float, default=None,
                        help="WCSPH kernel support radius / particle spacing (default: 2)")
    parser.add_argument("--wcsph-allow-negative-pressure", action="store_true",
                        help="Disable the pressure clamp for WCSPH comparisons")
    parser.add_argument("--report-every", type=positive_int, default=10)
    parser.add_argument("--metrics-every", type=positive_int, default=10,
                        help="Full statistics interval; finite-state checks still run every frame")
    parser.add_argument("--no-cuda-interop", action="store_true",
                        help="Use CPU uploads for OpenGL point transforms")
    args = parser.parse_args()
    if not 0.0 <= args.water_opacity <= 1.0:
        parser.error("--water-opacity must be in [0, 1]")
    if args.usd and Path(args.usd).suffix.lower() not in (".usd", ".usdc"):
        parser.error("--usd requires .usd or .usdc (binary USD)")
    if args.solver != "apic" and (args.voxel_size is not None or args.no_projection):
        parser.error("--voxel-size and --no-projection apply to APIC")
    if args.solver != "wcsph" and (
        args.wcsph_stiffness is not None or args.wcsph_viscosity is not None
        or args.wcsph_smoothing_length_coff is not None or args.wcsph_allow_negative_pressure
    ):
        parser.error("--wcsph-* options apply to WCSPH")
    args.output_dir = args.output_dir or Path(__file__).parent / "output" / args.solver
    return args


def main(method=None):
    args = parse_args(method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = make_scene(args.particles)
    print(f"{args.solver.upper()} dam_break: {scene.count} particles, "
          f"radius={scene.radius:.6g} m, spacing={scene.spacing:.6g} m", flush=True)
    print(f"tank={scene.spec.bound_lo} .. {scene.spec.bound_hi}; "
          f"{args.frames} frames at {scene.spec.fps:g} fps", flush=True)
    print(f"ideal flat height={scene.ideal_height:.6f} m; "
          f"ruler=100%, 90%, 80% of rest volume", flush=True)
    wcsph_options = {key: getattr(args, key) for key in
                    ("wcsph_stiffness", "wcsph_viscosity", "wcsph_smoothing_length_coff")
                    if getattr(args, key) is not None}
    sim = DamBreakSimulation(scene, args.solver, args.device, args.substeps,
                             args.voxel_size, not args.no_projection,
                             wcsph_clamp_negative_pressure=not args.wcsph_allow_negative_pressure,
                             **wcsph_options)
    if args.solver == "wcsph":
        print(f"WCSPH B={sim.material['stiffness']:g} Pa, "
              f"viscosity={sim.material['viscosity']:g} m^2/s, "
              f"h={sim.solver.smoothing_length:.6g} m, "
              f"non-negative pressure={sim.solver.config.clamp_negative_pressure}", flush=True)
    # Without a viewer or USD sink, skip expensive surface allocation.
    output = SurfaceOutput(sim, args) if args.viewer != "none" or args.usd else None
    viewer = None
    particle_colors = None
    guide_buffers = []
    if args.viewer == "gl":
        interop = newton.viewer.ViewerGL.CudaInterop
        flags = interop.DYNAMIC_MESH
        if sim.model.device.is_cuda and not args.no_cuda_interop:
            flags |= interop.POINTS
        viewer = newton.viewer.ViewerGL(paused=False, enable_cuda_interop=flags)
        viewer.set_model(sim.model)
        viewer.show_particles = False
        viewer.set_camera(wp.vec3(0.0, -2.6, 0.70), -0.18, 90.0)
        starts, ends = tank_edges(scene.spec)
        starts = wp.array(starts, dtype=wp.vec3, device=sim.model.device)
        ends = wp.array(ends, dtype=wp.vec3, device=sim.model.device)
        if args.show_particles or args.no_surface:
            # ViewerGL's point upload path requires an array, even for uniform colors.
            particle_colors = wp.full(scene.count, wp.vec3(0.25, 0.55, 0.95),
                                      dtype=wp.vec3, device=sim.model.device)
        for guide in height_guides(scene):
            guide_buffers.append((guide, wp.array(guide.starts, dtype=wp.vec3, device=sim.model.device),
                                  wp.array(guide.ends, dtype=wp.vec3, device=sim.model.device)))

        def height_panel(imgui):
            imgui.text(f"Rest-volume water height: {scene.ideal_height:.4f} m")
            imgui.text("100%: rest volume (orange)")
            imgui.text("90% / 80%: 10% / 20% volume reduction")
            imgui.text("Vertical ticks: 0.1 m")
            imgui.text("Compare settled water; waves change local height.")

        viewer.register_ui_callback(height_panel, position="panel")

    last_visible_mesh = None
    particle_colors_uploaded = False

    def render():
        nonlocal last_visible_mesh, particle_colors_uploaded
        if viewer is None:
            return
        viewer.begin_frame(sim.sim_time)
        viewer.log_lines("/dam_break/tank", starts=starts, ends=ends,
                         colors=(0.55, 0.55, 0.60), width=0.004)
        for guide, guide_starts, guide_ends in guide_buffers:
            viewer.log_lines(f"/dam_break/height/{guide.name}", starts=guide_starts,
                             ends=guide_ends, colors=guide.color, width=guide.width)
        if args.show_particles or args.no_surface:
            viewer.log_points("/dam_break/particles", points=sim.state_0.particle_q,
                              radii=sim.model.particle_radius,
                              colors=None if particle_colors_uploaded else particle_colors)
            particle_colors_uploaded = True
        if output.enabled:
            vertices, indices, normals = output.extract()
            if vertices is not None and vertices.shape[0] > 0 and indices.shape[0] > 0:
                last_visible_mesh = (vertices, indices, normals)
                viewer.log_mesh("/dam_break/surface", vertices, indices, normals,
                                dynamic=True, color=(0.12, 0.42, 0.65), roughness=0.08,
                                metallic=0.0, opacity=args.water_opacity)
            elif last_visible_mesh is not None:
                # ViewerGL requires valid buffers even when hiding an existing mesh.
                viewer.log_mesh("/dam_break/surface", *last_visible_mesh, dynamic=True, hidden=True)
        viewer.end_frame()

    rows = []
    started = time.perf_counter()
    try:
        rows.append(sim.metrics())
        if output:
            output.record()
        render()
        while sim.frame < args.frames and (viewer is None or viewer.is_running()):
            if viewer is not None and not viewer.should_step():
                render()
                continue
            sim.step()
            sim.check_finite()
            report_now = sim.frame % args.report_every == 0 or sim.frame == args.frames
            if sim.frame % args.metrics_every == 0 or report_now:
                metrics = sim.metrics()
                rows.append(metrics)
            if output and sim.frame % args.usd_every == 0:
                output.record()
            render()
            if report_now:
                print(f"frame {sim.frame}/{args.frames}, t={sim.sim_time:.4f} s, "
                      f"substeps={sim.last_substeps}, vmax={metrics['speed_max']:.4g}, finite=True", flush=True)
        if output:
            output.record()  # Always include the actual final state.
        if rows[-1]["frame"] != sim.frame:
            rows.append(sim.metrics())
    finally:
        if output:
            output.close()
        if viewer:
            viewer.close()
    elapsed = time.perf_counter() - started
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(scene=scene.report(), simulation=sim.report(), completed_frames=sim.frame,
                  simulation_seconds=sim.sim_time, elapsed_seconds=elapsed,
                  diagnostics=dict(metrics_every=args.metrics_every, finite_check_every=1),
                  environment=dict(python=sys.version.split()[0], newton=newton.__version__,
                                   warp=wp.__version__, device=str(sim.model.device)),
                  surface=output.report() if output else dict(enabled=False), final_metrics=rows[-1])
    (args.output_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Finished in {elapsed:.2f} s. Metrics: {args.output_dir}", flush=True)
    if args.usd:
        print(f"USD: {output.usd_path} ({output.samples} samples)", flush=True)


if __name__ == "__main__":
    main()
