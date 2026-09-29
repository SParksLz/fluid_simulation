"""Synchronized solver/display timings; run each comparison in a fresh process."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import newton
import numpy as np
import warp as wp

from examples.dam_break.guides import height_guides
from examples.dam_break.scene import make_scene
from examples.dam_break.simulation import DamBreakSimulation, SOLVERS
from examples.dam_break.surface import tank_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=SOLVERS, default="pbf")
    parser.add_argument("--particles", type=int, default=400000)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--baseline", action="store_true", help="Original CPU uploads and every-frame statistics")
    parser.add_argument("--reference-pbf", type=Path, help="Optional original solver source for PBF comparison")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output/performance/benchmark.json")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.frames <= args.warmup or args.warmup < 0:
        parser.error("--frames must exceed a nonnegative --warmup")
    scene = make_scene(args.particles)
    sim = DamBreakSimulation(scene, args.solver)
    if args.reference_pbf:
        if args.solver != "pbf":
            parser.error("--reference-pbf only applies to PBF")
        spec = importlib.util.spec_from_file_location("dam_break_pbf_reference", args.reference_pbf.resolve())
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        sim.solver = reference.SolverPBF(sim.model, reference.SolverPBF.PbfConfig(**asdict(sim.solver.config)))
    flags = newton.viewer.ViewerGL.CudaInterop.DYNAMIC_MESH
    if not args.baseline:
        flags |= newton.viewer.ViewerGL.CudaInterop.POINTS
    viewer = newton.viewer.ViewerGL(headless=True, width=args.width, height=args.height,
                                    paused=False, enable_cuda_interop=flags)
    viewer.set_model(sim.model)
    viewer.show_particles = False
    viewer.set_camera(wp.vec3(0.0, -2.6, 0.70), -0.18, 90.0)
    starts, ends = tank_edges(scene.spec)
    starts, ends = (wp.array(x, dtype=wp.vec3, device=sim.model.device) for x in (starts, ends))
    guides = [(g, wp.array(g.starts, dtype=wp.vec3, device=sim.model.device),
               wp.array(g.ends, dtype=wp.vec3, device=sim.model.device)) for g in height_guides(scene)]
    colors = wp.full(scene.count, wp.vec3(0.25, 0.55, 0.95), dtype=wp.vec3, device=sim.model.device)

    def sync():
        wp.synchronize_device(sim.model.device)
        viewer.renderer.gl.glFinish()

    def measure(fn):
        sync()
        started = time.perf_counter()
        fn()
        sync()
        return (time.perf_counter() - started) * 1000.0

    rows = []
    try:
        for frame in range(args.frames):
            row = {"solve_ms": measure(sim.step)}

            def stats():
                if args.baseline:
                    sim.metrics()
                else:
                    sim.check_finite()
                    if sim.frame % 10 == 0 or frame == args.frames - 1:
                        sim.metrics()

            row["diagnostics_ms"] = measure(stats)
            viewer.begin_frame(sim.sim_time)

            def upload():
                viewer.log_lines("/tank", starts, ends, colors=(0.55, 0.55, 0.60), width=0.004)
                for guide, a, b in guides:
                    viewer.log_lines(f"/height/{guide.name}", a, b, colors=guide.color, width=guide.width)
                viewer.log_points("/particles", sim.state_0.particle_q, sim.model.particle_radius,
                                  colors=colors if args.baseline or frame == 0 else None)

            row["upload_ms"] = measure(upload)
            row["draw_ms"] = measure(viewer.end_frame)
            if frame >= args.warmup:
                rows.append(row)
        pixels = viewer.get_frame().numpy()
        rgb = pixels.astype(np.int16)
        blue_pixels = int(np.sum((rgb[:, :, 2] > rgb[:, :, 0] + 20) & (rgb[:, :, 2] > 50)))
        if blue_pixels < 1000:
            raise RuntimeError("Particle image missing; do not use these display timings")
        if args.screenshot:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(args.screenshot)
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        mean["total_ms"] = sum(mean.values())
        mean["fps"] = 1000.0 / mean["total_ms"]
        report = dict(method="CUDA synchronize + OpenGL glFinish; headless particle viewer",
                      particles=scene.count, resolution=[args.width, args.height], surface=False,
                      physics=sim.report(), measured_frames=len(rows), warmup_frames=args.warmup,
                      baseline=args.baseline, reference_pbf=str(args.reference_pbf),
                      rendered_blue_pixels=blue_pixels, mean=mean, frame_timings=rows,
                      environment=dict(python=sys.version.split()[0], newton=newton.__version__,
                                       warp=wp.__version__, device=str(sim.model.device)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(mean, indent=2), flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
