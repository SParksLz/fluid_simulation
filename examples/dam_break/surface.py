"""Shared APIC surface settings and time-sampled USD output."""

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from examples.dam_break.guides import height_guides


def tank_edges(spec):
    lo, hi = spec.bound_lo, spec.bound_hi
    corners = np.asarray([
        (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
    ], dtype=np.float32)
    edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                        (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
    return corners[edges[:, 0]], corners[edges[:, 1]]


class SurfaceOutput:
    def __init__(self, sim, args):
        self.sim = sim
        self.enabled = not args.no_surface
        self.surface = None
        self.stage = None
        self.mesh = None
        self.points = None
        self.usd_path = Path(args.usd).resolve() if args.usd else None
        self.samples = 0
        self.max_triangles = 0
        self.last_extract_frame = None
        self.last_arrays = (None, None, None)
        self.parameters = {}
        self.frame_samples = []
        if self.enabled:
            if not sim.model.device.is_cuda:
                raise ValueError("ParticleSurface requires CUDA; use --no-surface on CPU")
            from newton.geometry import ParticleSurface
            voxel = args.surface_voxel_size
            radius = max(0.007, voxel)
            pad = 4.0 * voxel + radius
            extent = np.asarray(sim.scene.spec.bound_hi) - np.asarray(sim.scene.spec.bound_lo)
            estimated_cells = int(np.prod(np.ceil((extent + 2.0 * pad) / voxel)))
            capacity = args.surface_max_grid_cells or max(8_000_000, int(1.5 * estimated_cells))
            # Match backup/examples/newton_apic_test.py's water appearance for every solver.
            self.parameters = dict(
                voxel_size=voxel, max_grid_cells=capacity, kernel_radius=radius,
                threshold=0.441, smooth_lambda=0.0, anisotropic=not args.no_anisotropic,
                kernel_scale=0.801, anisotropy_ratio=12.768, anisotropy_scale=1.338,
                anisotropy_min_neighbors=4, anisotropy_binning=not args.no_anisotropic,
                anisotropy_strength=0.871, field_smooth_iterations=1,
                field_smooth_radius=2, field_mode="density", redistance_iterations=0,
                mesh_smooth_iterations=1,
            )
            self.surface = ParticleSurface(**self.parameters, device=sim.model.device)
        if self.usd_path:
            self.usd_path.parent.mkdir(parents=True, exist_ok=True)
            self.stage = Usd.Stage.CreateNew(str(self.usd_path))
            self.stage.SetTimeCodesPerSecond(sim.scene.spec.fps)
            self.stage.SetFramesPerSecond(sim.scene.spec.fps)
            self.stage.SetStartTimeCode(0.0)
            self.stage.SetEndTimeCode(0.0)
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/Fluid").GetPrim())
            self.stage.GetRootLayer().customLayerData = {
                "solver": sim.method, "scene": "APIC dam-break specifications",
                "initial_position_sha256": sim.scene.report()["initial_position_sha256"],
            }
            if self.enabled:
                self.mesh = UsdGeom.Mesh.Define(self.stage, "/Fluid/Surface")
                self.mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                self.mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                self.mesh.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.42, 0.65)])
                self._water_material(args.water_opacity)
            if args.show_particles or not self.enabled:
                self.points = UsdGeom.Points.Define(self.stage, "/Fluid/Particles")
                self.points.CreateWidthsAttr([sim.scene.radius * 2.0])
                self.points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                self.points.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.55, 0.95)])
            starts, ends = tank_edges(sim.scene.spec)
            lines = UsdGeom.BasisCurves.Define(self.stage, "/Fluid/TankGuide")
            lines.CreateTypeAttr(UsdGeom.Tokens.linear)
            lines.CreateCurveVertexCountsAttr([2] * len(starts))
            lines.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.stack([starts, ends], axis=1).reshape(-1, 3)))
            lines.CreateWidthsAttr([0.004])
            lines.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            lines.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.60)])
            lines.CreatePurposeAttr(UsdGeom.Tokens.guide)
            for guide in height_guides(sim.scene):
                curves = UsdGeom.BasisCurves.Define(self.stage, f"/Fluid/HeightGuides/{guide.name}")
                curves.CreateTypeAttr(UsdGeom.Tokens.linear)
                curves.CreateCurveVertexCountsAttr([2] * len(guide.starts))
                curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(
                    np.stack([guide.starts, guide.ends], axis=1).reshape(-1, 3)))
                curves.CreateWidthsAttr([guide.width])
                curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                curves.CreateDisplayColorAttr([Gf.Vec3f(*guide.color)])
                curves.CreatePurposeAttr(UsdGeom.Tokens.guide)
                if guide.volume_fraction is not None:
                    curves.GetPrim().SetCustomDataByKey("restVolumeFraction", guide.volume_fraction)
                    curves.GetPrim().SetCustomDataByKey("waterHeightMeters", sim.scene.ideal_height * guide.volume_fraction)

    def _water_material(self, opacity):
        material = UsdShade.Material.Define(self.stage, "/Fluid/Materials/Water")
        shader = UsdShade.Shader.Define(self.stage, "/Fluid/Materials/Water/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.42, 0.65))
        for name, value in dict(roughness=0.08, metallic=0.0, opacity=opacity, ior=1.333,
                                clearcoat=1.0, clearcoatRoughness=0.03).items():
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)

    def extract(self):
        if self.enabled and self.last_extract_frame != self.sim.frame:
            extraction = self.surface.extract(
                self.sim.state_0.particle_q, self.sim.model.particle_radius,
                compute_normals=True, particle_flags=self.sim.model.particle_flags,
            )
            self.last_arrays = extraction.to_arrays()
            self.last_extract_frame = self.sim.frame
        return self.last_arrays

    def record(self):
        if self.stage is None or (self.frame_samples and self.frame_samples[-1]["frame"] == self.sim.frame):
            return
        time_code = float(self.sim.frame)
        triangles = 0
        if self.enabled:
            vertices, indices, normals = self.extract()
            if vertices is None:
                x = np.empty((0, 3), dtype=np.float32)
                faces = np.empty(0, dtype=np.int32)
                n = x
            else:
                x, faces, n = vertices.numpy(), indices.numpy(), normals.numpy()
            if not np.isfinite(x).all() or not np.isfinite(n).all() or x.shape != n.shape:
                raise RuntimeError(f"Invalid surface vertices/normals at frame {self.sim.frame}")
            if faces.size % 3 or (faces.size and (faces.min() < 0 or faces.max() >= len(x))):
                raise RuntimeError(f"Invalid surface topology at frame {self.sim.frame}")
            triangles = faces.size // 3
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x), time_code)
            self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces), time_code)
            self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(triangles, 3, dtype=np.int32)), time_code)
            self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n), time_code)
            extent = np.stack([x.min(axis=0), x.max(axis=0)]) if len(x) else np.zeros((2, 3), dtype=np.float32)
            self.mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent), time_code)
            self.mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if len(x) else UsdGeom.Tokens.invisible, time_code)
        if self.points:
            x = self.sim.state_0.particle_q.numpy()
            if not np.isfinite(x).all():
                raise RuntimeError("Cannot export non-finite particle positions")
            self.points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x), time_code)
        self.stage.SetEndTimeCode(time_code)
        self.samples += 1
        self.max_triangles = max(self.max_triangles, triangles)
        self.frame_samples.append(dict(frame=self.sim.frame, triangles=triangles))

    def close(self):
        if self.stage:
            self.stage.GetRootLayer().Save()

    def report(self):
        return dict(enabled=self.enabled, parameters=self.parameters,
                    usd_path=str(self.usd_path) if self.usd_path else None,
                    usd_samples=self.samples, max_triangles=self.max_triangles,
                    frame_samples=self.frame_samples)
