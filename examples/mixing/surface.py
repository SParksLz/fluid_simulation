"""Particle concentration -> surface scalar -> pigment lookup texture.

Mesh UVs encode concentration, so shading interpolates the scalar before
the nonlinear color lookup. displayColor is also exported as a fallback.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import warp as wp
from newton import ParticleFlags
from newton.viewer import ViewerRTX

from examples.mixing.color import colors_numpy
from examples.mixing.viewer_rtx import MixingViewerRTX


LUT_SIZE = 1024


def palette_texture(color_model):
    """Canonical raw-linear, 8-bit RGB lookup, repeated vertically."""
    colors = colors_numpy(np.linspace(0.0, 1.0, LUT_SIZE), color_model)
    row = np.rint(255.0*colors).astype(np.uint8)
    return np.repeat(row[None, :, :], 4, axis=0)


def gl_palette_texture(raw_texture):
    # Newton 1.7's GL shader decodes texture RGB with pow(x, 2.2).
    # Compensate here so its albedo agrees with USD's raw-linear lookup.
    return np.rint(255.0*(raw_texture.astype(np.float64)/255.0)**(1.0/2.2)).astype(np.uint8)


@wp.kernel
def interpolate_surface_concentration(
    grid: wp.uint64, positions: wp.array(dtype=wp.vec3),
    concentration: wp.array(dtype=float), volume: wp.array(dtype=float),
    flags: wp.array(dtype=int), vertices: wp.array(dtype=wp.vec3),
    surface_concentration: wp.array(dtype=float), uvs: wp.array(dtype=wp.vec2),
    support: float, lut_size: int, missing: wp.array(dtype=int),
):
    i = wp.tid()
    x = vertices[i]
    total = float(0.0)
    weighted = float(0.0)
    query = wp.hash_grid_query(grid, x, support)
    for j in query:
        if (flags[j] & ParticleFlags.ACTIVE) != 0 and volume[j] > 0.0:
            offset = positions[j]-x
            q2 = wp.dot(offset, offset)/(support*support)
            if q2 < 1.0:
                kernel = 1.0-q2
                weight = volume[j]*kernel*kernel*kernel
                total += weight
                weighted += weight*concentration[j]
    if total <= 1.0e-20:
        wp.atomic_add(missing, 0, 1)
        surface_concentration[i] = 0.0
        uvs[i] = wp.vec2(0.5/float(lut_size), 0.5)
    else:
        value = weighted/total
        surface_concentration[i] = value
        # Address texel centers, including the yellow and blue endpoints.
        u = (wp.clamp(value, 0.0, 1.0)*float(lut_size-1)+0.5)/float(lut_size)
        uvs[i] = wp.vec2(u, 0.5)


@dataclass
class SurfaceFrame:
    vertices: wp.array
    indices: wp.array
    normals: wp.array
    concentration: wp.array
    uvs: wp.array


class MixingSurface:
    def __init__(self, sim, voxel_size=None, color_radius=None):
        if not sim.model.device.is_cuda:
            raise ValueError("ParticleSurface requires CUDA; use --no-surface for CPU particle runs")
        from newton.geometry import ParticleSurface
        voxel = sim.spacing/3.0 if voxel_size is None else voxel_size
        self.support = 2.5*sim.spacing if color_radius is None else color_radius
        if not all(math.isfinite(v) and v > 0 for v in (voxel, self.support)):
            raise ValueError("Surface voxel size and color radius must be finite and positive")
        extent = np.asarray(sim.bound_hi)-np.asarray(sim.bound_lo)
        radius = 2.0*sim.spacing
        capacity = max(250_000, int(1.25*np.prod(np.ceil((extent+4.0*radius)/voxel))))
        self.parameters = dict(voxel_size=voxel, kernel_radius=radius, kernel_scale=0.5,
                               threshold=0.5, smooth_lambda=0.0, anisotropic=False,
                               field_smooth_iterations=1, field_smooth_radius=1,
                               mesh_smooth_iterations=1, max_grid_cells=capacity)
        self.extractor = ParticleSurface(**self.parameters, device=sim.model.device)
        dims = np.maximum(4, np.ceil(extent/self.support).astype(int))
        self.grid = wp.HashGrid(*[int(v) for v in dims], device=sim.model.device)
        self.missing = wp.zeros(1, dtype=int, device=sim.model.device)
        self.texture = palette_texture(sim.color_model)
        self.gl_texture = gl_palette_texture(self.texture)
        self.last_frame = -1
        self.frame = None
        self.max_triangles = 0

    def extract(self, sim):
        if self.last_frame == sim.frame:
            return self.frame
        if sim.count:
            vertices, indices, normals = self.extractor.extract(
                sim.state_0.particle_q, sim.model.particle_radius,
                compute_normals=True, particle_flags=sim.model.particle_flags,
            ).to_arrays()
        else:
            vertices, indices, normals = None, None, None
        if vertices is None:
            vertices = wp.empty(0, dtype=wp.vec3, device=sim.model.device)
            indices = wp.empty(0, dtype=int, device=sim.model.device)
            normals = wp.empty(0, dtype=wp.vec3, device=sim.model.device)
        scalar = wp.empty(len(vertices), dtype=float, device=sim.model.device)
        uvs = wp.empty(len(vertices), dtype=wp.vec2, device=sim.model.device)
        if len(vertices):
            self.grid.build(points=sim.state_0.particle_q, radius=self.support)
            self.missing.zero_()
            wp.launch(interpolate_surface_concentration, dim=len(vertices),
                      inputs=[self.grid.id, sim.state_0.particle_q, sim.concentration,
                              sim.model.apic.particle_volume, sim.model.particle_flags,
                              vertices, scalar, uvs, self.support, LUT_SIZE, self.missing],
                      device=sim.model.device)
            count = int(self.missing.numpy()[0])
            if count:
                raise RuntimeError(f"{count} surface vertices have no dye neighbors; increase --surface-color-radius")
        self.frame = SurfaceFrame(vertices, indices, normals, scalar, uvs)
        self.last_frame = sim.frame
        self.max_triangles = max(self.max_triangles, len(indices)//3)
        return self.frame

    def draw(self, viewer, sim):
        mesh = self.extract(sim)
        name = "/mixing/surface"
        if isinstance(viewer, MixingViewerRTX):
            viewer.log_fluid_surface(mesh, self.texture)
            return
        if isinstance(viewer, ViewerRTX):
            if len(mesh.indices) or viewer.stage.GetPrimAtPath("/root"+name):
                viewer.log_mesh(name, mesh.vertices, mesh.indices, normals=mesh.normals,
                                uvs=mesh.uvs, texture=self.texture, hidden=not len(mesh.indices),
                                roughness=0.25, metallic=0.0, opacity=1.0,
                                dynamic=True, backface_culling=False)
            return
        if not len(mesh.indices):
            # An empty first frame creates no GL object; hide any previous mesh.
            if name in viewer.objects:
                viewer.objects[name].hidden = True
            return
        viewer.log_mesh(name, mesh.vertices, mesh.indices, normals=mesh.normals,
                        uvs=mesh.uvs, texture=self.gl_texture, color=(1.0, 1.0, 1.0),
                        roughness=0.25, metallic=0.0, opacity=1.0,
                        dynamic=True, backface_culling=False)
        # A lookup table must not be averaged through texture mip levels:
        # this would average RGB again across large concentration gradients.
        obj = viewer.objects[name]
        # Newton 1.7 log_mesh uploads the texture but leaves the mesh shader's
        # texture-enable material component at its default zero.
        roughness, metallic, _, _ = obj.material
        obj.material = (roughness, metallic, 0.0, 1.0)
        gl = viewer.renderer.gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, obj.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


class SurfaceUSD:
    def __init__(self, path, sim, surface, include_particles=False):
        from PIL import Image
        from pxr import Sdf, Usd, UsdGeom, UsdShade
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.surface = surface
        self.stage = Usd.Stage.CreateNew(str(self.path))
        self.stage.SetTimeCodesPerSecond(sim.fps)
        self.stage.SetFramesPerSecond(sim.fps)
        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(0.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        root = UsdGeom.Xform.Define(self.stage, "/Mixing")
        self.stage.SetDefaultPrim(root.GetPrim())
        self.stage.GetRootLayer().customLayerData = dict(
            method="APIC + LSMPS colored surface", color_model=sim.color_model,
            color_interpolation="particle scalar -> vertex scalar -> pigment lookup",
            diffusivity_m2_s=sim.diffusion.diffusivity,
        )
        self.mesh = UsdGeom.Mesh.Define(self.stage, "/Mixing/Surface")
        self.mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self.scalar = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar(
            "dyeConcentration", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        self.uv = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        self.colors = self.mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        # This asset is part of the export. Keep it next to the USD so moving
        # the pair preserves the relative material binding.
        palette_path = self.path.with_name(self.path.stem+"_palette.png")
        Image.fromarray(surface.texture).save(palette_path)
        material = UsdShade.Material.Define(self.stage, "/Mixing/Materials/Pigment")
        reader = UsdShade.Shader.Define(self.stage, "/Mixing/Materials/Pigment/ConcentrationUV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(self.stage, "/Mixing/Materials/Pigment/Palette")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(palette_path.name))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader = UsdShade.Shader.Define(self.stage, "/Mixing/Materials/Pigment/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
        self.points = None
        if include_particles:
            self.points = UsdGeom.Points.Define(self.stage, "/Mixing/Particles")
            self.points.CreateWidthsAttr([sim.spacing])
            self.points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            self.point_colors = self.points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            self.point_scalar = UsdGeom.PrimvarsAPI(self.points).CreatePrimvar(
                "dyeConcentration", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        self.last_frame = -1
        self.samples = 0
        self.tank = None
        if getattr(sim, "scene_kind", "") == "tank":
            from examples.mixing.tank_guides import TankUSD
            self.tank = TankUSD(self.stage, sim)

    def record(self, sim):
        if self.last_frame == sim.frame:
            return
        from pxr import UsdGeom, Vt
        mesh = self.surface.extract(sim)
        x, faces = mesh.vertices.numpy(), mesh.indices.numpy()
        normals, c, uv = mesh.normals.numpy(), mesh.concentration.numpy(), mesh.uvs.numpy()
        if not all(np.isfinite(v).all() for v in (x, normals, c, uv)):
            raise RuntimeError(f"Non-finite surface output at frame {sim.frame}")
        if faces.size % 3 or (faces.size and (faces.min() < 0 or faces.max() >= len(x))):
            raise RuntimeError("Invalid surface topology")
        time = sim.frame
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x), time)
        self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces), time)
        self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(faces.size//3, 3, dtype=np.int32)), time)
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals), time)
        bounds = np.stack([x.min(axis=0), x.max(axis=0)]) if len(x) else np.zeros((2, 3), dtype=np.float32)
        self.mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(bounds), time)
        self.mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if len(x) else UsdGeom.Tokens.invisible, time)
        self.scalar.Set(Vt.FloatArray.FromNumpy(c), time)
        self.uv.Set(Vt.Vec2fArray.FromNumpy(uv), time)
        self.colors.Set(Vt.Vec3fArray.FromNumpy(colors_numpy(c, sim.color_model)), time)
        if self.tank is not None:
            self.tank.record(sim)
        if self.points is not None:
            cp = sim.concentration.numpy()
            self.points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(sim.state_0.particle_q.numpy()), time)
            self.point_colors.Set(Vt.Vec3fArray.FromNumpy(colors_numpy(cp, sim.color_model)), time)
            self.point_scalar.Set(Vt.FloatArray.FromNumpy(cp), time)
            self.points.CreateIdsAttr().Set(Vt.Int64Array.FromNumpy(np.arange(sim.count, dtype=np.int64)), time)
        self.stage.SetEndTimeCode(time)
        self.last_frame = time
        self.samples += 1

    def close(self):
        self.stage.GetRootLayer().Save()
