"""Screen-space particle indexing and the OVRTX Sensor Processing Graph.

Depth filtering follows Truong and Yuksel (2018). Surface reconstruction and
optical composition follow van der Laan et al. (2009). The graph consumes
owned CUDA snapshots, so an asynchronous render never reads live solver state.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import warp as wp

from examples.mixing.splats import fit_render_kernels, prepare_render_positions
from examples.mixing.media import MAX_MEDIUM_LAYERS, MEDIUM_LAYERS, MEDIUM_MODES, MEDIUM_PRESETS


TILE_SIZE = 16


@dataclass(frozen=True)
class SSFRConfig:
    filter: str = "narrow"
    iterations: int = 2
    radius_scale: float = 1.6
    absorption: float = 12.0
    ior: float = 1.333
    debug: str = "color"
    roughness: float = 0.2
    splats: str = "anisotropic"
    medium: str = "pigment"
    medium_scale: float = 1.0
    medium_layers: int = MEDIUM_LAYERS

    def __post_init__(self):
        if self.filter not in ("narrow", "curvature"):
            raise ValueError("SSFR filter must be narrow or curvature")
        if self.debug not in ("color", "depth", "thickness", "normals", "dye", "specular",
                              "transmittance", "scattering"):
            raise ValueError("Unknown SSFR debug view")
        if not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("SSFR iterations must be a nonnegative integer")
        if not math.isfinite(self.radius_scale) or self.radius_scale <= 0:
            raise ValueError("SSFR radius scale must be finite and positive")
        if not math.isfinite(self.absorption) or self.absorption < 0:
            raise ValueError("SSFR absorption must be finite and nonnegative")
        if not math.isfinite(self.ior) or self.ior < 1:
            raise ValueError("SSFR index of refraction must be finite and >=1")
        if not math.isfinite(self.roughness) or not 0.0 < self.roughness <= 1.0:
            raise ValueError("SSFR roughness must be finite and in (0,1]")
        if self.splats not in ("sphere", "anisotropic"):
            raise ValueError("SSFR splats must be sphere or anisotropic")
        if self.medium not in MEDIUM_MODES:
            raise ValueError("Unknown SSFR medium preset")
        if not isinstance(self.medium_layers, int) or not 1 <= self.medium_layers <= MAX_MEDIUM_LAYERS:
            raise ValueError(f"SSFR medium layers must be an integer in [1,{MAX_MEDIUM_LAYERS}]")
        if (not math.isfinite(self.medium_scale) or self.medium_scale < 0
                or self.medium_scale > float(np.finfo(np.float32).max)):
            raise ValueError("SSFR medium scale must be finite, nonnegative and fit float32")


@wp.kernel
def project_particles(
    positions: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
    eye: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
    width: int, height: int, tiles_x: int, tile_size: int,
    tan_half_fov: float, near: float, far: float, radius_scale: float,
    projected: wp.array(dtype=wp.vec4), bounds: wp.array(dtype=wp.vec4i),
    counts: wp.array(dtype=int), world_kernels: wp.array(dtype=wp.mat33),
    view_kernels: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    delta = positions[i] - eye
    x, y, z = wp.dot(delta, right), wp.dot(delta, up), wp.dot(delta, forward)
    radius = radii[i] * radius_scale
    projected[i] = wp.vec4(x, y, z, radius)
    bounds[i] = wp.vec4i(-1)
    if not (wp.isfinite(x) and wp.isfinite(y) and wp.isfinite(z) and wp.isfinite(radius)):
        return
    if radius <= 0.0:
        return
    view_rotation = wp.transpose(wp.matrix_from_cols(right, up, forward))
    inverse_kernel = view_rotation * world_kernels[i] * wp.transpose(view_rotation)
    view_kernels[i] = inverse_kernel
    shape = wp.inverse(inverse_kernel)
    extent = wp.vec3(wp.length(shape[0]), wp.length(shape[1]), wp.length(shape[2]))
    if z + extent[2] <= near or z - extent[2] >= far:
        return
    xmin, xmax = float(0), float(width)
    ymin, ymax = float(0), float(height)
    if z > extent[2]:
        # Project a bounding cube. This conservatively encloses the sphere even
        # off-axis, unlike a center-depth-only point-sprite bounding rectangle.
        focal = float(height) / (2.0 * tan_half_fov)
        xa, xb = (x-extent[0])/(z-extent[2]), (x-extent[0])/(z+extent[2])
        xc, xd = (x+extent[0])/(z-extent[2]), (x+extent[0])/(z+extent[2])
        ya, yb = (y-extent[1])/(z-extent[2]), (y-extent[1])/(z+extent[2])
        yc, yd = (y+extent[1])/(z-extent[2]), (y+extent[1])/(z+extent[2])
        xmin = 0.5*float(width) + focal*wp.min(wp.min(xa, xb), wp.min(xc, xd))
        xmax = 0.5*float(width) + focal*wp.max(wp.max(xa, xb), wp.max(xc, xd))
        ymin = 0.5*float(height) - focal*wp.max(wp.max(ya, yb), wp.max(yc, yd))
        ymax = 0.5*float(height) - focal*wp.min(wp.min(ya, yb), wp.min(yc, yd))
    if xmax < 0.0 or ymax < 0.0 or xmin >= float(width) or ymin >= float(height):
        return
    tx0 = wp.max(0, int(wp.floor(xmin/float(tile_size))))
    tx1 = wp.min(tiles_x-1, int(wp.floor(xmax/float(tile_size))))
    ty0 = wp.max(0, int(wp.floor(ymin/float(tile_size))))
    ty1 = wp.min((height-1)/tile_size, int(wp.floor(ymax/float(tile_size))))
    bounds[i] = wp.vec4i(tx0, ty0, tx1, ty1)
    for ty in range(ty0, ty1+1):
        for tx in range(tx0, tx1+1):
            # The zero at counts[0] produces an inclusive scan containing each
            # tile's start at offsets[tile] and end at offsets[tile+1].
            wp.atomic_add(counts, 1+ty*tiles_x+tx, 1)


@wp.kernel
def fill_tiles(bounds: wp.array(dtype=wp.vec4i), offsets: wp.array(dtype=int),
               cursors: wp.array(dtype=int), indices: wp.array(dtype=int), tiles_x: int):
    i = wp.tid()
    b = bounds[i]
    if b[0] < 0:
        return
    for ty in range(b[1], b[3]+1):
        for tx in range(b[0], b[2]+1):
            tile = ty*tiles_x+tx
            slot = wp.atomic_add(cursors, tile, 1)
            indices[offsets[tile]+slot] = i


class FluidBuffers:
    """Owned CUDA snapshots and a compact, overflow-free screen tile index."""

    def __init__(self, width, height, device, config, particle_radius=None):
        if width < 1 or height < 1:
            raise ValueError("SSFR image dimensions must be positive")
        self.width, self.height = width, height
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("SSFR requires CUDA")
        self.config = config
        self.particle_radius = particle_radius
        self.grid = None
        if config.splats == "anisotropic" and particle_radius is not None:
            if not math.isfinite(particle_radius) or particle_radius <= 0.0:
                raise ValueError("SSFR particle radius must be finite and positive")
            self.grid = wp.HashGrid(64, 64, 64, device=self.device)
        self.tiles_x = (width+TILE_SIZE-1)//TILE_SIZE
        self.tile_count = self.tiles_x*((height+TILE_SIZE-1)//TILE_SIZE)
        self.counts = wp.zeros(self.tile_count+1, dtype=int, device=self.device)
        self.offsets = wp.zeros_like(self.counts)
        self.cursors = wp.zeros(self.tile_count, dtype=int, device=self.device)
        self.projected = wp.empty(1, dtype=wp.vec4, device=self.device)
        self.bounds = wp.empty(1, dtype=wp.vec4i, device=self.device)
        self.concentration = wp.empty(1, dtype=float, device=self.device)
        self.indices = wp.empty(1, dtype=int, device=self.device)
        self.render_positions = wp.empty(1, dtype=wp.vec3, device=self.device)
        self.grid_positions = wp.empty(1, dtype=wp.vec3, device=self.device)
        self.world_kernels = wp.empty(1, dtype=wp.mat33, device=self.device)
        self.view_kernels = wp.empty(1, dtype=wp.mat33, device=self.device)
        self.index_count = 0

    def prepare(self, points, radii, concentration, camera, near, far, tan_half_fov):
        if (points.dtype != wp.vec3 or radii.dtype != wp.float32
                or concentration.dtype != wp.float32):
            raise ValueError("SSFR requires vec3f positions and float32 radius/concentration")
        if any(a.device != self.device for a in (points, radii, concentration)):
            raise ValueError("All SSFR particle inputs must share the viewer CUDA device")
        count = len(points)
        if min(len(radii), len(concentration)) < count:
            raise ValueError("SSFR radius and concentration arrays must cover all particles")
        if count > len(self.projected):
            capacity = max(count, 2*len(self.projected))
            # Publish a complete set together so allocation failure leaves the
            # previous frame's compatible buffers available for a retry.
            storage_types = (("projected", wp.vec4), ("bounds", wp.vec4i), ("concentration", float),
                             ("render_positions", wp.vec3), ("grid_positions", wp.vec3),
                             ("world_kernels", wp.mat33), ("view_kernels", wp.mat33))
            storage = {name: wp.empty(capacity, dtype=dtype, device=self.device)
                       for name, dtype in storage_types}
            for name, array in storage.items():
                setattr(self, name, array)
        self.counts.zero_()
        if count:
            wp.copy(self.concentration, concentration, count=count)
            wp.launch(prepare_render_positions, dim=count,
                      inputs=[points, radii, self.render_positions, self.grid_positions,
                              self.world_kernels, self.config.radius_scale], device=self.device)
            if self.grid is not None:
                self.grid.build(self.grid_positions[:count], radius=5.0*self.particle_radius)
                wp.launch(fit_render_kernels, dim=count,
                          inputs=[self.grid.id, points, radii, self.render_positions,
                                  self.world_kernels, self.config.radius_scale], device=self.device)
            wp.launch(project_particles, dim=count,
                      inputs=[self.render_positions, radii, wp.vec3(camera[3, :3]), wp.vec3(camera[0, :3]),
                              wp.vec3(camera[1, :3]), wp.vec3(-camera[2, :3]), self.width, self.height,
                              self.tiles_x, TILE_SIZE, tan_half_fov, near, far, self.config.radius_scale,
                              self.projected, self.bounds, self.counts, self.world_kernels,
                              self.view_kernels], device=self.device)
        wp.utils.array_scan(self.counts, self.offsets, inclusive=True)
        # Only this allocation-size scalar crosses to the CPU. Particle data,
        # depth, thickness, dye and the final image stay in device memory.
        self.index_count = int(self.offsets[self.tile_count:self.tile_count+1].numpy()[0])
        if self.index_count > len(self.indices):
            self.indices = wp.empty(max(self.index_count, 2*len(self.indices)), dtype=int, device=self.device)
        self.cursors.zero_()
        if count and self.index_count:
            wp.launch(fill_tiles, dim=count,
                      inputs=[self.bounds, self.offsets, self.cursors, self.indices, self.tiles_x],
                      device=self.device)
        # SPG launches on the renderer's CUDA stream. Complete the producer
        # stream before handing over these process-local device addresses.
        wp.synchronize_stream(wp.get_stream(self.device))

    @property
    def pointers(self):
        return {"particles": self.projected.ptr, "dye": self.concentration.ptr,
                "offsets": self.offsets.ptr, "indices": self.indices.ptr,
                "kernels": self.view_kernels.ptr}


def build_graph(viewer, config, radius, color_model):
    """Author the SPG graph before Newton serializes the stage for RTX."""
    from pxr import Sdf, UsdGeom, UsdLux, UsdShade

    stage = viewer.stage
    product = stage.GetPrimAtPath(viewer._render_product_path)
    root = viewer._render_product_path
    source = Sdf.AssetPath(str(Path(__file__).with_name("spg") / "fluid.cu"))

    def node(name, entry, values, resources, outputs):
        shader = UsdShade.Shader.Define(stage, root+"/Fluid"+name)
        shader.CreateImplementationSourceAttr("sourceAsset")
        shader.SetSourceAsset(source, "spg")
        shader.SetSourceAssetSubIdentifier(entry, "spg")
        for key, (kind, value) in values.items():
            shader.CreateInput(key, kind).Set(value)
        for key, path in resources.items():
            shader.CreateInput(key, Sdf.ValueTypeNames.Opaque).GetAttr().AddConnection(path)
        for key in outputs:
            shader.CreateOutput(key, Sdf.ValueTypeNames.Opaque)
        return shader

    def aov(name):
        path = root+"/"+name
        prim = stage.DefinePrim(path, "RenderVar")
        prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(name)
        prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
        product.GetRelationship("orderedVars").AddTarget(prim.GetPath())
        return path+".omni:rtx:aov"

    scene_color = stage.GetPrimAtPath(viewer._render_var_path).CreateAttribute(
        "omni:rtx:aov", Sdf.ValueTypeNames.Opaque).GetPath()
    scene_depth = aov("DistanceToCameraSD")
    tan_half_fov = math.tan(math.radians(viewer.camera.fov)/2)
    float_type = Sdf.ValueTypeNames.Float
    raster_values = {key: (Sdf.ValueTypeNames.UInt64, value)
                     for key, value in viewer.fluid_buffers.pointers.items()}
    raster_values.update(near=(float_type, viewer.camera.near), far=(float_type, viewer.camera.far),
                         tanHalfFov=(float_type, tan_half_fov),
                         radiusScale=(float_type, config.radius_scale),
                         tileSize=(Sdf.ValueTypeNames.Int, TILE_SIZE),
                         opticalLayers=(Sdf.ValueTypeNames.Int,
                                        config.medium_layers if config.medium != "pigment" else 1),
                         layerWidth=(float_type, 4*radius))
    raster = node("Raster", "raster", raster_values,
                  {"SceneDepth": scene_depth, "SceneColor": scene_color}, ("Depth", "Optical", "Layers"))
    depth = raster.GetPath().AppendProperty("outputs:Depth")
    raw_depth = depth
    optical = raster.GetPath().AppendProperty("outputs:Optical")
    raw_optical = optical
    layered_optical = raster.GetPath().AppendProperty("outputs:Layers")
    raw_layers = layered_optical
    common = {"radius": (float_type, radius*config.radius_scale),
              "tanHalfFov": (float_type, tan_half_fov)}
    filters = []
    if config.iterations:
        expanded = node("Expand", "expand_depth", {}, {"Depth": raw_depth}, ("Depth",))
        closed = node("Close", "close_depth", {"radius": common["radius"]},
                      {"Original": raw_depth, "Expanded": expanded.GetPath().AppendProperty("outputs:Depth")},
                      ("Depth",))
        depth = closed.GetPath().AppendProperty("outputs:Depth")
    optical_guide = depth
    for i in range(config.iterations):
        axes = (0, 1) if config.filter == "narrow" else (3,)
        for axis in axes:
            shader = node(f"Filter{i}_{axis}", config.filter,
                          dict(common, axis=(Sdf.ValueTypeNames.Int, axis)), {"Depth": depth}, ("Depth",))
            depth = shader.GetPath().AppendProperty("outputs:Depth")
            filters.append(str(shader.GetPath()))
    if config.filter == "narrow" and config.iterations:
        shader = node("Cleanup", "narrow", dict(common, axis=(Sdf.ValueTypeNames.Int, 2)),
                      {"Depth": depth}, ("Depth",))
        depth = shader.GetPath().AppendProperty("outputs:Depth")
        filters.append(str(shader.GetPath()))
    # Filter thickness and dye mass together, then divide during shading. The
    # depth guide prevents transport across distinct front/back surface layers.
    for axis in (0, 1):
        shader = node(f"Optical{axis}", "smooth_optical",
                      dict(common, axis=(Sdf.ValueTypeNames.Int, axis)),
                      {"Depth": optical_guide, "Optical": optical}, ("Optical",))
        optical = shader.GetPath().AppendProperty("outputs:Optical")
        filters.append(str(shader.GetPath()))
    if config.medium != "pigment":
        for axis in (0, 1):
            shader = node(f"MediumOptical{axis}", "smooth_optical",
                          dict(common, axis=(Sdf.ValueTypeNames.Int, axis)),
                          {"Depth": optical_guide, "Optical": layered_optical}, ("Optical",))
            layered_optical = shader.GetPath().AppendProperty("outputs:Optical")
            filters.append(str(shader.GetPath()))
    else:
        layered_optical = optical
    matrix = viewer._compute_camera_matrix()
    key_direction = (0.0, 0.0, 1.0)
    key_color = (0.0, 0.0, 0.0)
    dome_color = (0.0, 0.0, 0.0)
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DistantLight):
            light = UsdLux.DistantLight(prim)
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            key_direction = tuple(float(transform[2][i]) for i in range(3))
            strength = float(light.GetIntensityAttr().Get()) / 3000.0
            key_color = tuple(float(c)*strength for c in light.GetColorAttr().Get())
        elif prim.IsA(UsdLux.DomeLight):
            light = UsdLux.DomeLight(prim)
            strength = float(light.GetIntensityAttr().Get()) / 400.0
            dome_color = tuple(float(c)*strength for c in light.GetColorAttr().Get())
    medium_values = {"mediumModel": (Sdf.ValueTypeNames.Int, int(config.medium != "pigment")),
                     "mediumScale": (float_type, config.medium_scale),
                     "opticalLayers": raster_values["opticalLayers"]}
    pair = MEDIUM_PRESETS.get(config.medium)
    for label, material in zip(("A", "B"), pair or (None, None)):
        for coefficient in ("absorption", "scattering"):
            value = getattr(material, coefficient) if material is not None else (0.0, 0.0, 0.0)
            medium_values[coefficient+label] = (Sdf.ValueTypeNames.Float3, value)
    shade_values = dict(tanHalfFov=(float_type, tan_half_fov),
                 absorption=(float_type, config.absorption), ior=(float_type, config.ior),
                 radius=(float_type, radius*config.radius_scale),
                 roughness=(float_type, config.roughness),
                 keyDirection=(Sdf.ValueTypeNames.Float3, key_direction),
                 keyColor=(Sdf.ValueTypeNames.Float3, key_color),
                 domeColor=(Sdf.ValueTypeNames.Float3, dome_color),
                 right=(Sdf.ValueTypeNames.Float3, tuple(matrix[0, :3])),
                 up=(Sdf.ValueTypeNames.Float3, tuple(matrix[1, :3])),
                 forward=(Sdf.ValueTypeNames.Float3, tuple(-matrix[2, :3])),
                 colorModel=(Sdf.ValueTypeNames.Int, color_model),
                 debug=(Sdf.ValueTypeNames.Int,
                        ("color", "depth", "thickness", "normals", "dye", "specular",
                         "transmittance", "scattering").index(config.debug)))
    shade_values.update(medium_values)
    shade = node("Shade", "shade", shade_values,
                 {"Depth": depth, "Optical": optical, "SceneColor": scene_color,
                  "SceneDepth": scene_depth, "Layers": layered_optical}, ("Color",))
    final = aov("FluidColor")
    stage.GetAttributeAtPath(final).AddConnection(shade.GetPath().AppendProperty("outputs:Color"))
    raw_aov = aov("FluidRawDepth")
    stage.GetAttributeAtPath(raw_aov).AddConnection(raw_depth)
    depth_aov = raw_aov
    if depth != raw_depth:
        depth_aov = aov("FluidDepth")
        stage.GetAttributeAtPath(depth_aov).AddConnection(depth)
    optical_aov = aov("FluidOptical")
    stage.GetAttributeAtPath(optical_aov).AddConnection(raw_optical)
    smooth_optical_aov = aov("FluidOpticalSmooth")
    stage.GetAttributeAtPath(smooth_optical_aov).AddConnection(optical)
    layers_aov, smooth_layers_aov = optical_aov, smooth_optical_aov
    if config.medium != "pigment":
        layers_aov = aov("FluidLayers")
        stage.GetAttributeAtPath(layers_aov).AddConnection(raw_layers)
        smooth_layers_aov = aov("FluidLayersSmooth")
        stage.GetAttributeAtPath(smooth_layers_aov).AddConnection(layered_optical)
    return {"raster": str(raster.GetPath()), "filters": filters, "shade": str(shade.GetPath()),
            "output": root+"/FluidColor", "raw_depth": raw_aov.rsplit(".", 1)[0],
            "depth": depth_aov.rsplit(".", 1)[0], "optical": optical_aov.rsplit(".", 1)[0],
            "optical_smooth": smooth_optical_aov.rsplit(".", 1)[0],
            "layers": layers_aov.rsplit(".", 1)[0],
            "layers_smooth": smooth_layers_aov.rsplit(".", 1)[0]}
