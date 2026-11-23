"""Replay APIC particle snapshots in OVRTX with the HDRI004 optics reference."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import newton
import numpy as np
import ovrtx
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
from scipy.ndimage import gaussian_filter
import warp as wp

from examples.mixing import ssfr
from examples.mixing.tank import Emitter
from examples.mixing.tank_guides import TankGL
from examples.mixing.viewer_ssfr import ScreenSpaceViewerRTX


HERE = Path(__file__).resolve().parent
TRAINSTATION = "TexturesCom_HDRPanorama0022_trainstation_hallway_4K_hdri_sphere.exr"
# Measured against OVRTX HdrColor for the two assets in this experiment.
RGBE_DOME_GAIN = 0.6173 / 700.0
TRAINSTATION_DOME_GAIN = 1.85573 / 2100.0
KEY_REFERENCE_INTENSITY = 320.0
IOR = 1.333
ROUGHNESS = 0.15


@dataclass(frozen=True)
class Lighting:
    dome_yaw: float
    key_direction: tuple[float, float, float]
    key_diameter: float
    dome_intensity: float = 700.0
    key_intensity: float = KEY_REFERENCE_INTENSITY


LIGHTING = {
    "baseline": Lighting(0.0, (-0.58, 0.72, 0.42), 3.0),
    "dome-only": Lighting(30.0, (-0.58, 0.72, 0.42), 3.0),
    "gloss": Lighting(30.0, (-0.46, 0.74, 0.49), 2.0),
    "trainstation": Lighting(-60.0, (-0.46, 0.74, 0.49), 2.0, 2100.0),
}


def downsample_panorama(rgb):
    height, width, _ = rgb.shape
    if width != 2 * height or height % 4:
        raise ValueError("Expected a 2:1 panorama with height divisible by four")
    return rgb.reshape(height//4,4,width//4,4,3).mean(axis=(1,3))


def read_exr(path):
    """Decode linear floating-point RGB, dropping the optional alpha plane."""
    video = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,pix_fmt", "-of", "json", str(path),
    ]))["streams"][0]
    if video["codec_name"] != "exr":
        raise ValueError("Expected an OpenEXR panorama")
    has_alpha = video["pix_fmt"].startswith("gbra")
    layout = "gbrapf32le" if has_alpha else "gbrpf32le"
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-frames:v", "1", "-c:v", "rawvideo", "-threads:v", "1",
        "-pix_fmt", layout, "-f", "rawvideo", "pipe:1",
    ])
    planes = np.frombuffer(raw, dtype="<f4").reshape(
        4 if has_alpha else 3, video["height"], video["width"],
    )
    rgb = planes[[2, 0, 1]].transpose(1, 2, 0)
    if not np.isfinite(rgb).all():
        raise ValueError("OpenEXR panorama contains non-finite radiance")
    return np.maximum(rgb, 0.0)


def read_rgbe(path):
    """Read the scanline RGBE encoding used by HDRI004."""
    with path.open("rb") as file:
        signature = file.readline().strip()
        if signature not in (b"#?RADIANCE", b"#?RGBE"):
            raise ValueError("Expected a Radiance RGBE environment")
        header = []
        for line in file:
            if not line.strip():
                break
            header.append(line.strip())
        if b"FORMAT=32-bit_rle_rgbe" not in header:
            raise ValueError("Expected 32-bit RLE RGBE")
        ny, h, nx, w = file.readline().split()
        height, width = int(h), int(w)
        if (ny, nx) != (b"-Y", b"+X") or width != 2*height or height % 4:
            raise ValueError("Expected a 2:1 top-down HDR panorama divisible by four")
        rgbe = np.empty((height, 4, width), dtype=np.uint8)
        for scanline in rgbe:
            marker = file.read(4)
            if marker != bytes((2, 2, width >> 8, width & 255)):
                raise ValueError("Unexpected Radiance scanline marker")
            for channel in scanline:
                col = 0
                while col < width:
                    code = file.read(1)
                    if not code or not code[0]:
                        raise ValueError("Truncated Radiance scanline")
                    repeat = code[0] > 128
                    count = code[0] - 128 if repeat else code[0]
                    if col + count > width:
                        raise ValueError("Radiance scanline packet overruns its row")
                    block = file.read(1 if repeat else count)
                    if len(block) != (1 if repeat else count):
                        raise ValueError("Truncated Radiance packet")
                    channel[col:col+count] = block[0] if repeat else np.frombuffer(block, dtype=np.uint8)
                    col += count
    exponents = rgbe[:, 3, :].astype(np.int32)
    rgb = np.ldexp(rgbe[:, :3, :].transpose(0, 2, 1).astype(np.float32), exponents[:, :, None]-136)
    rgb[exponents == 0] = 0
    return rgb


def read_hdri(path, sampling):
    reader = read_exr if path.suffix.lower() == ".exr" else read_rgbe
    rgb = reader(path)
    height, width, _ = rgb.shape
    if width != 2 * height:
        raise ValueError("Expected a 2:1 latitude-longitude panorama")
    return downsample_panorama(rgb) if sampling == "quarter" else rgb


def filter_panorama(image):
    angular_scale = image.shape[1] / 1000
    sigma = (2.7 * angular_scale, 5.2 * angular_scale)
    mode = ("nearest", "wrap")
    if image.ndim == 3:
        sigma += (0,)
        mode += ("nearest",)
    return gaussian_filter(image, sigma, mode=mode)


def distant_light_disk(width, height, diameter):
    """Place the light at the equator and center longitude of its own panorama."""
    radius = math.radians(diameter) / 2
    # Supersample only the small patch containing the light's angular disk.
    half_extent = math.ceil(radius * height / math.pi) + 1
    x0, x1 = max(0, width // 2 - half_extent), min(width, width // 2 + half_extent)
    y0, y1 = max(0, height // 2 - half_extent), min(height, height // 2 + half_extent)
    if radius > math.pi / 2:
        x0, x1 = 0, width
    x, y = np.arange(x0, x1), np.arange(y0, y1)
    coverage = np.zeros((len(y), len(x)), dtype=np.float32)
    offsets = (np.arange(4) + 0.5) / 4
    for offset_y in offsets:
        latitude = math.pi * ((y + offset_y) / height - 0.5)
        for offset_x in offsets:
            longitude = math.tau * ((x + offset_x) / width - 0.5)
            coverage += np.cos(latitude[:, None]) * np.cos(longitude) >= math.cos(radius)
    disk = np.zeros((height, width), dtype=np.float32)
    disk[y0:y1, x0:x1] = coverage / 16
    if not coverage.any():
        mid_y, mid_x = height // 2, width // 2
        disk[mid_y - 1:mid_y + 1, mid_x - 1:mid_x + 1] = 1
    return disk


def match_key_to_dome(rgb, blurred, env_scale, diameter):
    height, width, _ = rgb.shape
    disk = distant_light_disk(width, height, diameter)
    soft_disk = filter_panorama(disk)
    blend = min(1.0, ROUGHNESS * 5.0)
    luminance = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dome_luma = (1 - blend) * (rgb @ luminance) + blend * (blurred @ luminance)
    # Use the bright end of the filtered DomeLight as an exposure reference
    # for this rig. OVRTX light intensity and panorama texels use separate scales.
    reference = env_scale * np.quantile(dome_luma, 0.999)
    disk_peak = ((1 - blend) * disk + blend * soft_disk).max()
    scale = reference / disk_peak
    return disk * scale, soft_disk * scale


def copy_output(product, path):
    aov = product.frames[0].render_vars[path]
    with aov.map(device=ovrtx.Device.CPU) as mapping:
        return np.array(np.from_dlpack(mapping), copy=True)


class OpticsViewer(ScreenSpaceViewerRTX):
    def __init__(self, hdri, shader, rgb, spacing, diagnostics, lighting, *,
                 key_reflection="ggx", volume_lighting="ambient", **kwargs):
        super().__init__(particle_radius=0.5*spacing, ssfr_config=ssfr.SSFRConfig(medium="coffee-milk"), **kwargs)
        self.hdri = hdri
        self.shader = shader
        self.spacing = spacing
        self.diagnostics = diagnostics
        self.lighting = lighting
        self.key_reflection = key_reflection
        self.volume_lighting = volume_lighting
        gain = TRAINSTATION_DOME_GAIN if hdri.suffix.lower() == ".exr" else RGBE_DOME_GAIN
        self.env_scale = gain * lighting.dome_intensity
        self.key_direction = Gf.Vec3d(*lighting.key_direction).GetNormalized()
        h,w,_ = rgb.shape
        filtered = filter_panorama(rgb)
        data = np.zeros((2*h,w,4),dtype=np.float32)
        data[:h,:,:3] = rgb
        data[h:,:,:3] = filtered
        if key_reflection == "panorama":
            # Reuse the alpha channel for white DistantLight radiance.
            data[:h,:,3], data[h:,:,3] = match_key_to_dome(
                rgb, filtered, self.env_scale, lighting.key_diameter
            )
        self.panorama = wp.array(data.reshape(-1,4), dtype=wp.vec4, device=self.fluid_buffers.device)
        self.env_shape = (w,h)
        self._rear_uniforms = {}

    def _add_camera_lights_and_render_product(self):
        dome = UsdLux.DomeLight.Define(self.stage, "/root/HDRI004")
        dome.GetIntensityAttr().Set(self.lighting.dome_intensity)
        dome.GetTextureFileAttr().Set(Sdf.AssetPath(str(self.hdri)))
        dome.GetTextureFormatAttr().Set("latlong")
        if self.lighting.dome_yaw:
            UsdGeom.Xformable(dome).AddRotateZOp().Set(self.lighting.dome_yaw)
        key = UsdLux.DistantLight.Define(self.stage, "/root/DistantKey")
        key.GetIntensityAttr().Set(self.lighting.key_intensity)
        key.GetAngleAttr().Set(self.lighting.key_diameter)
        key.LightAPI().GetNormalizeAttr().Set(True)
        transform = Gf.Rotation(Gf.Vec3d(0,0,1),self.key_direction).GetQuat()
        UsdGeom.Xformable(key).AddOrientOp().Set(Gf.Quatf(transform))
        super()._add_camera_lights_and_render_product()
        path = self._render_product_path
        graph = self._ssfr_graph
        product = self.stage.GetPrimAtPath(path)
        product.GetAttribute("omni:rtx:dlss:frameGeneration").Set(False)
        product.CreateAttribute("omni:rtx:post:aa:op",Sdf.ValueTypeNames.Token).Set("none")
        product.CreateAttribute("omni:rtx:rt:fp32Target",Sdf.ValueTypeNames.Bool).Set(True)
        types = Sdf.ValueTypeNames

        def aov(name, source=None):
            prim = self.stage.DefinePrim(f"{path}/{name}","RenderVar")
            prim.CreateAttribute("sourceName",types.String).Set(name)
            attr = prim.CreateAttribute("omni:rtx:aov",types.Opaque)
            if source is not None:
                attr.SetConnections([source])
            product.GetRelationship("orderedVars").AddTarget(prim.GetPath())
            return attr.GetPath()

        def node(name, entry, constants, resources, output):
            shader = UsdShade.Shader.Define(self.stage, f"{path}/{name}")
            shader.CreateImplementationSourceAttr("sourceAsset")
            shader.SetSourceAsset(Sdf.AssetPath(str(self.shader)), "spg")
            shader.SetSourceAssetSubIdentifier(entry,"spg")
            for key,(kind,value) in constants.items():
                shader.CreateInput(key,kind).Set(value)
            for key,value in resources.items():
                shader.CreateInput(key,types.Opaque).GetAttr().SetConnections([value])
            target=shader.CreateOutput(output,types.Opaque).GetAttr().GetPath()
            return str(shader.GetPath()),target

        shade = UsdShade.Shader.Get(self.stage,graph["shade"])
        def original(key):
            return shade.GetInput(key).GetAttr().GetConnections()[0]

        hdr = aov("HdrColor")
        front = original("Depth")
        scene = original("SceneDepth")
        fov = shade.GetInput("tanHalfFov").Get()
        radius = self.spacing*0.5*self.ssfr_config.radius_scale
        rear_values = {
            "tileSize": (types.Int,ssfr.TILE_SIZE),
            "near": (types.Float,self.camera.near), "far": (types.Float,self.camera.far),
            "binWidth": (types.Float,0.5*self.spacing), "fov": (types.Float,fov),
        }
        rear_values.update({key:(types.UInt64,value) for key,value in self.fluid_buffers.pointers.items() if key != "dye"})
        graph["rear"], raw_rear = node(
            "MeasuredRear","rear_depth",rear_values,
            {"RawFront": Sdf.Path(graph["raster"]+".outputs:Depth"), "SceneDepth":scene},"Depth")
        _,rear = node("SmoothedRear","smooth_rear",{"radius":(types.Float,radius)},
                      {"Front":front,"Rear":raw_rear},"Depth")
        basis = {name:(types.Float3,shade.GetInput(name).Get()) for name in ("right","up","forward")}
        raster = UsdShade.Shader.Get(self.stage,graph["raster"])
        key_from_panorama = self.key_reflection == "panorama"
        key_intensity_scale = KEY_REFERENCE_INTENSITY if key_from_panorama else 1000.0
        constants = {
            **basis, "keyDirection":(types.Float3,Gf.Vec3f(*self.key_direction)),
            "fov":(types.Float,fov), "radius":(types.Float,radius),
            "roughness":(types.Float,ROUGHNESS), "ior":(types.Float,IOR),
            "keyPower":(types.Float,self.lighting.key_intensity/key_intensity_scale),
            "keyAngle":(types.Float,math.radians(self.lighting.key_diameter)),
            "keyFromPanorama":(types.Int,int(key_from_panorama)),
            "envScale":(types.Float,self.env_scale), "exposure":(types.Float,0.98),
            "ambient":(types.Float3,Gf.Vec3f(0.43,0.42,0.39)),
            "envAddress":(types.UInt64,self.panorama.ptr),
            "envWidth":(types.Int,self.env_shape[0]),"envHeight":(types.Int,self.env_shape[1]),
            "layerCount":(types.Int,self.ssfr_config.medium_layers),
            "envLongitudeShift":(types.Float,self.lighting.dome_yaw/360.0),
            "directionalVolume":(types.Int,int(self.volume_lighting == "directional")),
            "layerWidth":(types.Float,raster.GetInput("layerWidth").Get()),
            "cameraHeight":(types.Float,float(self.camera.pos[2])),
            "volumeKeyScale":(types.Float,self.lighting.key_intensity/KEY_REFERENCE_INTENSITY),
        }
        resources = {"Front":front,"Rear":rear,"Hdr":hdr,"SceneDepth":scene,
                     "Optical":original("Optical"),"Layers":original("Layers")}
        modes = (("Optics",0),("OneInterface",1))
        if self.diagnostics:
            modes += (("Paths",2),("Reflection",3),("DomeReflection",4),("KeySpecular",5),("VolumeScatter",6))
            graph["rear_depth"] = str(aov("FluidRear",rear).GetPrimPath())
        for name,mode in modes:
            _, color = node("Shade"+name,"hdri_shade",{**constants,"mode":(types.Int,mode)},resources,"Color")
            graph[name] = str(aov("Fluid"+name,color).GetPrimPath())

    def _update_ovrtx_mesh_points(self):
        super()._update_ovrtx_mesh_points()
        self._write_rear_pointers()

    def _write_rear_pointers(self):
        for key,address in self.fluid_buffers.pointers.items():
            if key != "dye" and self._rear_uniforms.get(key) != address:
                self._write_runtime_attribute([self._ssfr_graph["rear"]],"inputs:"+key,np.asarray([address],dtype=np.uint64))
                self._rear_uniforms[key] = address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of generated particle snapshots")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--hdri", type=Path, required=True, help="2:1 latitude-longitude .hdr or .exr panorama")
    parser.add_argument("--frames",type=int,nargs="+",default=[120,300])
    parser.add_argument("--width",type=int,default=768)
    parser.add_argument("--diagnostics",action="store_true")
    parser.add_argument("--lighting",choices=LIGHTING)
    parser.add_argument("--hdr-sampling", choices=("native", "quarter"), default="quarter",
                        help="SSFR panorama resolution, independent of the OVRTX DomeLight")
    parser.add_argument("--key-reflection", choices=("panorama", "ggx"),
                        help="SSFR DistantLight reflection model")
    parser.add_argument("--volume-lighting", choices=("ambient","directional"), default="ambient",
                        help="SSFR illumination of the front volume layers")
    args = parser.parse_args()
    lighting = args.lighting or ("trainstation" if args.hdri.name == TRAINSTATION else "baseline")
    key_reflection = args.key_reflection or ("panorama" if lighting == "trainstation" else "ggx")
    args.output.mkdir(parents=True,exist_ok=True)
    scene = json.loads((args.cache/"scene.json").read_text())
    scene["emitters"] = [Emitter(**item,color=(1,1,1)) for item in scene["emitters"]]
    device = wp.get_device("cuda:0")
    state = SimpleNamespace(**scene,model=SimpleNamespace(device=device),fps=60,color_model="km",
                            optical_medium="coffee-milk",emitter_concentration=lambda emitter:emitter.concentration)
    rgb = read_hdri(args.hdri, args.hdr_sampling)
    height, width, _ = rgb.shape
    print(f"SSFR HDR sampling: {width}×{height} ({args.hdr_sampling})", flush=True)
    print(f"SSFR DistantLight reflection: {key_reflection}", flush=True)
    print(f"SSFR volume lighting: {args.volume_lighting}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ssfr-optics-") as scratch:
        shader=Path(scratch)/"optics_with_helpers.cu"
        original=Path(ssfr.__file__).with_name("spg")/"fluid.cu"
        shader.write_text(original.read_text()+"\n"+(HERE/"optics.cu").read_text())
        shader.with_suffix(".cu.lua").write_text((HERE/"optics.cu.lua").read_text())
        viewer=OpticsViewer(args.hdri.resolve(),shader,rgb,state.spacing,args.diagnostics,LIGHTING[lighting],
                            key_reflection=key_reflection,
                            volume_lighting=args.volume_lighting,
                            device=device,headless=True,fps=60,up_axis="Z",environment="none",
                            width=args.width,height=args.width*9//16)
        try:
            viewer.set_model(newton.ModelBuilder().finalize(device=device))
            viewer.show_particles=False
            viewer.set_camera(wp.vec3(1.65,-1.5,1.25),-24,120)
            tank=TankGL(state)
            for k,number in enumerate(args.frames):
                with np.load(args.cache/f"{number:03d}.npz") as snap:
                    points=wp.array(snap["positions"],dtype=wp.vec3,device=device)
                    c=wp.array(snap["concentration"],dtype=wp.float32,device=device)
                state.count=len(points)
                radii=wp.full(state.count,0.5*state.spacing,dtype=wp.float32,device=device)
                for _ in range(3 if k==0 else 1):
                    viewer.begin_frame(number/60)
                    tank.draw(viewer,state,draw_walls=False)
                    viewer.log_fluid_ssfr(points,radii,c)
                    viewer.end_frame()
                    viewer.get_frame()
                product=next(iter(viewer._render_products.values()))
                graph=viewer._ssfr_graph
                for key in ("Optics","OneInterface","Paths","Reflection","DomeReflection","KeySpecular","VolumeScatter"):
                    if key in graph:
                        image=copy_output(product,graph[key])
                        Image.fromarray(image[...,:3]).save(args.output/f"{number:03d}-{key}.png")
                if args.diagnostics:
                    rear=copy_output(product,graph["rear_depth"])
                    front=copy_output(product,graph["depth"])
                    np.savez_compressed(args.output/f"{number:03d}-depth.npz",front=front,rear=rear)
                if k%15==0 or k==len(args.frames)-1:
                    print(number,state.count,flush=True)
        finally:
            viewer.close()


if __name__=="__main__":
    main()
