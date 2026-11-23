"""Matching GL/USD tank walls, source rings and emitted-volume height ruler."""

import math

import numpy as np
import warp as wp

from examples.mixing.media import display_colors


def tank_geometry(sim):
    lo, hi = np.asarray(sim.bound_lo), np.asarray(sim.bound_hi)
    corners = np.array([(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
                        (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
                        (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
                        (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])], dtype=np.float32)
    edge_ids = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
    lines = [(corners[a], corners[b]) for a, b in edge_ids]
    for z in np.arange(0.1, hi[2], 0.1):
        lines.append(((lo[0]-0.025, lo[1], z), (lo[0]+0.025, lo[1], z)))
        lines.append(((hi[0]-0.025, lo[1], z), (hi[0]+0.025, lo[1], z)))
    lines = np.asarray(lines, dtype=np.float32)
    wall_ids = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    walls = np.concatenate([corners[list(face)] for face in wall_ids])
    triangles = np.array([v for start in range(0, 16, 4)
                          for v in (start, start+1, start+2, start, start+2, start+3)], dtype=np.int32)
    floor = corners[:4].copy()
    floor[:, 2] -= 0.005
    return lines, walls, triangles, floor


def emitter_lines(emitter, radius):
    d = np.asarray(emitter.direction)
    u, v = emitter.tangent_axes()
    rings = []
    for center in (np.asarray(emitter.center), np.asarray(emitter.center)-0.08*d):
        theta = np.arange(33)*2*np.pi/32
        rings.append(center+radius*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v))
    lines = [(ring[i], ring[i+1]) for ring in rings for i in range(32)]
    lines += [(rings[0][i], rings[1][i]) for i in (0, 8, 16, 24)]
    return np.asarray(lines, dtype=np.float32)


def rest_height_line(sim):
    height = sim.count*sim.volume/((sim.bound_hi[0]-sim.bound_lo[0])*(sim.bound_hi[1]-sim.bound_lo[1]))
    return np.asarray([(sim.bound_lo[0], sim.bound_lo[1]-0.003, height),
                       (sim.bound_hi[0], sim.bound_lo[1]-0.003, height)], dtype=np.float32)


class TankGL:
    def __init__(self, sim):
        lines, walls, triangles, floor = tank_geometry(sim)
        device = sim.model.device
        self.starts = wp.array(lines[:, 0], dtype=wp.vec3, device=device)
        self.ends = wp.array(lines[:, 1], dtype=wp.vec3, device=device)
        self.walls = wp.array(walls, dtype=wp.vec3, device=device)
        self.indices = wp.array(triangles, dtype=int, device=device)
        self.floor = wp.array(floor, dtype=wp.vec3, device=device)
        self.floor_indices = wp.array([0, 1, 2, 0, 2, 3], dtype=int, device=device)
        self.emitters = []
        for emitter in sim.emitters:
            lines = emitter_lines(emitter, sim.emitter_radius)
            self.emitters.append((emitter, wp.array(lines[:, 0], dtype=wp.vec3, device=device),
                                  wp.array(lines[:, 1], dtype=wp.vec3, device=device)))

    def draw(self, viewer, sim, draw_walls=True):
        viewer.log_mesh("/mixing/tank/floor", self.floor, self.floor_indices,
                        color=(0.22, 0.24, 0.27), roughness=0.5, backface_culling=False)
        if draw_walls:
            viewer.log_mesh("/mixing/tank/walls", self.walls, self.indices,
                            color=(0.65, 0.8, 0.9), roughness=0.1, opacity=0.12, backface_culling=False)
        viewer.log_lines("/mixing/tank/ruler", self.starts, self.ends, colors=(0.65, 0.7, 0.75), width=0.002)
        for emitter, start, end in self.emitters:
            color = display_colors([sim.emitter_concentration(emitter)], sim.color_model,
                                   getattr(sim, "optical_medium", "pigment"),
                                   getattr(sim, "optical_medium_scale", 1.0))[0]
            viewer.log_lines("/mixing/emitters/"+emitter.name, start, end,
                             colors=tuple(float(v) for v in color), width=0.003)
        line = rest_height_line(sim)
        start = wp.array(line[:1], dtype=wp.vec3, device=sim.model.device)
        end = wp.array(line[1:], dtype=wp.vec3, device=sim.model.device)
        viewer.log_lines("/mixing/tank/rest_volume", start, end, colors=(1.0, 0.4, 0.1), width=0.002)


class TankUSD:
    def __init__(self, stage, sim):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        self.stage = stage
        base = "/Mixing/Tank"
        UsdGeom.Xform.Define(stage, base)
        lines, walls, triangles, floor = tank_geometry(sim)
        for name, positions, faces, color, opacity in (
            ("Floor", floor, [0, 1, 2, 0, 2, 3], (0.22, 0.24, 0.27), 1.0),
            ("Walls", walls, triangles, (0.65, 0.8, 0.9), 0.12),
        ):
            mesh = UsdGeom.Mesh.Define(stage, base+"/"+name)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))
            mesh.CreateFaceVertexCountsAttr([3]*(len(faces)//3))
            mesh.CreateFaceVertexIndicesAttr([int(v) for v in faces])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            material = UsdShade.Material.Define(stage, base+"/Materials/"+name)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        self._curve(base+"/Ruler", lines, (0.65, 0.7, 0.75), 0.002)
        self.emitter_curves = []
        self.last_emitter_frame = None
        for emitter in sim.emitters:
            color = display_colors([emitter.concentration], sim.color_model,
                                   getattr(sim, "optical_medium", "pigment"),
                                   getattr(sim, "optical_medium_scale", 1.0))[0]
            curve = self._curve("/Mixing/Emitters/"+emitter.name,
                                emitter_lines(emitter, sim.emitter_radius),
                                tuple(float(v) for v in color), 0.003)
            curve.GetPrim().SetCustomDataByKey("initialDyeConcentration", emitter.concentration)
            concentration = curve.GetPrim().CreateAttribute("dyeConcentration", Sdf.ValueTypeNames.Float)
            concentration.Set(emitter.concentration)
            self.emitter_curves.append((emitter, curve, concentration))
        self.level = self._curve(base+"/RestVolumeLevel", rest_height_line(sim)[None, :, :], (1.0, 0.4, 0.1), 0.002)
        self.height = self.level.GetPrim().CreateAttribute("expectedSettledHeight", Sdf.ValueTypeNames.Float)

    def _curve(self, path, lines, color, width):
        from pxr import Gf, UsdGeom, Vt
        curve = UsdGeom.BasisCurves.Define(self.stage, path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([2]*len(lines))
        curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(lines.reshape(-1, 3)))
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return curve

    def record(self, sim):
        from pxr import Gf, Vt
        line = rest_height_line(sim)
        self.level.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(line), sim.frame)
        self.height.Set(float(line[0, 2]), sim.frame)
        # Preserve sharp nozzle changes even when USD samples skip frames.
        samples = {float(sim.frame)}
        period = sim.emitter_color_period*sim.fps
        if period and self.last_emitter_frame is not None:
            for cycle in range(math.floor(self.last_emitter_frame/period)+1,
                               math.floor(sim.frame/period)+1):
                boundary = cycle*period
                epsilon = min(1.0e-3, 0.25*period)
                samples.update((boundary-epsilon, boundary))
        for time in sorted(samples):
            for emitter, curve, concentration in self.emitter_curves:
                c = sim.emitter_concentration(emitter, time/sim.fps)
                color = display_colors([c], sim.color_model,
                                       getattr(sim, "optical_medium", "pigment"),
                                       getattr(sim, "optical_medium_scale", 1.0))[0]
                curve.GetDisplayColorAttr().Set([Gf.Vec3f(*(float(v) for v in color))], time)
                concentration.Set(c, time)
        self.last_emitter_frame = float(sim.frame)
