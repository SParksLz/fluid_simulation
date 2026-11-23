"""CUDA attribute uploads for the mixing example's RTX particles and surface."""

import numpy as np
import warp as wp
from newton.viewer import ViewerRTX


@wp.kernel
def build_particle_attributes(
    radii: wp.array(dtype=float),
    scales: wp.array(dtype=wp.vec3),
    prototypes: wp.array(dtype=wp.int32),
    ids: wp.array(dtype=wp.int64),
    color_indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    scales[index] = wp.vec3(radii[index])
    prototypes[index] = 0
    ids[index] = wp.int64(index)
    color_indices[index] = index


_ATTRIBUTE_DTYPES = {
    wp.vec3: (np.float32, 3),
    wp.vec2: (np.float32, 2),
    wp.int32: (np.int32, 1),
    wp.int64: (np.int64, 1),
}

class MixingViewerRTX(ViewerRTX):
    def clear_model(self):
        super().clear_model()
        self._pending_gpu_points = {}
        self._pending_gpu_surfaces = {}
        self._particle_capacity = 0
        self._particle_attributes = {}
        self._face_counts = None

    def log_fluid_particles(self, points, radii, colors):
        name = "/mixing/particles"
        if not points.device.is_cuda:
            return super().log_points(name, points, radii=radii, colors=colors)

        from pxr import UsdGeom

        name = self._qualify(name)
        count = len(points)
        if name not in self._point_batch_paths:
            if count == 0:
                return
            # Only the USD schema and sphere prototype cross the CPU setup path.
            instancer = self._ensure_point_batch_primitive(name)
            instancer.CreatePositionsAttr([])
            instancer.CreateScalesAttr([])
            instancer.CreateProtoIndicesAttr([])
            instancer.CreateIdsAttr([])
            color = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
            color.Set([])
            color.SetIndices([])
            instancer.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            if self._phase == self._PHASE_RENDER:
                self._point_batch_paths[name] = self._replace_runtime_prim(str(instancer.GetPath()))

        if count > self._particle_capacity:
            self._particle_capacity = max(count, 2*self._particle_capacity)
            self._particle_attributes = {
                attribute: wp.empty(self._particle_capacity, dtype=dtype, device=points.device)
                for attribute, dtype in (("scales", wp.vec3), ("protoIndices", wp.int32),
                                         ("ids", wp.int64), ("primvars:displayColor:indices", wp.int32))
            }
        attributes = {attribute: storage[:count] for attribute, storage in self._particle_attributes.items()}
        if count:
            wp.launch(build_particle_attributes, dim=count,
                      inputs=[radii, attributes["scales"], attributes["protoIndices"],
                              attributes["ids"], attributes["primvars:displayColor:indices"]], device=points.device)
        attributes["positions"] = points
        attributes["primvars:displayColor"] = colors
        stream = wp.get_stream(points.device).cuda_stream
        self._pending_gpu_points[name] = (attributes, count > 0, stream)

    def log_fluid_surface(self, mesh, texture):
        from pxr import Sdf, UsdGeom

        name = self._qualify("/mixing/surface")
        face_count = len(mesh.indices)//3
        if name not in self._mesh_prim_paths:
            if face_count == 0:
                return
            path = self._get_path(name)
            self._ensure_scopes_for_path(self.stage, path)
            prim = UsdGeom.Mesh.Define(self.stage, path)
            prim.CreatePointsAttr([])
            prim.CreateNormalsAttr([])
            prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            prim.CreateFaceVertexCountsAttr([])
            prim.CreateFaceVertexIndicesAttr([])
            prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            prim.CreateDoubleSidedAttr(True)
            prim.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([])
            material = self._get_preview_surface_material(name, texture=texture,
                                                          roughness=0.25, metallic=0.0, opacity=1.0)
            self._bind_material(prim.GetPrim(), material)
            self._mesh_prim_paths[name] = path
            if self._phase == self._PHASE_RENDER:
                self._mesh_prim_paths[name] = self._replace_runtime_prim(path)

        if self._face_counts is None or len(self._face_counts) < face_count:
            previous = 0 if self._face_counts is None else len(self._face_counts)
            self._face_counts = wp.full(max(face_count, 2*previous), 3,
                                       dtype=wp.int32, device=mesh.vertices.device)
        attributes = {"points": mesh.vertices, "normals": mesh.normals,
                      "faceVertexIndices": mesh.indices, "faceVertexCounts": self._face_counts[:face_count],
                      "primvars:st": mesh.uvs}
        stream = wp.get_stream(mesh.vertices.device).cuda_stream
        self._pending_gpu_surfaces[name] = (attributes, face_count > 0, stream)

    def _flush_gpu_attributes(self, pending, prim_paths):
        import ovstage

        for name, (attributes, visible, stream) in pending.items():
            path = prim_paths[name]
            if visible:
                writes = []
                for attribute, array in attributes.items():
                    if not array.device.is_cuda:
                        raise ValueError(f"RTX fluid attribute {attribute} must reside on CUDA")
                    scalar, lanes = _ATTRIBUTE_DTYPES[array.dtype]
                    dtype = ovstage.numpy_to_dldatatype(np.dtype(scalar), lanes=lanes)
                    tensor = ovstage.make_dltensor(array, dtype=dtype)
                    # DLPack aliases each device allocation. The queued payload
                    # and operation keep its owner alive through the copy-in.
                    # ovstage uses 0 for no handoff, 1 for CUDA's default stream.
                    writes.append(ovstage.WriteDesc(attribute, tensors=[tensor], is_array=True,
                                                    cuda_stream=stream or 1))
                query = self._get_ovstage_query([path])
                self._ovstage.write_attributes(query, writes, ordinal=self._ovstage_ordinal).wait()
            self._write_runtime_attribute([path], "visibility", ["inherited" if visible else "invisible"])
        pending.clear()

    def _update_ovrtx_point_batches(self):
        super()._update_ovrtx_point_batches()
        self._flush_gpu_attributes(self._pending_gpu_points, self._point_batch_paths)

    def _update_ovrtx_mesh_points(self):
        super()._update_ovrtx_mesh_points()
        self._flush_gpu_attributes(self._pending_gpu_surfaces, self._mesh_prim_paths)

    def _get_preview_surface_material(self, mesh_name, **kwargs):
        from pxr import Sdf, UsdShade

        material = super()._get_preview_surface_material(mesh_name, **kwargs)
        if kwargs.get("texture") is not None:
            # The palette stores linear reflectance, including endpoint texels.
            # Author this before RTX publishes a newly appearing surface prim.
            texture = UsdShade.Shader(self.stage.GetPrimAtPath(f"{material.GetPath()}/DiffuseTexture"))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        return material
