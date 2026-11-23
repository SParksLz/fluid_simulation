"""Scalar transfer, nonlinear surface shading, portable animated USD."""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import warp as wp

from examples.mixing.apic import MixingSimulation
from examples.mixing.color import colors_numpy
from examples.mixing.surface import (LUT_SIZE, MixingSurface, SurfaceUSD,
                                     gl_palette_texture, palette_texture,
                                     interpolate_surface_concentration)


DEVICE = "cuda:0" if wp.get_cuda_device_count() else "cpu"


class TestSurfaceTransfer(unittest.TestCase):
    def transfer(self, values):
        positions = wp.array([[-0.5, 0, 0], [0.5, 0, 0]], dtype=wp.vec3, device=DEVICE)
        concentration = wp.array(values, dtype=float, device=DEVICE)
        volumes = wp.ones(2, dtype=float, device=DEVICE)
        flags = wp.ones(2, dtype=int, device=DEVICE)
        vertices = wp.array([[0, 0, 0], [0.2, 0, 0]], dtype=wp.vec3, device=DEVICE)
        scalar = wp.empty(2, dtype=float, device=DEVICE)
        uv = wp.empty(2, dtype=wp.vec2, device=DEVICE)
        missing = wp.zeros(1, dtype=int, device=DEVICE)
        grid = wp.HashGrid(4, 4, 4, device=DEVICE)
        grid.build(positions, 1.0)
        wp.launch(interpolate_surface_concentration, dim=2,
                  inputs=[grid.id, positions, concentration, volumes, flags,
                          vertices, scalar, uv, 1.0, LUT_SIZE, missing], device=DEVICE)
        self.assertEqual(int(missing.numpy()[0]), 0)
        np.testing.assert_array_equal(concentration.numpy(), values)
        return scalar.numpy(), uv.numpy()

    def test_scalar_is_interpolated_before_nonlinear_color(self):
        c, uv = self.transfer([0.0, 1.0])
        self.assertAlmostEqual(float(c[0]), 0.5)
        palette = palette_texture("km")[0].astype(float)/255.0
        texel = uv[0, 0]*LUT_SIZE-0.5
        left = int(np.floor(texel))
        actual = (1-(texel-left))*palette[left] + (texel-left)*palette[left+1]
        np.testing.assert_allclose(actual, colors_numpy(np.array([0.5]), "km")[0], atol=0.004)
        self.assertGreater(actual[1], 2.0*actual[0])
        averaged_rgb = colors_numpy(np.array([0.0, 1.0]), "km").mean(axis=0)
        self.assertGreater(np.linalg.norm(actual-averaged_rgb), 0.4)

    def test_constant_concentration_and_gl_texture_encoding(self):
        c, uv = self.transfer([0.375, 0.375])
        np.testing.assert_allclose(c, 0.375, atol=1.0e-6)
        np.testing.assert_allclose(uv[:, 0], (0.375*(LUT_SIZE-1)+0.5)/LUT_SIZE)
        raw = palette_texture("km")
        gl_texture = gl_palette_texture(raw)
        # Compare effective albedo after Newton GL's gamma decode with the
        # raw palette read by the USD shader, allowing 8-bit quantization.
        np.testing.assert_allclose((gl_texture/255.0)**2.2, raw/255.0, atol=0.005)


@unittest.skipUnless(wp.get_cuda_device_count(), "ParticleSurface requires CUDA")
class TestSurfaceUSD(unittest.TestCase):
    def test_animated_mesh_material_and_portable_palette(self):
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdShade, UsdUtils
        sim = MixingSimulation(12, DEVICE)
        surface = MixingSurface(sim)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mixing.usdc"
            output = SurfaceUSD(path, sim, surface, include_particles=True)
            output.record(sim)
            sim.step()
            output.record(sim)
            output.close()
            stage = Usd.Stage.Open(str(path))
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Mixing/Surface"))
            api = UsdGeom.PrimvarsAPI(mesh)
            scalar, st = api.GetPrimvar("dyeConcentration"), api.GetPrimvar("st")
            self.assertEqual(mesh.GetPointsAttr().GetTimeSamples(), [0.0, 1.0])
            self.assertEqual(scalar.GetInterpolation(), UsdGeom.Tokens.vertex)
            self.assertEqual(st.GetAttr().GetTimeSamples(), [0.0, 1.0])
            for time in (0, 1):
                x = np.asarray(mesh.GetPointsAttr().Get(time))
                faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(time))
                c, uv = np.asarray(scalar.Get(time)), np.asarray(st.Get(time))
                self.assertGreater(len(faces), 0)
                self.assertEqual(len(c), len(x))
                self.assertLess(int(faces.max()), len(x))
                self.assertGreaterEqual(int(faces.min()), 0)
                self.assertTrue(np.isfinite(x).all())
                np.testing.assert_allclose(uv[:, 0], (np.clip(c, 0, 1)*(LUT_SIZE-1)+0.5)/LUT_SIZE, atol=1.0e-6)
                np.testing.assert_allclose(np.asarray(mesh.GetDisplayColorPrimvar().Get(time)),
                                           colors_numpy(c, "km"), atol=1.0e-6)
            material, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
            self.assertEqual(str(material.GetPath()), "/Mixing/Materials/Pigment")
            preview = UsdShade.Shader(stage.GetPrimAtPath("/Mixing/Materials/Pigment/PreviewSurface"))
            source, name, _ = preview.GetInput("diffuseColor").GetConnectedSource()
            self.assertEqual(str(source.GetPath()), "/Mixing/Materials/Pigment/Palette")
            self.assertEqual(name, "rgb")
            texture = UsdShade.Shader(source.GetPrim())
            self.assertEqual(texture.GetInput("sourceColorSpace").Get(), "raw")
            asset = texture.GetInput("file").Get()
            self.assertFalse(Path(asset.path).is_absolute())
            self.assertTrue(Path(asset.resolvedPath).is_file())
            np.testing.assert_array_equal(np.asarray(Image.open(asset.resolvedPath)), surface.texture)
            particle_colors = UsdGeom.Points(stage.GetPrimAtPath("/Mixing/Particles")).GetDisplayColorPrimvar()
            self.assertEqual(len(particle_colors.Get(1)), sim.count)

            moved = Path(directory)/"moved"
            moved.mkdir()
            shutil.copy(path, moved/path.name)
            shutil.copy(asset.resolvedPath, moved/asset.path)
            relocated = Usd.Stage.Open(str(moved/path.name))
            texture2 = UsdShade.Shader(relocated.GetPrimAtPath(str(source.GetPath())))
            self.assertEqual(Path(texture2.GetInput("file").Get().resolvedPath), moved/asset.path)
            checker = UsdUtils.ComplianceChecker()
            checker.CheckCompliance(str(moved/path.name))
            self.assertEqual(checker.GetErrors(), [])
            self.assertEqual(checker.GetFailedChecks(), [])


if __name__ == "__main__":
    unittest.main()
