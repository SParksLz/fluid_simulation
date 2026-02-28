"""
OpenGL 渲染器：继承自 Warp 的 OpenGLRenderer，可在此扩展自定义渲染逻辑。
"""

from typing import Union
import numpy as np
import warp as wp
import warp.render  # 显式加载 render 子模块，否则 wp.render 不可用

# 从 warp 内部模块获取 ShapeInstancer（wp.render 中已弃用该符号）
from warp._src.render.render_opengl import ShapeInstancer as _ShapeInstancer
from warp._src.render.utils import tab10_color_map
from warp._src.render.render_opengl import OpenGLRenderer as _OpenGLRenderer
from warp._src.render.render_opengl import update_points_positions 


ShapeInstancer = _ShapeInstancer


class CustomOpenGLRenderer(_OpenGLRenderer):
    """继承 wp.render.OpenGLRenderer，便于在项目中扩展或重写渲染行为。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render_points(self, name: str, points, radius, colors=None, as_spheres: bool = True, visible: bool = True):
        """Add a set of points

        Args:
            points: The points to render
            radius: The radius of the points (scalar or list)
            colors: The colors of the points
            name: A name for the USD prim on the stage
        """

        if len(points) == 0:
            return

        if isinstance(points, wp.array):
            wp_points = points
        else:
            wp_points = wp.array(points, dtype=wp.vec3, device=self._device)

        if name not in self._shape_instancers:
            np_points = points.numpy() if isinstance(points, wp.array) else points
            instancer = ShapeInstancer(self._shape_shader, self._device)
            radius_is_scalar = np.isscalar(radius)
            if radius_is_scalar:
                vertices, indices = self._create_sphere_mesh(radius)
            else:
                vertices, indices = self._create_sphere_mesh(1.0)
            if colors is None:
                color = tab10_color_map(len(self._shape_geo_hash))
            elif len(colors) == 3:
                color = colors
            else:
                color = colors[0]
            instancer.register_shape(vertices, indices, color, color)
            scalings = None if radius_is_scalar else np.tile(radius, (3, 1)).T
            instancer.allocate_instances(np_points, colors1=colors, colors2=colors, scalings=scalings)
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]
            np_points = points.numpy() if isinstance(points, wp.array) else points
            n = len(points)
            # 归一化 colors 供 allocate_instances 或 update_colors 使用
            colors1 = colors2 = None
            if colors is not None:
                if len(colors) == 3:
                    colors1 = np.tile(colors, (n, 1)).astype(np.float32)
                    colors2 = colors1
                elif np.ndim(colors) == 2 and np.shape(colors) == (n, 3):
                    colors1 = np.asarray(colors, dtype=np.float32)
                    colors2 = colors1
                else:
                    colors1 = np.tile(np.asarray(colors[0]), (n, 1)).astype(np.float32)
                    colors2 = colors1
            if len(points) != instancer.num_instances:
                instancer.allocate_instances(np_points, colors1=colors1, colors2=colors2)
            elif colors1 is not None:
                instancer.update_colors(colors1, colors2)

        with instancer:
            wp.launch(
                update_points_positions,
                dim=len(points),
                inputs=[wp_points, instancer.instance_scalings],
                outputs=[instancer.vbo_transforms],
                device=self._device,
            )
