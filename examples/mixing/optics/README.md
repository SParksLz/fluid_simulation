# SSFR HDRI 光学回放

从粒子快照用 OVRTX 重建水面并输出逐帧 PNG。`Optics` 使用前后界面折射并读取 OVRTX 的线性场景颜色。`OneInterface` 使用单界面偏移供对照。`--diagnostics` 输出光路、DomeLight 高光、DistantLight 高光和体散射。

## 运行

在仓库根目录用[混色示例的 RTX 环境](../README.md)运行。安装 SciPy，并准备一张 2:1 经纬全景图。使用 OpenEXR 时，`ffmpeg` 和 `ffprobe` 需在 PATH 中。

```bash
python -m pip install -r requirements-mixing.txt scipy

python -m examples.mixing.optics.cache_vortex_pair \
  --output /tmp/ssfr-coffee-cache --secondary-acceleration 1.44 --track-milk-front

python -m examples.mixing.optics.render \
  --cache /tmp/ssfr-coffee-cache \
  --hdri /path/to/environment.exr --lighting trainstation \
  --volume-lighting directional --width 2560 --frames 90 180 216 300 \
  --output /tmp/ssfr-coffee-optics
```

`--frames` 使用 60 Hz 的仿真帧编号。完整 5 秒的 30 fps 回放可传入从 0 到 300 的全部偶数帧。试跑可用 `--width 640`。

`--lighting trainstation` 使用 DomeLight 绕 Z 轴 -60° 的灯位，与 DistantLight 的 2° 光盘高光。`--hdr-sampling native` 可让 SSFR 采样原始尺寸的环境图。DomeLight 使用输入文件的原始尺寸。`--volume-lighting directional` 为近表层加入有方向的入射光和介质内遮光。

次级涡的三档对照可用 `--secondary-acceleration 0.72`、`--secondary-acceleration 0.72 --track-milk-front` 和 `--secondary-acceleration 1.44 --track-milk-front` 生成。跟随模式额外记录次级涡的位置和权重。三档扩散系数均为 `0.003 m²/s`。
