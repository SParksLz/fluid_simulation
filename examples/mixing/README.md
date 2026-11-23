# APIC + LSMPS 双发射器混色水槽

默认场景从空水槽开始。两个 emitter 最初分别注入黄色、蓝色水流，每隔 2 秒交换注入颜色，朝槽内下方喷射，并分别向前后偏转，给槽内水体同一方向的回旋趋势。APIC 负责流体运动，LSMPS 提供粒子局部拟合，KM 负责颜色显示。

## 运行

在仓库根目录、已有 Newton/Warp 的环境中运行：

```bash
# 额外的 PNG 颜色表依赖（不会修改 Newton/Warp 版本）
python -m pip install -r requirements-mixing.txt

# 默认：空水槽、两个 emitter、KM 彩色粒子（不计算 surface）
python examples/mixing/apic.py

# 切换到彩色表面；加 --show-particles 可同时保留粒子
python examples/mixing/apic.py --surface

# OVRTX 彩色粒子，或彩色 surface
python examples/mixing/apic.py --viewer rtx
python examples/mixing/apic.py --viewer rtx --surface

# 无窗口运行 RTX，仍执行实际渲染
python examples/mixing/apic.py --viewer rtx --headless --resolution 24 --frames 8 --emitter-duration 0.1

# 无窗口导出 surface USD 和同目录的 mixing_palette.png
python examples/mixing/apic.py --surface --viewer none --usd-every 6 \
  --usd examples/mixing/output/tank/mixing.usdc \
  --output-dir examples/mixing/output/tank

# 粒子 USD，包含稳定粒子 ID、emitterId 与原始浓度
python examples/mixing/apic.py --viewer none --no-surface \
  --usd examples/mixing/output/tank_particles/mixing.usdc \
  --output-dir examples/mixing/output/tank_particles

# 调整注入速度、半径和持续时间
python examples/mixing/apic.py --emitter-speed 1.2 --emitter-radius 0.075 --emitter-duration 3

# 喷口每 2 秒交换黄/蓝；延长注入时间以观察多轮切换
python examples/mixing/apic.py --emitter-color-period 2 --emitter-duration 6 --frames 540

# 关闭换色，恢复每个喷口固定注入一种染料
python examples/mixing/apic.py --emitter-color-period 0

# 默认前后偏转 20°；0 恢复正对，负角度反转回旋趋势
python examples/mixing/apic.py --emitter-yaw-offset 20
python examples/mixing/apic.py --emitter-yaw-offset 0

# 仅恢复旧版整层发射，用于观察水柱的分节条纹
python examples/mixing/apic.py --emission-mode layered

# RGB 颜色对照，或关闭物理扩散
python examples/mixing/apic.py --color-model rgb
python examples/mixing/apic.py --diffusivity 0

# 原来的较弱扩散和短混合时长，便于对照
python examples/mixing/apic.py --diffusivity 0.001 --frames 240

# 小规模 CPU 检查
python examples/mixing/apic.py --viewer none --no-surface --device cpu \
  --resolution 24 --frames 12 --emitter-duration 0.2

# 保留原来的封闭水体旋涡例子，用于论文数值对照
python examples/mixing/apic.py --scene vortex
```

每次运行生成 `metrics.csv`、`run.json` 和 `final_state.npz`。状态快照保存实际活动粒子的位置、速度、浓度；tank 场景另外保存 `source_ids` 和量化后的 `birth_time_s`。输出目录中的历史结果会被同名运行覆盖，请为对照使用不同目录。

RTX 使用 CUDA 模拟设备，默认自动选择 `cuda:0`。在 `env_newton_1_5` 环境中运行，该环境实际安装 Newton 1.7.0、Warp 1.19.0 和 ovrtx 0.5.0.377615。当前 Newton 的 RTX 查看器需要 ovstage 0.2 系列，已补装 ovstage 0.2.1.385922。ovstage 0.3 修改了场景载入接口，会与这个 Newton 版本冲突。其他环境可用下面的命令安装同一组合。

```bash
conda activate env_newton_1_5
python -m pip install newton==1.7.0 warp-lang==1.19.0 ovrtx==0.5.0.377615 ovstage==0.2.1.385922
```

RTX 使用 studio 灯光，支持逐粒子颜色、喷口换色、透明槽壁和理论液位刻度。surface 使用线性 KM 颜色表，并随动态网格更新浓度 UV。`viewer_rtx.py` 将逐帧流体属性通过 CUDA DLPack 提交给 ovstage，覆盖粒子位置、半径缩放、颜色、ID，以及 surface 顶点、法线、拓扑和 UV。辅助索引和三角形计数在 GPU 上生成，缓冲随容量增长并复用。提交携带生产者 CUDA stream，并将源数组保留到写操作完成。

GPU 直传的范围是应用提交给 ovstage 的流体属性。ovstage 0.2.1 的写接口执行 copy-in，公开 read/map 返回 CPU buffer。场景和材质初始化、诊断统计、USD 导出与截图会使用 CPU 数据。`run.json` 记录查看器选项、RTX 依赖版本，以及 CUDA 模式下的 `rtx_fluid_upload=cuda_dlpack`。

RTX 粒子与 surface 均完成从空槽开始的 8 帧实际模拟渲染，窗口模式完成 3 帧启动检查。另用已有动画的 1.8、2.1、4.1 秒状态核对粒子数量增长、喷口换色和动态 surface 颜色，并读取实际 RTX 输出检查画面。

GPU 上传改造后完成了粒子与 surface 同时显示的 20 帧实际模拟渲染。新增测试在属性提交过程中禁止调用 Warp `.numpy()`，通过真实 ovstage 写入与结果校验，覆盖粒子数量增减、空帧、颜色、索引、动态拓扑和 UV，并检查非默认 CUDA stream 的数据交接。

## RTX 屏幕空间流体渲染

```bash
conda activate env_newton_1_5
python examples/mixing/apic.py --viewer rtx --ssfr

# 曲率流对照，默认 100 次更新
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-filter curvature

# 查看重建法线，也可选 depth、thickness、dye、specular
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-view normals

# 单独检查反射高光，roughness 默认 0.2
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-view specular --ssfr-roughness 0.2

# 默认使用各向异性椭球投影，可切换球形核进行对照
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-splats anisotropic
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-splats sphere

# 调整渲染半径与染料吸收强度
python examples/mixing/apic.py --viewer rtx --ssfr \
  --ssfr-radius-scale 1.6 --ssfr-absorption 12
```

`--ssfr` 使用 OVRTX SPG 的 CUDA 节点重建当前相机可见的水面。运行组合为上面记录的 Newton 1.7.0、Warp 1.19.0、ovrtx 0.5.0.377615、ovstage 0.2.1.385922。首次运行会编译 CUDA 节点。节点源码和同名 Lua 启动脚本位于 `spg/fluid.cu` 与 `spg/fluid.cu.lua`，USD 图由 `ssfr.py` 创建。SPG 的节点、资源端口和 CUDA 启动约定见官方文档。[OVRTX SPG](https://nvidia-omniverse.github.io/ovrtx/spg/overview.html)

实现使用 van der Laan 等的屏幕空间重建与光学合成流程，默认深度滤波采用 Truong 和 Yuksel 的窄范围算法。滤波包含深度钳制、对称采样偏差修正、动态深度范围，以及横纵分离滤波后的二维清理。曲率流模式使用局部显式步长和不连续边界约束，便于与窄范围结果比较。[Screen Space Fluid Rendering with Curvature Flow](https://doi.org/10.1145/1507149.1507164), [A Narrow-Range Filter for Screen-Space Fluid Rendering](https://doi.org/10.1145/3203201)

| 阶段 | 实现与数据 |
| --- | --- |
| 背景 | RTX 输出场景 LdrColor 与 DistanceToCameraSD |
| 粒子重建 | GPU 邻域加权 PCA 拟合椭球主方向，约束轴比和核体积，平滑渲染核中心 |
| 粒子索引 | GPU 计算椭球包围盒，计数、前缀和与填充构建 16×16 屏幕分块索引 |
| 深度与厚度 | SPG 按像素遍历分块中的椭球，计算解析射线交点和弦长，并由场景距离裁剪 |
| 轮廓 | 半径 2 像素的局部闭运算修补粒子交叠处的凹口，以深度范围约束分层表面 |
| 深度平滑 | 默认 2 轮横纵窄范围滤波，随后 5×5 二维清理，Gaussian 标准差连续变化，上限 32 像素 |
| 法线 | 5×5 深度加权平面拟合，从透视位置导数重建法线 |
| 混色 | 厚度和染料量联合滤波，再取比值恢复浓度并计算 KM 或 RGB 颜色 |
| 着色 | Schlick Fresnel、GGX 方向光高光、dome 颜色近似、双线性屏幕空间折射、逐通道指数吸收 |
| 显示 | SPG 的 FluidColor AOV 通过 Newton 原有 CUDA 显示路径进入窗口 |

渲染球半径默认是物理粒子半径的 1.6 倍，用于覆盖粒子间隙。厚度按等体积粒子的球体体积比例补偿，调整半径时保持体积权重一致。吸收系数由当前 KM/RGB 外观颜色近似构造，默认强度 `12 m⁻¹`，折射率默认 `1.333`。方向光高光读取 USD 中的方向与颜色，dome 使用颜色梯度近似。高光经过显示范围压缩后与 RTX 的 LdrColor 合成，曝光和 RGB 光学参数属于演示近似。

默认 `--ssfr-splats anisotropic` 使用 Yu 和 Turk 的邻域 PCA 与核中心平滑思路。`splats.py` 在 GPU 上搜索物理粒子半径 5 倍以内的邻居，构建加权协方差矩阵，通过 SVD 得到主方向。主轴比上限为 `4:1`，轴尺度乘积归一化为 1，以保持单个重建核体积。渲染核中心的位移上限为物理半径的 `0.25` 倍。邻域权重和连续控制形变强度，稀疏水滴使用球形核。本例将拟合椭球的解析交点用于屏幕空间深度与厚度生成。[Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels](https://faculty.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf)

各向异性核沿连续水面与水柱方向铺开，以减轻轮廓的小圆弧。该模式增加一轮邻域计算和 PCA，核中心平滑与主轴形变会略微改变薄流和边缘的可见厚度。`--ssfr-splats sphere` 使用球形投影。两种模式都由 CUDA 常驻缓冲区向 SPG 提供数据，拟合矩阵以设备地址传入。

高光稳定化包括连续滤波宽度、平滑截断 Gaussian 权重、局部平面拟合法线，以及按拟合残差拓宽 GGX 高光。`--ssfr-roughness` 范围为 `(0,1]`，默认 `0.2`。厚度滤波由深度引导，染料量使用同一权重，恒定浓度在滤波后保持恒定。轮廓修补最多影响边界附近 2 像素，可能轻微修整细小轮廓。GGX 使用 Smith 相关遮蔽项。[Filament 的镜面 BRDF 说明](https://google.github.io/filament/main/filament.html#materialsystem/specularbrdf)

当前 ovrtx 0.5.0.377615 环境的 SPG `float3` 参数路径在静态重复渲染中出现过颜色变化。CUDA 节点现在通过独立 float 分量接收相机基向量和灯光颜色，在核函数内部构造向量。回归测试要求静止水面的高光逐像素重复一致，并检查折射率为 1 时反射为零。

SSFR 水槽显示底板、线框槽壁、喷口与理论液位刻度。半透明槽壁进入 RTX 距离 AOV 后会遮挡后方流体，因此这一模式用线框表示槽壁。其余相机可见的场景几何参与遮挡和厚度裁剪。

粒子视空间位置和浓度使用独立 GPU 快照，SPG 接收这些常驻缓冲区的 CUDA 地址。生产者事件完成交接后，GPU 分块构建在提交 SPG 前同步，异步帧完成后才复用快照。逐帧仅将一个分块索引容量标量读回 CPU，用于安全扩容。地址、相机与渲染参数通过小型 USD 属性更新。模拟诊断、场景几何日志、导出和截图仍沿用各自现有的数据路径。`run.json` 记录 `ssfr_parameters` 和 `rtx_fluid_upload=cuda_device_address`。

图还发布 `FluidRawDepth`、`FluidDepth`、`FluidOptical` 与 `FluidOpticalSmooth` AOV。深度为正视空间 Z，零表示无流体。Optical 的两个通道分别为场景裁剪后的加权厚度与厚度乘浓度，Smooth 为滤波后的对应数据。零次滤波时通过 `FluidRawDepth` 读取未滤波深度。`--ssfr-view` 可切换最终显示的诊断图，`--ssfr-iterations` 可调整滤波次数。

`--ssfr` 与 `--surface` 互斥。SSFR 随当前相机重建屏幕水面，`--usd` 在该模式下保存粒子动画。SPG 图中的设备地址属于当前进程，动画重放时由查看器重新创建渲染图和 GPU 缓冲。

数值测试覆盖球体深度和弦长、遮挡与厚度裁剪、浓度累积、空帧、容量增长、近裁剪面、相机移动及曲率流。分块测试限制读回为一个 int32 标量，并验证 GPU 快照独立于输入数组。

测试覆盖光学滤波后的恒定浓度、厚度纹理平滑、空帧清理、静态高光逐像素一致，以及折射率为 1 时的零反射。各向异性测试还检查单核体积、轴比、中心位移、旋转等变性、孤立水滴与射线深度和厚度的数值参考值。

此前 37 项 mixing 测试通过。规则水柱的原始深度轮廓测试中，边界相对 15 像素移动平均的 RMS 起伏从约 `0.764 px` 降至 `0.380 px`。当时 resolution=96 完成了 24 帧实际混色模拟与 RTX 无窗口渲染，使用默认扩散和每帧 16 子步，末帧活动粒子数 `8656`。完整 3 秒发射计划的 `65088` 粒由采样计划核实。

实际入口完成空槽发射的 12 帧无窗口运行、3 帧窗口运行、默认 100 次曲率流的 2 帧运行，以及 vortex 场景的 2 帧运行。另重放已有粒子动画的 1.8、2.1、4.1 秒状态，检查密集水体与喷口换色。预览图来自这组动画重放。高光与颗粒感修正还使用同一段 24 个连续仿真帧对比前后结果。

24 帧对比在 `800×600` 图像的固定水体区域 `y=305:410, x=310:500` 测量帧间亮度差。平均绝对帧间差除以区域平均亮度，从 `0.0656` 降至 `0.00542`。该数值记录这段相同粒子运动下的图像变化。

```bash
python -m unittest unit_test.unit_test_mixing_ssfr -v
```

可用 [HDRI 光学回放](optics/README.md) 对同一缓存比较前后表面折射、方向光高光和近表层体散射。

## 默认水槽与发射器

| 参数 | 默认值 |
| --- | --- |
| Tank | `(0,0,0)` 到 `(1.2,0.6,0.65) m`，初始为空 |
| 重力 | `(0,0,-9.81) m/s²` |
| 粒子间距 | `1.2 / resolution`，默认 resolution=96，即 `0.0125 m` |
| 粒子半径 | `0.5 × spacing`，默认 `0.00625 m` |
| 粒子体积、质量 | `spacing³`、`1000 × spacing³ kg` |
| 黄色 nozzle 中心 | `(0.28,0.3,0.52) m` |
| 蓝色 nozzle 中心 | `(0.92,0.3,0.52) m` |
| 喷射方向 | 保持向下 45°，黄色朝前偏、蓝色朝后偏，水平偏角默认 20° |
| 喷射速度、半径 | `1.2 m/s`、`0.075 m` |
| 发射持续时间 | 3 s |
| 喷口换色周期 | 2 s；`--emitter-color-period 0` 关闭 |
| 动画 | 360 帧、60 fps、每帧 16 个物理子步，总计 6 s |
| 浓度扩散系数 | `0.003 m²/s`，用于演示，未标定真实染料 |
| 仿射矩阵稳定化 | `c_damping=1.0`、`c_norm_max=160 s⁻¹` |
| 粒子间距修正 | 强度 `0.10`、每子步 1 次 |

默认采用连续发射。圆盘的 113 个采样列使用固定随机排列，在 `spacing / emitter_speed` 周期内均匀错开出生时间，再量化到下一个物理子步。两个喷口同步、等流量注入，默认 3 秒发射计划共 `65088` 粒。相较 resolution=64 的 `18816` 粒，数量约为 `3.46` 倍，粒子半径为此前的 `2/3`。新版累计注入体积为 `0.127125 m³`，圆盘离散采样会带来少量流量差异。3 秒后停止发射，余下 3 秒继续流动和混色。可用 `--resolution 64` 对照前一规格。更高分辨率和喷射速度通常也需要增加 `--substeps`。

这个示例没有加入流体黏性项。此前每子步的仿射衰减 `0.998`、梯度上限 `40 s⁻¹` 和较强的粒子分离会抑制运动细节；默认每秒有 960 个子步，不能只把单步衰减看作一个很小的数。现在仅在 tank 示例中取消仿射衰减、放宽梯度上限并减轻间距修正。APIC 传输、压力离散和有限分辨率仍会产生数值耗散；这次调整不构成真实水黏度的标定。浓度的 diffusivity 只改变颜色，不参与速度计算。

喷口不断注入浓度为 0/1 的新水：0–2 秒为黄/蓝，2–4 秒为蓝/黄，4–6 秒恢复黄/蓝，随后重复。换色按仿真时间计算，只改变新粒子的初始染料浓度；已发射的粒子继续随流动和扩散演化。两个喷口始终注入等量的两种组分，换色不改变速度或流量。默认 3 秒停止发射，因此只发生 2 秒处的一次切换；观察更多轮次需要延长 `--emitter-duration` 和 `--frames`。

接触后的均匀化由扩散系数和流动时间决定，KM 只负责颜色转换。当前默认提高扩散并延长停喷后时间，让池水更快趋向浓度 0.5 的均匀绿色。喷口圆环在 GL/USD 中同步换色；USD 路径中的 Yellow/Blue 和 `emitterId` 保持来源标识，喷口属性 `dyeConcentration` 记录随时间变化的注入浓度，`initialDyeConcentration` 记录初始值。

`--emitter-yaw-offset` 范围为 `[-45,45]` 度，只改变喷口方向；速度、流量采样和粒子大小保持相同。偏转后两个喷口向竖直轴注入同号角动量，不额外施加旋涡体力。喷口采样平面和显示圆环都使用垂直于喷射方向的正交基，避免偏转后喷口变成椭圆。

粒子容量按完整发射计划预分配，只将已发射的连续前缀传给 APIC、LSMPS、显示和导出，不回收或复用旧粒子。每次出生都会初始化两个状态缓冲中的位置、速度、浓度、零仿射矩阵和零外力，保留旧粒子的演化结果。APIC 稀疏网格提前按整个水槽域预留空间，避免只根据第一层小水柱分配容量。

水槽的四壁、底面和两个喷口同时出现在 GL/USD 中；竖边短刻度每隔 `0.1 m`。橙线是**全部已注入静止体积 / 水槽底面积**对应的理论沉降液位，包含仍在空中的粒子体积，并非当前自由液面的测量值。

## LSMPS 与扩散稳定化

参考 Matsunaga 等的 [Hybrid grid-particle method for fluid mixing simulation (2015)](https://doi.org/10.1007/s40571-015-0046-7)。LSMPS 全称是 **Least Squares Moving Particle Semi-implicit**：用邻居位置和浓度拟合局部多项式，再计算拉普拉斯。三阶三维基底有 19 项，支撑半径 `2.5 × spacing`。浓度作为被动标量，使用显式 Euler 在旧粒子位置推进，再由 APIC 推进粒子。

**Tank 模式为自由表面和细水柱增加了稳定化：**

- 三阶拟合退化时尝试二阶拟合；仍退化则该粒子暂不交换浓度，并在指标中统计。
- 将拟合系数转换为**非负、对称的粒子对通量**，用实际通量行和限制显式步长。对于本例的等体积粒子，这保证染料积分守恒和浓度有界，不需要裁剪浓度或全局重归一化。
- 总量误差按实际注入的蓝色组分体积核算，不能与固定初始染料量比较。

这项稳定化改变了论文原始配点扩散算子，不应视为其高阶一致性在自由表面上的精确复现。对未知自由液面没有重建法线或精确流体连通域，支撑半径内的薄空气隙仍可能出现数值传色。

`--scene vortex` 保留原始封闭旋涡场景：`0.8 × 0.4 × 0.4 m`、零重力、初始左右浓度为 0/1、默认 3456 粒、180 帧、每帧 4 子步。该模式使用三阶 LSMPS 原始配点算子与箱壁镜像粒子，适合解析解和论文方法对照；它不自动保证严格染料守恒。`--initial-speed` 和 `--stirring-acceleration` 只作用于此模式。

## KM 颜色与彩色 surface

参考 [Kubelka-Munk Color Mixing in VEX](https://vanity-ibex.xyz/blog/kubelka_munk_colormixing/)。将 RGB 通道近似视为有效反射率 R，计算 `F(R)=K/S=(1-R)²/(2R)`。假设散射系数均为 S=1，按浓度混合 F，再求反射率。

黄色为 `(0.95,0.85,0.05)`，蓝色为 `(0.05,0.15,0.95)`。等量混合在 RGB 模式为灰色 `(0.5,0.5,0.5)`，默认 KM 模式约为深绿色 `(0.0915,0.2391,0.0915)`。这是三通道、固定散射的颜料外观近似，没有实测颜料光谱或透明染料水的体积吸收。

彩色 surface 的流程为：

1. 用 Newton ParticleSurface 重建几何，默认体素 `spacing/3`。
2. 在表面顶点以粒子体积及紧支撑核加权，插值**浓度**，默认支撑 `2.5 × spacing`。
3. 浓度编码为 UV；三角形内部先插值 UV，再查询 1024 项 KM 颜色表，避免重新平均黄色、蓝色 RGB。

默认只显示/导出粒子，不分配或计算 surface。使用 `--surface` 开启表面后，可用 `--surface-voxel-size`、`--surface-color-radius` 调整显示分辨率和插值支撑；没有查看器和 USD 输出时仍不分配表面资源。表面重建需要 CUDA，CPU 可直接运行默认粒子模式。

表面插值只影响显示；即使 `D=0`，也可能出现窄的颜色过渡带。物理混合指标始终由粒子原始浓度计算。`--show-particles` 同时保留表面和粒子，方便在 USD 中分别查看。

## 咖啡、清水与牛奶的光学混合

```bash
# 同一双喷口水槽，一路咖啡、一路牛奶
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-medium coffee-milk

# 高质量档，细化模拟网格并增加粒子，1080p、12 层介质
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-medium coffee-milk --quality high

# 水槽预先装咖啡，再由右侧喷口慢速导入牛奶
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-medium coffee-milk \
  --initial-coffee-height 0.12 --emitter-speed 0.6 --emitter-duration 3

# 清水与牛奶，观察透明度随局部浓度变化
python examples/mixing/apic.py --viewer rtx --ssfr --ssfr-medium water-milk

# 同时缩放吸收与散射系数，降低光学密度
python examples/mixing/apic.py --viewer rtx --ssfr \
  --ssfr-medium coffee-milk --ssfr-medium-scale 0.5

# 分别检查背景透射率与介质散射项
python examples/mixing/apic.py --viewer rtx --ssfr \
  --ssfr-medium coffee-milk --ssfr-view transmittance
python examples/mixing/apic.py --viewer rtx --ssfr \
  --ssfr-medium coffee-milk --ssfr-view scattering
```

这两个介质预设默认固定喷口材料，`c=0` 为咖啡或清水，`c=1` 为牛奶。可用 `--emitter-color-period 2` 恢复每 2 秒交换两路材料。默认 `pigment` 模式继续使用黄蓝 KM 混色与周期换色。

`--quality high` 的 tank 规格为 resolution=160、24 个物理子步、1920×1080 图像和 12 个光学深度层。粒子间距为 7.5 mm、半径为 3.75 mm、APIC 网格间距为 11.25 mm，3 秒注入计划共 304320 粒。`--resolution`、`--substeps`、`--render-width`、`--render-height` 和 `--ssfr-medium-layers` 的显式值优先于档位设置。光学层数支持 1 至 16 层。

`--initial-coffee-height` 会在模拟开始时用静止的 c=0 粒子填充咖啡，并自动只启用右侧 c=1 牛奶喷口。喷口速度决定导入速度，`--emitter-duration` 决定持续时间。初始咖啡粒子拥有 `emitterId=-1`，后续牛奶粒子保留右侧喷口的来源编号。

预填充咖啡示例已完成 5 秒 RTX 录制，使用 resolution=96、初始液面 0.15 m、牛奶速度 0.6 m/s、导入 3 秒，输出 1280×720、60 fps 视频 `output/coffee_prefilled_milk/coffee_prefilled_milk_r96.mp4`。末帧活动粒子为 `71568`，牛奶组分均值约 `0.227`，染料积分相对误差约 `3.3×10⁻⁷`。

浓度随粒子输运，并由 LSMPS 扩散。SPG 沿视线划分多个深度层，标准档为 8 层、高质量档为 12 层，前面的层各宽 4 倍物理粒子半径，最后一层包含剩余可见厚度。粒子弦长按相交区间分配到各层，分别滤波厚度 `H` 和组分厚度 `Hc`。每层由 `Hc/H` 线性混合 `σa`、`σs`，逐通道透射率为 `T=exp[-(σa+σs)H]`。每层散射项使用 `σs/(σa+σs) × (1-T) × J`，其中 `J=2.5×domeColor+0.5×keyColor` 为 studio 光照的常量源近似。按从前到后的顺序累积透射和散射，使前方咖啡能够遮挡后方牛奶。Fresnel、GGX 高光与屏幕空间折射继续参与最终合成。系数单位及消光定义参考 PBRT。[Volume Scattering Processes](https://pbr-book.org/4ed/Volume_Scattering/Volume_Scattering_Processes)

| 组分 | RGB 吸收系数 σa，m⁻¹ | RGB 散射系数 σs，m⁻¹ |
| --- | --- | --- |
| 咖啡 | 18、45、100 | 0.1、0.1、0.1 |
| 牛奶 | 0.15、0.25、0.5 | 80、80、80 |
| 清水 | 0.02、0.008、0.004 | 0.01、0.01、0.01 |

这些系数是针对当前水槽尺度选择的演示值。它们定义在 `media.py`，`run.json` 保存完整端点系数、单位、缩放与近似模型。沿用的 `dyeConcentration` 表示牛奶占比，粒子 USD 的 `displayColor` 是白背景、单位入射光、3 cm 厚度下的参考色。

此版本采用三通道、分层均匀介质与常量入射光源近似。每层内部的细节、远端尾层的深度次序和多次散射的表现受限。流体动力学采用同密度的被动组分模型。咖啡与牛奶实测参数混合的研究可参考 Narasimhan 等的项目。[Participating Media — Measuring Scattering by Dilution](https://www.cs.cmu.edu/~ILIM/projects/LT/dilution/)

此前 44 项 mixing 测试通过，新增检查覆盖端点系数、GPU 透射与散射、零光学密度、分层厚度守恒、前后材料顺序和 USD 参考色。咖啡／牛奶模式完成 resolution=96、360 帧实际模拟与 RTX 逐帧录制，输出 1280×720、60 fps、361 个视频帧。最终活动粒子 `65088`，组分总量相对误差约 `1.1×10⁻⁶`。清水／牛奶模式另完成 8 帧实际模拟、9 次渲染读回及粒子 USD 导出。

高质量档已完成 360 帧模拟和 361 帧 RTX 渲染，生成 1920×1080、60 fps 的完整视频，最终活动粒子为 `304320`，组分总量相对误差约 `3.4×10⁻⁶`。当前 45 项 mixing 测试通过，覆盖高质量档默认值、显式参数覆盖与 12 层厚度守恒。此次录制中途遇到其他进程显存占用增加，完整版本采用先缓存模拟、再独立渲染的流程。结果与缓存位于 `output/coffee_milk_high`。

高质量版本的薄片与飞溅仍有粒子重建痕迹。末帧粒子中心最高 z 约为 0.1431 m，理论静止液位为 0.1783 m，当前 APIC 的体积保持仍有误差。

## USD 与诊断

表面 USD 包含动态拓扑、法线、顶点 `dyeConcentration`、`st` 和兼容回退的 `displayColor`。实际绑定材质为标准 `UsdPrimvarReader_float2 → UsdUVTexture → UsdPreviewSurface`；请保留材质，单用回退 displayColor 会重新发生 RGB 插值。

USD 和同目录的 `<USD文件名>_palette.png` 使用相对路径绑定，**移动时要一起保留**。颜色表标记为 raw 线性数值，GL 纹理解码做对应补偿；8-bit 查表有少量量化误差，最终画面还受查看器灯光和显示变换影响。当前表面材质不透明。Tank 的初始采样为空网格，后续水体出现；粒子导出保留稳定 ID 与 emitterId。

`metrics.csv` 记录活动粒子数、累计注入体积、染料积分、相对注入量误差、浓度极值、混合指数和退化拟合计数。混合指数采用 `1 - σ/0.5`；持续注入时新粒子也会改变统计，它不等同于固定粒子集的混合率。

另外记录速度均方根、总动能、绕水槽中心竖直轴的单位质量角动量，及近底部池水的同类统计，用于观察耗散和回旋方向。它们是运动诊断，不代表形成了稳定涡核或满足能量守恒。

可选绘图脚本需要 matplotlib，tank 采用中间 y 截面的 x/z 粒子图：

```bash
python -m pip install matplotlib
python examples/mixing/apic.py --viewer none --diffusivity 0 --output-dir examples/mixing/output/reference
python examples/mixing/apic.py --viewer none --output-dir examples/mixing/output/tank
python examples/mixing/plot.py examples/mixing/output/reference examples/mixing/output/tank
```

## 验证与范围

```bash
python -m unittest unit_test.unit_test_apic_mixing unit_test.unit_test_mixing_surface unit_test.unit_test_mixing_tank -v
python -m unittest unit_test.unit_test_mixing_rtx unit_test.unit_test_mixing_ssfr -v
CUDA_VISIBLE_DEVICES="" python -m unittest unit_test.unit_test_mixing_tank -v
python -m unit_test.unit_test_apic
```

实测环境：Python 3.12.13、Newton 1.7.0、Warp 1.19.0、NumPy 2.5.2、usd-core 26.3、Pillow 12.3.0，GPU 为 RTX 4090。根目录 requirements.txt 为历史版本快照，本次新增功能没有在那个组合上重测。

此前 resolution=32 的 tank 已跑完 240 帧，导出 41 个表面采样（0、6、…、240）：

| 指标 | 结果 |
| --- | --- |
| 注入总粒子数 | 2496 |
| 已注入静止体积 | 0.131625 m³ |
| 理论沉降液位 | 0.1828125 m |
| 最终浓度范围 | 约 `[0.00969,0.99124]` |
| 最终染料总量相对误差 | 约 `1.1 × 10⁻⁷` |
| 最终混合指数 | 约 0.195 |

此前正对喷口（yaw=0）的 resolution=64、每帧 16 子步也已完成 240 帧粒子运行：最终 18816 粒，半径 `0.009375 m`，浓度约 `[0.0352,0.9621]`，染料总量相对误差约 `3.4 × 10⁻⁷`，混合指数约 `0.274`。不同分辨率的圆盘离散采样会带来少量注入体积差异；resolution=64 的累计体积为 `0.12403125 m³`。

此前前后偏转 20°、D=0.001 的配置完成了 240 帧运行，仍为 18816 粒；最终混合指数约 `0.658`，池水单位质量竖直角动量约 `0.0908 m²/s`，染料总量相对误差约 `3.2 × 10⁻⁷`。

此前采用整层发射及旧稳定化参数，保持同样的粒子运动与注入计划，扩散和时长对照为：

| 扩散系数 D | 总时长 | 停喷后时长 | 最终混合指数 |
| --- | --- | --- | --- |
| 0.001 m²/s | 4 s | 1 s | 0.658 |
| 0.001 m²/s | 6 s | 3 s | 0.816 |
| 0.003 m²/s | 4 s | 1 s | 0.784 |
| 0.003 m²/s | 6 s | 3 s | 0.886 |

上述 6 秒的两组运行最终粒子位置逐元素相同。D=0.003 的最终浓度约 `[0.3514,0.6983]`，染料总量相对误差约 `8.6 × 10⁻⁷`。

引入周期换色前，连续发射与较弱稳定化的配置已跑完 resolution=64、360 帧粒子 USD：最终仍为 18816 粒，浓度约 `[0.3745,0.6653]`，混合指数约 `0.8865`，染料总量相对误差约 `8.0 × 10⁻⁷`。另一个 resolution=48、关闭颜色扩散、注入 1 秒后继续流动 1 秒的运动对照中，新版最终动能约 `5.89 J`，旧版约 `5.10 J`；它只说明该场景中运动保留更多，不等同于流体准确性验证。所有这些结果都是本例的诊断，没有与实际流体实验对照。

周期换色已完成 resolution=48、注入 4.2 秒、总计 270 帧的粒子 USD 检查，覆盖 2 秒和 4 秒两次切换。最终 11692 粒，浓度约 `[0.0685,0.9258]`，染料总量相对误差约 `3.8 × 10⁻⁷`。测试还检查了已有浓度不被换色重置，以及稀疏 USD 采样下喷口切换时刻的准确性。

检查涵盖粒子出生与双缓冲初始化、注入停止、来源及染料总量核算、自由表面退化保护、对称通量守恒和有界性、原始 LSMPS 解析解、KM/GL/USD 一致性、动态网格及相对纹理资产、USD 规范。已检查空槽、喷流和沉降阶段的 OpenGL 画面。当前环境没有 USD Hydra 显示模块，尚未在 Houdini 内实际渲染核验。

这是仍在调整中的 APIC 示例；细水柱、飞溅和表面仍受粒子分辨率影响。当前不支持不同密度流体、内部障碍物或真实颜料光谱模型。
