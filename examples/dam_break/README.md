# Dam break（溃坝）

`dam_break` 拼写正确。四种求解器使用同一份 APIC 左侧水柱场景；统一入口负责时间推进、查看、表面重建和 USD 动画。

**APIC 和 DFSPH 仍在调整和验证中，尚不是最终版本。**

## 公共规格

配置来源为 [scene.py](scene.py)，归档的 [APIC 示例](../../backup/examples/newton_apic_test.py) 也引用这份粒子生成函数。

| 项目 | 默认值 |
| --- | --- |
| 单位 / 向上轴 | 米、千克、秒 / Z |
| 水槽边界 | `(-1.2, -0.28, 0)` 至 `(1.2, 0.28, 1.2)` |
| 水柱边界 | `(-1.17, -0.25, 0.03)` 至 `(-0.468, 0.25, 0.8508)` |
| 目标 / 实际粒子数 | `400000` / **401856**（`78 × 56 × 92`） |
| 粒子间距参数 / 半径 | 约 `0.00901978 m` / `0.00450989 m` |
| 粒子体积 / 质量 | `spacing³` / `1000 × spacing³`，约 `0.000733818 kg` |
| 静止密度 / 初速度 | `1000 kg/m³` / 零 |
| 重力 | `(0, 0, -10) m/s²` |
| 模拟帧率 / 默认帧数 | `60 fps` / `300`（5 秒） |
| 基础子步 | `4`；WCSPH/DFSPH 可按 CFL 细分，完整推进每帧 |
| 相机 | 原 APIC 侧视相机 |
| 表面 | 原 APIC 各向异性参数，体素 `0.007 m`，阈值 `0.441` |
| 理论静水高度 | 约 `0.219411 m`（默认 401,856 粒子） |

保持 APIC 原有 `linspace` 端点采样及整数分辨率取整；三轴实际间距略有差异，表中的 `spacing` 是最小轴间距。相同 `--particles` 在所有求解器中生成逐元素相同的初始位置、半径和质量。

## 运行

已验证的环境：**Python 3.12.13 / 3.13.5、Newton 1.6.0、Warp 1.17.0**，其余依赖见根目录 `requirements.txt`。用户当前 Conda 环境名称虽为 `env_newton_1_5`，实际安装的是 Newton 1.6.0。表面重建需要 CUDA；交互查看需要可用的 OpenGL 环境。从仓库根目录运行：

```bash
python examples/dam_break/apic.py
python examples/dam_break/wcsph.py
python examples/dam_break/dfsph.py
python examples/dam_break/pbf.py
```

也可以使用统一入口，例如 `python -m examples.dam_break.run --solver dfsph`。每个入口支持相同的 `--particles`、`--frames`、`--substeps`、`--no-viewer`、`--no-surface`、`--show-particles` 和 USD 选项。默认显示水面；`--show-particles` 同时显示粒子。CPU 运行需要 `--no-surface`。

```bash
# 小规模无窗口求解
python examples/dam_break/apic.py --particles 10000 --frames 60 --no-viewer --no-surface

# 导出表面动画；换成 wcsph.py、dfsph.py、pbf.py 即可使用同样规格
python examples/dam_break/apic.py --frames 300 --no-viewer --usd-every 2 \
  --usd examples/dam_break/output/apic/surface.usdc

# 仅导出粒子动画
python examples/dam_break/wcsph.py --frames 300 --no-viewer --no-surface \
  --usd examples/dam_break/output/wcsph/particles.usdc
```

USD 包含初态、每个指定采样帧以及终态，`timeCodesPerSecond=60`；300 个模拟帧对应时间码 `0–300`，即 5 秒。水面路径为 `/Fluid/Surface`，包含动态拓扑、顶点法线、包围盒及水材质；`--show-particles` 会额外写入 `/Fluid/Particles`，关闭 surface 则只输出粒子。`/Fluid/TankGuide` 是显示水槽范围的辅助线。

`--surface-voxel-size`、`--no-anisotropic`、`--surface-max-grid-cells` 对所有求解器生效；仅 APIC 使用 `--voxel-size` 和 `--no-projection`。默认表面容量根据整个水槽估计，至少 800 万网格单元。

输出的 `metrics.csv` 与 `run.json` 默认位于 `output/<solver>/`，记录实际粒子数量、初始位置 SHA-256、质量、环境、求解和表面参数、实际时间及子步数。使用 `--output-dir` 指定记录目录；`output/` 已被 Git 忽略。

## 水槽刻度

四个示例共用 [guides.py](guides.py) 中的刻度，同时用于粒子显示、表面显示和 USD 输出：

- **橙色实线 100%**：静止体积对应的理论平水高度，`H = Σmass / rest_density / tank_floor_area`。
- **90%、80% 虚线**：剩余体积为静止体积的 90%、80% 时的水位，分别对应体积减少 10%、20%。它们不是密度增加 10%、20% 的刻度。
- **灰色短刻度**：水槽前侧立柱上每 `0.1 m` 一条。界面面板显示理论水位及刻度含义。
- 默认三条水位线分别约 `0.219411`、`0.197470`、`0.175529 m`。更改粒子目标数量时按实际总质量重新计算。

水面趋于平静、铺满水槽后，可用刻度观察体积保持程度；溃坝和波浪阶段的局部水位会随运动改变，不能直接当作压缩率。USD 的 `/Fluid/HeightGuides` 包含静态刻度、百分比线标签及理论高度元数据，purpose 为 `guide`。

## 性能与诊断

默认通过 CUDA/OpenGL 互操作更新粒子变换，固定颜色只上传一次；`--no-cuda-interop` 可改用 CPU 上传。完整统计默认每 10 帧采样一次，可用 `--metrics-every 1` 恢复逐帧统计；**每帧仍在设备上检查位置、速度、密度的有限值，只回读一个整数**。初态、结束状态和控制台报告帧均保留完整统计。

PBF 的零人工压力、零核吸引系数和零相对法向速度分支跳过结果为零的核函数计算。粒子数、4 个子步、每步 4 次约束迭代及求解参数保持一致。

2026-09-29，在用户的 Python 3.12.13 / Newton 1.6.0 / Warp 1.17.0 环境中，401,856 粒子、1920×1080 无窗口 OpenGL 粒子显示的同步计时（25 帧，前 5 帧预热）：

| 阶段 | 优化前 | 优化后 |
| --- | --- | --- |
| 求解 | 68.31 ms | 64.65 ms |
| 诊断 | 6.84 ms | 1.67 ms |
| 上传 | 8.73 ms | 1.84 ms |
| 绘制 | 11.94 ms | 12.22 ms |
| 总耗时 / 对应帧率 | 95.83 ms / 10.44 FPS | 80.38 ms / 12.44 FPS |

两组均显示水槽刻度、使用相同物理配置；各自在新进程中测量，并确认画面中存在粒子。此机器同时运行其他 GPU 工作，且计时包含同步屏障，不能保证交互窗口获得相同帧率。GPU 上的 PBF 邻域计算仍占主要耗时。25 帧对照中，位置、速度、密度与原求解器逐元素一致。[性能记录](validation/performance_summary.json)、[刻度预览](validation/height_guides.png)。

可在本机重新测量求解、诊断、上传、绘制四个阶段（不运行表面重建）：

```bash
python examples/dam_break/benchmark.py --solver pbf --particles 400000 --frames 30
```

`--baseline` 测量原 CPU 上传与逐帧统计路径；每次基准应在独立进程中运行。若需要对照旧求解器，可使用 `--reference-pbf /path/to/old_solver_pbf.py`。

## 求解参数

公共场景规格用于比较相同输入。各算法的压力、边界离散、黏性和稳定控制仍由各自实现决定。

- **APIC**：保留原示例的 FEM 网格与压力投影配置；网格体素默认 `max(1.5 × spacing, 0.010)`。
- **SPH/PBF**：直接使用米制坐标以及 APIC 的 `rho0 × spacing³` 质量。没有旧示例的坐标放大 100 倍和质量乘 0.8；核函数支持半径设为 `2 × spacing`，以配合这套质量和邻域。
- **WCSPH**：示例刚度 `250000 Pa`、指数 `7`，默认使用非负表压 `p=max(0, B*((rho/rho0)^7-1))`，避免邻域不完整时产生负压力吸引。含声速、加速度和显式黏性时间步限制；CFL 最大速度/加速度在设备上归约，每个子步只回读两个标量。
- **DFSPH**：密度最多 5 次、散度最多 12 次迭代，关闭 warm start；米制示例的 `max_kappa=1`、`max_kappa_v=100`（旧放大坐标示例上限除以 `100²`），保留现有收敛判断。限幅仍属于待验证的稳定控制。
- **PBF**：每步 4 次约束迭代，`lambda_epsilon=100`、`xsph_c=0.05`，关闭人工压力，单次位移最多半个粒子半径。
- SPH/PBF 的非压力力使用黏性系数 `1e-6`、核吸引系数 `0`；核吸引系数不是直接以 N/m 输入的物理表面张力。参数记录在 `run.json` 中。

表面重建只读取粒子状态，不改变粒子。统一执行路径使用普通 kernel launch。

### WCSPH 调参

默认保持水的运动黏度 `1e-6 m²/s`、支持半径 `2 × spacing`、核吸引系数 `0` 和箱壁回弹系数 `0`。水柱高度约 `0.8208 m`，按 `U≈sqrt(2gH)≈4.05 m/s`、`c0≈10U` 和 `B=rho0*c0²/7` 得到约 `235000 Pa`，因此用 `250000 Pa` 作为起始值。此估算针对约 1% 量级的可压缩性，不能保证离散与撞击误差始终小于 1%。参见 [Becker & Teschner 2007](https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf)。

```bash
# 默认刚度、物理水黏度，显示粒子
python examples/dam_break/wcsph.py --no-surface

# 增加数值阻尼，观察粒子振荡与撞墙响应
python examples/dam_break/wcsph.py --no-surface --wcsph-viscosity 5e-4

# 降低刚度的预览起始值
python examples/dam_break/wcsph.py --no-surface --wcsph-stiffness 150000

# 调整核支持半径（相同质量与初始粒子位置）
python examples/dam_break/wcsph.py --no-surface --wcsph-smoothing-length-coff 2.2
```

`--wcsph-viscosity` 可取 `0`；其余上述选项必须为有限正数，仅 WCSPH 可用。`--wcsph-allow-negative-pressure` 用于复现有符号压力的对照。压力截断建议见 [SPH 不可压缩性教程](https://sph-tutorial.physics-simulation.org/slides/02_incompressibility.pdf)。`5e-4` 是额外阻尼，其飞溅和波浪衰减应单独检查。

当前 CFL 使用 `dt≤0.25h/(c0+vmax)`、`dt≤0.25sqrt(h/amax)` 与保守的黏性限制 `dt≤0.025h²/viscosity`，其中 `h` 是完整核支持半径；仍完整推进每个显示帧。按默认规模及 `vmax≈4 m/s` 估计，`B=250000` 约需 170 个子步/帧，加速度约束可能要求更多子步。

WCSPH 统计额外记录 `density_p99`、`density_compression_p99`（`max(P99(rho)/rho0-1, 0)`，比例值）及压力最小/最大值。密度与压力是在末个子步积分前计算的。自由表面和靠墙处的邻域缺失使全体密度均值不宜直接解释为体积变化。

当前箱壁仍采用位置限制和法向速度处理，没有 SPH 边界体积贡献；非负压力与刚度调整不能单独补齐墙面压力支撑。初始采样也保持 APIC 的三轴间距和质量规则。后续边界完善后再验证静水平衡及长期体积保持。

## 检查

```bash
python -m unittest unit_test.unit_test_dam_break -v
```

检查默认 APIC 规格、四种求解器的共同初态、完整帧时间推进、非采样帧的无效状态检测和理论刻度，以及 WCSPH 初始自由落体、压缩排斥与动量守恒、CFL 最大值与异常检测、高黏度速度衰减和 CLI 参数校验。小规模运行通过不代表默认规模的长期稳定性或物理准确性已得到验证。

2026-09-29 的检查结果：

- 十项 CPU 检查通过；默认规格与提取前的 APIC 生成函数在目标 1 万、10 万、40 万粒子时逐元素一致。
- 四种求解器各运行目标 1 万粒子、60 帧，均完整推进 1 秒，记录状态均为有限值。
- 四种方法各导出 4 帧的表面与粒子 USD，采样时间码为 `0, 3, 4`，验证初始粒子一致、拓扑与法线有效、包围盒正确、材质绑定有效。
- APIC 的公共 OpenGL 查看路径通过两帧无窗口渲染检查；CPU 的纯粒子 USD 路径也通过检查。
- 修正粒子查看器的颜色输入后，用户环境（Python 3.12.13、Newton 1.6.0、Warp 1.17.0）中 PBF 的默认 401,856 粒子、`--no-surface` 路径通过两帧无窗口 OpenGL 渲染检查。
- 表面检查使用 `0.02 m` 体素和 75 万网格单元容量。尚未运行默认 40 万粒子、300 帧的四组完整对比。
- WCSPH 参数调整后，默认 401,856 粒子在 `viscosity=1e-6` 与 `5e-4` 两种设置下各运行 60 帧（1 秒），均完整推进、保持有限状态和非负压力；第一帧最大速度均约 `0.166667 m/s`，符合接触地面前的自由落体。
- 两组 WCSPH 的密度 P99 正向误差峰值分别约 `3.41%`、`3.20%`，最终约 `2.54%`、`0.70%`。这不代表所有粒子的最大密度误差均低于上述值，也未验证长期静水体积。[运行记录](validation/wcsph_summary.json)、[对照曲线](validation/wcsph_comparison.png)。
