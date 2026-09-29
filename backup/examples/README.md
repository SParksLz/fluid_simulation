# 旧示例备份

原先位于仓库根目录的示例入口及其独立 SPH 核函数已归档到此目录。新版统一溃坝示例见 [examples/dam_break](../../examples/dam_break/README.md)。

| 文件 | 用途 |
| --- | --- |
| `newton_apic_test.py` | 原 APIC 波浪水槽与表面重建查看器 |
| `newton_tank_test.py` | 原 Newton WCSPH 示例 |
| `newton_dfsph_test.py` | 原 Newton DFSPH 示例 |
| `newton_pbf_test.py` | 原 Newton PBF 示例 |
| `tank_test.py` | 原独立 Warp 水槽实验，主入口选择 DFSPH |
| `suction_test.py` | 原独立 Warp 吸管/容器实验 |
| `wcsph_kernel.py` | 独立 Warp 实验及旧 DFSPH 单元测试使用的核函数 |

此目录保留旧入口与参数，导入和数据路径已适配新位置。Newton 示例引用仓库当前的 `solver/`；它们不是冻结所有依赖的历史版本。

## 运行

使用根目录 `requirements.txt` 中的环境，从仓库根目录执行：

```bash
python backup/examples/newton_tank_test.py --device cuda:0 --frames 600 --color-field pressure
python backup/examples/newton_dfsph_test.py --device cuda:0 --frames 600 --color-field kappa_v
python backup/examples/newton_pbf_test.py --device cuda:0 --frames 600 --color-field rho

# 原 APIC 查看器
python backup/examples/newton_apic_test.py --device cuda:0 --particles 100000 --frames 600

# 原 APIC 的无窗口小规模求解
CUDA_VISIBLE_DEVICES="" python backup/examples/newton_apic_test.py \
  --device cpu --particles 1000 --frames 2 --no-viewer --no-surface

# 原 APIC 的 USD 输出
python backup/examples/newton_apic_test.py --device cuda:0 --particles 100000 \
  --viewer usd --frames 120 --output-path apic_wave_tank.usdc
```

四个 Newton 入口支持 `--help`，也可用 `python -m backup.examples.newton_apic_test` 等模块形式运行。APIC 查看器需要 Newton 1.6.0 的 `ParticleSurface`；可选 RTX 查看器需要另行安装 `ovrtx`。

WCSPH、DFSPH、PBF 默认读取仓库根目录下的 `temp/fluid_particles.usd`。自定义 `--usd` 应包含 `/Fluid/Particles` 的 `points` 和 `widths` 属性。旧 WCSPH/DFSPH 将坐标放大 100 倍后求解，再缩回显示；PBF 直接使用输入坐标。它们与新版米制场景的参数不能直接互换。

`tank_test.py` 同样读取根目录的 `temp/fluid_particles.usd`；诊断输出仍写入根目录 `temp/`。`suction_test.py` 需要 `temp/fluid_suction_scene.usd`，该输入不在仓库内。两个独立 Warp 实验没有命令行参数解析，运行入口会直接启动模拟。
