# 检查记录快照

此目录保存 2026-09-29 检查中用于文档的摘要与图片。完整逐帧记录和 USD 等生成文件写入本地 `output/`。

| 文件 | 内容 |
| --- | --- |
| `performance_summary.json` | PBF 401,856 粒子、25 帧、前 5 帧预热的同步显示计时与数值对照 |
| `height_guides.png` | 默认粒子场景中的水槽刻度预览 |
| `wcsph_summary.json` | WCSPH 401,856 粒子、60 帧，两种黏度的数值检查 |
| `wcsph_comparison.png` | 两组 WCSPH 的最大速度与密度 P99 正向误差曲线 |

环境为 Python 3.12.13、Newton 1.6.0、Warp 1.17.0、RTX 4090。性能数据来自有其他 GPU 工作的机器；WCSPH 检查覆盖 1 秒模拟时间，没有验证长期静水体积。参数、命令与指标含义见[示例说明](../README.md)。
