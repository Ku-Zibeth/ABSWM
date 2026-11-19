# ABSWM：ABS 与物理增强世界模型

从四分之一车直线制动出发，比较传统辨识、直接最佳滑移率预测与物理增强动力学模型。
本项目是研究原型；仿真结果不构成实车安全或性能证明。

## ABSWM 2.0

**E4 A 是 ABSWM 2.0 的正式基础模型。** 参考模型、训练种子和权重来源见 [模型登记](configs/abswm_2.json)。以这组权重为起点的递归状态更新和动作条件多步预测见 [递归预测研究与验收](docs/recurrent_study.md)。
物理先验、结构化轮胎 Decoder、在线潜状态修正及跨函数族验收见 [物理轮胎先验与潜状态修正](docs/physics_prior_latent_refinement.md)。

## 快速开始

使用 Python 3.12。在 PowerShell 7 中执行：

```powershell
uv sync --extra dev
uv run pytest
uv run abswm run --config configs/demo.json --output outputs/demo
```

若 `uv` 不在 PATH，可使用 `& "$env:USERPROFILE/.local/bin/uv.exe"`。
默认安装 CPU 版 PyTorch，运行不依赖 CUDA。`uv.lock` 固定实际依赖版本。

完整规模实验：

```powershell
uv run abswm run --config configs/full.json --output outputs/full
```

完整配置为 300/60/60 组路面，三种学习结构、三个随机种子，每种训练 40 轮。
演示配置使用 24/6/6 组路面、10 轮，供验证全流程；不能用它替代完整研究结果。
同一输出目录只能用于同一配置，防止混用数据。

### 复现 ABSWM 2.0

以下命令使用模型登记里的默认输出路径。第二步生成 E4 A 权重，随后用于递归预测和物理先验实验。

```powershell
uv run abswm run --config configs/full.json --output outputs/diagnosis_reproduced
uv run python scripts/decoder_study.py
uv run python scripts/recurrent_study.py
uv run python scripts/physics_prior_study.py
```

## 分步运行

```powershell
uv run abswm data --config configs/demo.json --output outputs/demo
uv run abswm train --config configs/demo.json --output outputs/demo
uv run abswm evaluate --config configs/demo.json --output outputs/demo
```

输出包括：`config.json`、`environment.json`、`data/manifest.json` 和三组 NPZ 轨迹、
`models/*.pt` 权重、训练曲线、开放环与闭环 CSV、诊断图和 `report.md`。
再次运行相同命令会复用数据和完成的模型；评估重新生成。失败命令返回非零退出码。

详细假设、指标定义、因果时序、数值误差和扩展边界见 [方法说明](docs/method.md)。
开放环误差、低滑移辨识、潜在空间和 PI 闭环归因见 [ABSWM 2.0 第一阶段诊断](docs/research_diagnosis.md)。
冻结及联合训练的峰值和完整附着曲线实验见 [潜在状态解码研究](docs/decoder_study.md)。

## 已运行实验

完整配置在 60 组测试路面和两种初速度上完成了 1,440 次闭环仿真。
到 2 m/s 的平均制动距离为：真实峰值参考 72.67 m、在线非线性最小二乘辨识 73.10 m、
直接峰值回归 73.22 m、固定参考 74.52 m、世界模型峰值参考 78.50 m。
在这个路面函数族和控制器设置下，历史窗口改善了学习模型的轮速预测，世界模型的峰值参考仍弱于传统辨识。
这里的闭环仅用世界模型提供滑移率参考，尚未检验基于模型比较未来制动动作能否改善控制。

训练数据、权重、轨迹和图表由本地命令生成，保存在 Git 忽略的 `outputs/` 目录。
运行完整配置后可查看 `outputs/full/report.md`。

默认编码器增加从同一段历史计算的因果速度、滑移率差分，以处理数值尺度差异；对外仍只接收四种观测。
`current` 消融仅使用当前一行，不能读取这些历史差分。完整实验不以胜过传统辨识作为运行成功的条件。
