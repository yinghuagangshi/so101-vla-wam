# SO-101：ACT 验证与语言条件操作

面向 SO-101 主从单臂的实验项目。第一阶段验证示范采集、ACT 训练和远程执行；第二阶段开展 SmolVLA 四任务微调、语言能力诊断与数据增强实验。

## 开始使用

1. 按 [环境配置](docs/setup.md) 安装本地与 AutoDL 环境。
2. 按 [ACT 操作流程](docs/workflow.md) 填写设备配置，完成采集、训练和部署。
3. 按 [ACT 实验方案](experiments/act_validation/README.md) 记录实机测试。
4. 按 [学习计划](learning/README.md) 学习模型与实验方法。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `lerobot/` | Hugging Face 官方 LeRobot 子模块，固定到 v0.6.1 |
| `configs/hardware/` | 主从臂端口、设备 ID、摄像头配置模板 |
| `configs/deployment/` | 本地客户端、AutoDL 服务端配置模板 |
| `configs/policies/` | ACT 训练配置 |
| `scripts/local/` | 本地标定、遥操作、采集、回放、远程客户端入口 |
| `scripts/autodl/` | 云端训练与推理服务入口 |
| `src/so101_project/` | 配置组合、启动记录、数据检查和评估统计工具 |
| `experiments/` | ACT 验证与后续 SmolVLA 实验方案 |
| `learning/` | 分阶段学习计划、资料与验收清单 |
| `docs/` | 环境配置、操作流程、版本来源 |
| `tests/` | 无需连接机器人或安装模型依赖的工具测试 |
| `outputs/` | 本地运行记录、结果和视频，不进入 Git |

所有命令从项目根目录执行。启动器默认只预览参数，添加 `--execute` 才实际运行：

```text
python scripts/autodl/run.py train --config configs/policies/act.json --config experiments/act_validation/experiment.example.json
```

模板中的 `EDIT_` 参数需要替换。预览不会连接硬件、启动训练或生成实机成功率。

## 版本与成果边界

底层机械臂驱动、模型和远程推理由官方 LeRobot 提供，详见 [依赖来源](docs/dependencies.md)。本项目维护设备与实验配置、启动工具、评估记录和后续实验分析。

当前提供 ACT 工作流入口和离线检查工具；设备连接、GPU 训练和实机效果需在对应环境验证。SmolVLA 阶段目前提供实验计划，运行配置在 ACT 验证通过后补充。WAM 尚无独立实验，不设置空的模型目录。

原始数据、标定文件、训练权重和实验记录独立备份。结果报告必须注明数据版本、检查点、试验次数及失败类型。
