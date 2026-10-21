# 阅读清单

## 优先资料

1. [SO-101 官方文档](https://huggingface.co/docs/lerobot/so101)：理解连接、设备 ID、标定与主从臂关系。
2. [LeRobot 实机模仿学习](https://huggingface.co/docs/lerobot/il_robots)：理解采集、数据、训练、部署；实际命令以固定版本源码为准。
3. [ACT 论文](https://arxiv.org/abs/2304.13705)：重点阅读输入输出、动作序列、变分目标及误差累积问题。
4. [SmolVLA 论文](https://arxiv.org/abs/2506.01844)：重点阅读视觉语言条件、动作专家、流匹配与异步执行。
5. [SmolVLA 官方教程](https://huggingface.co/docs/lerobot/smolvla)：对照微调、相机输入与实机推理。
6. [官方异步推理](https://huggingface.co/docs/lerobot/async)：理解动作段、队列与观察更新。

## 代码阅读顺序

在固定版本 `lerobot/` 中依次检查：

- `src/lerobot/configs/dataset.py` 与 `datasets/`：一条轨迹保存了什么。
- `src/lerobot/policies/act/`：观察如何变成动作序列，训练目标是什么。
- `src/lerobot/policies/smolvla/`：图像与文字如何影响动作。
- `src/lerobot/async_inference/`：模型执行位置、时间戳、队列和动作合并。

每次只追踪一个具体样本，记录张量形状、单位、归一化和训练参数。资料中的新版本命令可能变化，遇到差异先核对固定源码。
