# 设备与环境信息

## 本地环境

| 信息 | 配置用途 | 检查方式 |
| --- | --- | --- |
| 系统、Python 与环境名称 | 安装依赖与设备访问 | `python --version`、`conda env list` |
| 主臂与从臂串口 | `teleop.port`、`robot.port` | Windows 设备管理器的端口列表；分别插拔确认 |
| 原设备 ID | 寻找对应标定文件 | 查看旧配置或标定 JSON 文件名 |
| 标定目录与文件 | `calibration_dir` 与 `id` | 文件为对应目录下的 `<id>.json` |
| 全局、腕部相机编号 | `camera1`、`camera2` 的 `index_or_path` | `lerobot-find-cameras opencv`，查看画面并对应实际安装位置 |
| 相机型号、分辨率与帧率 | 保持采集和执行输入一致 | 先验证两个相机同时输出 640×480、30 Hz 的可用画面 |

本地使用独立 Python 3.12 环境。推理模型在云端，本地仅采集和执行，CPU 版 PyTorch 即可。用摄像头探测命令时，画面会暂存到官方工具指定目录；不要将包含个人环境的画面上传到公开仓库。

## 标定文件

LeRobot 默认将标定文件保存在 `~/.cache/huggingface/lerobot/calibration/` 下，也可能由 `HF_HOME`、`HF_LEROBOT_HOME` 或 `HF_LEROBOT_CALIBRATION` 改变位置。主臂与从臂分别位于 `teleoperators/<类型>/` 和 `robots/<类型>/`。GitHub 中没有标定文件，不代表本地已经丢失。

复用前核对设备 ID、舵机配置与单位。确实缺失或与当前装配不兼容时，分别运行标定入口，按提示移动机械臂；不要自动给已配置的舵机重新写 ID。当前配置将主、从臂的 `use_degrees` 均设为 `true`；采集和执行必须保持一致。

主线使用全局与腕部两个相机。第三个相机可用于独立演示录像，或在后续新数据版本中做增加视角的对照。

## AutoDL 环境

| 信息 | 配置用途 |
| --- | --- |
| GPU 型号、显存、驱动 | 确认训练与推理支持；调整批量大小 |
| Python、PyTorch、CUDA 版本 | 核对依赖与 CUDA 可用性 |
| 项目绝对路径 | 运行入口和保存配置 |
| 数据集绝对路径 | `training.dataset.root`，指向实际包含 `meta/` 的目录 |
| 完整微调检查点目录 | `remote.pretrained_name_or_path`，指向 `pretrained_model/` |
| SSH 主机、端口、用户名 | `configs/deployment/ssh.json` |

密码、私钥和 Hub token 通过本机 SSH 或环境认证使用，不写入配置模板或运行命令。SSH、个人目录和设备配置仅保存到已忽略的个人文件。每次训练保留配置、实际依赖清单、数据版本与检查点来源。
