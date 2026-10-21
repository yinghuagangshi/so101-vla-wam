# 环境配置

## 本地与云端职责

本地负责机械臂连接、摄像头、遥操作、示范采集与动作执行。AutoDL 负责训练和模型推理。两端使用官方 LeRobot v0.6.1 的相同提交。

建议分别创建 Python 3.12 环境。PyTorch 与 CUDA 按实际驱动选择配套版本；安装完成后确认云端 `torch.cuda.is_available()` 为真。

## 安装

先按 [LeRobot 安装文档](https://huggingface.co/docs/lerobot/installation) 配置 Python、PyTorch 和视频编解码依赖，再从项目根目录执行。

本地：

```text
python -m pip install -e "./lerobot[core_scripts,feetech,async]"
python -m pip install -e .
```

AutoDL：

```text
python -m pip install -e "./lerobot[training,async]"
python -m pip install -e .
```

后续 SmolVLA 训练和服务端额外安装同一源码版本的 `smolvla` 依赖。不要将本项目根目录安装误认为已安装 LeRobot：根目录仅包含轻量实验工具。

## 配置

复制以下模板并填写，模板保留在 Git，个人配置被忽略：

| 模板 | 个人文件 | 需要填写 |
| --- | --- | --- |
| `configs/hardware/local.example.json` | `configs/hardware/local.json` | 主从臂端口、ID、相机编号和视角名称 |
| `configs/deployment/local.example.json` | `configs/deployment/local.json` | 服务地址、服务端检查点路径 |
| `configs/deployment/autodl.example.json` | `configs/deployment/autodl.json` | 服务端监听配置 |

Windows 的串口可为 `COM5` 等实际值，Linux 可为 `/dev/ttyACM0` 等实际值。通过 `lerobot-find-port` 和 `lerobot-find-cameras opencv` 查找设备。摄像头模板中的 0、1 仅是示例，必须确认与真实设备对应。当前以 `front`、`wrist` 两个视角为例，实际名称要在采集、训练和推理之间保持一致。

`policy_device=cuda` 指模型所在的云端 GPU；`client_device=cpu` 指本地接收动作的设备。两者不能混淆。

## 网络连接

默认让云端服务监听 `127.0.0.1:8080`，本地通过 AutoDL 控制台给出的 SSH 信息建立隧道：

```text
ssh -N -L 8080:127.0.0.1:8080 -p EDIT_SSH_PORT EDIT_USER@EDIT_SSH_HOST
```

隧道保持运行，本地客户端连接 `127.0.0.1:8080`。服务器路径由云端解释；本地路径不能直接作为云端检查点路径使用。

## 验证与记录

先验证三个摄像头各自能打开，选择两个互补视角进入主实验；再验证主从臂遥操作和已标定设备的回放。标定文件需要独立备份，确实丢失时再运行标定入口。不要为了运行项目重新给已配置好的舵机写 ID。

两端各保存依赖与设备信息到自己的 `outputs/` 目录。启动器检查 LeRobot 包版本；提交号和完整依赖清单仍需在环境记录中保存。
