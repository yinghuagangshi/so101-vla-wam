# SmolVLA 采集、微调与远程推理

## 1. 环境与配置

先按 [环境配置](setup.md) 安装依赖，在本地和 AutoDL 分别安装本项目。以下命令均从仓库根目录执行；默认只预览参数，添加 `--execute` 才实际启动。

```text
python scripts/local/init_configs.py
```

按 [设备信息](device-info.md) 填写生成的个人文件。模板配置见 `configs/`；个人文件已被 Git 忽略。云端也需具有实验配置，`training.dataset.root` 指向完整数据集实际所在位置。

硬件相机固定为 `camera1`（全局）和 `camera2`（腕部）。两个相机以相同名称、尺寸和帧率用于采集与推理；不叠加旧 ACT 硬件模板。摄像头编号示例 0、1 不代表实际设备。

## 2. 设备与标定

先找回已有标定文件。缺失时分别标定主臂、从臂，实际操作需人在设备旁按提示完成。配置中的 `id` 必须对应标定目录下 `<id>.json`。

```text
python scripts/local/run.py calibrate-leader --config configs/hardware/smolvla.local.json
python scripts/local/run.py calibrate-follower --config configs/hardware/smolvla.local.json
python scripts/local/run.py teleoperate --config configs/hardware/smolvla.local.json
```

确认全局画面覆盖两个物体与两个盒子，腕部画面能看到夹爪附近。先检查遥操作、动作单位、停止方式，再采集。标定入口不会替代首次舵机 ID 配置。

## 3. 四任务采集

`experiments/smolvla_language/experiment.json` 是采集和推理共用的指令表。每条轨迹只执行指定任务；所有任务进入同一个数据集。

| 任务 ID | 指令含义 |
| --- | --- |
| `red_left` | 红色物体放入左盒 |
| `red_right` | 红色物体放入右盒 |
| `blue_left` | 蓝色物体放入左盒 |
| `blue_right` | 蓝色物体放入右盒 |

首次创建数据集：

```text
python scripts/local/run.py record --config configs/hardware/smolvla.local.json --config experiments/smolvla_language/experiment.json --task red_left
```

后续每次向同一数据集追加，包括再次采集相同任务，使用 `--resume-recording`：

```text
python scripts/local/run.py record --config configs/hardware/smolvla.local.json --config experiments/smolvla_language/experiment.json --task red_right --resume-recording
python scripts/local/run.py record --config configs/hardware/smolvla.local.json --config experiments/smolvla_language/experiment.json --task blue_left --resume-recording
python scripts/local/run.py record --config configs/hardware/smolvla.local.json --config experiments/smolvla_language/experiment.json --task blue_right --resume-recording
```

每次默认增加 10 条轨迹，形成每任务 10 条的 pilot 数据。首次设备调试可先在个人配置中设为 2，之后改为本轮需要追加的条数。`num_episodes` 是本次新增数量，不是最终总数。保持 `no_stamp=true`、数据集标识与目录一致；任务切换时复位物体，不混入上次任务标签。左右以固定全局视角为准，两个物体同时存在并随机交换位置。

```text
python scripts/local/run.py replay --config configs/hardware/smolvla.local.json --config experiments/smolvla_language/experiment.json
python -m so101_project.data.inspect datasets/smolvla_sorting_pilot_v1
```

抽样检查视频、动作和标签。元信息检查不能替代逐帧解码与示范质量检查。正式实验另建数据版本，保持四任务平衡，并将独立开发摆放安排在每任务末尾：固定版本训练代码按任务留出末尾 `ceil(轨迹数×eval_split)` 条轨迹作为开发集。

## 4. AutoDL 微调

将数据集完整复制到云端，保留 `meta/`、`data/` 和 `videos/`。在 `configs/deployment/smolvla.autodl.json` 中填入数据集根目录。先检查元信息和抽样视频解码，再训练。

先运行 1,000 步的小规模流程验证：

```text
python scripts/autodl/run.py train --config configs/policies/smolvla.json --config experiments/smolvla_language/experiment.json --config configs/deployment/smolvla.autodl.json --config experiments/smolvla_language/pilot.json
```

正式基线去掉最后一层 pilot 配置，默认 20,000 步、batch size 8。按实际显存、开发损失与实机结果调整；pilot 的少量数据和短训练用于验证流程。

先按 [环境配置](setup.md) 准备固定版本的基础模型资源。AutoDL 配置将 `policy.path` 覆盖为 `/root/autodl-tmp/models/smolvla_local`，从本机完整检查点加载预训练权重，不同时传 `policy.type`。`policy.input_features=null` 清除预训练相机特征，由当前数据重建，因此两相机数据不会被当作原模型的三相机输入。初始配置冻结视觉编码器，以动作专家和状态投影为主要训练部分，保持四任务共同训练。

启动器保存完整参数与退出状态到 `outputs/runs/train_*/run.json`，模型在该次目录的 `model/` 下。另行保存训练终端日志、开发损失及依赖清单。不设置已有输出目录；断点续训按官方检查点配置流程执行。

## 5. 检查模型与启动服务

在云端确定实际检查点，例如 `outputs/runs/train_.../model/checkpoints/last/pretrained_model`。检查目录应包含权重、`config.json`、前后处理配置及其统计状态文件。

```text
python scripts/autodl/check_checkpoint.py --checkpoint EDIT_FINETUNED_MODEL_PATH --hardware configs/hardware/smolvla.local.example.json
python scripts/autodl/run.py server --config configs/deployment/smolvla.autodl.json
```

检查工具只读取文件与相机结构，不加载模型或连接机械臂。若更改过相机配置，`--hardware` 使用同步到云端的实际版本。云端服务默认监听 `127.0.0.1:8080`，客户端连接后才加载它所指定的模型。

## 6. 本地隧道与执行

在 `configs/deployment/ssh.json` 填写 AutoDL 给出的 SSH 主机、端口、用户名。另开本地终端建立隧道并保持运行：

```text
python scripts/local/tunnel.py --config configs/deployment/ssh.json
```

添加 `--execute` 才运行 SSH；使用已有 SSH 密钥或在终端交互认证。首次连接核实主机指纹，不关闭主机校验。修改服务端或转发端口时，同步更新服务配置、SSH 配置与本地客户端地址。

机械臂尚未连接时，可按 [无设备软件验证](software-check.md) 使用远程探针检查跨机通信与云端 GPU 动作返回。

在 `configs/deployment/smolvla.local.json` 填入云端实际微调检查点绝对路径。使用微调后的完整目录，不把未适配的基础模型直接用于实机执行。

```text
python scripts/local/run.py client --config configs/hardware/smolvla.local.json --config configs/deployment/smolvla.local.json --config experiments/smolvla_language/experiment.json --task red_left
```

逐次改变 `--task` 测试四条指令，每次停止客户端、复位场景后重新启动。固定版本客户端在一次进程中使用一条指令，不提供运行中修改任务或自动复位。

客户端通过项目兼容入口运行官方动作队列，串行化动作段合并与执行，修复固定版本的队列并发竞争；见 [软件验证](software-check.md)。30 Hz、每段 50 个动作、队列阈值 0.5 是初值；需观察实际推理耗时、网络延迟与队列是否断流。先做受控短执行，确认相机映射、动作单位和停止方式，再开展正式测试。当前不自动判定任务成功或录制评估视频。

## 7. 结果记录

```text
python -m so101_project.evaluation.results init outputs/evaluation/smolvla_trials.csv
python -m so101_project.evaluation.results summary outputs/evaluation/smolvla_trials.csv
```

只填写实际执行的试验；`success` 为 0 或 1，人工接管记为失败。保留任务、摆放、模型检查点、耗时、视频和失败原因；统计按模型与任务给出成功次数、总次数、成功率与 Wilson 95% 区间。实验设计见 [语言条件实验](../experiments/smolvla_language/README.md)，可选 ACT 链路排查见 [smoke test 流程](smoke-test.md)。
