# ACT 操作流程

命令从项目根目录执行。所有入口先预览，检查参数后添加 `--execute`。配置按给定顺序合并，后者覆盖前者。设备操作前确认工作区、动作范围和停止方式。

## 1. 标定与遥操作

仅在需要重新标定时分别使用 `calibrate-follower`、`calibrate-leader`：

```text
python scripts/local/run.py calibrate-follower --config configs/hardware/local.json
python scripts/local/run.py calibrate-leader --config configs/hardware/local.json
python scripts/local/run.py teleoperate --config configs/hardware/local.json
```

## 2. 采集与回放

填写 `experiments/act_validation/experiment.example.json` 中的用户标识、数据目录和任务。正式实验建议复制为新的实验 JSON 文件，保留具体版本。初次调试可将录制数量从 50 改为 2。

```text
python scripts/local/run.py record --config configs/hardware/local.json --config experiments/act_validation/experiment.example.json
python scripts/local/run.py replay --config configs/hardware/local.json --config experiments/act_validation/experiment.example.json
python -m so101_project.data.inspect datasets/act_pick_place_v1
```

录制默认不上传 Hub。`no_stamp=true` 保持数据标识不自动加日期，数据根目录与该标识需成对维护。恢复录制要显式设置 `record_options.resume=true`，不要覆盖已有数据。

元数据检查仅确认帧率、轨迹数量和特征字段，不能替代视频解码、时间同步和示范质量检查。采集后抽样回放，并记录失败示范与筛选依据。

## 3. AutoDL 训练

将数据集完整复制到云端，保持 `meta/`、`data/`、`videos/` 等实际目录结构。修改云端实验配置中的 `training.dataset.root` 为云端真实位置。

```text
python scripts/autodl/run.py train --config configs/policies/act.json --config experiments/act_validation/experiment.example.json
```

实际运行时添加 `--execute`。未指定 `training.output_dir` 时生成唯一目录，完整配置和命令保存到 `outputs/runs/train_*/run.json`，模型保存在对应 `model/` 下。官方训练的终端日志和指标需另外保存。`eval_split` 按轨迹留出开发数据，正式实机测试仍使用独立场景。

当前入口用于新训练。续训使用官方检查点配置流程，不把重复的新训练误作续训。

## 4. 云端推理与本地执行

云端：

```text
python scripts/autodl/run.py server --config configs/deployment/autodl.json
```

本地建立 SSH 隧道，填写 `configs/deployment/local.json` 的服务端检查点路径，通常是官方训练输出中的 `checkpoints/.../pretrained_model` 目录。路径内必须包括实际检查点及其预处理配置。

```text
python scripts/local/run.py client --config configs/hardware/local.json --config configs/deployment/local.json
```

模型预期的相机名称、分辨率、动作单位与实际输入应一致。模板的每段 25 步和 30 Hz 是初始配置，应根据模型的 `chunk_size`、网络延迟与动作连续性验证，不能视为已验证参数。

默认客户端使用官方异步推理与动作队列。当前项目启动器保存参数快照和退出状态，不自动判断成功，不提供完整的延迟测量或执行视频录制。首轮用外部录像、官方日志和人工计时建立记录；之后再补充带时间戳的遥测功能。

## 5. 实机结果

```text
python -m so101_project.evaluation.results init outputs/evaluation/act_trials.csv
python -m so101_project.evaluation.results summary outputs/evaluation/act_trials.csv
```

`init` 创建只有表头的文件，已存在时拒绝覆盖。逐次填写已执行试验，`success` 使用 1 或 0。失败试验必须选择：`wrong_object`、`wrong_bin`、`grasp`、`transport`、`release`、`timeout`、`communication`、`hardware` 或 `other`。

每次记录场景、任务、模型版本、完成时间、视频位置。人工接管计为失败；通信与设备异常单列原因，保留在总次数内。统计按模型与任务分组，给出成功次数、试验次数和 Wilson 95% 区间。相邻连续试验存在相关性时，区间仅作描述参考，应跨摆放和时段重复验证。
