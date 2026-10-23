# 无设备软件验证

此流程使用合成图像、关节状态和动作，验证模型资源、推理、一次参数更新及本机 gRPC 通信。全程不创建机器人客户端，不打开串口或摄像头。

## 准备模型资源

在已安装 SmolVLA 与异步推理依赖的环境中，从仓库根目录执行：

```text
python scripts/autodl/prepare_smolvla.py --output-dir /root/autodl-tmp/models --endpoint huggingface --workers 4 --prepare-local
```

AutoDL 上官方地址访问受限时，将端点显式改为 `--endpoint hf-mirror`。下载失败后重复同一命令，可复用已经校验的分片。工具按 [固定资源清单](../configs/models/smolvla-base-manifest.json) 检查大小与官方 Git blob 或 LFS 哈希，完整校验通过后才生成目标文件。

输出包含三个目录：

| 目录 | 内容 |
| --- | --- |
| `smolvla_base/` | 固定版本的完整基础检查点，保持官方原件 |
| `smolvlm/` | 固定版本的视觉语言骨干配置与分词器资源 |
| `smolvla_local/` | 当前机器的离线加载配置与完整基础权重 |

`smolvla_local` 将骨干与分词器指向本机绝对路径，并设置 `load_vlm_weights=false`：从架构创建模型后，用完整 SmolVLA 检查点覆盖参数，省去再次下载骨干权重。参数精度由骨干配置与库版本决定，检查报告记录实际加载精度；同组实验保持加载路径和依赖版本一致。来源与修改记录在 `sources.json`。跨机器移动后，在目标机器重新运行 `--prepare-local`，重建本机路径。

## 运行检查

云端执行：

```text
python scripts/autodl/check_smolvla.py --model-path /root/autodl-tmp/models/smolvla_local --output-dir outputs/setup/smolvla-software-check --device cuda
```

本地也可用 CPU 检查。先按 [环境配置](setup.md) 安装本地依赖，再补装 `python -m pip install -c requirements/runtime-constraints.txt -e "./lerobot[smolvla]"`；下载命令的输出目录改为本地路径，检查时使用 `--device cpu`。

输出目录必须是新目录。检查依次执行：

1. 严格加载完整权重，记录参数量、精度与加载耗时。
2. 用两个 `640×480` 视角及六维关节状态生成 `50×6` 动作段，检查形状与有限值。
3. 用一个合成批次反向传播，检查梯度与参数更新。
4. 保存模型及前后处理器，再由官方策略服务重新加载。
5. 通过仅监听本机的 gRPC 服务发送原始图像、状态和指令，检查返回的动作数量、时间步和数值。

成功后生成 `report.json`。`synthetic_checkpoint/` 只供软件诊断，不用于实机；它不包含任务学习成果。软件检查不能替代真实示范训练、网络延迟测量和机械臂评估。

## 异步客户端兼容入口

项目客户端入口沿用官方实现，并在动作段聚合和动作执行之间加锁，防止队列遍历时被修改，或将刚执行的动作重新放入队列。该兼容层位于 `src/so101_project/async_client.py`，官方 LeRobot 子模块保持不变。

离线测试直接读取固定版本的官方方法，验证动作加权合并、时间步过滤及并发互斥。运行全部轻量测试：

```text
python -m unittest discover -s tests -v
```
