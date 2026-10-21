# 依赖来源与版本

| 项目 | 固定值 |
| --- | --- |
| 仓库 | https://github.com/huggingface/lerobot |
| 发布版本 | v0.6.1 |
| 提交 | `7e241bd630a3719a56157a497ce5d08f244784f1` |
| 路径 | `lerobot/`，Git 子模块 |
| LeRobot 运行环境 | Python 3.12 或更高版本 |
| 本项目离线工具 | Python 3.11 或更高版本 |

LeRobot 采用其仓库中的 Apache-2.0 许可证。代码原始来源与许可证保留在子模块中。

主项目通过 Git 记录子模块提交。克隆时使用：

```text
git clone --recurse-submodules https://github.com/yinghuagangshi/so101-vla-wam.git
```

主仓库已经存在时使用：

```text
git submodule update --init --recursive
```

升级时单独记录新提交，先通过设备、数据和推理验证，再更新依赖。不同版本的教程、命令与数据格式需分别核对。本地和云端应使用相同的子模块提交；各自保存依赖清单、Python、PyTorch、CUDA 和驱动版本。
