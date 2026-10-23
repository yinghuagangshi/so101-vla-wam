"""Check exported SmolVLA files and camera compatibility without loading weights."""

import argparse
import json
from pathlib import Path

from .config import load
from .console import configure_console


def inspect_checkpoint(path: Path, hardware: dict) -> dict:
    if (path / "SYNTHETIC_ONLY.txt").exists():
        raise ValueError("这是合成数据软件诊断检查点，不能作为实机部署模型")
    policy = load([path / "config.json"])
    if policy.get("type") != "smolvla":
        raise ValueError("检查点不是 SmolVLA")
    cameras = hardware.get("robot", {}).get("cameras", {})
    if not cameras:
        raise ValueError("硬件配置没有摄像头")
    images = {key: value for key, value in policy.get("input_features", {}).items()
              if value.get("type") == "VISUAL"}
    expected = {f"observation.images.{name}" for name in cameras}
    if set(images) != expected:
        raise ValueError(f"检查点相机 {sorted(images)} 与硬件 {sorted(expected)} 不一致；请使用本项目微调后的检查点")
    for name, cam in cameras.items():
        if images[f"observation.images.{name}"].get("shape") != [3, cam["height"], cam["width"]]:
            raise ValueError(f"相机 {name} 分辨率与训练配置不一致")
    state_shape = policy["input_features"].get("observation.state", {}).get("shape")
    action_shape = policy.get("output_features", {}).get("action", {}).get("shape")
    if state_shape != [6] or action_shape != [6]:
        raise ValueError("SO-101 单臂配置要求 6 维状态与 6 维动作")
    if not (path / "model.safetensors").is_file() or (path / "model.safetensors").stat().st_size == 0:
        raise ValueError("缺少非空 model.safetensors")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = load([path / name])
        if not processor.get("steps"):
            raise ValueError(f"{name} 没有处理步骤")
        for step in processor["steps"]:
            if step.get("registry_name") == "rename_observations_processor" and step.get("config", {}).get("rename_map"):
                raise ValueError("此异步配置使用原生相机名称，检查点不能依赖非空 rename_map")
            if step.get("state_file"):
                state = (path / step["state_file"]).resolve()
                if not state.is_relative_to(path.resolve()) or not state.is_file() or state.stat().st_size == 0:
                    raise ValueError(f"{name} 引用的状态文件缺失或路径无效")
    return {"checkpoint": str(path.resolve()), "policy_type": "smolvla", "cameras": sorted(images),
            "state_shape": state_shape, "action_shape": action_shape, "chunk_size": policy.get("chunk_size")}


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="检查云端 SmolVLA 导出文件与相机配置")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.checkpoint, load([args.hardware]))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("文件与相机配置检查通过；模型加载、数值和实机动作需分别验证。")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"检查点检查失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
