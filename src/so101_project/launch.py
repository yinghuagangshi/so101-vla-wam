"""Build upstream commands, preview them, and launch with configuration snapshots."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import uuid

from . import LEROBOT_COMMIT, LEROBOT_VERSION
from .config import flatten, has_placeholder, load
from .console import configure_console

MODULES = {
    "calibrate-follower": "lerobot.scripts.lerobot_calibrate",
    "calibrate-leader": "lerobot.scripts.lerobot_calibrate",
    "teleoperate": "lerobot.scripts.lerobot_teleoperate",
    "record": "lerobot.scripts.lerobot_record",
    "replay": "lerobot.scripts.lerobot_replay",
    "train": "lerobot.scripts.lerobot_train",
    "server": "lerobot.async_inference.policy_server",
    "client": "lerobot.async_inference.robot_client",
}
LOCAL_ACTIONS = set(MODULES) - {"train", "server"}
CLOUD_ACTIONS = {"train", "server"}


def required(config: dict, name: str) -> dict:
    value = config.get(name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"缺少配置区段：{name}")
    return value


def parameters(action: str, config: dict) -> dict:
    if action == "train":
        result = dict(required(config, "training"))
        dataset = required(result, "dataset")
        if not dataset.get("repo_id"):
            raise ValueError("training.dataset.repo_id 不能为空")
        if not result.get("policy", {}).get("type") and not result.get("policy", {}).get("path"):
            raise ValueError("training.policy 需要 type 或 path")
        if result.get("resume"):
            raise ValueError("此入口用于新训练；断点续训请按官方文档使用检查点配置")
        return result
    if action == "server":
        result = dict(required(config, "server"))
        if not 1 <= result.get("port", 0) <= 65535:
            raise ValueError("server.port 必须在 1—65535 之间")
        return result

    section = "teleop" if action == "calibrate-leader" else "robot"
    hardware = dict(required(config, section))
    if not hardware.get("port") or not hardware.get("id"):
        raise ValueError(f"{section} 需要 port 和 id")
    if action.startswith("calibrate-"):
        hardware.pop("cameras", None)
        return {section: hardware}

    cameras = hardware.get("cameras", {})
    if action != "replay":
        if not cameras:
            raise ValueError("robot.cameras 至少需要一个摄像头")
        for name, camera in cameras.items():
            for key in ("width", "height", "fps"):
                if camera.get(key, 0) <= 0:
                    raise ValueError(f"摄像头 {name} 的 {key} 必须为正数")
        camera_paths = [cam.get("index_or_path") for cam in cameras.values()]
        if len(set(camera_paths)) != len(camera_paths):
            raise ValueError("多个摄像头不能使用相同的 index_or_path")
    else:
        hardware.pop("cameras", None)

    result = {"robot": hardware}
    if action in {"teleoperate", "record"}:
        result["teleop"] = required(config, "teleop")
        if not result["teleop"].get("port") or not result["teleop"].get("id"):
            raise ValueError("teleop 需要 port 和 id")
        if result["teleop"]["port"] == hardware["port"]:
            raise ValueError("主臂与从臂不能使用相同端口")
    if action == "teleoperate":
        result.update(config.get("teleoperation", {}))
    elif action == "record":
        result["dataset"] = required(config, "recording")
        if not result["dataset"].get("repo_id") or not result["dataset"].get("single_task"):
            raise ValueError("recording 需要 repo_id 和 single_task")
        for key in ("num_episodes", "episode_time_s", "fps"):
            if result["dataset"].get(key, 0) <= 0:
                raise ValueError(f"recording.{key} 必须为正数")
        result.update(config.get("record_options", {}))
    elif action == "replay":
        result["dataset"] = required(config, "replay")
        if not result["dataset"].get("repo_id") or result["dataset"].get("episode", -1) < 0:
            raise ValueError("replay 需要 repo_id 和非负 episode")
    elif action == "client":
        remote = required(config, "remote")
        for key in ("server_address", "policy_type", "pretrained_name_or_path", "task"):
            if not remote.get(key):
                raise ValueError(f"remote.{key} 不能为空")
        if remote.get("fps", 0) <= 0 or remote.get("actions_per_chunk", 0) <= 0:
            raise ValueError("remote.fps 和 actions_per_chunk 必须为正数")
        if not 0 <= remote.get("chunk_size_threshold", -1) <= 1:
            raise ValueError("remote.chunk_size_threshold 必须在 0—1 之间")
        result.update(remote)
    return result


def command(action: str, params: dict) -> list[str]:
    args = [sys.executable, "-m", MODULES[action]]
    for key, value in flatten(params).items():
        if value is None:
            continue
        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        args.append(f"--{key}={encoded}")
    return args


def main(allowed_actions: set[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="预览或运行 SO-101 官方 LeRobot 命令")
    parser.add_argument("action", choices=sorted(allowed_actions or MODULES))
    parser.add_argument("--config", action="append", required=True, help="配置层，可重复；后者覆盖前者")
    parser.add_argument("--execute", action="store_true", help="实际启动；默认仅显示参数")
    parser.add_argument("--runs-dir", default="outputs/runs")
    args = parser.parse_args()
    try:
        config = load(args.config)
        params = parameters(args.action, config)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_dir = Path(args.runs_dir).resolve() / f"{args.action}_{run_id}"
        if args.action == "train":
            params = dict(params)
            params.setdefault("output_dir", str(run_dir / "model"))
            if Path(params["output_dir"]).exists():
                raise ValueError("训练 output_dir 已存在；请使用新目录以保留旧结果")
        cmd = command(args.action, params)
        print(json.dumps({"action": args.action, "arguments": cmd}, ensure_ascii=False, indent=2))
        if not args.execute:
            print("预览完成，未连接设备或启动训练。填写 EDIT_ 参数后添加 --execute。")
            return 0
        if has_placeholder(params):
            raise ValueError("配置仍含 EDIT_ 占位符，请先填写设备、用户、模型或服务地址")
        if sys.version_info < (3, 12):
            raise ValueError("LeRobot v0.6.1 实际运行需要 Python 3.12 或更高版本")
        try:
            installed = version("lerobot")
        except PackageNotFoundError as exc:
            raise ValueError("未安装 LeRobot，请按 docs/setup.md 配置运行环境") from exc
        if installed != LEROBOT_VERSION:
            raise ValueError(f"需要 LeRobot {LEROBOT_VERSION}，当前为 {installed}")
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = {
            "action": args.action, "command": cmd, "parameters": params,
            "config_files": [str(Path(path).resolve()) for path in args.config],
            "lerobot_version": installed, "lerobot_reference_commit": LEROBOT_COMMIT,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep a real terminal: recording and teleoperation use keyboard controls.
        status = subprocess.run(cmd, check=False).returncode
        snapshot["exit_code"] = status
        snapshot["finished_at"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "run.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"配置或运行错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
