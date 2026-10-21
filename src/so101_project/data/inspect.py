"""Inspect LeRobot metadata without decoding video or loading model weights."""

import argparse
import json
from pathlib import Path

from ..console import configure_console


def inspect(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    missing = {"action", "observation.state"} - features.keys()
    if missing:
        raise ValueError(f"缺少特征：{sorted(missing)}")
    cameras = {key: value for key, value in features.items() if key.startswith("observation.images.")}
    if not cameras or info.get("total_episodes", 0) <= 0 or info.get("fps", 0) <= 0:
        raise ValueError("需要图像特征、正数 total_episodes 和 fps")
    return {"root": str(root.resolve()), "fps": info["fps"], "episodes": info["total_episodes"],
            "frames": info.get("total_frames"), "cameras": cameras,
            "state": features["observation.state"], "action": features["action"],
            "scope": "仅检查元数据；视频完整性、时间同步和轨迹质量仍需抽样回放"}


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="检查 LeRobot 数据集元数据")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.root), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"数据检查错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
