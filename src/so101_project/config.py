"""Merge JSON configuration layers without importing the robotics runtime."""

from copy import deepcopy
import json
from pathlib import Path


def merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load(paths: list[str | Path]) -> dict:
    config = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"配置文件必须包含 JSON 对象：{path}")
        config = merge(config, value)
    return config


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        # Cameras are parsed as one dictionary by the upstream config parser.
        if isinstance(item, dict) and name != "robot.cameras":
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def has_placeholder(value) -> bool:
    if isinstance(value, str):
        return "EDIT_" in value
    if isinstance(value, dict):
        return any(has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(has_placeholder(item) for item in value)
    return False
