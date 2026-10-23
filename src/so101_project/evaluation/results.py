"""Validate and summarize manually scored trials; never infer robot success."""

import argparse
import csv
import json
from pathlib import Path
import math
from statistics import mean

from ..console import configure_console

FIELDS = ["trial_id", "scene_id", "task", "model", "success", "failure_type", "duration_s", "video_path", "notes"]
FAILURES = {"wrong_object", "wrong_bin", "grasp", "transport", "release", "timeout", "communication", "hardware", "other"}


def read_trials(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"CSV 列必须依次为：{','.join(FIELDS)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if not all(row.get(key, "").strip() for key in ("trial_id", "scene_id", "task", "model")):
            raise ValueError("每条试验需要 trial_id、scene_id、task、model")
        if row["trial_id"] in seen:
            raise ValueError(f"trial_id 重复：{row['trial_id']}")
        seen.add(row["trial_id"])
        if row["success"] not in {"0", "1"}:
            raise ValueError("success 必须为 0 或 1；尚未执行的试验不要加入结果表")
        if row["success"] == "0" and row["failure_type"] not in FAILURES:
            raise ValueError("失败试验必须填写有效 failure_type")
        if row["success"] == "1" and row["failure_type"]:
            raise ValueError("成功试验的 failure_type 应为空")
        duration = float(row["duration_s"])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_s 必须是非负有限数")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["task"]), []).append(row)
    output = []
    for (model, task), group in sorted(groups.items()):
        n = len(group)
        wins = sum(int(row["success"]) for row in group)
        p = wins / n
        z = 1.96
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        failures = {}
        for row in group:
            if row["success"] == "0":
                key = row["failure_type"]
                failures[key] = failures.get(key, 0) + 1
        output.append({"model": model, "task": task, "successes": wins, "trials": n,
                       "success_rate": p, "wilson_95": [center - radius, center + radius],
                       "mean_duration_s": mean(float(row["duration_s"]) for row in group),
                       "failures": failures})
    return output


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="建立或汇总实机试验 CSV")
    parser.add_argument("action", choices=["init", "summary"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "init":
            args.path.parent.mkdir(parents=True, exist_ok=True)
            with args.path.open("x", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(FIELDS)
            print(f"已创建空结果表：{args.path}")
        else:
            rows = read_trials(args.path)
            if not rows:
                raise ValueError("结果表没有已完成的试验")
            print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"结果记录错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
