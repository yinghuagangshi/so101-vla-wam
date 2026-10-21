import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_project.config import flatten, has_placeholder, load, merge
from so101_project.data.inspect import inspect
from so101_project.evaluation.results import FIELDS, read_trials, summarize
from so101_project.launch import command, parameters


class ConfigTests(unittest.TestCase):
    def test_overlay_preserves_cameras_without_mutation(self):
        original = {"robot": {"port": "A", "cameras": {"front": {"fps": 30}}}}
        merged = merge(original, {"robot": {"port": "B"}})
        self.assertEqual(original["robot"]["port"], "A")
        self.assertEqual(merged["robot"]["cameras"]["front"]["fps"], 30)

    def test_camera_json_is_one_argument_and_strings_are_not_shell_code(self):
        params = {"robot": {"cameras": {"front": {"index_or_path": 0}}}, "task": "pick red; echo $x"}
        args = command("client", params)
        self.assertIn('--robot.cameras={"front": {"index_or_path": 0}}', args)
        self.assertIn("--task=pick red; echo $x", args)
        self.assertEqual(len(args), 5)

    def test_placeholder_is_recursive(self):
        self.assertTrue(has_placeholder({"a": ["EDIT_PORT"]}))
        self.assertFalse(has_placeholder({"port": "COM5"}))

    def test_all_act_preview_actions_build(self):
        config = load([ROOT / "configs/hardware/local.example.json",
                       ROOT / "configs/deployment/local.example.json",
                       ROOT / "configs/deployment/autodl.example.json",
                       ROOT / "configs/policies/act.json",
                       ROOT / "experiments/act_validation/experiment.example.json"])
        for action in ["calibrate-follower", "calibrate-leader", "teleoperate", "record", "replay", "train", "server", "client"]:
            with self.subTest(action=action):
                self.assertTrue(command(action, parameters(action, config)))
        self.assertNotIn("robot.cameras", flatten(parameters("calibrate-follower", config)))
        self.assertNotIn("teleop", parameters("client", config))
        self.assertNotIn("num_episodes", parameters("train", config)["dataset"])

    def test_conflicting_ports_and_camera_indices_rejected(self):
        config = load([ROOT / "configs/hardware/local.example.json"])
        config["teleop"]["port"] = config["robot"]["port"]
        with self.assertRaises(ValueError):
            parameters("teleoperate", config)
        config["robot"]["cameras"]["wrist"]["index_or_path"] = 0
        with self.assertRaises(ValueError):
            parameters("client", config)

    def test_execute_with_placeholder_never_creates_run_or_connects(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "runs"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/local/run.py"), "teleoperate", "--config",
                                     str(ROOT / "configs/hardware/local.example.json"), "--runs-dir", str(output), "--execute"],
                                    capture_output=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_preview_does_not_need_lerobot_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/autodl/run.py"), "train", "--config",
                                     str(ROOT / "configs/policies/act.json"), "--config",
                                     str(ROOT / "experiments/act_validation/experiment.example.json"),
                                     "--runs-dir", str(Path(folder) / "runs")],
                                    capture_output=True, encoding="utf-8", errors="replace",
                                    env={**os.environ, "PYTHONIOENCODING": "ascii"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(folder) / "runs").exists())


class EvaluationTests(unittest.TestCase):
    def test_grouped_counts_failures_and_interval(self):
        rows = [{"model": "a", "task": "pick", "success": "1", "duration_s": "10", "failure_type": ""},
                {"model": "a", "task": "pick", "success": "0", "duration_s": "20", "failure_type": "grasp"},
                {"model": "b", "task": "pick", "success": "1", "duration_s": "8", "failure_type": ""}]
        summary = summarize(rows)
        self.assertEqual(summary[0]["successes"], 1)
        self.assertEqual(summary[0]["trials"], 2)
        self.assertEqual(summary[0]["failures"], {"grasp": 1})
        self.assertEqual(summary[0]["mean_duration_s"], 15)
        self.assertLess(summary[0]["wilson_95"][0], 0.5)
        self.assertGreater(summary[0]["wilson_95"][1], 0.5)
        self.assertEqual(summary[1]["trials"], 1)

    def test_duplicate_trials_and_incomplete_rows_rejected(self):
        row = dict(zip(FIELDS, ["1", "s1", "pick", "act", "0", "grasp", "12", "video.mp4", ""]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "trial_id"):
                read_trials(path)
            row["trial_id"] = "2"
            row["duration_s"] = "nan"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaises(ValueError):
                read_trials(path)


class DatasetTests(unittest.TestCase):
    def test_metadata_requires_state_action_and_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta").mkdir()
            path = root / "meta/info.json"
            info = {"fps": 30, "total_episodes": 2, "total_frames": 60,
                    "features": {"action": {"shape": [6]}, "observation.state": {"shape": [6]},
                                 "observation.images.front": {"shape": [480, 640, 3]}}}
            path.write_text(json.dumps(info), encoding="utf-8")
            self.assertEqual(inspect(root)["episodes"], 2)
            del info["features"]["action"]
            path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError):
                inspect(root)


if __name__ == "__main__":
    unittest.main()
