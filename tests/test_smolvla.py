import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_project.checkpoint import inspect_checkpoint
from so101_project.config import load
from so101_project.launch import command, parameters
from so101_project.tunnel import tunnel_command


class SmolVLAConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load([ROOT / "configs/hardware/smolvla.local.example.json",
                            ROOT / "configs/deployment/smolvla.local.example.json",
                            ROOT / "experiments/smolvla_language/experiment.example.json"])

    def test_four_tasks_share_recording_and_execution_instructions_without_mutation(self):
        for task, text in self.config["tasks"].items():
            with self.subTest(task=task):
                record = parameters("record", self.config, task)
                client = parameters("client", self.config, task)
                self.assertEqual(record["dataset"]["single_task"], text)
                self.assertEqual(client["task"], text)
                self.assertNotIn("tasks", record)
                self.assertNotIn("teleop", client)
        self.assertNotIn("single_task", self.config["recording"])
        self.assertNotIn("task", self.config["remote"])

    def test_task_selection_is_explicit_and_unknown_task_rejected(self):
        for action in ("record", "client"):
            for task in (None, "wrong"):
                with self.subTest(action=action, task=task), self.assertRaisesRegex(ValueError, "--task"):
                    parameters(action, self.config, task)
        with self.assertRaises(ValueError):
            parameters("teleoperate", self.config, "red_left")

    def test_resume_only_appends_recording_and_not_config(self):
        params = parameters("record", self.config, "blue_right", resume_recording=True)
        self.assertIn("--resume=true", command("record", params))
        self.assertFalse(self.config["record_options"]["resume"])
        with self.assertRaises(ValueError):
            parameters("client", self.config, "blue_right", resume_recording=True)

    def test_finetune_uses_pretrained_path_and_clears_input_features(self):
        config = load([ROOT / "configs/policies/smolvla.json",
                       ROOT / "experiments/smolvla_language/experiment.example.json",
                       ROOT / "configs/deployment/smolvla.autodl.example.json",
                       ROOT / "experiments/smolvla_language/pilot.json"])
        params = parameters("train", config)
        args = command("train", params)
        self.assertIn("--policy.path=lerobot/smolvla_base", args)
        self.assertIn("--policy.input_features=null", args)
        self.assertFalse(any(arg.startswith("--policy.type=") for arg in args))
        self.assertEqual(params["steps"], 1000)
        self.assertEqual(params["dataset"]["root"], "/autodl-tmp/EDIT_DATASET_ROOT")
        self.assertEqual(params["dataset"]["eval_split"], 0.1)
        config["training"]["policy"]["type"] = "act"
        with self.assertRaisesRegex(ValueError, "不能同时"):
            parameters("train", config)

    def test_multitask_preview_does_not_connect_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/local/run.py"), "record",
                                     "--config", str(ROOT / "configs/hardware/smolvla.local.example.json"),
                                     "--config", str(ROOT / "experiments/smolvla_language/experiment.example.json"),
                                     "--task", "red_right", "--resume-recording", "--runs-dir", folder],
                                    capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("right bin", result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])


class TunnelTests(unittest.TestCase):
    def test_loopback_bind_and_placeholder_block_before_ssh(self):
        config = load([ROOT / "configs/deployment/ssh.example.json"])
        args = tunnel_command(config)
        self.assertIn("127.0.0.1:8080:127.0.0.1:8080", args)
        self.assertIn("ExitOnForwardFailure=yes", args)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/local/tunnel.py"), "--config",
                                 str(ROOT / "configs/deployment/ssh.example.json"), "--execute"],
                                capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        self.assertIn("请先填写", result.stderr)

    def test_invalid_port_or_option_host_rejected(self):
        config = load([ROOT / "configs/deployment/ssh.example.json"])
        config["ssh"]["port"] = 70000
        with self.assertRaises(ValueError):
            tunnel_command(config)
        config["ssh"]["port"] = 22
        config["ssh"]["host"] = "-option"
        with self.assertRaises(ValueError):
            tunnel_command(config)


class CheckpointTests(unittest.TestCase):
    def test_exported_camera_contract_and_missing_processor_state(self):
        hardware = load([ROOT / "configs/hardware/smolvla.local.example.json"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            config = {"type": "smolvla", "chunk_size": 50,
                      "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                                         **{f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 480, 640]}
                                            for name in ("camera1", "camera2")}},
                      "output_features": {"action": {"type": "ACTION", "shape": [6]}}}
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            # Test fixtures only: these bytes are never presented as trained weights.
            (path / "model.safetensors").write_bytes(b"fixture")
            processor = {"steps": [{"registry_name": "normalizer_processor", "state_file": "stats.safetensors"}]}
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                (path / name).write_text(json.dumps(processor), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "状态文件"):
                inspect_checkpoint(path, hardware)
            (path / "stats.safetensors").write_bytes(b"fixture")
            self.assertEqual(inspect_checkpoint(path, hardware)["state_shape"], [6])
            config["input_features"]["observation.images.camera3"] = {"type": "VISUAL", "shape": [3, 480, 640]}
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "相机"):
                inspect_checkpoint(path, hardware)


if __name__ == "__main__":
    unittest.main()
