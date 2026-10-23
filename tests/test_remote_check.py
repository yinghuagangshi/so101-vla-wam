"""Check the remote probe contract without a robot, model, network or NumPy."""

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_project.remote_check import (IMAGE_BYTES, MOTOR_NAMES, check_remote,
                                        robot_features, validate_actions)


@dataclass
class Action:
    timestep: int
    values: list
    shape: tuple = (6,)

    def get_timestep(self):
        return self.timestep

    def get_action(self):
        return self

    def tolist(self):
        return self.values


class RpcError(Exception):
    def code(self):
        return SimpleNamespace(name="DEADLINE_EXCEEDED")


class RemoteCheckTests(unittest.TestCase):
    def runtime(self):
        channel, stub = Mock(), Mock()
        observations = []

        def observe(messages, **kwargs):
            observations.append(pickle.loads(b"".join(message.data for message in messages)))

        def get_actions(*args, **kwargs):
            start = observations[-1].timestep
            return SimpleNamespace(data=pickle.dumps([Action(step, [0.0] * 6)
                                                      for step in range(start, start + 50)]))

        stub.SendObservations.side_effect = observe
        stub.GetActions.side_effect = get_actions
        runtime = SimpleNamespace(
            grpc=SimpleNamespace(insecure_channel=Mock(return_value=channel),
                                 channel_ready_future=Mock(return_value=Mock()),
                                 RpcError=RpcError, FutureTimeoutError=TimeoutError),
            np=SimpleNamespace(uint8="uint8", full=Mock(side_effect=lambda shape, value, dtype:
                                SimpleNamespace(shape=shape, value=value, dtype=dtype))),
            RemotePolicyConfig=lambda policy, checkpoint, features, count, device:
                SimpleNamespace(policy=policy, checkpoint=checkpoint, features=features,
                                count=count, device=device),
            TimedObservation=lambda **kwargs: SimpleNamespace(**kwargs),
            messages=SimpleNamespace(Empty=SimpleNamespace, PolicySetup=SimpleNamespace,
                                     Observation=SimpleNamespace),
            stub=Mock(return_value=stub), channel_options=Mock(return_value=[]),
            send_bytes=lambda payload, message_class, **kwargs: iter([message_class(data=payload)]))
        return runtime, channel, stub, observations

    def test_three_rounds_use_raw_robot_schema_force_processing_and_continue_timesteps(self):
        runtime, channel, stub, observations = self.runtime()
        with patch("so101_project.remote_check._runtime", return_value=runtime):
            report = check_remote("127.0.0.1:8080", "/cloud/model", task="blue right")
        self.assertTrue(report["passed"], report)
        runtime.grpc.channel_ready_future.return_value.result.assert_called_once_with(timeout=60)
        specs = pickle.loads(stub.SendPolicyInstructions.call_args.args[0].data)
        self.assertEqual((specs.policy, specs.checkpoint, specs.count, specs.device),
                         ("smolvla", "/cloud/model", 50, "cuda"))
        self.assertEqual(specs.features, robot_features())
        stub.Ready.assert_called_once()
        self.assertEqual(stub.SendPolicyInstructions.call_args.kwargs["timeout"], 300)
        self.assertEqual([item.timestep for item in observations], [0, 50, 100])
        for item in observations:
            self.assertTrue(item.must_go)
            self.assertEqual(set(item.observation), {*MOTOR_NAMES, "camera1", "camera2", "task"})
            self.assertEqual(item.observation["task"], "blue right")
            for camera in ("camera1", "camera2"):
                self.assertEqual(item.observation[camera].shape, (480, 640, 3))
                self.assertEqual(item.observation[camera].dtype, "uint8")
        self.assertEqual(report["image_bytes_per_observation"], 1_843_200)
        self.assertEqual(IMAGE_BYTES, 1_843_200)
        self.assertFalse(report["robot_connected"])
        self.assertEqual([row["first_timestep"] for row in report["rounds"]], [0, 50, 100])
        self.assertTrue(all(row["total_rpc_seconds"] >= row["send_observations_rpc_seconds"]
                            for row in report["rounds"]))
        channel.close.assert_called_once()

    def test_rejects_missing_count_nonfinite_shape_and_timestep_errors(self):
        good = [Action(index, [0.0] * 6) for index in range(50)]
        bad_cases = [[], good[:-1], [Action(0, [float("nan")] * 6)] + good[1:],
                     [Action(0, [0.0] * 5, (5,))] + good[1:],
                     [Action(1, [0.0] * 6)] + good[1:]]
        for actions in bad_cases:
            with self.subTest(actions=len(actions)), self.assertRaises(ValueError):
                validate_actions(actions, 0, 50)

    def test_empty_response_and_rpc_timeout_fail_with_partial_report_and_close_channel(self):
        for failure in (SimpleNamespace(data=b""), RpcError("sensitive server details")):
            runtime, channel, stub, _ = self.runtime()
            stub.GetActions.side_effect = failure if isinstance(failure, Exception) else None
            stub.GetActions.return_value = failure
            with patch("so101_project.remote_check._runtime", return_value=runtime):
                report = check_remote("127.0.0.1:8080", "/cloud/model", timeout=7)
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["rounds"]), 1)
            self.assertEqual(report["error"]["phase"], "round_1_get_actions")
            self.assertNotIn("sensitive server details", str(report))
            self.assertEqual(stub.GetActions.call_args.kwargs["timeout"], 7)
            channel.close.assert_called_once()

    def test_bad_configuration_never_imports_runtime_or_connects(self):
        with patch("so101_project.remote_check._runtime") as runtime:
            for address in ("127.0.0.1", "user:password@host:22", "host:0"):
                with self.subTest(address=address), self.assertRaises(ValueError):
                    check_remote(address, "/cloud/model")
            with self.assertRaises(ValueError):
                check_remote("127.0.0.1:8080", "/cloud/model", timeout=float("nan"))
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
