"""Probe the official remote SmolVLA service with synthetic observations only."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import json
import pickle
import time
from types import SimpleNamespace

from .console import configure_console


MOTOR_NAMES = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
DEFAULT_TASK = "Pick up the red block and place it in the left bin."
IMAGE_BYTES = 2 * 480 * 640 * 3


def robot_features() -> dict:
    """Match map_robot_keys_to_lerobot_features for a two-camera SO-101."""
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(MOTOR_NAMES)},
        **{f"observation.images.camera{i}": {"dtype": "image", "shape": (480, 640, 3),
             "names": ["height", "width", "channels"]} for i in (1, 2)},
    }


def _runtime():
    # Import neither RobotClient nor any robot factory. These protocol types are
    # the same classes used by the official client and its pickle transport.
    import grpc
    import numpy as np
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

    return SimpleNamespace(grpc=grpc, np=np, RemotePolicyConfig=RemotePolicyConfig,
                           TimedObservation=TimedObservation, messages=services_pb2,
                           stub=services_pb2_grpc.AsyncInferenceStub,
                           channel_options=grpc_channel_options,
                           send_bytes=send_bytes_in_chunks)


def validate_actions(actions, start: int, count: int) -> dict:
    if not isinstance(actions, list) or len(actions) != count:
        raise ValueError(f"动作数量错误，应为 {count} 条")
    timesteps = [action.get_timestep() for action in actions]
    if timesteps != list(range(start, start + count)):
        raise ValueError("动作时间步不连续或与本轮观察不一致")
    for action in actions:
        tensor = action.get_action()
        if tuple(tensor.shape) != (6,):
            raise ValueError("动作形状错误，应为 (6,)")
        if not all(math.isfinite(float(value)) for value in tensor.tolist()):
            raise ValueError("动作含非有限值")
    return {"actions": len(actions), "action_shape": [6], "finite": True,
            "timesteps_contiguous": True, "first_timestep": timesteps[0],
            "last_timestep": timesteps[-1]}


def check_remote(server_address: str, checkpoint: str, *, task: str = DEFAULT_TASK,
                 rounds: int = 3, timeout: float = 60, load_timeout: float = 300,
                 actions_per_chunk: int = 50) -> dict:
    host, separator, port = server_address.rpartition(":")
    if (not host or not separator or not port.isdigit() or not 1 <= int(port) <= 65535
            or "@" in server_address or "://" in server_address
            or any(character.isspace() for character in server_address)):
        raise ValueError("server-address 必须为 host:port，不能包含凭据或空白")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint 不能为空，路径由云端解释")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须为非空指令")
    if rounds < 1 or actions_per_chunk < 1:
        raise ValueError("rounds 和 actions-per-chunk 必须为正数")
    if any(not math.isfinite(value) or value <= 0 for value in (timeout, load_timeout)):
        raise ValueError("超时必须为有限正数")
    runtime = _runtime()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_address": server_address, "checkpoint": checkpoint,
        "policy_type": "smolvla", "policy_device": "cuda", "task": task,
        "robot_connected": False, "synthetic_observations": True,
        "measurement_scope": "两相机合成观察的顺序 RPC 验证；不表示实机控制频率，也不估计单程网络时延。",
        "send_timing_scope": "SendObservations 调用至服务端确认，包括传输、反序列化及入队耗时。",
        "image_bytes_per_observation": IMAGE_BYTES,
        "requested_rounds": rounds, "actions_per_chunk": actions_per_chunk,
        "timeout_seconds": timeout, "load_timeout_seconds": load_timeout,
        "rounds": [], "passed": False,
    }
    phase = "connect"
    start = time.perf_counter()
    channel = runtime.grpc.insecure_channel(
        server_address, options=runtime.channel_options(enable_retries=False))
    try:
        runtime.grpc.channel_ready_future(channel).result(timeout=timeout)
        report["connection_seconds"] = time.perf_counter() - start
        stub = runtime.stub(channel)
        phase = "ready"
        stub.Ready(runtime.messages.Empty(), timeout=timeout)
        specs = runtime.RemotePolicyConfig("smolvla", checkpoint, robot_features(),
                                           actions_per_chunk, "cuda")
        phase = "load_policy"
        start = time.perf_counter()
        stub.SendPolicyInstructions(runtime.messages.PolicySetup(data=pickle.dumps(specs)),
                                    timeout=load_timeout)
        report["policy_setup_rpc_seconds"] = time.perf_counter() - start
        timestep = 0
        for index in range(rounds):
            observation = {**dict.fromkeys(MOTOR_NAMES, 0.0), "task": task,
                "camera1": runtime.np.full((480, 640, 3), index % 256, dtype=runtime.np.uint8),
                "camera2": runtime.np.full((480, 640, 3), (128 + index) % 256,
                                            dtype=runtime.np.uint8)}
            timed = runtime.TimedObservation(timestamp=time.time(), timestep=timestep,
                                             observation=observation, must_go=True)
            payload = pickle.dumps(timed)
            result = {"round": index + 1, "observation_timestep": timestep,
                      "must_go": True, "payload_bytes": len(payload)}
            report["rounds"].append(result)
            phase = f"round_{index + 1}_send_observation"
            rpc_start = time.perf_counter()
            stream = runtime.send_bytes(payload, runtime.messages.Observation, silent=True)
            stub.SendObservations(stream, timeout=timeout)
            result["send_observations_rpc_seconds"] = time.perf_counter() - rpc_start
            phase = f"round_{index + 1}_get_actions"
            action_start = time.perf_counter()
            response = stub.GetActions(runtime.messages.Empty(), timeout=timeout)
            result["get_actions_rpc_seconds"] = time.perf_counter() - action_start
            result["total_rpc_seconds"] = time.perf_counter() - rpc_start
            if not response.data:
                raise ValueError("服务端返回空动作，请检查服务端推理日志")
            phase = f"round_{index + 1}_validate_actions"
            result.update(validate_actions(pickle.loads(response.data), timestep, actions_per_chunk))
            timestep = result["last_timestep"] + 1
        report["passed"] = True
    except runtime.grpc.FutureTimeoutError:
        report["error"] = {"phase": phase, "type": "connection_timeout",
                           "message": "连接服务超时，请检查 SSH 隧道及服务监听地址"}
    except runtime.grpc.RpcError as error:
        # Status identifies the failure without copying arbitrary server details
        # or authentication text into the reusable report.
        report["error"] = {"phase": phase, "type": "rpc_error", "code": error.code().name}
    except (ValueError, TypeError, AttributeError, pickle.UnpicklingError, EOFError) as error:
        report["error"] = {"phase": phase, "type": "invalid_response", "message": str(error)}
    finally:
        channel.close()
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="通过 SSH 隧道验证云端 SmolVLA RPC；仅发送合成数据，不连接设备")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--checkpoint", required=True, help="云端完整检查点目录，由服务端解释")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60, help="连接及各轮 RPC 超时秒数")
    parser.add_argument("--load-timeout", type=float, default=300, help="模型加载 RPC 超时秒数")
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--report", type=Path, help="可选 JSON 报告，使用新的文件名")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("报告已存在，请使用新的文件名")
    try:
        report = check_remote(args.server_address, args.checkpoint, task=args.task,
                              rounds=args.rounds, timeout=args.timeout,
                              load_timeout=args.load_timeout,
                              actions_per_chunk=args.actions_per_chunk)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"远程检查失败：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
