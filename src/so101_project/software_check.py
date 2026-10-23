"""Exercise SmolVLA weights, gradients and RPC transport without robot hardware."""

import argparse
from concurrent import futures
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import time

from .console import configure_console


def check(model_path: Path, output: Path, device: str, threads: int) -> dict:
    # This check requires the complete local resource bundle and must fail on a
    # missing file instead of silently waiting for a Hub request.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import grpc
    import numpy as np
    import torch
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import send_bytes_in_chunks

    torch.set_num_threads(threads)
    torch.manual_seed(42)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；CPU 检查请显式选择 --device cpu")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "synthetic software verification; no physical task evaluation",
        "model_path": str(model_path), "device": device,
        "torch": torch.__version__, "threads": threads,
        "robot_connected": False, "real_demonstrations_used": False,
    }
    config = PreTrainedConfig.from_pretrained(model_path)
    config.device = device
    config.push_to_hub = False
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
        **{f"observation.images.camera{i}": PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
           for i in (1, 2)},
    }
    start = time.perf_counter()
    print("Loading complete checkpoint with strict=True", flush=True)
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=True)
    report["strict_weights_loaded"] = True
    report["load_seconds"] = time.perf_counter() - start
    report["parameter_dtypes"] = sorted({str(p.dtype) for p in policy.parameters()})
    report["parameter_count"] = sum(p.numel() for p in policy.parameters())
    print("Checkpoint loaded; loading processors", flush=True)
    pre, post = make_pre_post_processors(
        config, pretrained_path=model_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"features": {**config.input_features, **config.output_features}},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    raw = {
        "observation.state": torch.zeros(6),
        "observation.images.camera1": torch.zeros(3, 480, 640),
        "observation.images.camera2": torch.ones(3, 480, 640) * 0.5,
        "task": "Pick up the red block and place it in the left bin.",
    }
    start = time.perf_counter()
    print("Running two-camera inference", flush=True)
    with torch.inference_mode():
        actions = policy.predict_action_chunk(pre(dict(raw)))
        action = post(actions[:, 0])
    if (tuple(actions.shape) != (1, config.chunk_size, 6)
            or not torch.isfinite(actions).all() or not torch.isfinite(action).all()):
        raise RuntimeError("推理动作形状或数值异常")
    report["inference"] = {"chunk_shape": list(actions.shape), "finite": True,
                           "seconds": time.perf_counter() - start}

    # One optimizer step only tests the training path; these are synthetic targets.
    policy.train()
    print("Running one synthetic optimizer step", flush=True)
    batch = pre({**raw, "action": torch.zeros(1, config.chunk_size, 6)})
    batch["action_is_pad"] = torch.zeros(1, config.chunk_size, dtype=torch.bool, device=device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    before = trainable[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    loss, _ = policy(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("合成数据损失非有限值")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
    optimizer.step()
    if torch.equal(before, trainable[0]):
        raise RuntimeError("优化器未更新参数")
    report["synthetic_update"] = {"steps": 1, "loss": loss.item(),
                                  "grad_norm": grad_norm.item(), "parameter_changed": True,
                                  "seconds": time.perf_counter() - start}
    checkpoint = output / "synthetic_checkpoint"
    print("Saving synthetic checkpoint and processors", flush=True)
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    (checkpoint / "SYNTHETIC_ONLY.txt").write_text(
        "Software diagnostic checkpoint trained on one synthetic batch. Do not deploy on a robot.\n",
        encoding="utf-8",
    )
    del policy, optimizer, trainable, before, batch, loss, actions, action, pre, post
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Use the official gRPC service and raw image/state protocol over loopback.
    # No RobotClient is instantiated, so no serial port or camera can be opened.
    service = PolicyServer(PolicyServerConfig(host="127.0.0.1", port=8080, fps=30))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    print("Reloading checkpoint through the official gRPC service", flush=True)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=15)
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        stub.Ready(services_pb2.Empty(), timeout=15)
        motor_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                       "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
        features = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": motor_names},
            **{f"observation.images.camera{i}": {"dtype": "image", "shape": (480, 640, 3),
                                                   "names": ["height", "width", "channels"]}
               for i in (1, 2)},
        }
        specs = RemotePolicyConfig("smolvla", str(checkpoint), features, config.chunk_size, device)
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(specs)), timeout=300)
        observation = {**dict.fromkeys(motor_names, 0.0),
                       "camera1": np.zeros((480, 640, 3), dtype=np.uint8),
                       "camera2": np.full((480, 640, 3), 128, dtype=np.uint8),
                       "task": raw["task"]}
        timed = TimedObservation(timestamp=time.time(), timestep=0,
                                 observation=observation, must_go=True)
        stream = send_bytes_in_chunks(pickle.dumps(timed), services_pb2.Observation, silent=True)
        stub.SendObservations(stream, timeout=30)
        start = time.perf_counter()
        response = stub.GetActions(services_pb2.Empty(), timeout=300)
        if not response.data:
            raise RuntimeError("gRPC 服务未返回动作；查看服务日志")
        received = pickle.loads(response.data)
        if len(received) != config.chunk_size:
            raise RuntimeError("gRPC 动作数量错误")
        if [a.get_timestep() for a in received] != list(range(config.chunk_size)):
            raise RuntimeError("gRPC 动作时间步错误")
        if any(tuple(a.get_action().shape) != (6,) or not torch.isfinite(a.get_action()).all()
               for a in received):
            raise RuntimeError("gRPC 动作形状或数值错误")
        report["grpc_loopback"] = {"checkpoint_reloaded": True, "actions": len(received),
                                   "finite": True, "seconds": time.perf_counter() - start}
    finally:
        channel.close()
        service.stop()
        server.stop(0).wait(timeout=10)
    report["passed"] = True
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="用合成数据检查 SmolVLA 权重、单步更新、检查点和 gRPC；不连接设备")
    parser.add_argument("--model-path", type=Path, required=True, help="prepare_smolvla 生成的 smolvla_local")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的诊断输出目录，不可为已有目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须为正数")
    report = check(args.model_path.resolve(), args.output_dir.resolve(), args.device, args.threads)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
