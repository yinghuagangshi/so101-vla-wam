"""Preview or open an SSH forward for the remote policy service."""

import argparse
import json
import subprocess

from .config import has_placeholder, load
from .console import configure_console


def tunnel_command(config: dict) -> list[str]:
    ssh = config.get("ssh", {})
    for key in ("host", "port", "user", "local_port", "remote_host", "remote_port"):
        if key not in ssh or not str(ssh[key]).strip():
            raise ValueError(f"ssh.{key} 不能为空")
    for key in ("port", "local_port", "remote_port"):
        if not has_placeholder(ssh[key]) and not 1 <= int(ssh[key]) <= 65535:
            raise ValueError(f"ssh.{key} 必须在 1—65535 之间")
    for key in ("user", "host", "remote_host"):
        value = str(ssh[key])
        if value.startswith("-") or any(c.isspace() for c in value) or "@" in value:
            raise ValueError(f"ssh.{key} 格式无效")
    forward = f"127.0.0.1:{ssh['local_port']}:{ssh['remote_host']}:{ssh['remote_port']}"
    return ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
            "-L", forward, "-p", str(ssh["port"]), f"{ssh['user']}@{ssh['host']}"]


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="预览或建立本地到 AutoDL 的 SSH 隧道")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        config = load([args.config])
        cmd = tunnel_command(config)
        print(json.dumps({"arguments": cmd}, ensure_ascii=False, indent=2))
        if not args.execute:
            print("预览完成；添加 --execute 才建立连接。")
            return 0
        if has_placeholder(config["ssh"]):
            raise ValueError("请先填写 SSH 主机、端口与用户名")
        return subprocess.run(cmd, check=False).returncode
    except (OSError, ValueError) as exc:
        parser.exit(2, f"SSH 配置错误：{exc}\n")
    except KeyboardInterrupt:
        print("SSH 隧道已停止。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
