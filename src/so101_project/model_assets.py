"""Download pinned public SmolVLA resources with resumable, verified shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib import error, parse, request

from .console import configure_console


CHUNK_SIZE = 8 * 1024 * 1024
ENDPOINTS = {"huggingface": "https://huggingface.co", "hf-mirror": "https://hf-mirror.com"}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/models/smolvla-base-manifest.json"


class DownloadError(RuntimeError):
    """An HTTP response or downloaded artifact failed validation."""


class _IncompleteResponse(DownloadError):
    pass


def _validate_spec(spec: dict) -> tuple[str, str]:
    size = spec.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("文件大小必须是非负整数")
    keys = [key for key in ("sha256", "git_blob_sha1") if key in spec]
    if len(keys) != 1:
        raise ValueError("每个文件必须提供一种完整性哈希")
    key = keys[0]
    digest = spec[key]
    length = 64 if key == "sha256" else 40
    if not isinstance(digest, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", digest) is None:
        raise ValueError(f"无效的 {key}")
    return key, digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: dict) -> bool:
    """Check size and the official LFS SHA256 or Git blob SHA1."""
    key, expected = _validate_spec(spec)
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    digest = hashlib.sha256() if key == "sha256" else hashlib.sha1()
    if key == "git_blob_sha1":
        digest.update(f"blob {spec['size']}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def _read_response(response, expected_size: int) -> bytes:
    content = bytearray()
    while len(content) <= expected_size:
        block = response.read(min(1024 * 1024, expected_size + 1 - len(content)))
        if not block:
            break
        content.extend(block)
    if len(content) != expected_size:
        raise _IncompleteResponse(f"响应大小错误：需要 {expected_size} 字节，收到 {len(content)} 字节")
    return bytes(content)


def _fetch(url: str, start: int, end: int, total: int, *, ranged: bool,
           retries: int, timeout: float) -> bytes:
    headers = {"User-Agent": "so101-project/0.1", "Accept-Encoding": "identity"}
    if ranged:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            # urllib uses normal HTTPS certificate verification and no Hugging Face token.
            with request.urlopen(request.Request(url, headers=headers), timeout=timeout) as response:
                if parse.urlsplit(response.geturl()).scheme != "https":
                    raise DownloadError("下载跳转目标必须使用 HTTPS")
                if ranged:
                    if response.status != 206:
                        raise DownloadError("服务器不支持 Range 分片下载；请切换 --endpoint 后重试")
                    content_range = response.headers.get("Content-Range", "")
                    if content_range != f"bytes {start}-{end}/{total}":
                        raise DownloadError(f"Content-Range 与请求不一致：{content_range!r}")
                elif response.status != 200:
                    raise DownloadError(f"完整文件下载需要 HTTP 200，收到 {response.status}")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise DownloadError("响应使用了意外的内容压缩")
                return _read_response(response, end - start + 1)
        except error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == retries:
                raise DownloadError(f"下载失败：HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, OSError, http.client.HTTPException, _IncompleteResponse) as exc:
            if attempt + 1 == retries:
                raise DownloadError(f"网络下载失败：{exc}") from exc
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def _download_shard(url: str, parts: Path, start: int, end: int, total: int,
                    *, retries: int, timeout: float) -> Path:
    part = parts / f"{start:012d}-{end:012d}.part"
    checksum = part.with_suffix(".sha256")
    if part.is_file() and part.stat().st_size == end - start + 1 and checksum.is_file():
        try:
            if checksum.read_text(encoding="ascii").strip() == _sha256(part):
                return part
        except (OSError, UnicodeError):
            pass
    content = _fetch(url, start, end, total, ranged=True, retries=retries, timeout=timeout)
    temporary = part.with_suffix(".partial")
    temporary.write_bytes(content)
    temporary.replace(part)
    checksum.write_text(hashlib.sha256(content).hexdigest() + "\n", encoding="ascii")
    return part


def download_file(url: str, destination: Path, spec: dict, *, workers: int = 4,
                  chunk_size: int = CHUNK_SIZE, retries: int = 3, timeout: float = 90) -> Path:
    """Download one artifact; only publish it after its final official hash matches.

    Large files require exact HTTP Range responses. Completed shards and their local
    checksums remain in ``<name>.parts`` after failure, and are reused on the next run.
    """
    _validate_spec(spec)
    address = parse.urlsplit(url)
    if address.scheme != "https" or not address.netloc or address.username or address.password:
        raise ValueError("下载地址必须是无内嵌凭据的 HTTPS 地址")
    if not 1 <= workers <= 16 or chunk_size <= 0 or retries <= 0 or timeout <= 0:
        raise ValueError("workers 范围为 1–16，分片大小、重试次数和超时必须大于零")
    destination = Path(destination)
    if verify_file(destination, spec):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    size = spec["size"]
    parts = destination.with_name(destination.name + ".parts")
    if size <= chunk_size:
        content = b"" if size == 0 else _fetch(url, 0, size - 1, size, ranged=False,
                                               retries=retries, timeout=timeout)
        temporary.write_bytes(content)
    else:
        parts.mkdir(exist_ok=True)
        # The cache is tied to the official artifact, and can be shared across endpoints.
        identity = {key: spec[key] for key in ("size", "sha256", "git_blob_sha1") if key in spec}
        identity_file = parts / "artifact.json"
        if identity_file.exists() and json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise DownloadError(f"分片缓存属于另一个文件版本：{parts}")
        identity_file.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
        ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(_download_shard, url, parts, start, end, size,
                                   retries=retries, timeout=timeout) for start, end in ranges]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        with temporary.open("wb") as stream:
            for start, end in ranges:
                with (parts / f"{start:012d}-{end:012d}.part").open("rb") as part:
                    shutil.copyfileobj(part, stream, length=1024 * 1024)
    if not verify_file(temporary, spec):
        raise DownloadError(f"完整文件哈希校验失败，未替换目标文件：{destination}")
    temporary.replace(destination)
    # Only remove the temporary shards created for this successfully verified artifact.
    if size > chunk_size:
        for start, end in ranges:
            part = parts / f"{start:012d}-{end:012d}.part"
            part.unlink()
            part.with_suffix(".sha256").unlink()
        identity_file.unlink()
        if not any(parts.iterdir()):
            parts.rmdir()
    return destination


def prepare_local_model(output_dir: Path, manifest: dict) -> Path:
    """Create a machine-local derivative without changing the verified originals."""
    output_dir = Path(output_dir).resolve()
    base = next(source for source in manifest["sources"] if source["directory"] == "smolvla_base")
    backbone = next(source for source in manifest["sources"] if source["directory"] == "smolvlm")
    for source in (base, backbone):
        for spec in source["files"]:
            if not verify_file(output_dir / source["directory"] / spec["name"], spec):
                raise DownloadError(f"创建本地派生目录前需要完整校验资源：{source['directory']}/{spec['name']}")
    original, local = output_dir / "smolvla_base", output_dir / "smolvla_local"
    local.mkdir(parents=True, exist_ok=True)
    backbone_path = str(output_dir / "smolvlm")
    config = json.loads((original / "config.json").read_text(encoding="utf-8"))
    preprocessor = json.loads((original / "policy_preprocessor.json").read_text(encoding="utf-8"))
    if config.get("type") != "smolvla":
        raise DownloadError("本地派生目录要求完整 SmolVLA 检查点")
    tokenizers = [step for step in preprocessor["steps"] if step.get("registry_name") == "tokenizer_processor"]
    if len(tokenizers) != 1:
        raise DownloadError("基础模型预处理配置必须有且仅有一个 tokenizer_processor")
    previous = {"vlm_model_name": config["vlm_model_name"], "load_vlm_weights": config["load_vlm_weights"],
                "tokenizer_name": tokenizers[0]["config"]["tokenizer_name"]}
    config.update(vlm_model_name=backbone_path, load_vlm_weights=False)
    tokenizers[0]["config"]["tokenizer_name"] = backbone_path
    modified = {"config.json": config, "policy_preprocessor.json": preprocessor}
    for spec in base["files"]:
        destination = local / spec["name"]
        temporary = destination.with_name(destination.name + ".partial")
        temporary.unlink(missing_ok=True)
        source_path = original / spec["name"]
        if spec["name"] in modified:
            temporary.write_text(json.dumps(modified[spec["name"]], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif spec["name"].endswith(".safetensors"):
            try:
                os.link(source_path, temporary)
            except OSError:
                shutil.copy2(source_path, temporary)
        else:
            shutil.copy2(source_path, temporary)
        temporary.replace(destination)
    sources = {"schema_version": 1, "sources": [{key: source[key] for key in ("repo_id", "revision", "directory")}
                                               for source in (base, backbone)],
               "changes": {"config.json": {"vlm_model_name": {"from": previous["vlm_model_name"], "to": backbone_path},
                                               "load_vlm_weights": {"from": previous["load_vlm_weights"], "to": False}},
                           "policy_preprocessor.json": {"tokenizer_processor.config.tokenizer_name": {
                               "from": previous["tokenizer_name"], "to": backbone_path}}},
               "usage": "Absolute paths belong to this machine. Regenerate here after moving the files. Load the full checkpoint strictly."}
    metadata = local / "sources.json.partial"
    metadata.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata.replace(local / "sources.json")
    return local


def prepare_assets(output_dir: Path, *, endpoint: str = "huggingface", workers: int = 4,
                   manifest_path: Path = DEFAULT_MANIFEST, prepare_local: bool = False) -> dict:
    """Fetch all files described by the project's pinned public manifest."""
    endpoint = ENDPOINTS.get(endpoint, endpoint).rstrip("/")
    if endpoint not in ENDPOINTS.values():
        raise ValueError("endpoint 仅支持 huggingface 或 hf-mirror")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("不支持的资源清单版本")
    output_dir = Path(output_dir).resolve()
    report = {"output_dir": str(output_dir), "endpoint": endpoint, "sources": []}
    for source in manifest["sources"]:
        directory, repo, revision = source["directory"], source["repo_id"], source["revision"]
        if directory not in ("smolvla_base", "smolvlm") or re.fullmatch(r"[\w.-]+/[\w.-]+", repo) is None:
            raise ValueError("无效的资源目录或仓库")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("资源必须固定到完整的提交哈希")
        for spec in source["files"]:
            name = spec["name"]
            if not isinstance(name, str) or re.fullmatch(r"[\w.-]+", name) is None or name in (".", ".."):
                raise ValueError("资源文件必须位于仓库顶层")
            destination = output_dir / directory / name
            print(f"准备 {directory}/{name}（{spec['size'] / 1024 / 1024:.2f} MiB）", flush=True)
            url = f"{endpoint}/{repo}/resolve/{revision}/{parse.quote(name)}"
            download_file(url, destination, spec, workers=workers)
        report["sources"].append({"directory": directory, "repo_id": repo, "revision": revision,
                                  "files_verified": len(source["files"])})
    if prepare_local:
        report["local_model_dir"] = str(prepare_local_model(output_dir, manifest))
        print("本地派生目录使用绝对路径；复制到另一台机器后，请在该机器重新运行 --prepare-local。", flush=True)
    return report


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="下载并校验固定版本的公开 SmolVLA 基础资源")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="huggingface",
                        choices=[*ENDPOINTS, *ENDPOINTS.values()],
                        help="默认官方 huggingface；hf-mirror 需明确选择")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--prepare-local", action="store_true",
                        help="生成 smolvla_local，供完整检查点严格加载；原始资源保持不变")
    args = parser.parse_args()
    try:
        report = prepare_assets(args.output_dir, endpoint=args.endpoint, workers=args.workers,
                                prepare_local=args.prepare_local)
    except (DownloadError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"资源准备失败：{exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
