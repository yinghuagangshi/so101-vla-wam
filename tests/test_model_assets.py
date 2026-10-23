import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_project.model_assets import (DEFAULT_MANIFEST, DownloadError, download_file,
                                        prepare_local_model, verify_file)


def spec_for(content, *, name="model.safetensors", git=False):
    if git:
        digest = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
        return {"name": name, "size": len(content), "git_blob_sha1": digest}
    return {"name": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, content, *, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}

    def geturl(self):
        return "https://example.org/model"


class ModelAssetsTests(unittest.TestCase):
    def test_verification_uses_git_blob_header_and_lfs_hash_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            content = b'{"type":"smolvla"}\n'
            path.write_bytes(content)
            self.assertTrue(verify_file(path, spec_for(content, git=True)))
            self.assertTrue(verify_file(path, spec_for(content)))
            self.assertFalse(verify_file(path, spec_for(content + b"x")))
            # A plain content SHA1 is not the Git blob identity.
            wrong = spec_for(content, git=True)
            wrong["git_blob_sha1"] = hashlib.sha1(content).hexdigest()
            self.assertFalse(verify_file(path, wrong))

    def test_small_file_is_verified_and_existing_verified_file_skips_network(self):
        content = b"valid resource"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "file"
            with patch("so101_project.model_assets.request.urlopen", return_value=Response(content)) as opener:
                download_file("https://example.org/file", path, spec_for(content))
                self.assertEqual(path.read_bytes(), content)
                self.assertNotIn("Authorization", dict(opener.call_args.args[0].header_items()))
            with patch("so101_project.model_assets.request.urlopen", side_effect=AssertionError("network")):
                download_file("https://example.org/file", path, spec_for(content))

    def test_hash_failure_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "file"
            path.write_bytes(b"old")
            with patch("so101_project.model_assets.request.urlopen", return_value=Response(b"bad")):
                with self.assertRaisesRegex(DownloadError, "哈希"):
                    download_file("https://example.org/file", path, spec_for(b"new"))
            self.assertEqual(path.read_bytes(), b"old")

    def test_range_resume_reuses_only_finished_verified_shards(self):
        content = b"0123456789abcdef"
        calls = []

        def serve(req, **kwargs):
            value = req.get_header("Range")
            calls.append(value)
            start, end = map(int, value.removeprefix("bytes=").split("-"))
            if start == 8 and calls.count(value) == 1:
                raise URLError("interrupted")
            return Response(content[start:end + 1], status=206,
                            headers={"Content-Range": f"bytes {start}-{end}/{len(content)}"})

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            with patch("so101_project.model_assets.request.urlopen", side_effect=serve):
                with self.assertRaisesRegex(DownloadError, "网络"):
                    download_file("https://example.org/model", path, spec_for(content),
                                  chunk_size=8, workers=1, retries=1)
                self.assertFalse(path.exists())
                parts = Path(str(path) + ".parts")
                self.assertEqual((parts / "000000000000-000000000007.part").read_bytes(), content[:8])
                download_file("https://example.org/model", path, spec_for(content),
                              chunk_size=8, workers=1, retries=1)
            self.assertEqual(calls.count("bytes=0-7"), 1)
            self.assertEqual(calls.count("bytes=8-15"), 2)
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse(parts.exists())

    def test_bad_range_headers_and_unsupported_range_cannot_publish(self):
        for status, content_range in ((200, ""), (206, "bytes 0-7/999"), (206, "bytes 8-15/16")):
            with self.subTest(status=status, range=content_range), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "file"

                def serve(req, **kwargs):
                    return Response(b"0" * 8, status=status, headers={"Content-Range": content_range})

                with patch("so101_project.model_assets.request.urlopen", side_effect=serve):
                    with self.assertRaises(DownloadError):
                        download_file("https://example.org/file", path, spec_for(b"0" * 16),
                                      chunk_size=8, workers=1, retries=1)
                self.assertFalse(path.exists())

    def test_truncated_range_is_retried_and_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "file"

            def serve(req, **kwargs):
                start, end = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
                return Response(b"x", status=206,
                                headers={"Content-Range": f"bytes {start}-{end}/16"})

            with patch("so101_project.model_assets.request.urlopen", side_effect=serve) as opener, \
                    patch("so101_project.model_assets.time.sleep"):
                with self.assertRaises(DownloadError):
                    download_file("https://example.org/file", path, spec_for(b"0" * 16),
                                  chunk_size=8, workers=1, retries=2)
            self.assertGreaterEqual(opener.call_count, 2)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(str(path) + ".parts").glob("*.part")), [])

    def test_corrupt_cached_shard_is_downloaded_again(self):
        content = b"0123456789abcdef"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            parts = Path(str(path) + ".parts")
            parts.mkdir()
            part = parts / "000000000000-000000000007.part"
            part.write_bytes(b"badbytes")
            part.with_suffix(".sha256").write_text(hashlib.sha256(content[:8]).hexdigest(), encoding="ascii")

            def serve(req, **kwargs):
                start, end = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
                return Response(content[start:end + 1], status=206,
                                headers={"Content-Range": f"bytes {start}-{end}/16"})

            with patch("so101_project.model_assets.request.urlopen", side_effect=serve) as opener:
                download_file("https://example.org/model", path, spec_for(content), chunk_size=8, workers=1)
            self.assertEqual(opener.call_count, 2)
            self.assertTrue(verify_file(path, spec_for(content)))

    def test_pinned_manifest_has_full_base_and_only_top_level_backbone_metadata(self):
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        base, backbone = manifest["sources"]
        self.assertEqual(base["revision"], "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
        self.assertEqual(backbone["revision"], "7b375e1b73b11138ff12fe22c8f2822d8fe03467")
        self.assertEqual(len(base["files"]), 6)
        weights = next(item for item in base["files"] if item["name"] == "model.safetensors")
        self.assertEqual(weights["size"], 906712520)
        self.assertTrue(all("/" not in item["name"] and item["name"].endswith((".json", ".txt"))
                            for item in backbone["files"]))

    def test_local_derivative_preserves_verified_originals_and_supports_link_fallback(self):
        original_config = {"type": "smolvla", "vlm_model_name": "repo/backbone", "load_vlm_weights": True}
        original_preprocessor = {"steps": [{"registry_name": "tokenizer_processor",
                                            "config": {"tokenizer_name": "repo/backbone"}}]}
        base_contents = {"config.json": json.dumps(original_config).encode(),
                         "policy_preprocessor.json": json.dumps(original_preprocessor).encode(),
                         "model.safetensors": b"full checkpoint",
                         "policy_postprocessor.json": b"{}",
                         "policy_preprocessor_step_5_normalizer_processor.safetensors": b"norm",
                         "policy_postprocessor_step_0_unnormalizer_processor.safetensors": b"norm"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"sources": []}
            for directory, contents in (("smolvla_base", base_contents), ("smolvlm", {"config.json": b"{}"})):
                (root / directory).mkdir()
                source = {"directory": directory, "repo_id": "public/" + directory,
                          "revision": "a" * 40, "files": []}
                for name, content in contents.items():
                    (root / directory / name).write_bytes(content)
                    source["files"].append(spec_for(content, name=name))
                manifest["sources"].append(source)
            with patch("so101_project.model_assets.os.link", side_effect=OSError("unsupported")):
                local = prepare_local_model(root, manifest)
            config = json.loads((local / "config.json").read_text())
            self.assertFalse(config["load_vlm_weights"])
            self.assertEqual(config["vlm_model_name"], str(root.resolve() / "smolvlm"))
            preprocessor = json.loads((local / "policy_preprocessor.json").read_text())
            self.assertEqual(preprocessor["steps"][0]["config"]["tokenizer_name"], config["vlm_model_name"])
            self.assertEqual((local / "model.safetensors").read_bytes(), b"full checkpoint")
            self.assertEqual(json.loads((root / "smolvla_base/config.json").read_text()), original_config)
            self.assertEqual(json.loads((root / "smolvla_base/policy_preprocessor.json").read_text()), original_preprocessor)
            self.assertTrue((local / "sources.json").is_file())
            for source in manifest["sources"]:
                for spec in source["files"]:
                    self.assertTrue(verify_file(root / source["directory"] / spec["name"], spec))


if __name__ == "__main__":
    unittest.main()
