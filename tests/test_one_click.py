"""One-click lifecycle with a mocked runtime; no Docker, download or GPUs."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("start_tool", ROOT / "tools/start.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class OneClick(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "private config"
        self.env = dict(tool.DEFAULTS, IMAGE="test-image")
        self.runtime = Mock()
        self.runtime.image.return_value = "sha256:test-image"
        self.runtime.container.return_value = None
        self.output = io.StringIO()

    def run_start(self, **kwargs):
        with contextlib.redirect_stdout(self.output):
            tool.start(self.env, self.runtime, [], self.directory, **kwargs)

    def test_private_key_is_generated_once_and_reused(self):
        key, path = tool.api_key({}, self.directory)
        self.assertEqual(len(key), 64)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(tool.api_key({}, self.directory), (key, path))

    def test_explicit_key_does_not_write_a_file(self):
        self.assertEqual(tool.api_key({"VLLM_API_KEY": "test-key"}, self.directory), ("test-key", None))
        self.assertFalse(self.directory.exists())

    def test_empty_or_public_key_file_is_rejected(self):
        self.directory.mkdir()
        path = self.directory / "api-key"
        path.write_text("")
        path.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            tool.api_key({}, self.directory)
        path.write_text("test-key")
        path.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "owner-only"):
            tool.api_key({}, self.directory)

    def test_dry_run_has_no_runtime_or_credential_side_effects(self):
        self.run_start(dry_run=True)
        self.assertEqual(self.runtime.mock_calls, [])
        self.assertFalse(self.directory.exists())
        self.assertIn("no commands", self.output.getvalue())

    def test_prerequisite_failure_happens_before_key_or_launch(self):
        self.runtime.check_environment.side_effect = RuntimeError("missing Docker")
        with self.assertRaisesRegex(RuntimeError, "missing Docker"):
            self.run_start()
        self.assertFalse(self.directory.exists())
        self.runtime.launch.assert_not_called()

    def test_existing_image_skips_build_and_waits_after_launch(self):
        self.run_start()
        self.runtime.build.assert_not_called()
        self.runtime.launch.assert_called_once_with([])
        self.runtime.wait_ready.assert_called_once_with(3600)
        self.assertNotIn(self.env["VLLM_API_KEY"], self.output.getvalue())

    def test_missing_image_builds_once(self):
        self.runtime.image.side_effect = [None, "sha256:built"]
        self.run_start(wait=False)
        self.runtime.build.assert_called_once_with()
        self.runtime.launch.assert_called_once_with([])
        self.runtime.wait_ready.assert_not_called()

    def test_matching_container_is_reused(self):
        self.env["VLLM_API_KEY"] = "test-key"
        value = tool.fingerprint(self.env, "sha256:test-image", [])
        self.runtime.container.return_value = {"Config": {"Labels": {"io.qwen-sm75.launch-fingerprint": value}}, "State": {"Running": True}}
        self.run_start()
        self.runtime.launch.assert_not_called()
        self.runtime.wait_ready.assert_called_once()

    def test_mismatched_or_stopped_container_is_not_replaced(self):
        self.env["VLLM_API_KEY"] = "test-key"
        value = tool.fingerprint(self.env, "sha256:test-image", [])
        for label, running in [("different", True), (value, False)]:
            self.runtime.container.return_value = {"Config": {"Labels": {"io.qwen-sm75.launch-fingerprint": label}}, "State": {"Running": running}}
            with self.assertRaises(RuntimeError):
                self.run_start()
        self.runtime.launch.assert_not_called()
        self.runtime.build.assert_not_called()

    def test_rebuild_does_not_touch_existing_container(self):
        self.runtime.container.return_value = {"State": {"Running": True}}
        with self.assertRaisesRegex(RuntimeError, "refusing to rebuild"):
            self.run_start(rebuild=True)
        self.runtime.build.assert_not_called()

    def test_identity_override_is_rejected_even_in_dry_run(self):
        with self.assertRaises(RuntimeError):
            tool.start(self.env, self.runtime, ["--api-key=private"], self.directory, dry_run=True)
        self.assertEqual(self.runtime.mock_calls, [])

    def test_gpu_selection_checks_selected_cards_not_only_host_count(self):
        runtime = tool.Runtime(self.env)
        rows = "0, GPU-a, 7.5, 16384\n1, GPU-b, 7.5, 16384\n2, GPU-c, 7.5, 16384\n3, GPU-d, 7.5, 16384\n"
        def command(argv, **kwargs):
            return SimpleNamespace(stdout='{"nvidia":{}}' if argv[0] == "docker" else rows)
        runtime.command = Mock(side_effect=command)
        with patch.object(tool.platform, "system", return_value="Linux"), patch.object(tool.platform, "machine", return_value="x86_64"), patch.object(tool.shutil, "which", return_value="/fake/tool"):
            self.env["GPUS"] = '"device=0,1"'
            with self.assertRaisesRegex(RuntimeError, "at least 4"):
                runtime.check_environment()
            self.env["GPUS"] = '"device=GPU-d,GPU-b,GPU-c,GPU-a"'
            runtime.check_environment()
            self.env["GPUS"] = '"device=0,1,2,2"'
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                runtime.check_environment()

    def test_authenticated_listing_is_required_before_ready(self):
        self.env["VLLM_API_KEY"] = "test-key"
        runtime = tool.Runtime(self.env)
        runtime.container = Mock(return_value={"State": {"Running": True}})
        response = io.BytesIO(b'{"data":[{"id":"VLLM-Qwen3.8-27B"}]}')
        with patch.object(tool.urllib.request, "urlopen", return_value=response) as request, contextlib.redirect_stdout(self.output):
            runtime.wait_ready(5)
        self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer test-key")
        self.assertIn("generation not tested", self.output.getvalue())
        self.assertNotIn("test-key", self.output.getvalue())

    def test_container_exit_is_not_reported_as_ready(self):
        self.env["VLLM_API_KEY"] = "test-key"
        runtime = tool.Runtime(self.env)
        runtime.container = Mock(return_value={"State": {"Running": False}})
        with self.assertRaisesRegex(RuntimeError, "exited before"):
            runtime.wait_ready(5)


if __name__ == "__main__":
    unittest.main()
