"""Native lifecycle with a mocked systemd runtime; no install or GPU access."""

import contextlib, importlib.util, io, json, os, stat, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("start_tool", ROOT / "tools/start.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class OneClick(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.directory = self.home / "private"
        self.env = dict(tool.DEFAULTS, QWEN_SM75_HOME=str(self.home))
        self.runtime = Mock()
        self.runtime.running.return_value = False
        self.output = io.StringIO()

    def run_start(self, **kwargs):
        with contextlib.redirect_stdout(self.output):
            tool.start(self.env, self.runtime, [], self.directory, **kwargs)

    def test_key_generated_once_private_and_reused(self):
        key, path = tool.api_key({}, self.directory)
        self.assertEqual(len(key), 64)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(tool.api_key({}, self.directory), (key, path))

    def test_explicit_key_does_not_write_file(self):
        self.assertEqual(
            tool.api_key({"VLLM_API_KEY": "test-key"}, self.directory),
            ("test-key", None),
        )
        self.assertFalse(self.directory.exists())

    def test_empty_and_public_key_file_rejected(self):
        self.directory.mkdir()
        p = self.directory / "api-key"
        p.write_text("")
        p.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            tool.api_key({}, self.directory)
        p.write_text("test-key")
        p.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "owner-only"):
            tool.api_key({}, self.directory)

    def test_dry_run_has_no_runtime_or_key_side_effects(self):
        self.run_start(dry_run=True)
        self.assertEqual(self.runtime.mock_calls, [])
        self.assertFalse(self.directory.exists())
        self.assertIn('"requires_docker": false', self.output.getvalue())

    def test_prerequisite_failure_before_key(self):
        self.runtime.check_environment.side_effect = RuntimeError("missing driver")
        with self.assertRaises(RuntimeError):
            self.run_start()
        self.assertFalse(self.directory.exists())
        self.runtime.launch.assert_not_called()

    def test_launch_and_wait(self):
        self.run_start()
        self.runtime.launch.assert_called_once()
        self.runtime.wait_ready.assert_called_once_with(3600)
        self.assertNotIn(self.env["VLLM_API_KEY"], self.output.getvalue())

    def test_matching_service_reused(self):
        self.env["VLLM_API_KEY"] = "test-key"
        self.runtime.running.return_value = True
        (self.home / "run").mkdir()
        (self.home / "run/server.json").write_text(
            json.dumps(
                {"fingerprint": tool.fingerprint(self.env, tool.profile(self.env))}
            )
        )
        self.run_start()
        self.runtime.launch.assert_not_called()

    def test_conflicting_service_preserved(self):
        self.runtime.running.return_value = True
        with self.assertRaisesRegex(RuntimeError, "not stopped or replaced"):
            self.run_start()
        self.runtime.launch.assert_not_called()

    def test_profile_has_fp16_marlin_native_settings(self):
        args = tool.profile(self.env)
        self.assertEqual(args[args.index("--dtype") + 1], "float16")
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "4")
        self.assertIn("flashqla_sm75", args)
        self.assertNotIn("--kv-transfer-config", args)

    def test_reserved_identity_override_rejected(self):
        with self.assertRaises(ValueError):
            tool.start(
                self.env,
                self.runtime,
                ["--api-key=private"],
                self.directory,
                dry_run=True,
            )
        self.assertEqual(self.runtime.mock_calls, [])

    def test_service_isolation_and_private_config(self):
        rt = tool.Runtime(self.env)
        rt.system = False
        rt.ctl = ["systemctl", "--user"]
        rt.command = Mock()
        self.env["VLLM_API_KEY"] = "test-key"
        with patch.object(tool.Path, "home", return_value=self.home):
            rt.launch(tool.profile(self.env), "test-fingerprint")
        unit = (self.home / ".config/systemd/user/qwen38-27b-sm75.service").read_text()
        self.assertIn(" -I ", unit)
        self.assertNotIn("test-key", unit)
        self.assertIn("MemoryMax=6G", unit)
        config = self.home / "run/server.json"
        self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o600)
        self.assertEqual(json.loads(config.read_text())["env"]["VLLM_FIREFLY_AR"], "0")
        self.assertIn("start", rt.command.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
