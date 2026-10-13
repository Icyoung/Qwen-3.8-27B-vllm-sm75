"""Native install isolation, wheel checksum, and zero-side-effect bootstrap."""

import importlib.util, io, json, os, subprocess, tarfile, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "installer_tool", ROOT / "tools/install-native.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class NativeInstall(unittest.TestCase):
    def test_every_python_install_targets_private_interpreter(self):
        home = Path("/private/native")
        lock = json.loads((ROOT / "runtime/native-lock.json").read_text())
        commands = tool.commands(home, lock)
        for argv in commands:
            self.assertNotIn("--system", argv)
            self.assertNotIn("docker", argv)
            if "pip" in argv:
                self.assertEqual(
                    argv[argv.index("--python") + 1], str(home / "venv/bin/python")
                )
        self.assertTrue(any("#sha256=" in arg for argv in commands for arg in argv))
        compiler = next(argv for argv in commands if "micromamba" in argv[0])
        self.assertEqual(compiler[compiler.index("--prefix") + 1], str(home / "cuda"))
        self.assertIn("gcc_linux-64=13", compiler)

    def test_ambient_python_and_cuda_library_paths_are_isolated(self):
        with patch.dict(
            os.environ,
            {
                "PYTHONPATH": "/other/python",
                "PYTHONHOME": "/other",
                "VIRTUAL_ENV": "/other/venv",
                "LD_LIBRARY_PATH": "/other/cuda/lib",
            },
        ):
            env = tool.native_env("/private/native")
        for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"]:
            self.assertNotIn(key, env)
        self.assertNotIn("/other", env["LD_LIBRARY_PATH"])
        self.assertNotIn("stubs", env["LD_LIBRARY_PATH"])
        self.assertEqual(env["CUDA_HOME"], "/private/native/cuda")

    def test_foreign_venv_is_not_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "venv").mkdir()
            sentinel = home / "venv/keep"
            sentinel.write_text("unchanged")
            with (
                patch.dict(os.environ, {"QWEN_SM75_HOME": tmp}),
                patch.object(tool, "download") as download,
            ):
                with self.assertRaisesRegex(RuntimeError, "unowned"):
                    tool.main()
            self.assertEqual(sentinel.read_text(), "unchanged")
            download.assert_not_called()

    def test_checksum_mismatch_does_not_publish_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tool"
            with patch.object(
                tool.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    tool.download(
                        {"url": "https://example.invalid/tool", "sha256": "0" * 64}, out
                    )
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".partial").exists())

    def test_bootstrap_dry_run_and_help_have_no_python_or_apt_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin = Path(tmp) / "bin"
            bin.mkdir()
            marker = Path(tmp) / "called"
            for name in ["apt-get", "sudo", "python3", "nvidia-smi"]:
                p = bin / name
                p.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\nexit 99\n')
                p.chmod(0o755)
            for flag in ["--dry-run", "--help"]:
                r = subprocess.run(
                    ["/bin/bash", str(ROOT / "start.sh"), flag],
                    env={
                        **os.environ,
                        "PATH": str(bin) + ":" + os.environ["PATH"],
                        "QWEN_SM75_HOME": str(Path(tmp) / "native"),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((Path(tmp) / "native").exists())

    def test_owned_interrupted_install_resumes_after_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)

            def archive(record, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path, "w:bz2") as stream:
                    entry = tarfile.TarInfo("bin/micromamba")
                    entry.size = 4
                    stream.addfile(entry, io.BytesIO(b"tool"))

            calls = []

            def first(argv, **kwargs):
                calls.append(argv)
                if argv == ["second"]:
                    raise subprocess.CalledProcessError(1, argv)
                (home / "venv").mkdir()

            with (
                patch.dict(os.environ, {"QWEN_SM75_HOME": tmp}),
                patch.object(tool, "source_hash", return_value="test-contract"),
                patch.object(tool, "download", side_effect=archive),
                patch.object(tool, "commands", return_value=[["first"], ["second"]]),
                patch.object(
                    tool.shutil,
                    "disk_usage",
                    return_value=type("Space", (), {"free": 100 * 1024**3})(),
                ),
            ):
                with patch.object(tool.subprocess, "run", side_effect=first):
                    with self.assertRaises(subprocess.CalledProcessError):
                        tool.main()
                self.assertEqual(
                    json.loads((home / "install-progress.json").read_text())[
                        "completed"
                    ],
                    1,
                )
                with patch.object(tool.subprocess, "run") as retry:
                    tool.main()
                self.assertEqual(retry.call_count, 1)
                self.assertEqual(retry.call_args.args[0], ["second"])
            self.assertEqual(
                json.loads((home / "installed.json").read_text())["source_hash"],
                "test-contract",
            )


if __name__ == "__main__":
    unittest.main()
