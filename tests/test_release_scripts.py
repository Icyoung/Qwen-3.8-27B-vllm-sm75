"""CPU-only command construction; fake Docker never starts containers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "Bash required")
class ReleaseScripts(unittest.TestCase):
    def invoke(self, script, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fake = directory / "docker"
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$COMMAND_LOG"\n')
            fake.chmod(0o755)
            logfile = directory / "calls"
            env = {**os.environ, "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                   "COMMAND_LOG": str(logfile), "SOURCE_REVISION": "test-source",
                   "VLLM_API_KEY": "synthetic-test-key", "VLLM_SM75_CACHE_ROOT": str(directory / "cache with space"),
                   **settings}
            env = {k: v for k, v in env.items() if v is not None}
            result = subprocess.run([BASH, str(ROOT / script)], env=env, cwd=ROOT, text=True, capture_output=True)
            calls = logfile.read_text().splitlines() if logfile.exists() else []
            return result, calls

    def test_standard_full_and_fast_builds(self):
        version = (ROOT / "docker/VERSION").read_text().strip()
        for mode, suffix in [("full", ""), ("fast", "-fast")]:
            result, args = self.invoke("docker/build.sh", BUILD_MODE=mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("vllm-sm75:v" + version + suffix, args)

    def test_removed_editions_and_ui_cannot_call_docker(self):
        for script, settings in [("docker/build.sh", {"EDITION": "ultra"}), ("docker/run.sh", {"EDITION": "ultra"}),
                                 ("docker/build.sh", {"BUILD_MODE": "ui"}), ("docker/build.sh", {"EDITION": "unknown"})]:
            result, args = self.invoke(script, **settings)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(args, [])

    def test_default_modelscope_fp8_and_small_host_budget(self):
        result, args = self.invoke("docker/run.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ["Qwen/Qwen3.8-27B-FP8", "VLLM_USE_MODELSCOPE=true", "VLLM_FIREFLY_AR=0", "NCCL_P2P_DISABLE=0", "--disable-custom-all-reduce"]:
            self.assertIn(value, args)
        self.assertNotIn("--kv-transfer-config", args)
        self.assertFalse(any(a.startswith("NCCL_P2P_LEVEL=") for a in args))
        for flag, value in [("--max-num-seqs", "1"), ("--max-num-batched-tokens", "512"), ("--max-model-len", "32768"),
                            ("--memory", "6g"), ("--memory-swap", "6g"), ("--shm-size", "512m"),
                            ("--tensor-parallel-size", "4"), ("--safetensors-load-strategy", "lazy"),
                            ("--auto-sleep-offload-target", "exit")]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(json.loads(args[args.index("--model-loader-extra-config") + 1]), {"enable_multithread_load": False})
        mounts = [a for a in args if ":/root/.cache/" in a]
        self.assertTrue(any("/models/modelscope:/root/.cache/modelscope" in a for a in mounts))
        self.assertTrue(any("/models/huggingface:/root/.cache/huggingface" in a for a in mounts))

    def test_huggingface_local_model_and_revision_overrides(self):
        result, args = self.invoke("docker/run.sh", MODEL="/models/local", MODEL_ROOT="/tmp", MODEL_PROVIDER="huggingface", MODEL_REVISION="fixed-revision")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VLLM_USE_MODELSCOPE=false", args)
        self.assertIn("/tmp:/models:ro", args)
        self.assertIn("/models/local", args)
        self.assertEqual(args[args.index("--revision") + 1], "fixed-revision")

    def test_explicit_cpu_kv_and_context_settings(self):
        override = json.dumps({"dtype": "float16", "text_config": {"max_position_embeddings": 1048576}})
        result, args = self.invoke("docker/run.sh", CPU_KV_GIB="2", GPU_KV_BYTES="4294967296", MAX_MODEL_LEN="1048576", HF_OVERRIDES=override)
        self.assertEqual(result.returncode, 0, result.stderr)
        kv = json.loads(args[args.index("--kv-transfer-config") + 1])
        self.assertEqual(kv["kv_connector_extra_config"]["cpu_bytes_to_use"], 2 * 1024**3)
        self.assertEqual(args[args.index("--hf-overrides") + 1], override)
        self.assertEqual(args[args.index("--kv-cache-memory-bytes") + 1], "4294967296")

    def test_p2p_off_affects_both_nccl_and_firefly(self):
        result, args = self.invoke("docker/run.sh", P2P_MODE="off", COMM_BACKEND="firefly")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NCCL_P2P_DISABLE=1", args)
        self.assertIn("VLLM_FIREFLY_AR_BACKEND=shm", args)
        self.assertIn("VLLM_FIREFLY_AR=auto", args)

    def test_nccl_level_and_firefly_buffer_overrides(self):
        result, args = self.invoke("docker/run.sh", NCCL_P2P_LEVEL="PIX", COMM_BACKEND="firefly", VLLM_FIREFLY_AR_CHUNK_BYTES="33554432")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args.count("NCCL_P2P_LEVEL=PIX"), 1)
        self.assertIn("VLLM_FIREFLY_AR_CHUNK_BYTES=33554432", args)

    def test_invalid_or_conflicting_communication_settings(self):
        cases = [{"P2P_MODE": "bad"}, {"COMM_BACKEND": "bad"}, {"NCCL_P2P_LEVEL": "bad"},
                 {"P2P_MODE": "off", "NCCL_P2P_DISABLE": "0"}, {"NCCL_P2P_DISABLE": "1", "VLLM_FIREFLY_AR_BACKEND": "p2p"},
                 {"NCCL_P2P_DISABLE": "2"}, {"MODEL_PROVIDER": "bad"}]
        for settings in cases:
            with self.subTest(settings=settings):
                result, args = self.invoke("docker/run.sh", **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(args, [])

    def test_download_only_has_no_gpu_or_api_key_requirement(self):
        result, args = self.invoke("docker/download.sh", VLLM_API_KEY=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--gpus", args)
        self.assertNotIn("--detach", args)
        self.assertIn("MODEL_ID=Qwen/Qwen3.8-27B-FP8", args)
        self.assertIn("snapshot_download", args[-1])
        self.assertNotIn("serve", args)
        self.assertEqual(args[args.index("--entrypoint") + 1], "python3")

    def test_download_revision_and_local_path_rejection(self):
        result, args = self.invoke("docker/download.sh", MODEL="publisher/FP8", MODEL_REVISION="fixed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MODEL_REVISION=fixed", args)
        result, args = self.invoke("docker/download.sh", MODEL="/models/local")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(args, [])


if __name__ == "__main__":
    unittest.main()
