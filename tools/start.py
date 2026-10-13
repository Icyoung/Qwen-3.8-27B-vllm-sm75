#!/usr/bin/env python3
"""Start the isolated native venv as an owned systemd service."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import secrets
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "MODEL": "Qwen/Qwen3.8-27B-FP8",
    "MODEL_PROVIDER": "modelscope",
    "TP_SIZE": "4",
    "PORT": "8000",
    "HOST": "0.0.0.0",
    "SERVED_MODEL_NAME": "VLLM-Qwen3.8-27B",
    "COMM_BACKEND": "nccl",
    "P2P_MODE": "auto",
    "MAX_MODEL_LEN": "32768",
    "MAX_NUM_SEQS": "1",
    "MAX_NUM_BATCHED_TOKENS": "512",
    "GPU_MEMORY_UTILIZATION": "0.87",
    "CPU_KV_GIB": "0",
    "AUTO_SLEEP_IDLE_TIMEOUT": "30",
    "NATIVE_MEMORY_MAX": "6G",
}


def installer():
    spec = importlib.util.spec_from_file_location(
        "native_installer", ROOT / "tools/install-native.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def api_key(env, directory):
    if env.get("VLLM_API_KEY"):
        return env["VLLM_API_KEY"], None
    directory = Path(directory).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "api-key"
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o077
            or info.st_uid != os.getuid()
        ):
            raise RuntimeError(f"API key must be an owner-only regular file: {path}")
        key = path.read_text().strip()
        if not key:
            raise RuntimeError(f"API key file is empty: {path}")
        return key, path
    with os.fdopen(fd, "w") as stream:
        key = secrets.token_hex(32)
        stream.write(key + "\n")
    return key, path


def profile(env, extra=()):
    if not re.fullmatch(r"[1-9][0-9]*[KMGTP]?", env["NATIVE_MEMORY_MAX"]):
        raise ValueError("Invalid NATIVE_MEMORY_MAX")
    if not 1 <= int(env["PORT"]) <= 65535:
        raise ValueError("Invalid PORT")
    reserved = {
        "--api-key",
        "--host",
        "--port",
        "--served-model-name",
        "--tensor-parallel-size",
    }
    if any(a.split("=", 1)[0] in reserved for a in extra):
        raise ValueError(
            "Configure API identity, port and TP through environment variables"
        )
    tp = int(env["TP_SIZE"])
    if tp not in {1, 2, 4, 8}:
        raise ValueError("TP_SIZE must be 1,2,4,8")
    if env["MODEL_PROVIDER"] not in {"modelscope", "huggingface"}:
        raise ValueError("Invalid MODEL_PROVIDER")
    if env["COMM_BACKEND"] not in {"nccl", "firefly"}:
        raise ValueError("COMM_BACKEND must be nccl or firefly")
    if env["P2P_MODE"] not in {"auto", "off"}:
        raise ValueError("P2P_MODE must be auto or off")
    args = [
        env["MODEL"],
        "--served-model-name",
        env["SERVED_MODEL_NAME"],
        "--host",
        env["HOST"],
        "--port",
        env["PORT"],
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        "float16",
        "--hf-overrides",
        '{"dtype":"float16"}',
        "--max-model-len",
        env["MAX_MODEL_LEN"],
        "--max-num-seqs",
        env["MAX_NUM_SEQS"],
        "--max-num-batched-tokens",
        env["MAX_NUM_BATCHED_TOKENS"],
        "--gpu-memory-utilization",
        env["GPU_MEMORY_UTILIZATION"],
        "--attention-config",
        '{"backend":"FLASHINFER"}',
        "--gdn-prefill-backend",
        "flashqla_sm75",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--block-size",
        "32",
        "--disable-custom-all-reduce",
        "--safetensors-load-strategy",
        "lazy",
        "--model-loader-extra-config",
        '{"enable_multithread_load":false}',
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--async-scheduling",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
    ]
    if env.get("MODEL_REVISION"):
        args += ["--revision", env["MODEL_REVISION"]]
    if int(env["CPU_KV_GIB"]) < 0:
        raise ValueError("CPU_KV_GIB must be nonnegative")
    if int(env["CPU_KV_GIB"]):
        cfg = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "CPUOffloadingSpec",
                "cpu_bytes_to_use": int(env["CPU_KV_GIB"]) * 1024**3,
            },
        }
        args += ["--kv-transfer-config", json.dumps(cfg, separators=(",", ":"))]
    if env.get("GPU_KV_BYTES"):
        args += ["--kv-cache-memory-bytes", env["GPU_KV_BYTES"]]
    if float(env["AUTO_SLEEP_IDLE_TIMEOUT"]) < 0:
        raise ValueError("Negative idle timeout")
    if float(env["AUTO_SLEEP_IDLE_TIMEOUT"]):
        args += [
            "--auto-sleep-idle-timeout",
            env["AUTO_SLEEP_IDLE_TIMEOUT"],
            "--auto-sleep-offload-target",
            "exit",
        ]
    return args + list(extra)


def fingerprint(env, args):
    selected = {
        k: v
        for k, v in env.items()
        if k in DEFAULTS
        or k.startswith(("VLLM_", "NCCL_", "QWEN_", "CUDA_VISIBLE_DEVICES"))
    }
    return hashlib.sha256(
        json.dumps(
            {"env": selected, "args": args, "source": installer().source_hash()},
            sort_keys=True,
        ).encode()
    ).hexdigest()


class Runtime:
    def __init__(self, env):
        self.env = env
        self.home = Path(env["QWEN_SM75_HOME"])
        self.system = os.geteuid() == 0
        self.ctl = ["systemctl"] if self.system else ["systemctl", "--user"]
        self.unit = env.get("QWEN_SM75_SERVICE", "qwen38-27b-sm75")
        if not self.unit.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid service name")

    def command(self, argv, check=True):
        r = subprocess.run(argv, text=True, capture_output=True, env=self.env)
        if check and r.returncode:
            raise RuntimeError(f"{argv[0]} failed: {r.stderr[-1000:]}")
        return r

    def check_environment(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Native runtime requires Linux x86_64")
        self.command(self.ctl + ["show-environment"])
        receipt = self.home / "installed.json"
        if (
            not receipt.is_file()
            or json.loads(receipt.read_text()).get("source_hash")
            != installer().source_hash()
        ):
            raise RuntimeError("Run the matching native bootstrap before starting")
        r = self.command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = list(csv.reader(io.StringIO(r.stdout)))
        selection = self.env.get("GPUS", "all").strip('"')
        if selection != "all":
            if not selection.startswith("device="):
                raise ValueError("GPUS must be all or device=indices/UUIDs")
            chosen = []
            for device in selection[7:].split(","):
                matches = [
                    row
                    for row in rows
                    if device.strip() in {row[0].strip(), row[1].strip()}
                ]
                if len(matches) != 1 or matches[0] in chosen:
                    raise ValueError("Unknown/duplicate selected GPU")
                chosen.append(matches[0])
            rows = chosen
        if len(rows) < int(self.env["TP_SIZE"]) or any(
            row[2].strip() != "7.5" for row in rows
        ):
            raise RuntimeError("Selected GPUs do not satisfy SM75 TP configuration")
        self.env["CUDA_VISIBLE_DEVICES"] = ",".join(row[1].strip() for row in rows)

    def running(self):
        return (
            self.command(
                self.ctl + ["is-active", self.unit], check=False
            ).stdout.strip()
            == "active"
        )

    def launch(self, args, value):
        state = self.home / "run"
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        native = installer().native_env(self.home)
        # Ignore another Python/CUDA environment. Only public engine knobs
        # and the private API key are forwarded from the caller.
        names = set(DEFAULTS) | {
            "VLLM_API_KEY",
            "MODEL_REVISION",
            "CUDA_VISIBLE_DEVICES",
        }
        native.update(
            {
                k: v
                for k, v in self.env.items()
                if k in names or k.startswith(("VLLM_", "NCCL_", "AUTO_SLEEP_"))
            }
        )
        cache = self.home / "cache"
        native.update(
            VLLM_USE_MODELSCOPE="true"
            if self.env["MODEL_PROVIDER"] == "modelscope"
            else "false",
            MODELSCOPE_CACHE=str(cache / "modelscope"),
            HF_HOME=str(cache / "huggingface"),
            VLLM_CACHE_ROOT=str(cache / "vllm"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
            VLLM_GDN_DECODE_KERNEL="triton",
            VLLM_USE_FLASHINFER_SAMPLER="0",
            VLLM_FIREFLY="1",
            VLLM_FIREFLY_AR="0" if self.env["COMM_BACKEND"] == "nccl" else "auto",
            VLLM_FIREFLY_AR_CHUNK_BYTES="16777216",
            NCCL_P2P_DISABLE="1" if self.env["P2P_MODE"] == "off" else "0",
            VLLM_FIREFLY_AR_BACKEND="shm" if self.env["P2P_MODE"] == "off" else "auto",
            NCCL_IB_DISABLE="1",
            VLLM_USE_NCCL_SYMM_MEM="0",
            VLLM_ALLREDUCE_USE_SYMM_MEM="0",
            FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH=str(
                self.home / "extensions/flash_qla_sm75_gdn.so"
            ),
            FLASH_QLA_SM75_ALLOW_JIT="0",
            SM75_FA2_SMALLQ_GRAPH="1",
            SM75_GDN_B1_METADATA="1",
        )
        config = state / "server.json"
        fd = os.open(
            config,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(fd, "w") as f:
            json.dump({"env": native, "args": args, "fingerprint": value}, f)
        os.chmod(config, 0o600)
        directory = (
            Path("/etc/systemd/system")
            if self.system
            else Path.home() / ".config/systemd/user"
        )
        directory.mkdir(parents=True, exist_ok=True)

        def quote(value):
            return (
                '"'
                + str(value)
                .replace("%", "%%")
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                + '"'
            )

        executable = (
            quote(self.home / "venv/bin/python")
            + " -I "
            + quote(ROOT / "runtime/run-server.py")
            + " "
            + quote(config)
        )
        unit = f"[Unit]\nDescription=Native Qwen3.8 27B SM75 API\n\n[Service]\nType=simple\nExecStart={executable}\nRestart=no\nKillMode=control-group\nTimeoutStopSec=60\nMemoryMax={self.env['NATIVE_MEMORY_MAX']}\nMemorySwapMax=0\n\n[Install]\nWantedBy={'multi-user.target' if self.system else 'default.target'}\n"
        (directory / (self.unit + ".service")).write_text(unit)
        self.command(self.ctl + ["daemon-reload"])
        self.command(self.ctl + ["start", self.unit])
        print(
            f"Native service started: {self.unit}; inspect logs with journalctl {'--user ' if not self.system else ''}-u {self.unit} -f"
        )

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.env['PORT']}/v1/models",
            headers={"Authorization": "Bearer " + self.env["VLLM_API_KEY"]},
        )
        while time.monotonic() < deadline:
            if not self.running():
                raise RuntimeError(
                    "Native service stopped before readiness; inspect journal logs"
                )
            try:
                with urllib.request.urlopen(req, timeout=3) as response:
                    if any(
                        x.get("id") == self.env["SERVED_MODEL_NAME"]
                        for x in json.load(response).get("data", [])
                    ):
                        print(
                            f"API ready: http://127.0.0.1:{self.env['PORT']}/v1; generation not tested"
                        )
                        return
            except (OSError, ValueError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError("Readiness timed out; native service retained")


def start(env, runtime, extra, key_directory, dry_run=False, wait=True, timeout=3600):
    args = profile(env, extra)
    if dry_run:
        print(
            json.dumps(
                {
                    "deployment": "native venv",
                    "model": env["MODEL"],
                    "home": env["QWEN_SM75_HOME"],
                    "requires_docker": False,
                    "credential_or_gpu_side_effects": False,
                    "extra_argument_count": len(extra),
                },
                indent=2,
            )
        )
        return
    runtime.check_environment()
    key, path = api_key(env, key_directory)
    env["VLLM_API_KEY"] = key
    if path:
        print(f"Private API key file: {path}; contents not printed")
    value = fingerprint(env, args)
    if runtime.running():
        config = Path(env["QWEN_SM75_HOME"]) / "run/server.json"
        if (
            not config.is_file()
            or json.loads(config.read_text()).get("fingerprint") != value
        ):
            raise RuntimeError(
                "Existing native service has different configuration; it was not stopped or replaced"
            )
        print("Reusing matching native service")
    else:
        runtime.launch(args, value)
    if wait:
        runtime.wait_ready(timeout)
    else:
        print("Startup submitted; API readiness not checked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=3600)
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.wait_timeout <= 0:
        parser.error("--wait-timeout must be positive")
    env = {**DEFAULTS, **os.environ}
    env.setdefault(
        "QWEN_SM75_HOME", str(Path.home() / ".local/share/qwen-3.8-27b-vllm-sm75")
    )
    env["QWEN_SM75_HOME"] = str(Path(env["QWEN_SM75_HOME"]).expanduser().resolve())
    args = (
        options.engine_args[1:]
        if options.engine_args[:1] == ["--"]
        else options.engine_args
    )
    try:
        start(
            env,
            Runtime(env),
            args,
            Path(env["QWEN_SM75_HOME"]) / "private",
            options.dry_run,
            not options.no_wait,
            options.wait_timeout,
        )
    except (RuntimeError, OSError, ValueError) as e:
        parser.exit(1, f"Native startup stopped: {e}\n")


if __name__ == "__main__":
    main()
