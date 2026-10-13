#!/usr/bin/env python3
"""Prepare a private API key, build if needed, and start the standard API."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import secrets
import shlex
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"VARIANT": "base", "FORMAT": "fp8", "MODEL": "Qwen/Qwen3.8-27B-FP8",
            "MODEL_PROVIDER": "modelscope", "TP_SIZE": "4", "POWER_MODE": "sleep",
            "COMM_BACKEND": "nccl", "P2P_MODE": "auto", "PORT": "8000",
            "CONTAINER_NAME": "qwen38-27b-sm75", "SERVED_MODEL_NAME": "VLLM-Qwen3.8-27B"}


def api_key(env, directory):
    if env.get("VLLM_API_KEY"):
        return env["VLLM_API_KEY"], None
    directory = Path(directory).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "api-key"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise RuntimeError(f"API key must be an owner-only regular file: {path}")
        key = path.read_text().strip()
        if not key:
            raise RuntimeError(f"API key file is empty: {path}")
        return key, path
    with os.fdopen(fd, "w") as stream:
        key = secrets.token_hex(32)
        stream.write(key + "\n")
    return key, path


def fingerprint(env, image_id, args):
    # Include all launcher knobs, including defaults from the current scripts.
    # Do not print this payload: it includes the API key for mismatch detection.
    names = set(DEFAULTS) | {"IMAGE", "MODEL_ROOT", "MODEL_REVISION", "GPUS", "BUILD_MODE", "EDITION",
                            "DRAFT_MODEL", "HF_OVERRIDES", "CPU_KV_GIB", "GPU_KV_BYTES", "CONTAINER_MEMORY",
                            "SHM_SIZE", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION"}
    selected = {k: v for k, v in env.items() if k in names or k.startswith(("VLLM_", "NCCL_", "PSTATE_", "AUTO_SLEEP_", "NVCC_"))}
    source = hashlib.sha256((ROOT / "docker/run.sh").read_bytes()).hexdigest()
    payload = json.dumps({"env": selected, "image": image_id, "args": args, "launcher": source}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class Runtime:
    def __init__(self, env):
        self.env = env

    def command(self, argv, check=True):
        result = subprocess.run(argv, env=self.env, text=True, capture_output=True)
        if check and result.returncode:
            # Do not print process environments or native API arguments.
            raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()[-1500:]}")
        return result

    def check_environment(self):
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
            raise RuntimeError("Run on a Linux x86_64 GPU host; --dry-run works without that host")
        for name in ["docker", "nvidia-smi", "bash"]:
            if not shutil.which(name):
                raise RuntimeError(f"Missing {name}; install the host prerequisites first")
        info = json.loads(self.command(["docker", "info", "--format", "{{json .Runtimes}}"] ).stdout)
        if "nvidia" not in info:
            raise RuntimeError("Docker NVIDIA runtime is missing; configure NVIDIA Container Toolkit first")
        query = "index,uuid,compute_cap,memory.total"
        rows = list(csv.reader(io.StringIO(self.command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]).stdout)))
        selection = self.env.get("GPUS", "all").strip('"')
        if selection != "all":
            if not selection.startswith("device="):
                raise RuntimeError('GPUS must be all or an explicit device selection, e.g. GPUS=\'"device=0,1,2,3"\'')
            requested = selection[len("device="):].split(",")
            selected = []
            for device in requested:
                matches = [row for row in rows if device.strip() in {row[0].strip(), row[1].strip()}]
                if len(matches) != 1 or matches[0] in selected:
                    raise RuntimeError("GPU selection contains an unknown or duplicate index/UUID")
                selected.append(matches[0])
            rows = selected
        tp = int(self.env["TP_SIZE"])
        if tp not in {1, 2, 4, 8}:
            raise RuntimeError("TP_SIZE must be 1, 2, 4, or 8")
        if len(rows) < tp or any(row[2].strip() != "7.5" for row in rows):
            raise RuntimeError(f"Need at least {tp} visible SM75 GPUs; inspect GPU selection before starting")

    def image(self):
        r = self.command(["docker", "image", "inspect", self.env["IMAGE"], "--format", "{{.Id}}"], check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def build(self):
        if self.env.get("BUILD_MODE", "full") != "full":
            raise RuntimeError("One-click serving requires a full image, not the fast development base")
        # An existing/imported image skips this source-build budget check.
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
        if available < 8 * 1024**3 or shutil.disk_usage(ROOT).free < 200 * 1024**3:
            raise RuntimeError("Source build requires 8GiB available RAM and 200GiB free space. Build on another host and import the image; NUC9 should reuse an image")
        print("Building the standard image (MAX_JOBS defaults to 1)...", flush=True)
        subprocess.run(["bash", str(ROOT / "docker/build.sh")], env=self.env, check=True)

    def container(self):
        r = self.command(["docker", "container", "inspect", self.env["CONTAINER_NAME"]], check=False)
        return json.loads(r.stdout)[0] if r.returncode == 0 else None

    def launch(self, args):
        print("Starting ModelScope FP8 API; missing weights download automatically...", flush=True)
        subprocess.run(["bash", str(ROOT / "docker/run.sh"), *args], env=self.env, check=True)

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        port = self.env["PORT"]
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer " + self.env["VLLM_API_KEY"]})
        expected = self.env.get("SERVED_MODEL_NAME", "VLLM-Qwen3.8-27B")
        while time.monotonic() < deadline:
            container = self.container()
            if not container or not container["State"].get("Running"):
                raise RuntimeError("Container exited before API readiness; inspect its logs with docker logs")
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    data = json.load(response)
                    if any(model.get("id") == expected for model in data.get("data", [])):
                        print(f"API ready: http://127.0.0.1:{port}/v1 (model listing verified; generation not tested)")
                        return
            except (OSError, ValueError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError("API readiness timed out. Container retained; inspect logs, then rerun the same command")


def start(env, runtime, args, key_directory, dry_run=False, rebuild=False, wait=True, timeout=3600):
    reserved = {"--api-key", "--host", "--port", "--served-model-name", "--tensor-parallel-size"}
    if any(arg.split("=", 1)[0] in reserved for arg in args):
        raise RuntimeError("Set API identity, port and TP through environment variables, not extra native arguments")
    if dry_run:
        print(json.dumps({"image": env["IMAGE"], "model": env["MODEL"], "provider": env["MODEL_PROVIDER"],
                          "container": env["CONTAINER_NAME"], "api": f"http://127.0.0.1:{env['PORT']}/v1",
                          "plan": ["check Linux, Docker/NVIDIA runtime and SM75 GPUs", "reuse or generate private API key",
                                   "reuse full image or build with resource checks", "reject conflicting existing container",
                                   "start API and auto-download missing weights", "wait for authenticated model listing"],
                          "launch": shlex.join(["bash", str(ROOT / "docker/run.sh")]),
                          "extra_engine_argument_count": len(args),
                          "dry_run": "no commands, credentials, download or GPU work"}, indent=2))
        return
    if env.get("EDITION", "standard") != "standard" or env.get("BUILD_MODE", "full") != "full":
        raise RuntimeError("One-click serving requires EDITION=standard and BUILD_MODE=full")
    runtime.check_environment()
    key, path = api_key(env, key_directory)
    env["VLLM_API_KEY"] = key
    if path:
        print(f"API key file: {path} (contents are not printed)")
    image_id = runtime.image()
    if rebuild or not image_id:
        if runtime.container():
            raise RuntimeError("Existing container found; refusing to rebuild underneath it")
        runtime.build()
        image_id = runtime.image()
        if not image_id:
            raise RuntimeError("Build did not produce the requested image")
    desired = fingerprint(env, image_id, args)
    env["QWEN_SM75_LAUNCH_FINGERPRINT"] = desired
    existing = runtime.container()
    if existing:
        labels = existing["Config"].get("Labels") or {}
        if labels.get("io.qwen-sm75.launch-fingerprint") != desired:
            raise RuntimeError("Existing container has different configuration; it was not stopped or replaced")
        if not existing["State"].get("Running"):
            raise RuntimeError("Existing container is stopped; inspect its logs and remove or restart it explicitly")
        print("Reusing the matching running container.", flush=True)
    else:
        runtime.launch(args)
    if wait:
        runtime.wait_ready(timeout)
    else:
        print("Startup submitted; API readiness has not been checked.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=3600)
    parser.add_argument("engine_args", nargs=argparse.REMAINDER, help="Pass native vLLM args after --")
    options = parser.parse_args()
    if options.wait_timeout <= 0:
        parser.error("--wait-timeout must be positive")
    env = {**DEFAULTS, **os.environ}
    version = (ROOT / "docker/VERSION").read_text().strip()
    env.setdefault("IMAGE", f"vllm-sm75:v{version}")
    key_directory = env.get("QWEN_SM75_CONFIG_DIR", str(Path.home() / ".config/qwen-3.8-27b-vllm-sm75"))
    args = options.engine_args[1:] if options.engine_args[:1] == ["--"] else options.engine_args
    try:
        start(env, Runtime(env), args, key_directory, options.dry_run, options.rebuild, not options.no_wait, options.wait_timeout)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Startup stopped: {error}\n")


if __name__ == "__main__":
    main()
