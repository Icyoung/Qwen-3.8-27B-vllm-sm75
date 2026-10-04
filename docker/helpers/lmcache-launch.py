"""Optional same-container LMCache MP lifecycle for the standard image.

Uses the Ultra install/probe contract; imports no GPU package in the supervisor.
The standard-v1 namespace deliberately cannot reuse an Ultra cache directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

PREFIX = "SM75_LMCACHE_PROBE="
PROBE = "/opt/vllm-sm75/console/lmcache_probe.py"
PSTATE = "/opt/vllm-sm75/pstate-supervisor.sh"
GIB = 2 ** 30
CACHE_ENV = {"LMCACHE_TRACK_USAGE": "false", "LMCACHE_REQUEST_TELEMETRY_TYPE": "noop"}


def arg_value(args, flag, default=None):
    value = default
    for i, arg in enumerate(args):
        if arg == flag:
            value = args[i + 1] if i + 1 < len(args) else None
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def has_flag(args, flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in args)


def set_arg(args, flag, value, boolean=False):
    result, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg == flag:
            skip = not boolean
        elif not arg.startswith(flag + "="):
            result.append(arg)
    if value is not None:
        result.append(flag)
        if value is not True:
            result.append(str(value))
    return result


def env_number(env, key, default=None, *, integer=False, zero=False):
    raw = env.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a {'non-negative' if zero else 'positive'} number") from None
    if (not math.isfinite(value) or (value < 0 if zero else value <= 0)
            or (integer and not value.is_integer()) or value * GIB > 2 ** 53 - 1):
        raise ValueError(f"Invalid {key}")
    return int(value) if integer else value


def configuration(args, env):
    if not args or not args[0] or args[0].startswith("--"):
        raise ValueError("LMCache requires the model as the first argument")
    if env.get("POWER_MODE", "pstate") != "pstate":
        raise ValueError("LMCache requires POWER_MODE=pstate")
    if float(env.get("CPU_KV_GIB", "0")) > 0 or env.get("KV_TRANSFER_CONFIG"):
        raise ValueError("LMCache conflicts with explicit native CPU KV/custom KV_TRANSFER_CONFIG")
    for flag in ("--kv-transfer-config", "--kv-offloading-size", "--kv-offloading-backend",
                 "--speculative-config", "--enable-sleep-mode", "--enable-cumem-allocator",
                 "--disable-hybrid-kv-cache-manager"):
        if has_flag(args, flag):
            raise ValueError(f"LMCache conflicts with {flag}; remove it explicitly")
    if (float(arg_value(args, "--auto-sleep-idle-timeout", "0")) > 0
            or float(env.get("VLLM_AUTO_SLEEP_IDLE_TIMEOUT", "0")) > 0
            or float(env.get("AUTO_SLEEP_IDLE_TIMEOUT", "0")) > 0):
        raise ValueError("LMCache requires automatic sleep to be disabled")
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if re.search(r"expandable_segments\s*:\s*true", env.get(key, ""), re.I):
            raise ValueError("LMCache does not support expandable_segments")
    c = {
        "cpuGiB": env_number(env, "LMCACHE_CPU_GIB", "8"),
        "chunkSize": env_number(env, "LMCACHE_CHUNK_SIZE", integer=True),
        "minFreeDiskGiB": env_number(env, "LMCACHE_MIN_FREE_DISK_GIB", "4", zero=True),
        "port": env_number(env, "LMCACHE_PORT", "5555", integer=True),
        "httpPort": env_number(env, "LMCACHE_HTTP_PORT", "5556", integer=True),
        "diskPath": env.get("LMCACHE_DISK_PATH", ""),
    }
    directory = c["diskPath"]
    if directory and (not directory.startswith("/") or not directory.strip("/")
                      or ".." in directory.split("/") or re.search(r"[\x00-\x1f\x7f]", directory)):
        raise ValueError("LMCACHE_DISK_PATH must be an absolute Linux subdirectory")
    ports = [c["port"], c["httpPort"]]
    if (any(port < 1024 or port > 65535 for port in ports) or ports[0] == ports[1]
            or int(arg_value(args, "--port", "8000")) in ports):
        raise ValueError("LMCache RPC, HTTP and model ports must be distinct (1024-65535)")
    return c


def plan(args, env, config, runtime_fingerprint=""):
    connector = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_connector_extra_config": {"lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": config["port"], "lmcache.mp.isolated_ipc": False}}
    engine_args = set_arg(args, "--no-enable-prefix-caching", None, boolean=True)
    engine_args = set_arg(engine_args, "--enable-prefix-caching", True, boolean=True)
    engine_args = set_arg(engine_args, "--mamba-cache-mode", "align")
    engine_args = set_arg(engine_args, "--kv-transfer-config", json.dumps(connector, separators=(",", ":")))
    identity = []
    model = Path(args[0])
    if model.is_dir():
        for file in sorted(model.iterdir()):
            if file.is_file() and file.suffix in {".json", ".safetensors", ".bin"}:
                stat = file.stat()
                identity.append([file.name, stat.st_size, stat.st_mtime_ns,
                                 file.read_text(encoding="utf-8") if file.name == "config.json" else None])
    if config["diskPath"] and not identity:
        raise ValueError("Persistent LMCache requires a local model directory with identifiable weights/config")
    layout_env = {key: value for key, value in sorted(env.items())
        if re.match(r"^(VLLM_|LMCACHE_|FLASH|SM75_|PYTORCH_|TORCH_|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)", key)
        and not re.search(r"KEY|TOKEN|PASSWORD|SECRET", key, re.I)}
    namespace = "standard-v1-" + hashlib.sha256(json.dumps({
        "schema": "sm75-standard-lmcache-v1", "lmcache": "0.5.5", "vllm": "0.30.0",
        "args": engine_args, "cacheLayout": {"chunkSize": config["chunkSize"]},
        "env": layout_env, "modelIdentity": identity,
        "runtimeFingerprint": runtime_fingerprint,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    disk_path = str(Path(config["diskPath"]) / namespace) if config["diskPath"] else ""
    server_args = ["-m", "lmcache.v1.multiprocess.http_server", "--host", "127.0.0.1",
        "--port", str(config["port"]), "--http-host", "127.0.0.1", "--http-port", str(config["httpPort"]),
        "--chunk-size", str(config["chunkSize"]), "--separate-object-groups", "--no-isolated-ipc",
        "--l1-size-gb", str(config["cpuGiB"]), "--l1-init-size-gb", "1", "--eviction-policy", "LRU"]
    if disk_path:
        server_args += ["--l2-adapter", json.dumps({"type": "fs", "base_path": disk_path, "use_odirect": False})]
    return {"engineArgs": engine_args, "serverArgs": server_args, "diskPath": disk_path,
            "namespace": namespace, "healthURL": f"http://127.0.0.1:{config['httpPort']}/healthcheck"}


def check_disk(config, active_plan, *, prepare=False):
    if not active_plan["diskPath"]:
        return
    parent = Path(config["diskPath"])
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError("LMCache disk directory is not writable")
    if shutil.disk_usage(parent).free < config["minFreeDiskGiB"] * GIB:
        raise ValueError("LMCache disk free space is below LMCACHE_MIN_FREE_DISK_GIB")
    if prepare:
        directory = Path(active_plan["diskPath"])
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        check = directory / (".sm75-write-check-" + uuid.uuid4().hex)
        with check.open("xb"):
            pass
        check.unlink()


def assert_free_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"LMCache port {port} is occupied; refusing an unknown server") from None


def healthy(url):
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status == 200 and json.load(response).get("status") == "healthy"
    except (OSError, ValueError):
        return False


def terminate_process(child, *, root_only=False):
    # Each child owns a process group. Reap descendants even if the leader exits.
    def signal_group(sig):
        try:
            os.killpg(child.pid, sig)
            return True
        except ProcessLookupError:
            return False
    if root_only and child.poll() is None:
        child.terminate()  # P-State supervisor restores its daemon with SIGINT.
    else:
        signal_group(signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    signal_group(signal.SIGKILL)
    child.wait()


class Launcher:
    def __init__(self, *, env=None, popen=subprocess.Popen, terminate=terminate_process,
                 health=healthy, port_check=assert_free_port, disk_check=check_disk,
                 clock=time.monotonic, sleep=time.sleep, timeout=45, probe=None):
        self.env = {**(os.environ if env is None else env), **CACHE_ENV}
        self.popen, self.terminate, self.health = popen, terminate, health
        self.port_check, self.disk_check, self.clock, self.sleep = port_check, disk_check, clock, sleep
        self.timeout, self.probe = timeout, probe or self.run_probe
        self.cancelled = False
        self.children = {}

    def cancel(self, *_):
        self.cancelled = True

    def check_cancelled(self):
        if self.cancelled:
            raise RuntimeError("LMCache startup/run cancelled")

    def launch(self, role, command, **kwargs):
        self.check_cancelled()
        child = self.popen(command, env=self.env, start_new_session=True, **kwargs)
        self.children[role] = child
        self.check_cancelled()
        return child

    def run_probe(self, active_plan):
        # Reuse the exact Ultra ABI/patch/real-CLI/runtime-fingerprint probe.
        payload = json.dumps({"serverArgs": active_plan["serverArgs"][2:]})
        child = self.launch("probe", [sys.executable, PROBE, payload],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = self.clock() + self.timeout
        try:
            while self.clock() < deadline:
                self.check_cancelled()
                try:
                    stdout, _ = child.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
            else:
                raise RuntimeError("LMCache dependency probe timed out")
            self.check_cancelled()
            line = next((line[len(PREFIX):] for line in stdout.splitlines() if line.startswith(PREFIX)), "{}")
            result = json.loads(line)
            if child.returncode or result.get("ok") is not True or not result.get("runtimeFingerprint"):
                raise RuntimeError("LMCache dependency probe failed: " + str(result.get("error", "no result"))[:1200])
            return result
        finally:
            self.terminate(child)
            self.children.pop("probe", None)

    def run(self, args):
        try:
            config = configuration(args, self.env)
            provisional = plan(args, self.env, config)
            self.disk_check(config, provisional)
            for port in (config["port"], config["httpPort"]):
                self.port_check(port)
            result = self.probe(provisional)
            self.check_cancelled()
            if not result.get("runtimeFingerprint"):
                raise RuntimeError("LMCache dependency probe omitted runtime fingerprint")
            active_plan = plan(args, self.env, config, result["runtimeFingerprint"])
            # Check again after the slow dependency probe, before creating children.
            for port in (config["port"], config["httpPort"]):
                self.port_check(port)
            self.disk_check(config, active_plan, prepare=True)
            cache = self.launch("cache", [sys.executable, *active_plan["serverArgs"]])
            deadline = self.clock() + self.timeout
            while True:
                self.check_cancelled()
                if cache.poll() is not None:
                    raise RuntimeError("LMCache server exited before becoming ready")
                ready = self.health(active_plan["healthURL"])
                self.check_cancelled()  # A late successful response cannot undo cancellation.
                if cache.poll() is not None:
                    raise RuntimeError("LMCache server exited while checking readiness")
                if ready:
                    break
                if self.clock() >= deadline:
                    raise RuntimeError("LMCache server did not become healthy before timeout")
                self.sleep(0.2)
            pstate = self.launch("pstate", ["bash", PSTATE])
            if pstate.poll() is not None:
                raise RuntimeError("P-State supervisor exited during startup")
            engine = self.launch("engine", ["vllm", "serve", *active_plan["engineArgs"]])
            while True:
                self.check_cancelled()
                code = engine.poll()
                if code is not None:
                    return code if code >= 0 else 128 - code
                if cache.poll() is not None or pstate.poll() is not None:
                    raise RuntimeError("LMCache or P-State service exited while model was running")
                self.sleep(0.2)
        finally:
            # Release model/CUDA IPC first; never remove any user disk cache.
            cleanup_errors = []
            for role in ("engine", "pstate", "cache", "probe"):
                child = self.children.pop(role, None)
                if child is not None:
                    try:
                        self.terminate(child, root_only=role == "pstate")
                    except Exception as error:
                        cleanup_errors.append(error)
            if cleanup_errors and sys.exc_info()[0] is None:
                raise RuntimeError("LMCache process cleanup failed") from cleanup_errors[0]


def main():
    if sys.platform != "linux":
        raise RuntimeError("LMCache standard entrypoint requires Linux CUDA")
    launcher = Launcher()
    signal.signal(signal.SIGTERM, launcher.cancel)
    signal.signal(signal.SIGINT, launcher.cancel)
    return launcher.run(sys.argv[1:])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as error:
        print("[lmcache] " + str(error), file=sys.stderr)
        sys.exit(1)
