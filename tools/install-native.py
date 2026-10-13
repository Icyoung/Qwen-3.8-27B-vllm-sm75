#!/usr/bin/env python3
"""Install only into owned private prefixes, never into system Python."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    digest = hashlib.sha256()
    for directory in [ROOT / "runtime", ROOT / "vllm"]:
        for p in sorted(directory.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                digest.update(str(p.relative_to(ROOT)).encode())
                digest.update(p.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def download(record, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(record["url"], timeout=60) as response,
        temp.open("wb") as stream,
    ):
        for data in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(data)
            digest.update(data)
    if digest.hexdigest() != record["sha256"]:
        temp.unlink()
        raise RuntimeError("Downloaded tool SHA-256 mismatch")
    temp.replace(output)


def commands(home, lock):
    home = Path(home)
    py = str(home / "venv/bin/python")
    uv = str(home / "bootstrap/bin/uv")
    constraints = str(ROOT / "runtime/constraints.txt")
    wheel = lock["vllm_wheel"]
    pip = [
        uv,
        "pip",
        "install",
        "--python",
        py,
        "--constraint",
        constraints,
        "--index-strategy",
        "unsafe-best-match",
    ]
    compiler = [
        str(home / "tools/micromamba"),
        "install" if (home / "cuda/conda-meta").is_dir() else "create",
        "--yes",
        "--prefix",
        str(home / "cuda"),
        "--channel",
        "nvidia/label/cuda-12.9.1",
        "--channel",
        "conda-forge",
        "--strict-channel-priority",
        *lock["cuda_specs"],
    ]
    return [
        [
            uv,
            "venv",
            "--python",
            "3.12",
            str(home / "venv"),
            "--seed",
            "--allow-existing",
        ],
        compiler,
        pip
        + [
            wheel["url"] + "#sha256=" + wheel["sha256"],
            "torch==2.13.0+cu129",
            "triton==3.7.1",
            "transformers==5.15.1",
            "modelscope",
            "psutil",
            "ninja",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu129",
        ],
        [
            uv,
            "pip",
            "uninstall",
            "--python",
            py,
            "flashinfer-jit-cache",
            "nvidia-cudnn-cu13",
            "nvidia-cusparselt-cu13",
            "nvidia-nccl-cu13",
            "nvidia-nvshmem-cu13",
        ],
        pip
        + [
            "--no-deps",
            "--reinstall",
            "flashinfer-python==0.6.18",
            "flashinfer-cubin==0.6.18",
            "nvidia-cudnn-cu12==9.20.0.48",
            "nvidia-cusparselt-cu12==0.8.1",
            "nvidia-nccl-cu12==2.30.7",
            "nvidia-nvshmem-cu12==3.4.5",
            "cuda-python==12.9.7",
            "cuda-bindings==12.9.7",
            "--extra-index-url",
            "https://flashinfer.ai/whl/",
        ],
        [py, str(ROOT / "runtime/helpers/verify_cuda_libraries.py")],
        [py, str(ROOT / "runtime/helpers/patch_transformers_startup.py")],
        [
            py,
            str(ROOT / "runtime/helpers/install_sm75_overlay.py"),
            str(ROOT / "vllm"),
            "--source-copy",
            str(home / "source/vllm"),
        ],
        [py, str(ROOT / "runtime/helpers/install_speculative.py")],
        [
            py,
            str(ROOT / "vllm/third_party/flash_qla_sm75/build_extension.py"),
            "--build-directory",
            str(home / "build/flashqla"),
            "--output",
            str(home / "extensions/flash_qla_sm75_gdn.so"),
        ],
        [py, str(ROOT / "runtime/helpers/verify_sm75_image.py")],
    ]


def native_env(home):
    home = Path(home)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    env.update(
        QWEN_SM75_HOME=str(home),
        CUDA_HOME=str(home / "cuda"),
        MAMBA_ROOT_PREFIX=str(home / "mamba"),
        PYTHONNOUSERSITE="1",
        MAX_JOBS="1",
        TORCH_CUDA_ARCH_LIST="7.5",
        NVCC_THREADS="1",
        UV_PYTHON_INSTALL_DIR=str(home / "python"),
        UV_CACHE_DIR=str(home / "cache/uv"),
        CC=str(home / "cuda/bin/x86_64-conda-linux-gnu-cc"),
        CXX=str(home / "cuda/bin/x86_64-conda-linux-gnu-c++"),
    )
    env["PATH"] = f"{home}/venv/bin:{home}/cuda/bin:" + os.environ.get("PATH", "")
    libraries = [home / "venv/lib/python3.12/site-packages/torch/lib"]
    libraries += sorted(
        (home / "venv/lib/python3.12/site-packages/nvidia").glob("*/lib")
    )
    libraries += [home / "cuda/lib", home / "cuda/targets/x86_64-linux/lib"]
    # Never add CUDA stub libraries or inherit another environment's libraries.
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in libraries if p.is_dir())
    return env


def main():
    home = Path(os.environ["QWEN_SM75_HOME"]).resolve()
    home.mkdir(parents=True, exist_ok=True)
    with (home / "install.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = source_hash()
        receipt = home / "installed.json"
        if (
            receipt.exists()
            and json.loads(receipt.read_text()).get("source_hash") == expected
        ):
            if (
                not (home / "venv/bin/python").is_file()
                or not (home / "extensions/flash_qla_sm75_gdn.so").is_file()
            ):
                raise RuntimeError(
                    "Native environment is incomplete; preserve it and repair before starting"
                )
            print("Reusing the matching native environment.")
            return
        progress = home / "install-progress.json"
        completed = 0
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved.get("source_hash") != expected:
                raise RuntimeError(
                    "Existing native install differs from source; use a new QWEN_SM75_HOME"
                )
            completed = saved["completed"]
        elif (home / "venv").exists() or (home / "cuda").exists():
            raise RuntimeError("Refusing to modify an unowned Python/CUDA prefix")
        if shutil.disk_usage(home).free < 60 * 1024**3:
            raise RuntimeError(
                "Need 60GiB free space for runtime, downloads and the default FP8 model"
            )
        lock = json.loads((ROOT / "runtime/native-lock.json").read_text())
        progress.write_text(
            json.dumps({"source_hash": expected, "completed": completed}) + "\n"
        )
        archive = home / "downloads/micromamba.tar.bz2"
        download(lock["micromamba"], archive)
        (home / "tools").mkdir(exist_ok=True)
        with tarfile.open(archive) as source:
            entry = source.getmember("bin/micromamba")
            if not entry.isfile():
                raise RuntimeError("Invalid micromamba executable in pinned archive")
            with (
                source.extractfile(entry) as src,
                (home / "tools/micromamba").open("wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        (home / "tools/micromamba").chmod(0o755)
        for index, command in enumerate(commands(home, lock)):
            if index < completed:
                continue
            print(f"Native setup stage {index + 1}/11", flush=True)
            # Refresh paths after each stage, since pip/compiler populate dirs.
            subprocess.run(command, env=native_env(home), check=True)
            temp = progress.with_suffix(".tmp")
            temp.write_text(
                json.dumps({"source_hash": expected, "completed": index + 1}) + "\n"
            )
            temp.replace(progress)
        receipt.write_text(
            json.dumps(
                {
                    "source_hash": expected,
                    "venv": str(home / "venv"),
                    "python": lock["python"],
                    "gpu_inference_validated": False,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
