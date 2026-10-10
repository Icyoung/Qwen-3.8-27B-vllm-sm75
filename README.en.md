# vLLM-SM75 v0.1.7 beta2

[中文](README.md) · [Release notes](docs/releases/v0.1.7-beta2.md) · [Model profiles](docs/configuration-v0.1.7.md) · [BenchLocal](docs/benchlocal.md)

## 1. Overview

vLLM 0.30.0 adapted for NVIDIA Turing / SM75, including Tesla T4 and T10. Supports Qwen3.8 27B FP8 / AWQ / NVFP4 and Flash Next W4A16, with GDN, FP8 KV, MTP, DFlash2, DSpark, Firefly, CPU KV, power management and monitoring.

Use one image: **`vllm-sm75:v0.1.7-ultra-beta2`**. Ultra includes the console, chat workspace and model tests. Set `SM75_EDITION=standard` to run only the inference API using the same model profiles and engine.

QQ community: **878924874**.

## 2. Updates

- DSpark k5 profiles for 27B FP8 / NVFP4, draft KV grouping and graph capture support.
- Prefix cache configuration, CPU KV shared-memory checks and LMCache configuration hints.
- Nine native BenchLocal suites, all execution dependencies in the Ultra image, generation settings, per-case results, radar charts and portable reports.
- Model/profile launch improvements, profile deletion, API address/key copying, monitoring and UI fixes.

## 3. Standard mode and Ultra

Both use the same inference engine, model profiles, KV settings and power controls. Standard mode exposes port 8000; Ultra adds the management UI on port 1615, Harness workspace, Watcher usage insights and integrated model tests.

## 4. Performance reference

Existing measurements and conditions are available for [27B](docs/validation/v0.1.6-full-images.md) and [Flash Next](docs/flash-next-performance.md). The overview table in the [Chinese README](README.md#4-性能参考) identifies the measured versions.

## 5. Build and start

Requires Linux x86_64, Docker / BuildKit, Git, and an NVIDIA driver plus NVIDIA Container Toolkit on the runtime host. Allow access to the pinned upstream package/download services and about 200 GiB of build storage.

```bash
git clone --branch v0.1.7-beta2 --single-branch https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
EDITION=ultra MAX_JOBS=1 bash docker/build.sh

export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export MODEL_ROOT=/srv/models
export VLLM_API_KEY='replace-with-your-own-long-random-key'
export PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1  # actual host path
EDITION=ultra bash docker/run.sh
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

The full build produces one Ultra image containing all nine BenchLocal suites. The startup script reuses that image for resource-limited CPU execution containers; no additional verifier images are needed.

Open `http://<server>:1615`, register a model and select its profile. For 27B FP8 with DSpark, use `SM75_PROFILE=qwen38-27b-fp8-dspark5` and set `MODEL` / `DRAFT_MODEL`. CPU KV 16 GiB uses `CPU_KV_GIB=16` and `SHM_SIZE=32g`.

Keep existing `/data`, cache, `/dsh/home` and `/dsh/workspace` mounts when upgrading. See [build details](docker/BUILD.md), [model profiles](docs/configuration-v0.1.7.md) and [BenchLocal setup](docs/benchlocal.md).
