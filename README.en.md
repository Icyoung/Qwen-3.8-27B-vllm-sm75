# Qwen-3.8-27B-vllm-sm75

A standard-only SM75 inference fork based on [fishensw/VLLM-SM75](https://github.com/fishensw/VLLM-SM75/tree/ecee5c70ee4e8cbd915edc49340d07fa5f5545ad). Ultra, Harness, and Node/npm build inputs have been removed. OpenAI serving, GDN, FP8/AWQ, FP8 KV, speculation, monitoring, and engine exit sleep remain. Licenses are preserved.

This is a **source-only, GPU-unqualified candidate**. No images, inference, or P2P benchmarks have been run for this fork. Historical upstream reports are not validation of these changes.

```bash
git clone https://github.com/Icyoung/Qwen-3.8-27B-vllm-sm75.git
cd Qwen-3.8-27B-vllm-sm75
MAX_JOBS=1 bash docker/build.sh
# Optional download only; no GPUs or inference process.
bash docker/download.sh
export VLLM_API_KEY='replace-with-your-api-key'
bash docker/run.sh
```

The default is official `Qwen/Qwen3.8-27B-FP8` from ModelScope. Missing files download automatically; download-only and serving share the persistent provider cache. The four-T10, 8GB-host candidate uses TP4, one sequence, 32K target context, 512-token prefill chunks, lazy loading, no CPU KV offload, a 6GiB memory limit, and exit sleep after 30 idle minutes. These memory and capacity settings still need GPU qualification.

Default communication uses NCCL with FP16 reductions and automatic topology selection. FP8 Firefly communication is opt-in; all ranks must agree on P2P availability. Its default chunk buffer is 16MiB, and changed chunking may change FP8 rounding. P2P capability does not establish payload correctness or speed.

See [configuration](docs/configuration-icy.md), [build](docker/BUILD.md), [review](docs/optimization-review.md), and [source validation](docs/releases/v0.1.6-icy.1.md). [中文说明](README.md).

## One-command entry

On a Linux x86_64 host with Docker, NVIDIA drivers and NVIDIA Container Toolkit, run `bash start.sh`. It checks the SM75 devices/runtime, creates or reuses an owner-only API key, reuses a full image or builds one if missing, starts the ModelScope FP8 API, and waits for the authenticated model listing. It does not perform generation. `bash start.sh --dry-run` has no credential, Docker, download or GPU side effects.

NUC9 currently lacks Docker and should import a full image built elsewhere. Source builds require 8GiB available RAM and 200GiB free disk; no host packages/drivers are automatically installed. Existing conflicting/stopped containers are preserved. This entry has only been validated with a mocked runtime, not real GPU startup.
