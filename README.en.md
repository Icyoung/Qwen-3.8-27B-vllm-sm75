# Qwen-3.8-27B-vllm-sm75

Native four-T10 SM75 deployment in an isolated Python 3.12 venv. Docker and Ultra are removed from the deployment path. The default model is official `Qwen/Qwen3.8-27B-FP8` from ModelScope; Marlin uses FP8 weights with FP16 activations.

```bash
bash start.sh
bash start.sh --dry-run
bash start.sh --setup-only
```

Ubuntu 22.04/24.04/26.04 x86_64, network, sudo/root and 60GiB free disk are required by the bootstrap. Healthy drivers are reused; missing drivers are installed via Ubuntu's GPGPU flow and exit 75 requires a manual reboot. No automatic reboot or healthy-driver upgrade.

Runtime components live under `$HOME/.local/share/qwen-3.8-27b-vllm-sm75`: a private Python venv, CUDA12.9/GCC13 compiler prefix, caches, API key and service config. Neither system Python nor global CUDA links are modified. Foreign environments are rejected. Matching interrupted installs can resume; changed contracts require a new `QWEN_SM75_HOME`.

The official cu129 vLLM wheel and compiler bootstrap tool are SHA-256 pinned. A native systemd service handles the API and a 6GiB memory limit. Matching services are reused; conflicting active services are preserved. Model listing readiness is checked, not generation. Auto-boot/linger is not enabled.

This source-only native candidate has not been installed or GPU-tested. See [native deployment](docs/native-deployment.md), [configuration](docs/configuration-icy.md), and [中文](README.md).
