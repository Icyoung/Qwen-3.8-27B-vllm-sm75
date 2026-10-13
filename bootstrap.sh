#!/usr/bin/env bash
# Native Ubuntu setup. All Python/CUDA tooling lives in this app's prefix.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then
    echo 'Usage: bash start.sh [--dry-run|--setup-only|--no-wait] [--wait-timeout SECONDS] [-- ENGINE_ARGS]'
    exit 0
  fi
  if [[ "$arg" == --dry-run ]]; then
    cat <<'PLAN'
Native deployment plan (no commands or writes performed):
1. Check Ubuntu x86_64; install missing Python/venv, compiler, curl and systemd tools using apt.
2. Reuse a working NVIDIA driver. If absent, install Ubuntu's GPGPU driver and stop with exit 75 for a manual reboot.
3. Install uv in a private bootstrap venv and provision a private Python 3.12 runtime venv.
4. Install the SHA-256-pinned official vLLM cu129 wheel and pinned dependencies.
5. Provision CUDA 12.9/GCC13 in a separate private compiler prefix; compile SM75 FlashQLA with MAX_JOBS=1.
6. Apply and verify overlays inside the owned venv only.
7. Start a native systemd service, auto-download the ModelScope FP8 model, and wait for the authenticated model listing.
Docker is neither installed nor required. No automatic reboot or existing-driver upgrade.
PLAN
    exit 0
  fi
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Native setup requires Linux x86_64' >&2; exit 2; }
. /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" =~ ^(22\.04|24\.04|26\.04)$ ]] || { echo 'Supported bootstrap: Ubuntu 22.04/24.04/26.04' >&2; exit 2; }
export QWEN_SM75_HOME="${QWEN_SM75_HOME:-$HOME/.local/share/qwen-3.8-27b-vllm-sm75}"
[[ "$QWEN_SM75_HOME" == /* ]] || { echo 'QWEN_SM75_HOME must be absolute' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX LD_LIBRARY_PATH
if (( EUID == 0 )); then privileged=(); else privileged=(sudo); fi
packages=(python3 python3-venv build-essential cmake ninja-build curl ca-certificates bzip2 xz-utils pkg-config systemd dbus-user-session)
missing=()
for package in "${packages[@]}"; do
  if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != 'install ok installed' ]]; then missing+=("$package"); fi
done
if (( ${#missing[@]} )); then
  ${privileged[@]+"${privileged[@]}"} apt-get update
  ${privileged[@]+"${privileged[@]}"} apt-get install -y --no-install-recommends "${missing[@]}"
fi
if (( EUID == 0 )); then
  systemctl show-environment >/dev/null
else
  systemctl --user show-environment >/dev/null || { echo 'Log in to a systemd user session before native setup' >&2; exit 2; }
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  if [[ -e /proc/driver/nvidia/version ]]; then
    echo 'An NVIDIA driver is present but not working. Preserve it; repair the driver/kernel mismatch before rerunning.' >&2
    exit 2
  fi
  ${privileged[@]+"${privileged[@]}"} apt-get update
  ${privileged[@]+"${privileged[@]}"} apt-get install -y ubuntu-drivers-common "linux-headers-$(uname -r)"
  ${privileged[@]+"${privileged[@]}"} ubuntu-drivers install --gpgpu
  echo 'Driver installed. Reboot manually, select the correct OS disk, then run bash start.sh again. No reboot was performed.' >&2
  exit 75
fi
mkdir -p "$QWEN_SM75_HOME"
# Refuse a caller's arbitrary existing Python environment.
if [[ -e "$QWEN_SM75_HOME/bootstrap" && ! -f "$QWEN_SM75_HOME/bootstrap/.qwen-owned" ]]; then
  echo 'Refusing to modify an unowned bootstrap environment' >&2; exit 2
fi
if [[ ! -f "$QWEN_SM75_HOME/bootstrap/.qwen-owned" ]]; then
  /usr/bin/python3 -m venv "$QWEN_SM75_HOME/bootstrap"
  touch "$QWEN_SM75_HOME/bootstrap/.qwen-owned"
fi
if [[ ! -x "$QWEN_SM75_HOME/bootstrap/bin/uv" ]]; then
  "$QWEN_SM75_HOME/bootstrap/bin/python" -m pip install 'uv==0.13.0'
fi
export UV_PYTHON_INSTALL_DIR="$QWEN_SM75_HOME/python"
export UV_CACHE_DIR="$QWEN_SM75_HOME/cache/uv"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
"$QWEN_SM75_HOME/bootstrap/bin/uv" python install 3.12
"$QWEN_SM75_HOME/bootstrap/bin/python" "$ROOT/tools/install-native.py"
for arg in "$@"; do [[ "$arg" != --setup-only ]] || exit 0; done
exec "$QWEN_SM75_HOME/venv/bin/python" "$ROOT/tools/start.py" "$@"
