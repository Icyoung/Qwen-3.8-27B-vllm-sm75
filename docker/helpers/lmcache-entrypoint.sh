#!/usr/bin/env bash
set -euo pipefail
exec python3 /opt/vllm-sm75/lmcache-launch.py "$@"
