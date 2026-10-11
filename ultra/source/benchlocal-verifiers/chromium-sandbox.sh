#!/usr/bin/env bash
# agent-browser 0.27 detects Docker and adds --no-sandbox automatically.
# Benchmark browsers run unprivileged; retain their own sandbox as well.
set -euo pipefail
if [[ $EUID -eq 0 ]]; then echo '评测浏览器必须以非 root 用户运行。' >&2; exit 2; fi
args=()
for value in "$@"; do
 case "${value%%=*}" in
  --no-sandbox|--disable-setuid-sandbox|--disable-namespace-sandbox|--disable-seccomp-filter-sandbox|--disable-gpu-sandbox) ;;
  *) args+=("$value") ;;
 esac
done
exec /opt/sm75-bench/chrome/chrome-linux64/chrome "${args[@]}"
