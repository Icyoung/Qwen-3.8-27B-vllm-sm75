#!/usr/bin/env bash
# Internal implementation; public entry: EDITION=ultra bash docker/run.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d ' \r\n' < "$SCRIPT_DIR/VERSION")"
ULTRA_VERSION="${VERSION%-beta}-ultra-beta"
[[ "$VERSION" == *-beta ]] || ULTRA_VERSION="$VERSION-ultra"
: "${ULTRA_DATA_ROOT:?Set ULTRA_DATA_ROOT to an absolute persistent host directory}"
: "${MODEL_ROOT:?Set MODEL_ROOT to your existing absolute model directory}"
[[ "$ULTRA_DATA_ROOT" == /* && "$ULTRA_DATA_ROOT" != / && "$MODEL_ROOT" == /* && -d "$MODEL_ROOT" ]] || {
  echo 'Use a dedicated absolute data directory and an existing absolute model directory' >&2; exit 2;
}
cache="${VLLM_SM75_CACHE_ROOT:-$ULTRA_DATA_ROOT/cache}"
[[ "$cache" == /* && "$cache" != / ]] || exit 2
mkdir -p "$ULTRA_DATA_ROOT/console" "$ULTRA_DATA_ROOT/home" "$ULTRA_DATA_ROOT/workspace" "$cache"
extra=()
if [[ -n "${PSTATE_NVAPI_LIB:-}" ]]; then
  [[ -f "$PSTATE_NVAPI_LIB" && "$PSTATE_NVAPI_LIB" == /* ]] || exit 2
  extra+=(--volume "$PSTATE_NVAPI_LIB:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro")
fi
# A dedicated optional persistent volume; leave both unset for CPU-only LMCache.
if [[ -n "${LMCACHE_DISK_ROOT:-}" ]]; then
  [[ "$LMCACHE_DISK_ROOT" == /* && "$LMCACHE_DISK_ROOT" != / ]] || { echo 'LMCACHE_DISK_ROOT must be a dedicated absolute host directory' >&2; exit 2; }
  export LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-/lmcache}"
  [[ "$LMCACHE_DISK_PATH" == /* && "$LMCACHE_DISK_PATH" != / && "$LMCACHE_DISK_PATH" != *'/../'* && "$LMCACHE_DISK_PATH" != */.. ]] || { echo 'LMCACHE_DISK_PATH must be an absolute container subdirectory' >&2; exit 2; }
  mkdir -p "$LMCACHE_DISK_ROOT"
  extra+=(--volume "$LMCACHE_DISK_ROOT:$LMCACHE_DISK_PATH")
fi
# The manager imports core settings once into its visible startup profile.
# Later panel edits win until the next manager/container startup.
runtime_env=()
while IFS= read -r setting || [[ -n "$setting" ]]; do
  setting="${setting%$'\r'}"
  [[ -n "$setting" && "$setting" != \#* ]] || continue
  if [[ -v "$setting" ]]; then runtime_env+=(--env "$setting"); fi
done < "$SCRIPT_DIR/helpers/runtime-env.list"
# Existing data is neither migrated nor recursively chowned by this launcher.
# The manager creates its own files; DSH drops to UID/GID 1000 itself.
exec docker run --detach --name "${CONTAINER_NAME:-vllm-sm75-ultra}" \
  --gpus "${GPUS:-all}" --shm-size 16g --ulimit nofile=1048576:1048576 \
  --cap-drop ALL --cap-add CHOWN --cap-add FOWNER --cap-add DAC_OVERRIDE \
  --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges \
  --publish "${CONSOLE_PORT:-1615}:1615" --publish "${PORT:-8000}:8000" \
  --env SM75_CONSOLE_ROOT=/data --env SM75_SINGLE_CONTAINER=1 \
  --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL-SYS}" \
  --env MAX_JOBS="${MAX_JOBS-1}" --env NVCC_THREADS="${NVCC_THREADS-1}" "${runtime_env[@]}" \
  --volume "$ULTRA_DATA_ROOT/console:/data" --volume "$cache:/data/cache" \
  --volume "$ULTRA_DATA_ROOT/home:/dsh/home" --volume "$ULTRA_DATA_ROOT/workspace:/dsh/workspace" \
  --volume "$MODEL_ROOT:/models:ro" "${extra[@]}" \
  "${IMAGE:-vllm-sm75:v${ULTRA_VERSION}}"
