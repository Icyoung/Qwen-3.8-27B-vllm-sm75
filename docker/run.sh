#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${EDITION:-standard}" in
  ultra) exec bash "$SCRIPT_DIR/helpers/run-ultra.sh" "$@" ;;
  standard) ;;
  *) echo 'EDITION must be standard or ultra' >&2; exit 2 ;;
esac
: "${VLLM_API_KEY:?Set VLLM_API_KEY to your own API key}"
lmcache_enabled="${LMCACHE_ENABLED:-0}"
case "${lmcache_enabled,,}" in
  1|true|yes|on) lmcache_enabled=1 ;;
  0|false|no|off) lmcache_enabled=0 ;;
  *) echo 'LMCACHE_ENABLED must be true/false or 1/0' >&2; exit 2 ;;
esac
# Match the panel: decimal GiB -> nearest safe integer bytes; zero disables.
# awk is available on the supported shell hosts; no host Python is required.
if [[ "$lmcache_enabled" == 1 ]]; then
  CPU_KV_GIB="${CPU_KV_GIB-0}"
else
  CPU_KV_GIB="${CPU_KV_GIB-8}"
fi
[[ "$CPU_KV_GIB" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || { echo 'CPU_KV_GIB must be a non-negative decimal GiB value' >&2; exit 2; }
if ! cpu_bytes="$(LC_ALL=C awk -v value="$CPU_KV_GIB" 'BEGIN {
  bytes = value * 1073741824
  if (bytes > 9007199254740991) exit 2
  rounded = int(bytes)
  if (bytes - rounded >= 0.5) rounded++
  if (rounded > 9007199254740991 || (value > 0 && rounded < 1)) exit 2
  printf "%.0f", rounded
}')"; then
  echo 'CPU_KV_GIB must round to a safe positive integer byte count, or be zero to disable' >&2; exit 2
fi
if [[ "$lmcache_enabled" == 1 ]]; then
  : "${LMCACHE_CHUNK_SIZE:?Set LMCACHE_CHUNK_SIZE from the actual model layout}"
  [[ "${POWER_MODE:-pstate}" == pstate ]] || { echo 'LMCache requires POWER_MODE=pstate' >&2; exit 2; }
  [[ "$cpu_bytes" == 0 && -z "${KV_TRANSFER_CONFIG:-}" ]] || { echo 'LMCache conflicts with explicit CPU_KV_GIB or KV_TRANSFER_CONFIG' >&2; exit 2; }
  [[ "${AUTO_SLEEP_IDLE_TIMEOUT:-0}" == 0 ]] || { echo 'LMCache requires automatic sleep to be disabled' >&2; exit 2; }
  if [[ -v SPECULATIVE_CONFIG ]]; then
    [[ -z "$SPECULATIVE_CONFIG" || "$SPECULATIVE_CONFIG" == null ]] || { echo 'LMCache conflicts with speculative decoding' >&2; exit 2; }
  elif [[ "${VARIANT:-base}" != base ]]; then
    echo 'LMCache requires target-only VARIANT=base' >&2; exit 2
  fi
fi
VARIANT="${VARIANT:-base}"
FORMAT="${FORMAT:-fp8}"
case "$VARIANT" in base|mtp|dflash2) ;; *) echo 'VARIANT must be base, mtp, or dflash2' >&2; exit 2;; esac
case "$FORMAT" in fp8|awq) ;; *) echo 'FORMAT must be fp8 or awq' >&2; exit 2;; esac
if [[ "$FORMAT" == awq ]]; then
  : "${MODEL:?Set MODEL to your downloaded AWQ model directory under /models}"
else
  MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
fi
SERVE_NAME="${SERVED_MODEL_NAME:-VLLM-Qwen3.8-27B}"
CACHE_ROOT="${VLLM_SM75_CACHE_ROOT:?Set VLLM_SM75_CACHE_ROOT to an absolute host cache directory}"
[[ "$CACHE_ROOT" == /* ]] || { echo 'Cache path must be absolute' >&2; exit 2; }
MODEL_CACHE_ROOT="${VLLM_SM75_MODEL_CACHE_ROOT:-$CACHE_ROOT/models}"
[[ "$MODEL_CACHE_ROOT" == /* ]] || { echo 'Model cache path must be absolute' >&2; exit 2; }
mkdir -p "$MODEL_CACHE_ROOT" "$CACHE_ROOT/$FORMAT/vllm" "$CACHE_ROOT/shared/flashinfer" \
  "$CACHE_ROOT/$FORMAT/triton" "$CACHE_ROOT/shared/torch_extensions"
mounts=(--volume "$MODEL_CACHE_ROOT:/root/.cache/modelscope"
  --volume "$MODEL_CACHE_ROOT:/root/.cache/huggingface"
  --volume "$CACHE_ROOT/$FORMAT/vllm:/root/.cache/vllm"
  --volume "$CACHE_ROOT/shared/flashinfer:/root/.cache/flashinfer"
  --volume "$CACHE_ROOT/$FORMAT/triton:/root/.triton/cache"
  --volume "$CACHE_ROOT/shared/torch_extensions:/root/.cache/torch_extensions")
if [[ -n "${MODEL_ROOT:-}" ]]; then
  [[ "$MODEL_ROOT" == /* && -d "$MODEL_ROOT" ]] || { echo 'MODEL_ROOT must be an existing absolute directory' >&2; exit 2; }
  mounts+=(--volume "$MODEL_ROOT:/models:ro")
fi
# A dedicated optional persistent volume; leave both unset for CPU-only LMCache.
if [[ -n "${LMCACHE_DISK_ROOT:-}" ]]; then
  [[ "$LMCACHE_DISK_ROOT" == /* && "$LMCACHE_DISK_ROOT" != / ]] || { echo 'LMCACHE_DISK_ROOT must be a dedicated absolute host directory' >&2; exit 2; }
  export LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-/lmcache}"
  [[ "$LMCACHE_DISK_PATH" == /* && "$LMCACHE_DISK_PATH" != / && "$LMCACHE_DISK_PATH" != *'/../'* && "$LMCACHE_DISK_PATH" != */.. ]] || { echo 'LMCACHE_DISK_PATH must be an absolute container subdirectory' >&2; exit 2; }
  mkdir -p "$LMCACHE_DISK_ROOT"
  mounts+=(--volume "$LMCACHE_DISK_ROOT:$LMCACHE_DISK_PATH")
fi
tp="${TENSOR_PARALLEL_SIZE:-4}"
[[ "$tp" =~ ^[1-9][0-9]*$ ]] || { echo 'TENSOR_PARALLEL_SIZE must be a positive integer' >&2; exit 2; }
pstate_default_gpus=0
for ((gpu_index=1; gpu_index<tp; gpu_index++)); do pstate_default_gpus+=",$gpu_index"; done
seq=4; batch=8192; util=0.87; length=auto; kv=0
[[ "$FORMAT" != awq ]] || { seq=8; batch=16384; }
VERSION="$(tr -d ' \r\n' < "$SCRIPT_DIR/VERSION")"
image="vllm-sm75:v${VERSION/-beta/-ultra-beta}"
[[ "$VERSION" == *-beta* ]] || image="vllm-sm75:v${VERSION}-ultra"
image="${IMAGE:-$image}"
graph='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
extra=()
POWER_MODE="${POWER_MODE:-pstate}"
case "$POWER_MODE" in sleep|pstate) ;; *) echo 'POWER_MODE must be sleep or pstate' >&2; exit 2;; esac
entrypoint=()
pstate_env=()
if [[ "$POWER_MODE" == pstate ]]; then
  pstate_env=(--env PSTATE_IDLE_TIMEOUT="${PSTATE_IDLE_TIMEOUT:-1800}" \
    --env PSTATE_UTIL="${PSTATE_UTIL:-5}" --env PSTATE_CONFIRM="${PSTATE_CONFIRM:-60}" \
    --env PSTATE_LOW="${PSTATE_LOW:-8}" --env PSTATE_HIGH="${PSTATE_HIGH:-16}" \
    --env PSTATE_POLL="${PSTATE_POLL:-5}" --env PSTATE_GPUS="${PSTATE_GPUS:-$pstate_default_gpus}")
  # pstate: 关闭 vLLM 休眠; 镜像内置 nvidia-pstated 在容器内管理 P8/16(高位必须 16)
  extra+=(--auto-sleep-idle-timeout 0)
  if [[ "${AUTO_SLEEP_IDLE_TIMEOUT:-0}" != 0 ]]; then
    echo 'note: POWER_MODE=pstate ignores AUTO_SLEEP_* settings' >&2
  fi
  pstate_lib="${PSTATE_NVAPI_LIB:-}"
  if [[ -z "$pstate_lib" ]]; then
    for c in /usr/lib64/libnvidia-api.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-api.so.1 /usr/lib/libnvidia-api.so.1; do
      [[ -f "$c" ]] && { pstate_lib="$c"; break; }
    done
  fi
  if [[ -z "$pstate_lib" || ! -f "$pstate_lib" ]]; then
    echo 'POWER_MODE=pstate requires host libnvidia-api.so.1; set PSTATE_NVAPI_LIB' >&2; exit 2
  fi
  mounts+=(--volume "$pstate_lib:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro")
  entrypoint=(--entrypoint /opt/vllm-sm75/pstate-entrypoint.sh)
  if pgrep -f "[n]vidia-pstated" >/dev/null 2>&1; then
    echo 'warning: host nvidia-pstated detected; stop it to avoid fighting the in-container one' >&2
  fi
else
  AUTO_SLEEP_IDLE_TIMEOUT="${AUTO_SLEEP_IDLE_TIMEOUT:-30}"
  AUTO_SLEEP_OFFLOAD_TARGET="${AUTO_SLEEP_OFFLOAD_TARGET:-exit}"
  if [[ "$AUTO_SLEEP_IDLE_TIMEOUT" != 0 ]]; then
    extra+=(--auto-sleep-idle-timeout "$AUTO_SLEEP_IDLE_TIMEOUT"
      --auto-sleep-offload-target "$AUTO_SLEEP_OFFLOAD_TARGET")
    case "$AUTO_SLEEP_OFFLOAD_TARGET" in
      cpu|reload|disk) extra+=(--enable-sleep-mode) ;;
      exit) ;;
      *) echo 'AUTO_SLEEP_OFFLOAD_TARGET must be cpu, reload, exit, or disk' >&2; exit 2 ;;
    esac
    if [[ "$AUTO_SLEEP_OFFLOAD_TARGET" == disk ]]; then
      mkdir -p "$CACHE_ROOT/sleep/$VARIANT-$FORMAT"
      mounts+=(--volume "$CACHE_ROOT/sleep/$VARIANT-$FORMAT:/sleep-state")
      extra+=(--auto-sleep-disk-path /sleep-state)
    fi
    if [[ -n "${AUTO_SLEEP_RELOAD_PATH:-}" ]]; then
      extra+=(--auto-sleep-reload-path "$AUTO_SLEEP_RELOAD_PATH")
    fi
    extra+=(--auto-sleep-page-cache-keep-interval "${AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL:-600}")
  fi
fi
# 投机 variant 用 SM75Scheduler 子类(支持运行中开/关投机, 见 /monitor 按钮)。
SCHED_CLS="vllm.v1.core.sched.scheduler_sm75.SM75Scheduler"
if [[ -v SPECULATIVE_CONFIG ]]; then
  if [[ -n "$SPECULATIVE_CONFIG" && "$SPECULATIVE_CONFIG" != null ]]; then
    extra+=(--speculative-config "$SPECULATIVE_CONFIG" --scheduler-cls "${SCHEDULER_CLS:-$SCHED_CLS}")
  fi
elif [[ "$VARIANT" == mtp ]]; then
  extra+=(--speculative-config '{"method":"mtp","num_speculative_tokens":5}' --scheduler-cls "$SCHED_CLS")
  graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6]}'
elif [[ "$VARIANT" == dflash2 ]]; then
  : "${DRAFT_MODEL:?Set DRAFT_MODEL to the matching DFlash draft directory under /models}"
  [[ "$DRAFT_MODEL" == /models/* && -n "${MODEL_ROOT:-}" ]] || { echo 'Mount MODEL_ROOT and set DRAFT_MODEL=/models/your-draft' >&2; exit 2; }
  [[ "$DRAFT_MODEL" != *'"'* && "$DRAFT_MODEL" != *'\'* ]] || exit 2
  kv=3288334336; util=0.92; length=262144
  [[ "$FORMAT" != awq ]] || { kv=4294967296; util=0.87; length=auto; }
  extra+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":$tp,\"max_model_len\":262144,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASHINFER\",\"draft_sample_method\":\"probabilistic\"}" --scheduler-cls "$SCHED_CLS")
  graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8]}'
fi

# The implicit standard CPU 8 GiB default already yielded to LMCache above.
if [[ "$lmcache_enabled" == 1 ]]; then
  entrypoint=(--entrypoint /opt/vllm-sm75/lmcache-entrypoint.sh)
fi
kv="${GPU_KV_BYTES:-${kv:-0}}"
[[ "$kv" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'GPU_KV_BYTES must be a non-negative integer' >&2; exit 2; }
[[ "$kv" == 0 ]] || extra+=(--kv-cache-memory-bytes "$kv")
if [[ -v KV_TRANSFER_CONFIG ]]; then
  [[ -z "$KV_TRANSFER_CONFIG" ]] || extra+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
elif (( cpu_bytes > 0 )); then
  extra+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"CPUOffloadingSpec\",\"cpu_bytes_to_use\":$cpu_bytes}}")
fi
hf_overrides="${HF_OVERRIDES-}"
[[ -v HF_OVERRIDES ]] || hf_overrides='{"dtype":"float16"}'
[[ -z "$hf_overrides" ]] || extra+=(--hf-overrides "$hf_overrides")
# Optional engine settings use the same names as the Ultra startup importer.
for setting in QUANTIZATION:--quantization TOOL_CALL_PARSER:--tool-call-parser \
  REASONING_PARSER:--reasoning-parser TOOL_PARSER_PLUGIN:--tool-parser-plugin \
  REASONING_PARSER_PLUGIN:--reasoning-parser-plugin CHAT_TEMPLATE:--chat-template \
  CHAT_TEMPLATE_CONTENT_FORMAT:--chat-template-content-format \
  OVERRIDE_GENERATION_CONFIG:--override-generation-config SCHEDULER_CLS:--scheduler-cls; do
  variable="${setting%%:*}"; flag="${setting#*:}"
  if [[ -v "$variable" && -n "${!variable}" ]]; then extra+=("$flag" "${!variable}"); fi
done
add_boolean() {
  local variable="$1" default="$2" yes="$3" no="$4" value
  value="${!variable-$default}"
  case "${value,,}" in
    1|true|yes|on) extra+=("$yes") ;;
    0|false|no|off) [[ -z "$no" ]] || extra+=("$no") ;;
    *) echo "$variable must be true/false or 1/0" >&2; exit 2 ;;
  esac
}
add_boolean ENABLE_PREFIX_CACHING 1 --enable-prefix-caching --no-enable-prefix-caching
add_boolean ASYNC_SCHEDULING 1 --async-scheduling --no-async-scheduling
add_boolean DISABLE_CUSTOM_ALL_REDUCE 1 --disable-custom-all-reduce --no-disable-custom-all-reduce
add_boolean ENABLE_AUTO_TOOL_CHOICE 0 --enable-auto-tool-choice ''
[[ ! -v COMPILATION_CONFIG ]] || graph="$COMPILATION_CONFIG"
[[ -z "$graph" ]] || extra+=(--compilation-config "$graph")
if [[ -n "${AUTO_SLEEP_DISK_PATH:-}" && "$POWER_MODE" == sleep ]]; then
  extra+=(--auto-sleep-disk-path "$AUTO_SLEEP_DISK_PATH")
fi
runtime_env=()
for variable in NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_DEBUG CUDA_VISIBLE_DEVICES \
  LMCACHE_ENABLED LMCACHE_CPU_GIB LMCACHE_CHUNK_SIZE LMCACHE_DISK_PATH \
  LMCACHE_MIN_FREE_DISK_GIB LMCACHE_PORT LMCACHE_HTTP_PORT; do
  if [[ -v "$variable" ]]; then runtime_env+=(--env "$variable"); fi
done

default_attention='{"backend":"FLASHINFER"}'
docker run --detach --name "${CONTAINER_NAME:-vllm-sm75-$VARIANT-$FORMAT}" \
  --gpus "${GPUS:-all}" --shm-size "${SHM_SIZE:-32g}" --ulimit nofile=1048576:1048576 \
  --publish "${PORT:-8000}:8000" "${mounts[@]}" \
  --env VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE-true}" --env MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
  --env VLLM_GDN_DECODE_KERNEL="${VLLM_GDN_DECODE_KERNEL-triton}" --env VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER-0}" \
  --env VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-0}" \
  --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL-SYS}" \
  --env VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM-0}" --env VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM-0}" \
  --env TRITON_CACHE_DIR=/root/.triton/cache --env TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  --env VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}" \
  --env VLLM_FIREFLY="${VLLM_FIREFLY:-1}" --env VLLM_FIREFLY_AR="${VLLM_FIREFLY_AR:-auto}" \
  --env VLLM_FIREFLY_AR_BACKEND="${VLLM_FIREFLY_AR_BACKEND:-auto}" \
  --env VLLM_FIREFLY_AR_MIN_SIZE="${VLLM_FIREFLY_AR_MIN_SIZE:-1048576}" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS-2}" --env MAX_JOBS="${MAX_JOBS-1}" --env NVCC_THREADS="${NVCC_THREADS:-1}" --env TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS-1}" \
  --env SM75_EDITION=standard --env VLLM_API_KEY "${runtime_env[@]}" "${pstate_env[@]}" "${entrypoint[@]}" "$image" "$MODEL" --served-model-name "$SERVE_NAME" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size "$tp" \
  --max-num-seqs "${MAX_NUM_SEQS:-$seq}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-$batch}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-$util}" --max-model-len "${MAX_MODEL_LEN:-$length}" \
  --attention-config "${ATTENTION_CONFIG:-$default_attention}" --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-flashqla_sm75}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" --block-size "${BLOCK_SIZE:-32}" --dtype "${DTYPE:-float16}" \
  --generation-config "${GENERATION_CONFIG:-vllm}" \
  "${extra[@]}" "$@"
