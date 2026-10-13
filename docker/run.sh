#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${EDITION:-standard}" in
  standard) ;;
  *) echo 'This fork only provides EDITION=standard' >&2; exit 2 ;;
esac
: "${VLLM_API_KEY:?Set VLLM_API_KEY to your own API key}"
VARIANT="${VARIANT:-base}"
FORMAT="${FORMAT:-fp8}"
TP_SIZE="${TP_SIZE:-4}"
[[ "$TP_SIZE" =~ ^(1|2|4|8)$ ]] || { echo 'TP_SIZE must be 1, 2, 4, or 8' >&2; exit 2; }
case "$VARIANT" in base|mtp|dflash2) ;; *) echo 'VARIANT must be base, mtp, or dflash2' >&2; exit 2;; esac
case "$FORMAT" in fp8|awq) ;; *) echo 'FORMAT must be fp8 or awq' >&2; exit 2;; esac
if [[ "$FORMAT" == awq ]]; then
  : "${MODEL:?Set MODEL to your downloaded AWQ model directory under /models}"
else
  MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
fi
SERVE_NAME="${SERVED_MODEL_NAME:-VLLM-Qwen3.8-27B}"
CACHE_ROOT="${VLLM_SM75_CACHE_ROOT:-$HOME/.cache/vllm-sm75}"
[[ "$CACHE_ROOT" == /* ]] || { echo 'Cache path must be absolute' >&2; exit 2; }
MODEL_CACHE_ROOT="${VLLM_SM75_MODEL_CACHE_ROOT:-$CACHE_ROOT/models}"
[[ "$MODEL_CACHE_ROOT" == /* ]] || { echo 'Model cache path must be absolute' >&2; exit 2; }
mkdir -p "$MODEL_CACHE_ROOT/modelscope" "$MODEL_CACHE_ROOT/huggingface" "$CACHE_ROOT/$FORMAT/vllm" "$CACHE_ROOT/shared/flashinfer" \
  "$CACHE_ROOT/$FORMAT/triton" "$CACHE_ROOT/shared/torch_extensions"
mounts=(--volume "$MODEL_CACHE_ROOT/modelscope:/root/.cache/modelscope"
  --volume "$MODEL_CACHE_ROOT/huggingface:/root/.cache/huggingface"
  --volume "$CACHE_ROOT/$FORMAT/vllm:/root/.cache/vllm"
  --volume "$CACHE_ROOT/shared/flashinfer:/root/.cache/flashinfer"
  --volume "$CACHE_ROOT/$FORMAT/triton:/root/.triton/cache"
  --volume "$CACHE_ROOT/shared/torch_extensions:/root/.cache/torch_extensions")
if [[ -n "${MODEL_ROOT:-}" ]]; then
  [[ "$MODEL_ROOT" == /* && -d "$MODEL_ROOT" ]] || { echo 'MODEL_ROOT must be an existing absolute directory' >&2; exit 2; }
  mounts+=(--volume "$MODEL_ROOT:/models:ro")
fi
seq=1; batch=512; util=0.87; length=32768; kv=0
[[ "$FORMAT" != awq ]] || { seq=8; batch=16384; }
VERSION="$(tr -d ' \r\n' < "$SCRIPT_DIR/VERSION")"
image="vllm-sm75:v${VERSION}"
image="${IMAGE:-$image}"
graph='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
extra=()
POWER_MODE="${POWER_MODE:-sleep}"
case "$POWER_MODE" in sleep|pstate) ;; *) echo 'POWER_MODE must be sleep or pstate' >&2; exit 2;; esac
entrypoint=()
pstate_env=()
if [[ "$POWER_MODE" == pstate ]]; then
  pstate_env=(--env PSTATE_IDLE_TIMEOUT="${PSTATE_IDLE_TIMEOUT:-1800}" \
    --env PSTATE_UTIL="${PSTATE_UTIL:-5}" --env PSTATE_CONFIRM="${PSTATE_CONFIRM:-60}" \
    --env PSTATE_LOW="${PSTATE_LOW:-8}" --env PSTATE_HIGH="${PSTATE_HIGH:-16}" \
    --env PSTATE_POLL="${PSTATE_POLL:-5}" --env PSTATE_GPUS="${PSTATE_GPUS:-0,1,2,3}")
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
if [[ "$VARIANT" == mtp ]]; then
  extra+=(--speculative-config '{"method":"mtp","num_speculative_tokens":5}' --scheduler-cls "$SCHED_CLS")
  graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6]}'
elif [[ "$VARIANT" == dflash2 ]]; then
  [[ "$TP_SIZE" == 4 ]] || { echo 'This DFlash2 template requires TP_SIZE=4' >&2; exit 2; }
  : "${DRAFT_MODEL:?Set DRAFT_MODEL to the matching DFlash draft directory under /models}"
  [[ "$DRAFT_MODEL" == /models/* && -n "${MODEL_ROOT:-}" ]] || { echo 'Mount MODEL_ROOT and set DRAFT_MODEL=/models/your-draft' >&2; exit 2; }
  [[ "$DRAFT_MODEL" != *'"'* && "$DRAFT_MODEL" != *'\'* ]] || exit 2
  kv=3288334336; util=0.92; length=262144
  [[ "$FORMAT" != awq ]] || { kv=4294967296; util=0.87; length=auto; }
  extra+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":4,\"max_model_len\":262144,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASHINFER\",\"draft_sample_method\":\"probabilistic\"}" --scheduler-cls "$SCHED_CLS")
  graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8]}'
fi

# Public tuning knobs use engine-native units. Zero CPU KV disables offload.
CPU_KV_GIB="${CPU_KV_GIB:-0}"
[[ "$CPU_KV_GIB" =~ ^(0|[1-9][0-9]{0,5})$ ]] || { echo 'CPU_KV_GIB must be a non-negative integer' >&2; exit 2; }
kv="${GPU_KV_BYTES:-${kv:-0}}"
[[ "$kv" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'GPU_KV_BYTES must be a non-negative integer' >&2; exit 2; }
[[ "$kv" == 0 ]] || extra+=(--kv-cache-memory-bytes "$kv")
if (( CPU_KV_GIB > 0 )); then
  cpu_bytes=$((CPU_KV_GIB * 1073741824))
  extra+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"CPUOffloadingSpec\",\"cpu_bytes_to_use\":$cpu_bytes}}")
fi
hf_overrides="${HF_OVERRIDES:-}"
[[ -n "$hf_overrides" ]] || hf_overrides='{"dtype":"float16"}'

# FP8 checkpoints and FP8 communication are independent choices. The default
# NCCL path retains FP16 reductions; Firefly's lossy reductions are opt-in.
COMM_BACKEND="${COMM_BACKEND:-nccl}"
case "$COMM_BACKEND" in
  nccl) ar_default=0 ;;
  firefly|auto) ar_default=auto ;;
  *) echo 'COMM_BACKEND must be nccl, firefly, or auto' >&2; exit 2 ;;
esac
P2P_MODE="${P2P_MODE:-auto}"
case "$P2P_MODE" in
  auto) p2p_default=0 ;;
  off) p2p_default=1 ;;
  on) p2p_default=0 ;;
  *) echo 'P2P_MODE must be auto, off, or on' >&2; exit 2 ;;
esac
p2p_disable="${NCCL_P2P_DISABLE:-$p2p_default}"
[[ "$p2p_disable" == 0 || "$p2p_disable" == 1 ]] || { echo 'NCCL_P2P_DISABLE must be 0 or 1' >&2; exit 2; }
[[ "$P2P_MODE" != off || "$p2p_disable" == 1 ]] || { echo 'P2P_MODE=off conflicts with NCCL_P2P_DISABLE=0' >&2; exit 2; }
ar_backend="${VLLM_FIREFLY_AR_BACKEND:-auto}"
case "$ar_backend" in auto|shm|p2p) ;; *) echo 'VLLM_FIREFLY_AR_BACKEND must be auto, shm, or p2p' >&2; exit 2;; esac
if [[ "$p2p_disable" == 1 ]]; then
  [[ "$ar_backend" != p2p ]] || { echo 'Disabled P2P conflicts with the Firefly p2p backend' >&2; exit 2; }
  ar_backend=shm
fi
comm_env=(--env "NCCL_P2P_DISABLE=$p2p_disable" --env "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}")
# Leave the distance policy unset unless requested: NCCL decides from the
# actual topology, rather than assuming a Thunderbolt bridge supports SYS.
if [[ -n "${NCCL_P2P_LEVEL:-}" ]]; then
  case "$NCCL_P2P_LEVEL" in LOC|NVL|PIX|PXB|PHB|SYS) ;; *) echo 'Invalid NCCL_P2P_LEVEL' >&2; exit 2;; esac
  comm_env+=(--env "NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL")
fi
MODEL_PROVIDER="${MODEL_PROVIDER:-modelscope}"
case "$MODEL_PROVIDER" in
  modelscope) use_modelscope=true ;;
  huggingface) use_modelscope=false ;;
  *) echo 'MODEL_PROVIDER must be modelscope or huggingface' >&2; exit 2 ;;
esac
[[ -z "${MODEL_REVISION:-}" ]] || extra+=(--revision "$MODEL_REVISION")
extra+=(--safetensors-load-strategy lazy --model-loader-extra-config '{"enable_multithread_load":false}')
resources=(--memory "${CONTAINER_MEMORY:-6g}" --memory-swap "${CONTAINER_MEMORY:-6g}")

docker run --detach --name "${CONTAINER_NAME:-vllm-sm75-$VARIANT-$FORMAT}" \
  --label "io.qwen-sm75.launch-fingerprint=${QWEN_SM75_LAUNCH_FINGERPRINT:-manual}" \
  --gpus "${GPUS:-all}" --shm-size "${SHM_SIZE:-512m}" "${resources[@]}" --ulimit nofile=1048576:1048576 \
  --publish "${PORT:-8000}:8000" "${mounts[@]}" \
  --env "VLLM_USE_MODELSCOPE=$use_modelscope" --env MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
  --env VLLM_GDN_DECODE_KERNEL=triton --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-0}" \
  "${comm_env[@]}" \
  --env VLLM_USE_NCCL_SYMM_MEM=0 --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env TRITON_CACHE_DIR=/root/.triton/cache --env TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  --env VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}" \
  --env VLLM_FIREFLY="${VLLM_FIREFLY:-1}" --env VLLM_FIREFLY_AR="${VLLM_FIREFLY_AR:-$ar_default}" \
  --env VLLM_FIREFLY_AR_BACKEND="$ar_backend" \
  --env VLLM_FIREFLY_AR_MIN_SIZE="${VLLM_FIREFLY_AR_MIN_SIZE:-1048576}" \
  --env VLLM_FIREFLY_AR_CHUNK_BYTES="${VLLM_FIREFLY_AR_CHUNK_BYTES:-16777216}" \
  --env OMP_NUM_THREADS=2 --env MAX_JOBS=1 --env NVCC_THREADS="${NVCC_THREADS:-1}" --env TORCHINDUCTOR_COMPILE_THREADS=1 \
  ${pstate_env[@]+"${pstate_env[@]}"} ${entrypoint[@]+"${entrypoint[@]}"} "$image" "$MODEL" --served-model-name "$SERVE_NAME" \
  --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size "${TP_SIZE:-4}" --disable-custom-all-reduce \
  --max-num-seqs "${MAX_NUM_SEQS:-$seq}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-$batch}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-$util}" --max-model-len "${MAX_MODEL_LEN:-$length}" \
  --attention-config '{"backend":"FLASHINFER"}' --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 --block-size 32 --dtype float16 \
  --hf-overrides "$hf_overrides" --generation-config vllm \
  --enable-prefix-caching --async-scheduling --compilation-config "$graph" \
  "${extra[@]}" "$@"
