#!/usr/bin/env bash
# Download weights only: no GPU access, inference process, or host Python deps.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
[[ "$MODEL" != /* && "$MODEL" == */* ]] || { echo 'MODEL must be a ModelScope repository ID' >&2; exit 2; }
CACHE_ROOT="${VLLM_SM75_CACHE_ROOT:-$HOME/.cache/vllm-sm75}"
MODEL_CACHE_ROOT="${VLLM_SM75_MODEL_CACHE_ROOT:-$CACHE_ROOT/models}"
[[ "$MODEL_CACHE_ROOT" == /* ]] || { echo 'Model cache path must be absolute' >&2; exit 2; }
mkdir -p "$MODEL_CACHE_ROOT/modelscope"
VERSION="$(tr -d ' \r\n' < "$SCRIPT_DIR/VERSION")"
docker run --rm --name "${DOWNLOAD_CONTAINER_NAME:-vllm-sm75-download}" \
  --volume "$MODEL_CACHE_ROOT/modelscope:/root/.cache/modelscope" \
  --env MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
  --env "MODEL_ID=$MODEL" --env "MODEL_REVISION=${MODEL_REVISION:-master}" \
  --entrypoint python3 "${IMAGE:-vllm-sm75:v${VERSION}}" -c \
  'import os; from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download(model_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], cache_dir=os.environ["MODELSCOPE_CACHE"], max_workers=2))'
