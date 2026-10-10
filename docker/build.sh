#!/usr/bin/env bash
# 鏋勫缓姝ｅ紡闀滃儚 vllm-sm75:v$(cat docker/VERSION)
set -euo pipefail
# Production Unraid has previously stalled while exporting full image layers.
# Temporary guard for the unisolated exporter that caused the incident.
# Full builds are allowed once resource isolation/export/recovery are validated.
if [[ -f /etc/unraid-version && "${BUILD_MODE:-full}" != ui ]]; then
    echo 'This unisolated build path is blocked on Unraid pending resource/export safeguards. Use the existing fast runtime or an isolated build host.' >&2
    exit 2
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d ' \r\n' < "$ROOT/docker/VERSION")"
[[ -n "$VERSION" ]] || { echo 'docker/VERSION is empty' >&2; exit 2; }
ULTRA_VERSION="${VERSION/-beta/-ultra-beta}"
[[ "$VERSION" == *-beta* ]] || ULTRA_VERSION="$VERSION-ultra"
EDITION="${EDITION:-ultra}"
BUILD_MODE="${BUILD_MODE:-full}"
case "$EDITION:$BUILD_MODE" in standard:full|standard:fast|ultra:full|ultra:ui) ;;
    *) echo 'Use EDITION=standard|ultra; BUILD_MODE=full, standard fast, or ultra ui' >&2; exit 2;; esac
command -v docker >/dev/null
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'source-archive')}"
if [[ "$EDITION" == ultra ]]; then
    if [[ "$BUILD_MODE" == ui ]]; then
        : "${RUNTIME_IMAGE:?Set RUNTIME_IMAGE to an audited ultra runtime image}"
        ui_recipe="$ROOT/ultra/Dockerfile.ui"
        if [[ "${INSTALL_LMCACHE:-0}" == 1 ]]; then ui_recipe="$ROOT/ultra/Dockerfile.candidate"; fi
        docker image inspect "$RUNTIME_IMAGE" >/dev/null
        export DOCKER_BUILDKIT=0
        exec docker build --network "${BUILD_NETWORK:-default}" --memory 1g --memory-swap 1g \
            --cpu-period 100000 --cpu-quota 100000 --file "$ui_recipe" \
            --build-arg BASE_IMAGE="$RUNTIME_IMAGE" --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
            --build-arg SOURCE_SHA256="${SOURCE_SHA256:-}" --build-arg RELEASE_VERSION="$VERSION" \
            --build-arg INSTALL_LMCACHE="${INSTALL_LMCACHE:-0}" \
            --tag "${IMAGE:-vllm-sm75:v${ULTRA_VERSION}-ui}" "$ROOT/ultra"
    fi
    exec docker build --file "$ROOT/docker/Dockerfile.ultra-full" --target final \
        --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg MAX_JOBS="${MAX_JOBS:-1}" \
        --build-arg SOURCE_REVISION="$SOURCE_REVISION" --build-arg RELEASE_VERSION="$ULTRA_VERSION" \
        --build-arg VLLM_SM75_RELEASE="vllm-sm75-v${VERSION}" \
        --build-arg INSTALL_LMCACHE="${INSTALL_LMCACHE:-0}" \
        --tag "${IMAGE:-vllm-sm75:v${ULTRA_VERSION}}" "$ROOT"
fi
file=Dockerfile
default_image="vllm-sm75:v${VERSION}"
if [[ "$BUILD_MODE" == fast ]]; then
    file=Dockerfile.fast
    default_image="vllm-sm75:v${VERSION}-fast"
fi
docker build --file "$ROOT/docker/$file" --target final \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MAX_JOBS="${MAX_JOBS:-1}" \
    --build-arg INSTALL_LMCACHE="${INSTALL_LMCACHE:-0}" \
    --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
    --build-arg BASE_IMAGE_ID="$BASE_IMAGE" \
    --build-arg VLLM_SM75_RELEASE="vllm-sm75-v${VERSION}" \
    --tag "${IMAGE:-$default_image}" "$ROOT"
