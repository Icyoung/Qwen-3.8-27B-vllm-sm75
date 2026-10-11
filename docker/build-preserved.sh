#!/usr/bin/env bash
# Complete preserved runtime: bounded preparation/packing, guarded local import.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${PRESERVE_WORK_ROOT:?Set an empty private build/output directory}"
: "${PRESERVE_BASE_IMAGE:?Set the audited immutable Ultra donor digest}"
: "${IMAGE:?Set a new local/ candidate tag}"
[[ "$IMAGE" == local/* ]] || { echo 'Preserved candidates must use a local/ tag' >&2; exit 2; }
[[ "$PRESERVE_BASE_IMAGE" == sha256:bb9f2af2b99be6eff3cbda9cad67d09a9d1218f2549af017f3c6e0d3e0aafe9f ]] || { echo 'Donor requires a new preservation audit' >&2; exit 2; }
command -v docker >/dev/null; command -v flock >/dev/null; command -v jq >/dev/null
[[ "$(docker info --format '{{.Driver}}')" == btrfs ]] || { echo 'This audited local packer currently requires the btrfs driver' >&2; exit 2; }
[[ "$PRESERVE_WORK_ROOT" == /* && "$PRESERVE_WORK_ROOT" != / && "$PRESERVE_WORK_ROOT" != /var/lib/docker* ]] || exit 2
mkdir -p "$PRESERVE_WORK_ROOT"
WORK=$(readlink -f -- "$PRESERVE_WORK_ROOT")
[[ "$WORK" == "$PRESERVE_WORK_ROOT" ]] || { echo 'Build path cannot resolve through symlinks' >&2; exit 2; }
[[ -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo 'Build output directory must be empty' >&2; exit 2; }
chmod 700 "$WORK"
exec 9>"$WORK/build.lock";flock -n 9 || { echo 'Build directory is busy' >&2; exit 2; }
if docker image inspect "$IMAGE" >/dev/null 2>&1; then echo 'Refusing to overwrite an existing image tag' >&2; exit 2; fi
[[ ! -e "$WORK/fold/runtime.layer.tar" && ! -e "$WORK/fold/candidate.docker.tar" ]] || { echo 'Previous outputs require inspection' >&2; exit 2; }
mkdir -p "$WORK/fold" "$WORK/evidence"
[[ "$(awk '/MemAvailable:/{print $2}' /proc/meminfo)" -gt 16000000 ]]
[[ "$(df -Pk "$WORK" | awk 'NR==2 {print $4}')" -gt 100000000 ]]
PREPARED="$IMAGE-prepared"
if docker image inspect "$PREPARED" >/dev/null 2>&1; then echo 'Prepared tag already exists' >&2; exit 2; fi
DOCKER_BUILDKIT=0 docker build --network none --memory 4g --memory-swap 4g --cpu-period 100000 --cpu-quota 200000 \
 --file "$ROOT/docker/Dockerfile.ultra-preserved" --build-arg BASE_IMAGE="$PRESERVE_BASE_IMAGE" \
 --build-arg SOURCE_REVISION="${SOURCE_REVISION:-local-preserve}" --tag "$PREPARED" "$ROOT" > "$WORK/evidence/preparation.log" 2>&1
CPU_ARGS=(--rm --runtime runc --network none --memory 4g --memory-swap 4g --cpu-period 100000 --cpu-quota 200000 --pids-limit 128 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=-1)
docker image inspect "$PRESERVE_BASE_IMAGE" "$PREPARED" | \
 docker run "${CPU_ARGS[@]}" -i --read-only --cap-drop ALL --security-opt no-new-privileges \
 -v "$ROOT/tools:/tools:ro" -v "$WORK/fold:/output" --entrypoint python3 "$PREPARED" \
 /tools/preserved_image_config.py --output /output > "$WORK/evidence/configuration-check.json"
chain=$(cat "$WORK/fold/source-chain-id.txt");[[ "$chain" =~ ^[0-9a-f]{64}$ ]] || exit 2
docker_root=$(docker info --format '{{.DockerRootDir}}')
cache=$(cat "$docker_root/image/btrfs/layerdb/sha256/$chain/cache-id");[[ "$cache" =~ ^[0-9a-f]{64}$ ]] || exit 2
snapshot="$docker_root/btrfs/subvolumes/$cache"
[[ "$(readlink -f "$snapshot")" == "$snapshot" && -f "$snapshot/opt/vllm-sm75/runtime-entrypoint.py" ]] || exit 2
# Docker storage is read only here; this helper has neither Docker socket nor user data.
docker run "${CPU_ARGS[@]}" --read-only --cap-drop ALL --cap-add DAC_READ_SEARCH --security-opt no-new-privileges \
 --mount "type=bind,src=$snapshot,dst=/image-root,readonly" -v "$WORK/fold:/output" -v "$ROOT/tools:/tools:ro" \
 --entrypoint python3 "$PREPARED" /tools/pack-preserved-rootfs.py > "$WORK/evidence/pack.log" 2>&1
docker run "${CPU_ARGS[@]}" -i --read-only --cap-drop ALL --security-opt no-new-privileges \
 -v "$WORK/fold:/output" -v "$ROOT/tools:/tools:ro" --entrypoint python3 "$PREPARED" \
 /tools/assemble-preserved-oci.py --layer /output/runtime.layer.tar --layer-info /output/runtime-layer.json \
 --output /output/candidate.docker.tar --tag "$IMAGE" < "$WORK/fold/validated-runtime-config.json" > "$WORK/evidence/assembly.json"
if [[ "${PRESERVE_IMPORT:-1}" != 1 ]]; then echo "Portable candidate: $WORK/fold/candidate.docker.tar"; exit 0; fi
[[ "$(df -Pk "$docker_root" | awk 'NR==2 {print $4}')" -gt 100000000 ]]
docker load -i "$WORK/fold/candidate.docker.tar" > "$WORK/evidence/import.log" 2>&1 & loader=$!
finish(){ trap - EXIT INT TERM;kill -TERM "$loader" 2>/dev/null || true; }
trap finish EXIT INT TERM
while kill -0 "$loader" 2>/dev/null; do
 available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
 rss=$(ps -C dockerd -o rss= | awk '{s+=$1} END {print s+0}')
 disk=$(df -Pk "$docker_root" | awk 'NR==2 {print $4}')
 printf '%s memory_available_kib=%s dockerd_rss_kib=%s docker_free_kib=%s\n' "$(date -Iseconds)" "$available" "$rss" "$disk" >> "$WORK/evidence/resources.log"
 if [[ "$available" -lt 8000000 || "$rss" -gt 4194304 || "$disk" -lt 80000000 ]]; then echo 'Import resource limit reached' >&2;exit 1;fi
 sleep 3
done
wait "$loader";trap - EXIT INT TERM
expected=$(jq -r .image "$WORK/fold/candidate.json")
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$expected" ]] || { echo 'Imported image ID mismatch' >&2; exit 1; }
docker image inspect --format '{{.Id}}|{{.Size}}|{{len .RootFS.Layers}}' "$IMAGE" | tee "$WORK/evidence/imported-image.txt"
echo 'Candidate built; run preservation checks and GPU validation before promotion. No production tag or container was changed.'
