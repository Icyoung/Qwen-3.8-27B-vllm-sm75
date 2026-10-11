#!/usr/bin/env bash
# Same-image BenchLocal jobs: fixed host launcher, no Docker socket in Ultra.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NAME=${CONTAINER_NAME:-vllm-sm75-ultra}
[[ "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}$ ]] || { echo 'Invalid controller name' >&2; exit 2; }
if [[ -d /boot/config/plugins && -f /etc/unraid-version ]]; then STATE=/boot/config/plugins/vllm-sm75-benchlocal; else STATE=/etc/vllm-sm75-benchlocal; fi
INSTANCE_ID=$(printf '%s' "$NAME" | sha256sum | cut -c1-24)
RUNTIME="/run/sm75-benchlocal/$INSTANCE_ID"
INSTANCE="$STATE/instances/$NAME"
case "${1:-install}" in
 path) printf '%s\n' "$RUNTIME/control"; exit 0 ;;
 status) [[ -f "$RUNTIME/status.json" ]] && cat "$RUNTIME/status.json" || printf '%s\n' '{"ready":false}'; exit 0 ;;
 ensure|stop) exec bash "$STATE/service.sh" "$1" "$NAME" ;;
 install) ;;
 *) echo 'Usage: benchlocal.sh install | ensure | stop | status | path' >&2; exit 2 ;;
esac
[[ $EUID -eq 0 ]] || { echo '按需评测启动器需要由宿主管理员安装。' >&2; exit 2; }
: "${IMAGE:?Set IMAGE to the Ultra image to use}"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
[[ "$IMAGE_ID" =~ ^sha256:[a-f0-9]{64}$ ]] || exit 2
[[ "$(docker image inspect --format '{{index .Config.Labels "io.sm75.benchlocal.execution"}}' "$IMAGE_ID")" == ondemand-v1 ]] || { echo '当前镜像未包含按需评测组件，请先构建新版 Ultra。' >&2; exit 2; }
if [[ -s "$RUNTIME/status.json" ]] && grep -q '"active":true' "$RUNTIME/status.json"; then echo '评测任务仍在执行，安装前请先停止评测。' >&2; exit 2; fi
if [[ -f "$STATE/service.sh" ]]; then bash "$STATE/service.sh" stop "$NAME"; fi
umask 077
mkdir -p "$INSTANCE" "$RUNTIME/control"
chmod 700 "$STATE" "$STATE/instances" "$INSTANCE" "$RUNTIME" "$RUNTIME/control"
[[ ! -L "$STATE" && ! -L "$INSTANCE" && "$(readlink -f "$RUNTIME")" == "$RUNTIME" ]] || { echo '启动器目录不能是符号链接。' >&2; exit 2; }
docker run --rm --runtime runc --network none -e NVIDIA_VISIBLE_DEVICES=void --entrypoint /bin/cat "$IMAGE_ID" /opt/sm75-workbench/benchlocal-verifiers/host-broker > "$INSTANCE/host-broker.new"
chmod 700 "$INSTANCE/host-broker.new"
mv -f -- "$INSTANCE/host-broker.new" "$INSTANCE/host-broker"
PROFILE=
if docker run --rm --runtime runc --network none -e NVIDIA_VISIBLE_DEVICES=void --entrypoint /bin/cat "$IMAGE_ID" /opt/sm75-workbench/benchlocal-verifiers/browser-seccomp.json > "$INSTANCE/browser-seccomp.json.new" 2>/dev/null; then
 chmod 600 "$INSTANCE/browser-seccomp.json.new"; mv -f -- "$INSTANCE/browser-seccomp.json.new" "$INSTANCE/browser-seccomp.json"
 PROFILE=",\"seccomp\":\"$INSTANCE/browser-seccomp.json\""
else rm -f -- "$INSTANCE/browser-seccomp.json.new"; fi
printf '{"controller":"%s","image":"%s","root":"%s"%s}\n' "$NAME" "$IMAGE_ID" "$RUNTIME" "$PROFILE" > "$INSTANCE/config.json.new"
chmod 600 "$INSTANCE/config.json.new"; mv -f -- "$INSTANCE/config.json.new" "$INSTANCE/config.json"
cp "$SCRIPT_DIR/helpers/benchlocal-service.sh" "$STATE/service.sh"; chmod 700 "$STATE/service.sh"
if [[ "$STATE" == /boot/config/plugins/vllm-sm75-benchlocal ]]; then
 # This hook only starts the fixed launchers; it never starts inference models.
 if ! grep -Fq '# SM75 BenchLocal on-demand launcher' /boot/config/go; then
  cat >> /boot/config/go <<'BOOT'

# SM75 BenchLocal on-demand launcher
bash /boot/config/plugins/vllm-sm75-benchlocal/service.sh ensure-all
BOOT
 fi
elif command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
 cat > /etc/systemd/system/vllm-sm75-benchlocal.service <<'UNIT'
[Unit]
Description=VLLM-SM75 fixed BenchLocal job launcher
After=docker.service
Requires=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash /etc/vllm-sm75-benchlocal/service.sh ensure-all
ExecStop=/bin/bash /etc/vllm-sm75-benchlocal/service.sh stop-all
[Install]
WantedBy=multi-user.target
UNIT
 systemctl daemon-reload
 systemctl enable vllm-sm75-benchlocal.service >/dev/null
fi
bash "$STATE/service.sh" ensure "$NAME"
printf '按需评测启动器已安装；Ultra 通道：%s\n' "$RUNTIME/control"
