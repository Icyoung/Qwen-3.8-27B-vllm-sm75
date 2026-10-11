#!/usr/bin/env bash
# Host supervisor; copied by benchlocal.sh into a root-controlled installation.
set -euo pipefail
STATE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
valid_name(){ [[ "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}$ ]]; }
paths(){ valid_name "$1" || exit 2; INSTANCE="$STATE/instances/$1"; INSTANCE_ID=$(printf '%s' "$1" | sha256sum | cut -c1-24); RUNTIME="/run/sm75-benchlocal/$INSTANCE_ID"; EXECROOT="/usr/local/libexec/vllm-sm75-benchlocal/$INSTANCE_ID"; }
owned_pid(){
 local file="$1" expected="$2" pid
 [[ -s "$file" ]] || return 1
 read -r pid < "$file"
 [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
 [[ "$(tr '\0' ' ' < "/proc/$pid/cmdline")" == *"$STATE/service.sh supervise $expected "* ]] || return 1
 printf '%s' "$pid"
}
start_one(){
 paths "$1"; [[ -s "$INSTANCE/config.json" && -s "$INSTANCE/host-broker" ]] || { echo "启动器尚未安装：$1" >&2; return 1; }
 mkdir -p "$RUNTIME/control"; chmod 700 "$RUNTIME" "$RUNTIME/control"
 if owned_pid "$RUNTIME/supervisor.pid" "$1" >/dev/null; then return; fi
 # Unraid mounts both /boot and /run with noexec. Keep executable code in /usr/local.
 mkdir -p "$EXECROOT"; chmod 700 /usr/local/libexec/vllm-sm75-benchlocal "$EXECROOT"
 [[ "$(readlink -f "$EXECROOT")" == "$EXECROOT" ]] || { echo '启动器程序目录不能是符号链接。' >&2; return 1; }
 cp "$INSTANCE/host-broker" "$EXECROOT/host-broker.new"; chmod 700 "$EXECROOT/host-broker.new"; mv -f -- "$EXECROOT/host-broker.new" "$EXECROOT/host-broker"
 nohup bash "$STATE/service.sh" supervise "$1" >> "$RUNTIME/service.log" 2>&1 </dev/null &
}
stop_one(){
 paths "$1"; local pid
 if pid=$(owned_pid "$RUNTIME/supervisor.pid" "$1"); then
  kill -TERM "$pid"
  for _ in $(seq 1 200); do owned_pid "$RUNTIME/supervisor.pid" "$1" >/dev/null || break; sleep .1; done
  if owned_pid "$RUNTIME/supervisor.pid" "$1" >/dev/null; then echo '启动器仍在清理任务，请稍后重试。' >&2; return 1; fi
 fi
}
case "${1:-}" in
 ensure-all)
  [[ -d "$STATE/instances" ]] || exit 0
  for instance in "$STATE"/instances/*; do [[ -d "$instance" ]] || continue; name=${instance##*/}; valid_name "$name" || continue; start_one "$name"; done
  ;;
 stop-all)
  for instance in "$STATE"/instances/*; do [[ -d "$instance" ]] || continue; name=${instance##*/}; valid_name "$name" || continue; stop_one "$name"; done
  ;;
 ensure) start_one "${2:?controller name required}" ;;
 stop) stop_one "${2:?controller name required}" ;;
 supervise)
  paths "${2:?controller name required}"
  mkdir -p "$RUNTIME/control"; chmod 700 "$RUNTIME" "$RUNTIME/control"
  exec 9>"$RUNTIME/supervisor.lock"
  flock -n 9 || exit 0
  printf '%s\n' "$BASHPID" > "$RUNTIME/supervisor.pid"
  child=
  finish(){ trap - TERM INT EXIT; if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; rm -f -- "$RUNTIME/supervisor.pid"; exit 0; }
  trap finish TERM INT EXIT
  while true; do
   GOMAXPROCS=2 GOMEMLIMIT=128MiB "$EXECROOT/host-broker" --config "$INSTANCE/config.json" &
   child=$!
   wait "$child" || true
   child=
   sleep 2 & child=$!; wait "$child" || true; child=
  done
  ;;
 *) echo 'Usage: service.sh ensure-all | stop-all | ensure NAME | stop NAME | supervise NAME' >&2; exit 2 ;;
esac
