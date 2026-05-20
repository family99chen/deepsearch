#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT_DIR="/root/deepsearch"
REDIS_URL_VALUE="redis://172.26.2.150:6379/0"
MONGODB_URL_VALUE="mongodb://172.26.2.150:27018/"

REMOTE_WORKERS=(
  "worker1:worker1"
  "worker2:worker2"
  "worker3:worker3"
)

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o StrictHostKeyChecking=accept-new
)

worker_command() {
  local action="$1"
  case "$action" in
    restart)
      cat <<'CMD'
set -e
cd /root/deepsearch
export REDIS_URL="redis://172.26.2.150:6379/0"
export REDIS_URL="${WORKER_REDIS_URL:-$REDIS_URL}"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://172.26.2.150:27018/"
export MONGODB_URL="${WORKER_MONGODB_URL:-$MONGODB_URL}"

find_celery() {
  for candidate in \
    /root/miniconda3/envs/webprof/bin/celery \
    /root/deepsearch/.venv/bin/celery \
    /root/.venvs/deepsearch-worker/bin/celery \
    /root/.venvs/deepsearch-worker/bin/celery3 \
    celery
  do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

kill_existing_celery() {
  python3 - <<'PY'
import os
import signal

current = {os.getpid(), os.getppid()}
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in current:
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if "celery" in cmdline and "tasks:celery_app" in cmdline and " worker" in cmdline:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"killed_celery_pid={pid}")
        except ProcessLookupError:
            pass
PY
}

CELERY_BIN="$(find_celery)"
PM2_BIN="$(command -v pm2 || true)"
if [ -n "$PM2_BIN" ] && "$PM2_BIN" describe deepsearch-worker >/dev/null 2>&1; then
  "$PM2_BIN" restart deepsearch-worker --update-env
elif [ -n "$PM2_BIN" ]; then
  kill_existing_celery
  "$PM2_BIN" start "$CELERY_BIN" \
    --name deepsearch-worker \
    --interpreter none \
    -- -A tasks:celery_app worker --loglevel=INFO --concurrency=20 --queues=celery,patent
  "$PM2_BIN" save
else
  kill_existing_celery
  mkdir -p /root/deepsearch/logs
  nohup "$CELERY_BIN" -A tasks:celery_app worker --loglevel=INFO --concurrency=20 --queues=celery,patent \
    > /root/deepsearch/logs/deepsearch-worker.log 2>&1 &
  echo "started deepsearch-worker with nohup pid=$!"
fi
CMD
      ;;
    stop)
      cat <<'CMD'
set -e
if command -v pm2 >/dev/null 2>&1 && pm2 describe deepsearch-worker >/dev/null 2>&1; then
  pm2 stop deepsearch-worker
  pm2 save
else
  python3 - <<'PY'
import os
import signal

current = {os.getpid(), os.getppid()}
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in current:
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if "celery" in cmdline and "tasks:celery_app" in cmdline and " worker" in cmdline:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"killed_celery_pid={pid}")
        except ProcessLookupError:
            pass
PY
fi
CMD
      ;;
    status)
      cat <<'CMD'
set -e
hostname
if command -v pm2 >/dev/null 2>&1; then
  pm2 status
else
  echo "pm2_not_found=true"
  ps -eo pid,ppid,stat,pcpu,pmem,rss,comm,args | grep -E 'celery|deepsearch-worker' | grep -v grep || true
fi
CMD
      ;;
    ping)
      cat <<'CMD'
set -e
cd /root/deepsearch
export REDIS_URL="redis://172.26.2.150:6379/0"
export REDIS_URL="${WORKER_REDIS_URL:-$REDIS_URL}"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://172.26.2.150:27018/"
export MONGODB_URL="${WORKER_MONGODB_URL:-$MONGODB_URL}"
find_celery() {
  for candidate in \
    /root/miniconda3/envs/webprof/bin/celery \
    /root/deepsearch/.venv/bin/celery \
    /root/.venvs/deepsearch-worker/bin/celery \
    /root/.venvs/deepsearch-worker/bin/celery3 \
    celery
  do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}
"$(find_celery)" -A tasks:celery_app inspect ping --timeout=8
CMD
      ;;
    *)
      echo "Unsupported worker action: $action" >&2
      return 2
      ;;
  esac
}

run_local() {
  local action="$1"
  echo "===== local ====="
  bash -lc "$(worker_command "$action")"
}

run_remote() {
  local name="$1"
  local host="$2"
  local action="$3"
  echo "===== ${name} (${host}) ====="
  ssh "${SSH_OPTS[@]}" "root@${host}" \
    "WORKER_REDIS_URL=redis://47.250.116.163:6379/0 WORKER_MONGODB_URL=mongodb://47.250.116.163:27018/ bash -lc $(printf '%q' "$(worker_command "$action")")"
}

usage() {
  cat <<'EOF'
Usage: scripts/manage_workers.sh <action>

Actions:
  status   Show PM2 status on local + remote workers
  restart  Restart or start deepsearch-worker on local + remote workers
  stop     Stop deepsearch-worker on local + remote workers
  ping     Run Celery ping from each node
EOF
}

case "$ACTION" in
  status|restart|stop|ping)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [ "$ACTION" = "ping" ]; then
  run_local "$ACTION"
  exit $?
fi

run_local "$ACTION" &
pids=("$!")

for worker in "${REMOTE_WORKERS[@]}"; do
  name="${worker%%:*}"
  host="${worker#*:}"
  run_remote "$name" "$host" "$ACTION" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

exit "$failed"
