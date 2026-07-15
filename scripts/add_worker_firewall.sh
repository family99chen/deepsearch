#!/usr/bin/env bash
# Add UFW whitelist rules on the main node for a DeepSearch worker IP.
# NOTE: Aliyun ECS Security Group must also allow the same IP/ports in the console.
set -euo pipefail

WORKER_IP="${1:-}"
if [ -z "$WORKER_IP" ]; then
  echo "Usage: $0 <worker_public_ip>" >&2
  echo "Example: $0 47.81.210.142" >&2
  exit 2
fi

for port in 6379 27018; do
  if ufw status | grep -q "${WORKER_IP}.*${port}/tcp"; then
    echo "already_allowed: ${WORKER_IP}:${port}"
  else
    ufw allow from "${WORKER_IP}" to any port "${port}" proto tcp comment "deepsearch worker ${WORKER_IP}:${port}"
    echo "added: ${WORKER_IP}:${port}"
  fi
done

echo
echo "UFW rules for ${WORKER_IP}:"
ufw status | grep "${WORKER_IP}" || true
echo
echo "Reminder: also add inbound rules in Aliyun ECS Security Group for ${WORKER_IP}:"
echo "  TCP 6379  (Redis)"
echo "  TCP 27018 (MongoDB)"
