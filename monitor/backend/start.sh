#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /root/miniconda3/envs/webprof/bin/uvicorn main:app --host 127.0.0.1 --port 8091
