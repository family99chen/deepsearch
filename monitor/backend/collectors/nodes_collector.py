from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional

from config import SSH_OPTS, WORKERS

_REMOTE_BASH = r"""#!/usr/bin/env bash
set -e
echo "HOSTNAME:$(hostname)"
echo "UPTIME:$(uptime -p 2>/dev/null || uptime)"
echo "LOAD:$(uptime | sed -n 's/.*load average: \([^,]*\).*/\1/p')"
echo "MEM:$(free -h | awk '/Mem:/ {print $3"/"$2}')"
echo "DISK:$(df -h / | awk 'NR==2 {print $3"/"$2" ("$5")"}')"
echo "CELERY_PROCS:$(ps aux | grep -E 'celery.*tasks:celery_app worker' | grep -v grep | wc -l | tr -d ' ')"
ps aux | grep -E 'celery.*tasks:celery_app worker' | grep -v grep | head -1 | sed 's/^/CELERY_CMD:/'
if command -v pm2 >/dev/null 2>&1; then
  pm2 jlist 2>/dev/null | tr -d '\n' | sed 's/^/PM2_JSON:/' || true
  pm2 status 2>/dev/null | grep deepsearch-worker | sed 's/^/PM2_STATUS:/' || true
else
  echo "PM2:none"
fi
"""


def _parse_output(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("CELERY_CMD:"):
            data["CELERY_CMD"] = line[len("CELERY_CMD:"):].strip()
            continue
        if line.startswith("PM2_JSON:"):
            data["PM2_JSON"] = line[len("PM2_JSON:"):].strip()
            continue
        if line.startswith("PM2_STATUS:"):
            data["PM2_STATUS"] = line[len("PM2_STATUS:"):].strip()
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            data[key.strip()] = val.strip()
    return data


def _parse_pm2(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = data.get("PM2_JSON")
    if raw:
        try:
            items = json.loads(raw)
            for item in items:
                if item.get("name") == "deepsearch-worker":
                    env = item.get("pm2_env", {})
                    return {
                        "status": env.get("status"),
                        "restarts": env.get("restart_time", 0),
                        "memory_mb": round((item.get("monit", {}).get("memory") or 0) / 1024 / 1024, 1),
                    }
            return {"status": "no_deepsearch_worker"}
        except Exception:
            return {"status": "parse_error"}
    status = data.get("PM2_STATUS", "")
    if "online" in status:
        return {"status": "online", "detail": status}
    if data.get("PM2") == "none":
        return {"status": "no_pm2"}
    return None


def _build_node(worker_id: str, label: str, parsed: Dict[str, Any], ok: bool, error: Optional[str] = None) -> Dict[str, Any]:
    celery_cmd = parsed.get("CELERY_CMD", "")
    queues = "celery, patent" if "patent" in celery_cmd else "celery"
    node = {
        "id": worker_id,
        "label": label,
        "hostname": parsed.get("HOSTNAME"),
        "uptime": parsed.get("UPTIME"),
        "load": parsed.get("LOAD"),
        "memory": parsed.get("MEM"),
        "disk": parsed.get("DISK"),
        "celery_processes": int(parsed.get("CELERY_PROCS") or 0),
        "celery_queues": queues,
        "pm2": _parse_pm2(parsed),
        "ok": ok,
    }
    if error:
        node["error"] = error
    return node


def _run_local() -> Dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-c", _REMOTE_BASH],
        capture_output=True,
        text=True,
        timeout=15,
    )
    parsed = _parse_output(proc.stdout)
    return _build_node("main", "主节点", parsed, proc.returncode == 0, proc.stderr.strip() or None)


def _run_remote(worker_id: str, label: str, ssh_host: str) -> Dict[str, Any]:
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, f"root@{ssh_host}", "bash", "-s"],
        input=_REMOTE_BASH,
        capture_output=True,
        text=True,
        timeout=25,
    )
    if proc.returncode != 0:
        return {
            "id": worker_id,
            "label": label,
            "ok": False,
            "error": (proc.stderr or proc.stdout or "ssh failed").strip()[:300],
        }
    parsed = _parse_output(proc.stdout)
    return _build_node(worker_id, label, parsed, True)


def collect_nodes() -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    errors: List[str] = []
    for w in WORKERS:
        try:
            if w["ssh_host"] is None:
                nodes.append(_run_local())
            else:
                nodes.append(_run_remote(w["id"], w["label"], w["ssh_host"]))
        except Exception as exc:
            errors.append(f"{w['id']}: {exc}")
            nodes.append({"id": w["id"], "label": w["label"], "ok": False, "error": str(exc)})

    online_workers = [n for n in nodes if n.get("celery_processes", 0) > 0]
    return {
        "ok": len(errors) == 0 and all(n.get("ok", False) for n in nodes),
        "errors": errors,
        "nodes": nodes,
        "workers_with_celery": len(online_workers),
    }
