from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List

from config import CELERY_APP, CELERY_BIN, CELERY_CWD, CELERY_ENV


def _run_inspect(command: str, timeout: int = 12) -> Dict[str, Any]:
    proc = subprocess.run(
        [CELERY_BIN, "-A", CELERY_APP, "inspect", command, f"--timeout={timeout}"],
        cwd=CELERY_CWD,
        env={**os.environ, **CELERY_ENV},
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"inspect {command} failed")
    return _parse_inspect_output(proc.stdout)


def _parse_inspect_output(text: str) -> Dict[str, Any]:
    """Parse celery inspect pseudo-json output into {hostname: data}."""
    result: Dict[str, Any] = {}
    blocks = re.split(r"->\s+celery@(\S+):\s+OK", text)
    for i in range(1, len(blocks), 2):
        name = blocks[i]
        body = blocks[i + 1].strip()
        if body.startswith("{"):
            depth = 0
            end = 0
            for j, ch in enumerate(body):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end:
                try:
                    result[name] = json.loads(body[:end])
                except json.JSONDecodeError:
                    result[name] = {"raw": body[:200]}
        elif "pong" in body.lower():
            result[name] = {"pong": True}
        elif "empty" in body.lower():
            result[name] = {}
    return result


def collect_celery() -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    try:
        ping = _run_inspect("ping", timeout=8)
        stats = _run_inspect("stats", timeout=10)
        active = _run_inspect("active", timeout=8)

        all_hosts = sorted(set(ping.keys()) | set(stats.keys()))
        for host in all_hosts:
            stat = stats.get(host, {})
            pool = stat.get("pool", {}) if isinstance(stat, dict) else {}
            total_tasks = stat.get("total", {}) if isinstance(stat, dict) else {}
            task_count = sum(total_tasks.values()) if isinstance(total_tasks, dict) else 0
            active_list = active.get(host, []) if isinstance(active.get(host), list) else []
            nodes.append({
                "hostname": host,
                "online": host in ping,
                "concurrency": pool.get("max-concurrency"),
                "uptime_seconds": stat.get("uptime") if isinstance(stat, dict) else None,
                "total_tasks": task_count,
                "task_breakdown": total_tasks if isinstance(total_tasks, dict) else {},
                "active_count": len(active_list),
                "broker": (stat.get("broker") or {}).get("hostname") if isinstance(stat, dict) else None,
            })

        return {
            "ok": True,
            "online_count": len([n for n in nodes if n["online"]]),
            "total_concurrency": sum(n["concurrency"] or 0 for n in nodes),
            "active_tasks": sum(n["active_count"] for n in nodes),
            "nodes": nodes,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "nodes": nodes}
