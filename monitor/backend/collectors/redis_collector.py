from __future__ import annotations

from typing import Any, Dict

import redis

from config import REDIS_URL


def collect_redis() -> Dict[str, Any]:
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        info = client.info("memory")
        return {
            "ok": True,
            "used_memory_human": info.get("used_memory_human"),
            "used_memory_mb": round(float(info.get("used_memory", 0)) / 1024 / 1024, 2),
            "maxmemory_human": info.get("maxmemory_human") or "未限制",
            "dbsize": client.dbsize(),
            "queue_celery": client.llen("celery"),
            "queue_patent": client.llen("patent"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
