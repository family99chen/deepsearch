"""
Redis-backed job metadata, result, and event storage.

The API process writes jobs here and streams events from here; worker
processes update the same records while running pipeline tasks.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_JOB_TTL_SECONDS = 7 * 24 * 3600
DEFAULT_STREAM_MAXLEN = 10000


def get_redis_url() -> str:
    return os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL") or DEFAULT_REDIS_URL


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class JobStore:
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        stream_maxlen: Optional[int] = None,
    ):
        self.redis_url = redis_url or get_redis_url()
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("JOB_TTL_SECONDS", str(DEFAULT_JOB_TTL_SECONDS))
        )
        self.stream_maxlen = stream_maxlen or int(
            os.getenv("JOB_STREAM_MAXLEN", str(DEFAULT_STREAM_MAXLEN))
        )
        self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)

    @staticmethod
    def job_key(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def events_key(job_id: str) -> str:
        return f"job:{job_id}:events"

    def ping(self) -> bool:
        return bool(self.redis.ping())

    def create_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        job_id: Optional[str] = None,
    ) -> str:
        job_id = job_id or uuid.uuid4().hex
        key = self.job_key(job_id)
        now = _utc_now()
        self.redis.hset(
            key,
            mapping={
                "job_id": job_id,
                "job_type": job_type,
                "status": "pending",
                "payload_json": _json_dumps(payload),
                "created_at": now,
                "updated_at": now,
            },
        )
        self.redis.expire(key, self.ttl_seconds)
        self.redis.expire(self.events_key(job_id), self.ttl_seconds)
        return job_id

    def set_running(self, job_id: str, celery_task_id: Optional[str] = None) -> None:
        mapping = {
            "status": "running",
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        if celery_task_id:
            mapping["celery_task_id"] = celery_task_id
        self.redis.hset(self.job_key(job_id), mapping=mapping)
        self._refresh_ttl(job_id)

    def set_dispatched(self, job_id: str, celery_task_id: str) -> None:
        self.redis.hset(
            self.job_key(job_id),
            mapping={
                "celery_task_id": celery_task_id,
                "updated_at": _utc_now(),
            },
        )
        self._refresh_ttl(job_id)

    def set_success(self, job_id: str, result: Dict[str, Any]) -> None:
        now = _utc_now()
        self.redis.hset(
            self.job_key(job_id),
            mapping={
                "status": "success",
                "result_json": _json_dumps(result),
                "finished_at": now,
                "updated_at": now,
            },
        )
        self._refresh_ttl(job_id)

    def set_failed(
        self,
        job_id: str,
        error: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _utc_now()
        mapping = {
            "status": "failed",
            "error": error,
            "finished_at": now,
            "updated_at": now,
        }
        if result is not None:
            mapping["result_json"] = _json_dumps(result)
        self.redis.hset(self.job_key(job_id), mapping=mapping)
        self._refresh_ttl(job_id)

    def append_event(
        self,
        job_id: str,
        tag: str,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        fields = {
            "tag": tag,
            "message": message or "",
            "payload_json": _json_dumps(payload) if payload is not None else "",
            "ts": str(time.time()),
        }
        event_id = self.redis.xadd(
            self.events_key(job_id),
            fields,
            maxlen=self.stream_maxlen,
            approximate=True,
        )
        self.redis.expire(self.events_key(job_id), self.ttl_seconds)
        return event_id

    def append_log(self, job_id: str, line: str) -> str:
        return self.append_event(job_id, "[LOG]", message=line)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        data = self.redis.hgetall(self.job_key(job_id))
        if not data:
            return None
        return {
            "job_id": data.get("job_id", job_id),
            "job_type": data.get("job_type"),
            "status": data.get("status"),
            "payload": _json_loads(data.get("payload_json"), {}),
            "result": _json_loads(data.get("result_json"), None),
            "error": data.get("error"),
            "created_at": data.get("created_at"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
            "updated_at": data.get("updated_at"),
            "celery_task_id": data.get("celery_task_id"),
        }

    def read_events(
        self,
        job_id: str,
        last_id: str = "0-0",
        block_ms: int = 5000,
        count: int = 100,
    ) -> Tuple[List[Tuple[str, Dict[str, str]]], str]:
        response = self.redis.xread(
            {self.events_key(job_id): last_id},
            block=block_ms,
            count=count,
        )
        events: List[Tuple[str, Dict[str, str]]] = []
        next_id = last_id
        for _, stream_events in response:
            for event_id, fields in stream_events:
                events.append((event_id, fields))
                next_id = event_id
        return events, next_id

    def _refresh_ttl(self, job_id: str) -> None:
        self.redis.expire(self.job_key(job_id), self.ttl_seconds)
        self.redis.expire(self.events_key(job_id), self.ttl_seconds)


_STORE: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _STORE
    if _STORE is None:
        _STORE = JobStore()
    return _STORE
