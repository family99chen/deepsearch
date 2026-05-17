#!/usr/bin/env python3
"""
Redis memory guard for DeepSearch.

Deletes only historical data that is safe to regenerate:
- completed job hashes and their event streams
- Celery result backend keys

It never deletes queue lists, pending jobs, or running jobs.
After cleanup it triggers BGSAVE so a later reboot does not reload stale data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import redis


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_THRESHOLD_MB = 8192
DEFAULT_TARGET_MB = 4096
DEFAULT_RETAIN_HOURS = 6
DEFAULT_BATCH_SIZE = 5000
LOCK_PATH = Path("/tmp/deepsearch_redis_auto_cleanup.lock")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def used_memory_mb(client: redis.Redis) -> float:
    return float(client.info("memory").get("used_memory", 0)) / 1024 / 1024


def acquire_lock() -> int:
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("cleanup_already_running=true")
        sys.exit(0)
    except Exception:
        # If locking is unavailable, keep going; cron overlap is still unlikely.
        pass
    return fd


def flush_pipeline(pipe: redis.client.Pipeline, pending_ops: int) -> int:
    if pending_ops:
        pipe.execute()
    return 0


def cleanup_completed_jobs(
    client: redis.Redis,
    cutoff: datetime,
    batch_size: int,
) -> tuple[int, int, int]:
    pipe = client.pipeline(transaction=False)
    pending_ops = 0
    deleted_jobs = 0
    deleted_events = 0
    kept_active = 0

    for key in client.scan_iter("job:*", count=batch_size):
        if key.endswith(":events"):
            continue
        data = client.hgetall(key)
        status = data.get("status") or ""
        if status in {"pending", "running"}:
            kept_active += 1
            continue
        if status not in {"success", "failed"}:
            continue
        finished_at = parse_dt(data.get("finished_at")) or parse_dt(data.get("updated_at"))
        if finished_at and finished_at > cutoff:
            continue

        pipe.delete(key)
        pipe.delete(f"{key}:events")
        pending_ops += 2
        deleted_jobs += 1
        deleted_events += 1
        if pending_ops >= batch_size:
            pending_ops = flush_pipeline(pipe, pending_ops)

    flush_pipeline(pipe, pending_ops)
    return deleted_jobs, deleted_events, kept_active


def cleanup_celery_results(client: redis.Redis, batch_size: int) -> int:
    # DeepSearch stores user-visible results in JobStore, so Celery backend keys
    # are only duplicate transport metadata and are safe to drop under pressure.
    pipe = client.pipeline(transaction=False)
    pending_ops = 0
    deleted = 0
    for key in client.scan_iter("celery-task-meta-*", count=batch_size):
        pipe.delete(key)
        pending_ops += 1
        deleted += 1
        if pending_ops >= batch_size:
            pending_ops = flush_pipeline(pipe, pending_ops)
    flush_pipeline(pipe, pending_ops)
    return deleted


def save_if_needed(client: redis.Redis) -> str:
    persistence = client.info("persistence")
    if int(persistence.get("rdb_bgsave_in_progress", 0) or 0):
        return "bgsave_already_in_progress"
    try:
        client.bgsave()
        return "bgsave_started"
    except redis.ResponseError as exc:
        if "Background save already in progress" in str(exc):
            return "bgsave_already_in_progress"
        return f"bgsave_error:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", DEFAULT_REDIS_URL))
    parser.add_argument("--threshold-mb", type=int, default=int(os.getenv("REDIS_CLEANUP_THRESHOLD_MB", DEFAULT_THRESHOLD_MB)))
    parser.add_argument("--target-mb", type=int, default=int(os.getenv("REDIS_CLEANUP_TARGET_MB", DEFAULT_TARGET_MB)))
    parser.add_argument("--retain-hours", type=float, default=float(os.getenv("REDIS_CLEANUP_RETAIN_HOURS", DEFAULT_RETAIN_HOURS)))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("REDIS_CLEANUP_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lock_fd = acquire_lock()
    started = time.time()
    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    before_mb = used_memory_mb(client)
    print(f"started_at={utc_now().isoformat()}")
    print(f"used_memory_before_mb={before_mb:.2f}")
    print(f"threshold_mb={args.threshold_mb}")
    print(f"retain_hours={args.retain_hours}")

    if before_mb < args.threshold_mb and not args.force:
        print("cleanup_skipped=below_threshold")
        os.close(lock_fd)
        return 0

    cutoff = utc_now() - timedelta(hours=args.retain_hours)
    deleted_jobs, deleted_events, kept_active = cleanup_completed_jobs(client, cutoff, args.batch_size)
    deleted_meta = cleanup_celery_results(client, args.batch_size)
    after_mb = used_memory_mb(client)

    save_status = "not_needed"
    if deleted_jobs or deleted_events or deleted_meta:
        save_status = save_if_needed(client)

    print(f"deleted_completed_jobs={deleted_jobs}")
    print(f"deleted_event_streams={deleted_events}")
    print(f"deleted_celery_task_meta={deleted_meta}")
    print(f"kept_pending_running_jobs={kept_active}")
    print(f"used_memory_after_mb={after_mb:.2f}")
    print(f"bgsave_status={save_status}")
    print(f"elapsed_seconds={time.time() - started:.2f}")
    os.close(lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
