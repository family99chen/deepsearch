"""
Org Pipeline 详细统计模块

记录 organization / social_media / arbitrary 三类联网搜索的详细统计信息：
- 总请求次数
- 缓存命中次数
- 成功 / 未找到 / 错误次数
- 链接与 worker 执行统计
"""

import json
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional


PIPELINE_TYPES = ("organization", "social_media", "arbitrary")


def _empty_worker_stats() -> Dict:
    return {
        "total": 0,
        "success": 0,
        "failed": 0,
        "timeout": 0,
        "by_mode": {},
    }


def _empty_pipeline_stats() -> Dict:
    return {
        "total_requests": 0,
        "cache_hits": 0,
        "success": 0,
        "not_found": 0,
        "error": 0,
        "links_found_total": 0,
        "links_processed_total": 0,
        "worker_success_total": 0,
        "worker": _empty_worker_stats(),
    }


class OrgPipelineStats:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, storage_path: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "total_usage" / "org_pipeline_stats.json"

        self.storage_path = Path(storage_path)
        self._file_lock = threading.Lock()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load_data()
        self._initialized = True

    def _load_data(self) -> Dict:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return self._get_empty_stats()

    def _get_empty_stats(self) -> Dict:
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "success": 0,
            "not_found": 0,
            "error": 0,
            "links_found_total": 0,
            "links_processed_total": 0,
            "worker_success_total": 0,
            "worker": _empty_worker_stats(),
            "by_pipeline": {name: _empty_pipeline_stats() for name in PIPELINE_TYPES},
            "daily": {},
            "last_updated": None,
        }

    def _get_empty_daily_stats(self) -> Dict:
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "success": 0,
            "not_found": 0,
            "error": 0,
            "links_found_total": 0,
            "links_processed_total": 0,
            "worker_success_total": 0,
            "worker": _empty_worker_stats(),
            "by_pipeline": {name: _empty_pipeline_stats() for name in PIPELINE_TYPES},
        }

    def _save_data(self):
        with self._file_lock:
            self._data["last_updated"] = datetime.now().isoformat()
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _get_today(self) -> str:
        return date.today().isoformat()

    def _ensure_daily(self, date_str: str):
        if "daily" not in self._data:
            self._data["daily"] = {}
        if date_str not in self._data["daily"]:
            self._data["daily"][date_str] = self._get_empty_daily_stats()

    def _ensure_pipeline(self, bucket: Dict, pipeline_type: str):
        if "by_pipeline" not in bucket:
            bucket["by_pipeline"] = {}
        if pipeline_type not in bucket["by_pipeline"]:
            bucket["by_pipeline"][pipeline_type] = _empty_pipeline_stats()

    def _record_worker(self, worker_bucket: Dict, success: bool, mode: str, error: Optional[str]):
        worker_bucket["total"] += 1
        mode_key = mode or "unknown"
        worker_bucket["by_mode"][mode_key] = worker_bucket["by_mode"].get(mode_key, 0) + 1
        if success:
            worker_bucket["success"] += 1
        else:
            worker_bucket["failed"] += 1
            if error and "超时" in error:
                worker_bucket["timeout"] += 1

    def record_request(self, pipeline_type: str):
        today = self._get_today()
        with self._lock:
            self._ensure_daily(today)
            self._ensure_pipeline(self._data, pipeline_type)
            self._ensure_pipeline(self._data["daily"][today], pipeline_type)

            self._data["total_requests"] += 1
            self._data["by_pipeline"][pipeline_type]["total_requests"] += 1
            self._data["daily"][today]["total_requests"] += 1
            self._data["daily"][today]["by_pipeline"][pipeline_type]["total_requests"] += 1
            self._save_data()

    def record_cache_hit(self, pipeline_type: str):
        today = self._get_today()
        with self._lock:
            self._ensure_daily(today)
            self._ensure_pipeline(self._data, pipeline_type)
            self._ensure_pipeline(self._data["daily"][today], pipeline_type)

            self._data["cache_hits"] += 1
            self._data["by_pipeline"][pipeline_type]["cache_hits"] += 1
            self._data["daily"][today]["cache_hits"] += 1
            self._data["daily"][today]["by_pipeline"][pipeline_type]["cache_hits"] += 1
            self._save_data()

    def record_error(self, pipeline_type: str):
        today = self._get_today()
        with self._lock:
            self._ensure_daily(today)
            self._ensure_pipeline(self._data, pipeline_type)
            self._ensure_pipeline(self._data["daily"][today], pipeline_type)

            self._data["error"] += 1
            self._data["by_pipeline"][pipeline_type]["error"] += 1
            self._data["daily"][today]["error"] += 1
            self._data["daily"][today]["by_pipeline"][pipeline_type]["error"] += 1
            self._save_data()

    def record_not_found(self, pipeline_type: str, links_found: int = 0, links_processed: int = 0):
        today = self._get_today()
        with self._lock:
            self._ensure_daily(today)
            self._ensure_pipeline(self._data, pipeline_type)
            self._ensure_pipeline(self._data["daily"][today], pipeline_type)

            self._data["not_found"] += 1
            self._data["links_found_total"] += links_found
            self._data["links_processed_total"] += links_processed
            self._data["by_pipeline"][pipeline_type]["not_found"] += 1
            self._data["by_pipeline"][pipeline_type]["links_found_total"] += links_found
            self._data["by_pipeline"][pipeline_type]["links_processed_total"] += links_processed

            self._data["daily"][today]["not_found"] += 1
            self._data["daily"][today]["links_found_total"] += links_found
            self._data["daily"][today]["links_processed_total"] += links_processed
            self._data["daily"][today]["by_pipeline"][pipeline_type]["not_found"] += 1
            self._data["daily"][today]["by_pipeline"][pipeline_type]["links_found_total"] += links_found
            self._data["daily"][today]["by_pipeline"][pipeline_type]["links_processed_total"] += links_processed
            self._save_data()

    def record_success(
        self,
        pipeline_type: str,
        links_found: int,
        links_processed: int,
        worker_success: int,
    ):
        today = self._get_today()
        with self._lock:
            self._ensure_daily(today)
            self._ensure_pipeline(self._data, pipeline_type)
            self._ensure_pipeline(self._data["daily"][today], pipeline_type)

            self._data["success"] += 1
            self._data["links_found_total"] += links_found
            self._data["links_processed_total"] += links_processed
            self._data["worker_success_total"] += worker_success
            self._data["by_pipeline"][pipeline_type]["success"] += 1
            self._data["by_pipeline"][pipeline_type]["links_found_total"] += links_found
            self._data["by_pipeline"][pipeline_type]["links_processed_total"] += links_processed
            self._data["by_pipeline"][pipeline_type]["worker_success_total"] += worker_success

            self._data["daily"][today]["success"] += 1
            self._data["daily"][today]["links_found_total"] += links_found
            self._data["daily"][today]["links_processed_total"] += links_processed
            self._data["daily"][today]["worker_success_total"] += worker_success
            self._data["daily"][today]["by_pipeline"][pipeline_type]["success"] += 1
            self._data["daily"][today]["by_pipeline"][pipeline_type]["links_found_total"] += links_found
            self._data["daily"][today]["by_pipeline"][pipeline_type]["links_processed_total"] += links_processed
            self._data["daily"][today]["by_pipeline"][pipeline_type]["worker_success_total"] += worker_success
            self._save_data()

    def record_worker_result(self, pipeline_type: str, success: bool, mode: str, error: Optional[str] = None):
        today = self._get_today()
        with self._lock:
            self._ensure_daily(today)
            self._ensure_pipeline(self._data, pipeline_type)
            self._ensure_pipeline(self._data["daily"][today], pipeline_type)

            self._record_worker(self._data["worker"], success, mode, error)
            self._record_worker(self._data["by_pipeline"][pipeline_type]["worker"], success, mode, error)
            self._record_worker(self._data["daily"][today]["worker"], success, mode, error)
            self._record_worker(self._data["daily"][today]["by_pipeline"][pipeline_type]["worker"], success, mode, error)
            self._save_data()

    def get_stats(self) -> Dict:
        with self._lock:
            return self._data.copy()

    def get_today_stats(self) -> Dict:
        today = self._get_today()
        with self._lock:
            if today in self._data.get("daily", {}):
                stats = self._data["daily"][today].copy()
                stats["date"] = today
                return stats
            return {"date": today, **self._get_empty_daily_stats()}


_stats: Optional[OrgPipelineStats] = None


def get_org_pipeline_stats() -> OrgPipelineStats:
    global _stats
    if _stats is None:
        _stats = OrgPipelineStats()
    return _stats


def record_request(pipeline_type: str):
    get_org_pipeline_stats().record_request(pipeline_type)


def record_cache_hit(pipeline_type: str):
    get_org_pipeline_stats().record_cache_hit(pipeline_type)


def record_error(pipeline_type: str):
    get_org_pipeline_stats().record_error(pipeline_type)


def record_not_found(pipeline_type: str, links_found: int = 0, links_processed: int = 0):
    get_org_pipeline_stats().record_not_found(pipeline_type, links_found, links_processed)


def record_success(pipeline_type: str, links_found: int, links_processed: int, worker_success: int):
    get_org_pipeline_stats().record_success(pipeline_type, links_found, links_processed, worker_success)


def record_worker_result(pipeline_type: str, success: bool, mode: str, error: Optional[str] = None):
    get_org_pipeline_stats().record_worker_result(pipeline_type, success, mode, error)
