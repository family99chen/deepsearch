"""
DeepSearch 调用统计模块。

统计口径：
- 每次 deepsearch 调用只记 1 次 total_requests
- success / not_found / error 三者互斥，理论上加总等于 total_requests
- cache_hits 独立统计，可与 success / not_found / error 同时发生
"""

import copy
import json
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - 非 Unix 环境回退
    fcntl = None


OUTCOME_FIELDS = ("success", "not_found", "error")


def _empty_stats() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "cache_hits": 0,
        "success": 0,
        "not_found": 0,
        "error": 0,
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
        self._lock_path = self.storage_path.with_name(f"{self.storage_path.name}.lock")
        self._file_lock = threading.Lock()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load_data()
        self._initialized = True

    def _load_data(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return self._normalize_data(json.loads(content))
            except (json.JSONDecodeError, IOError):
                pass
        return self._get_empty_container()

    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 用户要求重置旧的 stage 级统计口径；若读到旧结构则直接清空。
        if not isinstance(data, dict):
            return self._get_empty_container()

        if "by_pipeline" in data or "links_found_total" in data or "worker_success_total" in data:
            return self._get_empty_container()

        normalized = self._get_empty_container()
        for key in ("total_requests", "cache_hits", "success", "not_found", "error", "last_updated"):
            if key in data:
                normalized[key] = data[key]

        daily = data.get("daily")
        if isinstance(daily, dict):
            normalized_daily = {}
            for date_str, bucket in daily.items():
                if not isinstance(bucket, dict):
                    continue
                item = _empty_stats()
                for key in (*OUTCOME_FIELDS, "total_requests", "cache_hits"):
                    if key in bucket:
                        item[key] = bucket[key]
                normalized_daily[date_str] = item
            normalized["daily"] = normalized_daily

        return normalized

    def _get_empty_container(self) -> Dict[str, Any]:
        return {
            **_empty_stats(),
            "daily": {},
            "last_updated": None,
        }

    def _save_data(self):
        with self._lock:
            with self._locked_storage():
                self._save_data_locked()

    def _get_today(self) -> str:
        return date.today().isoformat()

    @contextmanager
    def _locked_storage(self):
        """跨进程文件锁，避免并发写覆盖"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None:
            yield
            return

        with open(self._lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _reload_from_disk_locked(self):
        """在持有文件锁时刷新内存中的统计"""
        self._data = self._load_data()

    def _save_data_locked(self):
        """在持有文件锁时保存数据"""
        with self._file_lock:
            self._data["last_updated"] = datetime.now().isoformat()
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _ensure_daily(self, date_str: str):
        if date_str not in self._data["daily"]:
            self._data["daily"][date_str] = _empty_stats()

    def record_request(self):
        today = self._get_today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
                self._ensure_daily(today)
                self._data["total_requests"] += 1
                self._data["daily"][today]["total_requests"] += 1
                self._save_data_locked()

    def record_cache_hit(self):
        today = self._get_today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
                self._ensure_daily(today)
                self._data["cache_hits"] += 1
                self._data["daily"][today]["cache_hits"] += 1
                self._save_data_locked()

    def record_success(self):
        today = self._get_today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
                self._ensure_daily(today)
                self._data["success"] += 1
                self._data["daily"][today]["success"] += 1
                self._save_data_locked()

    def record_not_found(self):
        today = self._get_today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
                self._ensure_daily(today)
                self._data["not_found"] += 1
                self._data["daily"][today]["not_found"] += 1
                self._save_data_locked()

    def record_error(self):
        today = self._get_today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
                self._ensure_daily(today)
                self._data["error"] += 1
                self._data["daily"][today]["error"] += 1
                self._save_data_locked()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            return copy.deepcopy(self._data)

    def get_today_stats(self) -> Dict[str, Any]:
        today = self._get_today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            bucket = self._data["daily"].get(today, _empty_stats())
            return {
                "date": today,
                **copy.deepcopy(bucket),
            }


_stats: Optional[OrgPipelineStats] = None


def get_org_pipeline_stats() -> OrgPipelineStats:
    global _stats
    if _stats is None:
        _stats = OrgPipelineStats()
    return _stats


def record_request():
    get_org_pipeline_stats().record_request()


def record_cache_hit():
    get_org_pipeline_stats().record_cache_hit()


def record_success():
    get_org_pipeline_stats().record_success()


def record_not_found():
    get_org_pipeline_stats().record_not_found()


def record_error():
    get_org_pipeline_stats().record_error()
