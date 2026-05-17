"""
Patent pipeline usage and outcome statistics.

Records one request per patent pipeline job and mutually exclusive outcomes:
- success: pipeline finished and found at least one confirmed patent
- not_found: pipeline finished but found no confirmed patents
- identity_not_found: input did not provide enough identity data to search
- error: the job raised or an external dependency failed
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
except ImportError:  # pragma: no cover
    fcntl = None


SOURCES = ("direct", "orcid", "google_scholar")
OUTCOMES = ("success", "not_found", "identity_not_found", "error")


def _empty_source_stats() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "success": 0,
        "not_found": 0,
        "identity_not_found": 0,
        "error": 0,
        "confirmed_total": 0,
        "possible_total": 0,
        "rejected_total": 0,
    }


def _empty_stats() -> Dict[str, Any]:
    return {
        **_empty_source_stats(),
        "by_source": {source: _empty_source_stats() for source in SOURCES},
    }


class PatentPipelineStats:
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
            storage_path = Path(__file__).parent.parent / "total_usage" / "patent_pipeline_stats.json"
        self.storage_path = Path(storage_path)
        self._lock_path = self.storage_path.with_name(f"{self.storage_path.name}.lock")
        self._file_lock = threading.Lock()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load_data()
        self._initialized = True

    def _get_empty_container(self) -> Dict[str, Any]:
        return {
            **_empty_stats(),
            "daily": {},
            "last_updated": None,
        }

    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return self._get_empty_container()
        normalized = self._get_empty_container()
        for key in _empty_source_stats():
            if isinstance(data.get(key), int):
                normalized[key] = data[key]
        by_source = data.get("by_source")
        if isinstance(by_source, dict):
            for source in SOURCES:
                if isinstance(by_source.get(source), dict):
                    for key in _empty_source_stats():
                        if isinstance(by_source[source].get(key), int):
                            normalized["by_source"][source][key] = by_source[source][key]
        daily = data.get("daily")
        if isinstance(daily, dict):
            for date_str, bucket in daily.items():
                normalized["daily"][date_str] = self._normalize_bucket(bucket)
        normalized["last_updated"] = data.get("last_updated")
        return normalized

    def _normalize_bucket(self, bucket: Any) -> Dict[str, Any]:
        normalized = _empty_stats()
        if not isinstance(bucket, dict):
            return normalized
        for key in _empty_source_stats():
            if isinstance(bucket.get(key), int):
                normalized[key] = bucket[key]
        by_source = bucket.get("by_source")
        if isinstance(by_source, dict):
            for source in SOURCES:
                source_bucket = by_source.get(source)
                if isinstance(source_bucket, dict):
                    for key in _empty_source_stats():
                        if isinstance(source_bucket.get(key), int):
                            normalized["by_source"][source][key] = source_bucket[key]
        return normalized

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

    @contextmanager
    def _locked_storage(self):
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
        self._data = self._load_data()

    def _save_data_locked(self):
        with self._file_lock:
            self._data["last_updated"] = datetime.now().isoformat()
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _today(self) -> str:
        return date.today().isoformat()

    def _ensure_daily(self, date_str: str):
        if date_str not in self._data["daily"]:
            self._data["daily"][date_str] = _empty_stats()

    def record_result(
        self,
        source: str,
        outcome: str,
        confirmed_count: int = 0,
        possible_count: int = 0,
        rejected_count: int = 0,
    ):
        source = source if source in SOURCES else "direct"
        outcome = outcome if outcome in OUTCOMES else "error"
        today = self._today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
                self._ensure_daily(today)
                for bucket in (
                    self._data,
                    self._data["daily"][today],
                    self._data["by_source"][source],
                    self._data["daily"][today]["by_source"][source],
                ):
                    bucket["total_requests"] += 1
                    bucket[outcome] += 1
                    bucket["confirmed_total"] += confirmed_count
                    bucket["possible_total"] += possible_count
                    bucket["rejected_total"] += rejected_count
                self._save_data_locked()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            return copy.deepcopy(self._data)

    def get_today_stats(self) -> Dict[str, Any]:
        today = self._today()
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            return {"date": today, **copy.deepcopy(self._data["daily"].get(today, _empty_stats()))}


_stats: Optional[PatentPipelineStats] = None


def get_patent_pipeline_stats() -> PatentPipelineStats:
    global _stats
    if _stats is None:
        _stats = PatentPipelineStats()
    return _stats


def record_result(
    source: str,
    outcome: str,
    confirmed_count: int = 0,
    possible_count: int = 0,
    rejected_count: int = 0,
):
    get_patent_pipeline_stats().record_result(
        source=source,
        outcome=outcome,
        confirmed_count=confirmed_count,
        possible_count=possible_count,
        rejected_count=rejected_count,
    )
