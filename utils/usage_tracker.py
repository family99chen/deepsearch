"""
API 使用统计模块

工业级实现：
- 线程安全（使用锁）
- 按天自动分割
- 实时持久化到 JSON 文件
- 支持多个 API 端点统计
"""

import copy
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - 非 Unix 环境回退
    fcntl = None


class UsageTracker:
    """
    API 使用统计追踪器
    
    特性：
    - 线程安全
    - 多进程安全（基于文件锁）
    - 按天自动分割（24点自动创建新的一天）
    - 实时写入 JSON 文件
    - 按路径缓存（同一路径返回同一实例，不同路径返回不同实例）
    """
    
    _instances: Dict[str, "UsageTracker"] = {}
    _lock = threading.Lock()
    
    def __new__(cls, storage_path: Optional[str] = None, *args, **kwargs):
        """按路径缓存的多实例模式"""
        # 确定存储路径
        if storage_path is None:
            storage_path = str(Path(__file__).parent.parent / "total_usage" / "mapping_api.json")
        else:
            storage_path = str(Path(storage_path))
        
        with cls._lock:
            if storage_path not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[storage_path] = instance
            return cls._instances[storage_path]
    
    def __init__(
        self, 
        storage_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        初始化追踪器
        
        Args:
            storage_path: JSON 文件存储路径
            auto_save: 是否自动保存（每次更新后立即写入文件）
        """
        # 确定存储路径（与 __new__ 保持一致）
        if storage_path is None:
            storage_path = str(Path(__file__).parent.parent / "total_usage" / "mapping_api.json")
        else:
            storage_path = str(Path(storage_path))
        
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save
        self._lock_path = self.storage_path.with_name(f"{self.storage_path.name}.lock")
        self._file_lock = threading.Lock()
        
        # 确保目录存在
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载已有数据
        self._data = self._load_data()
        
        self._initialized = True
    
    @staticmethod
    def _initial_data() -> Dict:
        """返回初始数据结构"""
        return {
            "total": 0,
            "daily": {},
            "endpoints": {},
            "last_updated": None
        }

    def _load_data(self) -> Dict:
        """从文件加载数据"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        
        return self._initial_data()

    @contextmanager
    def _locked_storage(self):
        """跨进程文件锁，避免并发写覆盖"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None:
            yield
            return

        with open(self._lock_path, 'a+', encoding='utf-8') as lock_file:
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
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
    
    def _save_data(self):
        """保存数据到文件"""
        with self._lock:
            with self._locked_storage():
                self._save_data_locked()
    
    def _get_today(self) -> str:
        """获取今天的日期字符串 (YYYY-MM-DD)"""
        return date.today().isoformat()

    @staticmethod
    def _normalize_metadata(metadata: Optional[Dict]) -> Dict[str, float]:
        """过滤并保留可聚合的数值元数据"""
        if not metadata:
            return {}

        normalized: Dict[str, float] = {}
        for key, value in metadata.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                normalized[key] = value
        return normalized

    def _ensure_daily_bucket(self, today: str) -> Dict:
        """确保当天统计 bucket 存在"""
        if "daily" not in self._data:
            self._data["daily"] = {}

        if today not in self._data["daily"]:
            self._data["daily"][today] = {
                "total": 0,
                "endpoints": {},
                "first_request": datetime.now().isoformat(),
                "last_request": None
            }
        return self._data["daily"][today]

    @staticmethod
    def _merge_metrics(bucket: Dict, metadata: Dict[str, float], count: int):
        """聚合长度等数值指标，并维护平均值"""
        for key, value in metadata.items():
            total_key = f"{key}_total"
            samples_key = f"{key}_samples"
            avg_key = f"avg_{key}"

            bucket[total_key] = bucket.get(total_key, 0) + value
            bucket[samples_key] = bucket.get(samples_key, 0) + count

            samples = bucket.get(samples_key, 0)
            bucket[avg_key] = round(bucket[total_key] / samples, 2) if samples else 0

    def _refresh_from_disk(self):
        """读取其他进程已写入的最新状态"""
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
    
    def record(
        self, 
        endpoint: str, 
        count: int = 1,
        metadata: Optional[Dict] = None
    ):
        """
        记录一次 API 调用
        
        Args:
            endpoint: API 端点路径（如 "/find", "/find/sync"）
            count: 调用次数（默认 1）
            metadata: 额外元数据（可选）
        """
        today = self._get_today()
        numeric_metadata = self._normalize_metadata(metadata)
        
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()

                self._data["total"] = self._data.get("total", 0) + count

                daily_bucket = self._ensure_daily_bucket(today)
                daily_bucket["total"] += count
                daily_bucket["last_request"] = datetime.now().isoformat()

                daily_endpoints = daily_bucket.setdefault("endpoints", {})
                daily_endpoints[endpoint] = daily_endpoints.get(endpoint, 0) + count

                global_endpoints = self._data.setdefault("endpoints", {})
                global_endpoints[endpoint] = global_endpoints.get(endpoint, 0) + count

                if numeric_metadata:
                    self._merge_metrics(
                        self._data.setdefault("metrics", {}),
                        numeric_metadata,
                        count,
                    )
                    self._merge_metrics(
                        daily_bucket.setdefault("metrics", {}),
                        numeric_metadata,
                        count,
                    )

                    endpoint_metrics = self._data.setdefault("endpoint_metrics", {})
                    self._merge_metrics(
                        endpoint_metrics.setdefault(endpoint, {}),
                        numeric_metadata,
                        count,
                    )

                    daily_endpoint_metrics = daily_bucket.setdefault("endpoint_metrics", {})
                    self._merge_metrics(
                        daily_endpoint_metrics.setdefault(endpoint, {}),
                        numeric_metadata,
                        count,
                    )

                if self.auto_save:
                    self._save_data_locked()
    
    def get_today_count(self, endpoint: Optional[str] = None) -> int:
        """
        获取今天的调用次数
        
        Args:
            endpoint: 指定端点（可选，不指定则返回总数）
            
        Returns:
            调用次数
        """
        today = self._get_today()
        
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            if today not in self._data.get("daily", {}):
                return 0
            
            if endpoint:
                return self._data["daily"][today].get("endpoints", {}).get(endpoint, 0)
            else:
                return self._data["daily"][today].get("total", 0)
    
    def get_total_count(self, endpoint: Optional[str] = None) -> int:
        """
        获取总调用次数
        
        Args:
            endpoint: 指定端点（可选，不指定则返回总数）
            
        Returns:
            调用次数
        """
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            if endpoint:
                return self._data.get("endpoints", {}).get(endpoint, 0)
            else:
                return self._data.get("total", 0)
    
    def get_stats(self) -> Dict:
        """获取完整统计数据"""
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            return copy.deepcopy(self._data)
    
    def get_daily_stats(self, date_str: Optional[str] = None) -> Dict:
        """
        获取指定日期的统计
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)，不指定则返回今天
            
        Returns:
            当天统计数据
        """
        if date_str is None:
            date_str = self._get_today()
        
        with self._lock:
            with self._locked_storage():
                self._reload_from_disk_locked()
            return copy.deepcopy(self._data.get("daily", {}).get(date_str, {
                "total": 0,
                "endpoints": {}
            }))


# 全局单例实例
_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    """获取全局 UsageTracker 实例"""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


def record_api_call(endpoint: str, count: int = 1, metadata: Optional[Dict] = None):
    """
    便捷函数：记录 API 调用
    
    Args:
        endpoint: API 端点
        count: 调用次数
    """
    get_tracker().record(endpoint, count, metadata=metadata)


def get_today_usage(endpoint: Optional[str] = None) -> int:
    """
    便捷函数：获取今日使用量
    
    Args:
        endpoint: 指定端点（可选）
        
    Returns:
        调用次数
    """
    return get_tracker().get_today_count(endpoint)


def get_total_usage(endpoint: Optional[str] = None) -> int:
    """
    便捷函数：获取总使用量
    
    Args:
        endpoint: 指定端点（可选）
        
    Returns:
        调用次数
    """
    return get_tracker().get_total_count(endpoint)


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("API 使用统计模块测试")
    print("=" * 60)
    
    tracker = get_tracker()
    
    # 模拟 API 调用
    print("\n模拟 API 调用...")
    record_api_call("/find")
    record_api_call("/find")
    record_api_call("/find/sync")
    
    print(f"\n今日 /find 调用次数: {get_today_usage('/find')}")
    print(f"今日 /find/sync 调用次数: {get_today_usage('/find/sync')}")
    print(f"今日总调用次数: {get_today_usage()}")
    print(f"历史总调用次数: {get_total_usage()}")
    
    print("\n完整统计:")
    import pprint
    pprint.pprint(tracker.get_stats())

