"""
Pipeline 详细统计模块

记录 ORCID → Google Scholar 查找的详细统计信息：
- 总请求次数
- 缓存命中次数
- 成功/失败次数
- 按搜索方法和迭代次数分类的成功次数
"""

import json
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional


class PipelineStats:
    """
    Pipeline 详细统计追踪器
    
    统计项：
    - total_requests: 总请求次数
    - cache_hits: 缓存命中次数（orcid_googleaccount_map）
    - success: 成功获取作者次数
    - error: 错误次数
    - name_search_success: 名字搜索成功次数（按迭代次数分类）
    - paper_search_success: 论文搜索成功次数（按论文数分类）
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, storage_path: Optional[str] = None):
        """初始化"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "total_usage" / "pipeline_stats.json"
        
        self.storage_path = Path(storage_path)
        self._file_lock = threading.Lock()
        
        # 确保目录存在
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载已有数据
        self._data = self._load_data()
        self._initialized = True
    
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
        
        # 返回初始结构
        return self._get_empty_stats()
    
    def _get_empty_stats(self) -> Dict:
        """获取空的统计结构"""
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "success": 0,
            "not_found": 0,
            "error": 0,
            "name_search": {
                "total": 0,
                "by_iterations": {}  # {"1": 10, "2": 5, "3": 2, ...}
            },
            "paper_search": {
                "total": 0,
                "by_papers": {}  # {"1": 8, "2": 2, ...}
            },
            "daily": {},
            "last_updated": None
        }
    
    def _get_empty_daily_stats(self) -> Dict:
        """获取空的每日统计结构"""
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "success": 0,
            "not_found": 0,
            "error": 0,
            "name_search": {
                "total": 0,
                "by_iterations": {}
            },
            "paper_search": {
                "total": 0,
                "by_papers": {}
            }
        }
    
    def _save_data(self):
        """保存数据到文件"""
        with self._file_lock:
            self._data["last_updated"] = datetime.now().isoformat()
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
    
    def _get_today(self) -> str:
        """获取今天的日期字符串"""
        return date.today().isoformat()
    
    def _ensure_daily(self, date_str: str):
        """确保每日统计存在"""
        if "daily" not in self._data:
            self._data["daily"] = {}
        if date_str not in self._data["daily"]:
            self._data["daily"][date_str] = self._get_empty_daily_stats()
    
    def record_request(self):
        """记录一次请求"""
        today = self._get_today()
        
        with self._lock:
            self._data["total_requests"] = self._data.get("total_requests", 0) + 1
            
            self._ensure_daily(today)
            self._data["daily"][today]["total_requests"] += 1
            
            self._save_data()
    
    def record_cache_hit(self):
        """记录缓存命中"""
        today = self._get_today()
        
        with self._lock:
            self._data["cache_hits"] = self._data.get("cache_hits", 0) + 1
            
            self._ensure_daily(today)
            self._data["daily"][today]["cache_hits"] += 1
            
            self._save_data()
    
    def record_error(self):
        """记录错误"""
        today = self._get_today()
        
        with self._lock:
            self._data["error"] = self._data.get("error", 0) + 1
            
            self._ensure_daily(today)
            self._data["daily"][today]["error"] += 1
            
            self._save_data()
    
    def record_success_by_name_search(self, iterations: int):
        """
        记录名字搜索成功
        
        Args:
            iterations: 使用了多少次 Google Search 迭代才成功
        """
        today = self._get_today()
        iter_key = str(iterations)
        
        with self._lock:
            # 更新总计
            self._data["success"] = self._data.get("success", 0) + 1
            
            if "name_search" not in self._data:
                self._data["name_search"] = {"total": 0, "by_iterations": {}}
            
            self._data["name_search"]["total"] += 1
            
            if "by_iterations" not in self._data["name_search"]:
                self._data["name_search"]["by_iterations"] = {}
            
            self._data["name_search"]["by_iterations"][iter_key] = \
                self._data["name_search"]["by_iterations"].get(iter_key, 0) + 1
            
            # 更新每日统计
            self._ensure_daily(today)
            self._data["daily"][today]["success"] += 1
            self._data["daily"][today]["name_search"]["total"] += 1
            self._data["daily"][today]["name_search"]["by_iterations"][iter_key] = \
                self._data["daily"][today]["name_search"]["by_iterations"].get(iter_key, 0) + 1
            
            self._save_data()
    
    def record_success_by_paper_search(self, papers_searched: int):
        """
        记录论文搜索成功
        
        Args:
            papers_searched: 搜索了多少篇论文才成功
        """
        today = self._get_today()
        paper_key = str(papers_searched)
        
        with self._lock:
            # 更新总计
            self._data["success"] = self._data.get("success", 0) + 1
            
            if "paper_search" not in self._data:
                self._data["paper_search"] = {"total": 0, "by_papers": {}}
            
            self._data["paper_search"]["total"] += 1
            
            if "by_papers" not in self._data["paper_search"]:
                self._data["paper_search"]["by_papers"] = {}
            
            self._data["paper_search"]["by_papers"][paper_key] = \
                self._data["paper_search"]["by_papers"].get(paper_key, 0) + 1
            
            # 更新每日统计
            self._ensure_daily(today)
            self._data["daily"][today]["success"] += 1
            self._data["daily"][today]["paper_search"]["total"] += 1
            self._data["daily"][today]["paper_search"]["by_papers"][paper_key] = \
                self._data["daily"][today]["paper_search"]["by_papers"].get(paper_key, 0) + 1
            
            self._save_data()
    
    def record_not_found(self):
        """记录未找到（所有方法都尝试了但没找到）"""
        today = self._get_today()
        
        with self._lock:
            self._data["not_found"] = self._data.get("not_found", 0) + 1
            
            self._ensure_daily(today)
            self._data["daily"][today]["not_found"] = self._data["daily"][today].get("not_found", 0) + 1
            
            self._save_data()
    
    def get_stats(self) -> Dict:
        """获取完整统计数据"""
        with self._lock:
            return self._data.copy()
    
    def get_today_stats(self) -> Dict:
        """获取今日统计"""
        today = self._get_today()
        
        with self._lock:
            if today in self._data.get("daily", {}):
                stats = self._data["daily"][today].copy()
                stats["date"] = today
                return stats
            else:
                return {"date": today, **self._get_empty_daily_stats()}


# 全局实例
_stats: Optional[PipelineStats] = None


def get_pipeline_stats() -> PipelineStats:
    """获取全局 PipelineStats 实例"""
    global _stats
    if _stats is None:
        _stats = PipelineStats()
    return _stats


# 便捷函数
def record_request():
    """记录请求"""
    get_pipeline_stats().record_request()


def record_cache_hit():
    """记录缓存命中"""
    get_pipeline_stats().record_cache_hit()


def record_error():
    """记录错误"""
    get_pipeline_stats().record_error()


def record_name_search_success(iterations: int):
    """记录名字搜索成功"""
    get_pipeline_stats().record_success_by_name_search(iterations)


def record_paper_search_success(papers_searched: int):
    """记录论文搜索成功"""
    get_pipeline_stats().record_success_by_paper_search(papers_searched)


def record_not_found():
    """记录未找到"""
    get_pipeline_stats().record_not_found()


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline 统计模块测试")
    print("=" * 60)
    
    stats = get_pipeline_stats()
    
    # 模拟一些调用
    print("\n模拟调用...")
    
    # 模拟缓存命中
    record_request()
    record_cache_hit()
    
    # 模拟名字搜索成功（1次迭代）
    record_request()
    record_name_search_success(1)
    
    # 模拟名字搜索成功（2次迭代）
    record_request()
    record_name_search_success(2)
    
    # 模拟论文搜索成功（1篇论文）
    record_request()
    record_paper_search_success(1)
    
    # 模拟错误
    record_request()
    record_error()
    
    # 模拟未找到
    record_request()
    record_not_found()
    
    print("\n完整统计:")
    import pprint
    pprint.pprint(stats.get_stats())
    
    print("\n今日统计:")
    pprint.pprint(stats.get_today_stats())

