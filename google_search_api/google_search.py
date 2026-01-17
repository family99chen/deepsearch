"""
Google Custom Search API 通用模块
支持高并发请求和指数退避重试

特性：
- MongoDB 缓存（按 query 缓存，TTL 15 天）
- API 调用次数统计（按天记录）
- 指数退避重试

文档: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
"""

import sys
import asyncio
import random
import hashlib
import aiohttp
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入缓存模块
HAS_CACHE = False
try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    pass

# 导入使用统计模块
HAS_USAGE_TRACKER = False
try:
    from utils.usage_tracker import UsageTracker
    HAS_USAGE_TRACKER = True
except ImportError:
    pass

# 缓存配置
CACHE_COLLECTION = "google_search_cache"
CACHE_TTL = 15 * 24 * 3600  # 15 天（秒）


def _get_cache() -> Optional["MongoCache"]:
    """获取缓存实例"""
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=CACHE_COLLECTION)
        return cache if cache.is_connected() else None
    except Exception:
        return None


def _get_usage_tracker() -> Optional["UsageTracker"]:
    """获取使用统计实例"""
    if not HAS_USAGE_TRACKER:
        return None
    try:
        storage_path = Path(__file__).parent.parent / "total_usage" / "google_search.json"
        return UsageTracker(storage_path=storage_path, auto_save=True)
    except Exception:
        return None


def _generate_cache_key(query: str, num: int, start: int, **kwargs) -> str:
    """
    生成缓存键
    
    基于查询参数生成唯一的缓存键
    """
    # 构建用于哈希的字符串
    key_parts = [
        f"q:{query}",
        f"num:{num}",
        f"start:{start}",
    ]
    
    # 添加其他可能影响结果的参数
    for k, v in sorted(kwargs.items()):
        if v is not None:
            key_parts.append(f"{k}:{v}")
    
    key_string = "|".join(key_parts)
    
    # 使用 MD5 生成较短的 key
    return hashlib.md5(key_string.encode()).hexdigest()


def _record_api_call():
    """记录一次 API 调用"""
    tracker = _get_usage_tracker()
    if tracker:
        tracker.record("google_custom_search", count=1)


@dataclass
class SearchResult:
    """搜索结果项"""
    title: str
    link: str
    snippet: str
    display_link: str
    html_title: Optional[str] = None
    html_snippet: Optional[str] = None
    formatted_url: Optional[str] = None
    

@dataclass
class SearchResponse:
    """搜索响应"""
    success: bool
    query: str
    total_results: int
    items: List[SearchResult]
    search_time: float
    error: Optional[str] = None


class RateLimiter:
    """简单的速率限制器"""
    
    def __init__(self, max_requests: int = 10, time_window: float = 1.0):
        """
        初始化速率限制器
        
        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取请求许可"""
        async with self._lock:
            now = time.time()
            # 清理过期的请求记录
            self.requests = [t for t in self.requests if now - t < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # 需要等待
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(time.time())


# 默认可重试的状态码
DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# 默认可重试的异常
DEFAULT_RETRYABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    aiohttp.ClientError,
    aiohttp.ServerTimeoutError,
    aiohttp.ClientConnectionError,
    ConnectionError,
    TimeoutError,
    OSError,
)


class GoogleSearchAPI:
    """Google Custom Search API 封装（带指数退避重试 + MongoDB 缓存）"""
    
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        rate_limit: int = 10,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        use_cache: bool = True,
    ):
        """
        初始化 Google Search API
        
        Args:
            api_key: Google API Key（不提供则从配置文件读取）
            cx: Custom Search Engine ID（不提供则从配置文件读取）
            rate_limit: 每秒最大请求数
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            use_cache: 是否使用缓存（默认 True）
        """
        config = self._load_config()
        google_config = config.get("google", {})
        
        self.api_key = api_key or google_config.get("api_key")
        self.cx = cx or google_config.get("cx")
        self.rate_limit = rate_limit or google_config.get("rate_limit", 10)
        
        # 重试配置
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # 缓存配置
        self.use_cache = use_cache
        self._cache = _get_cache() if use_cache else None
        
        if not self.api_key or not self.cx:
            raise ValueError("必须提供 api_key 和 cx，或在 config.yaml 中配置")
        
        self.limiter = RateLimiter(max_requests=self.rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        # 尝试多个可能的路径
        possible_paths = [
            Path(__file__).parent.parent / "config.yaml",
            Path("config.yaml"),
            Path(__file__).parent / "config.yaml"
        ]
        
        for config_path in possible_paths:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        return {}
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟时间
        
        Args:
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            延迟时间（秒）
        """
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay
        )
        # 添加随机抖动 (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """关闭 HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def search(
        self,
        query: str,
        num: int = 10,
        start: int = 1,
        language: str = "zh-CN",
        safe: str = "off",
        site_search: Optional[str] = None,
        file_type: Optional[str] = None,
        date_restrict: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None
    ) -> SearchResponse:
        """
        执行搜索（带指数退避重试 + MongoDB 缓存）
        
        Args:
            query: 搜索关键词
            num: 返回结果数量（1-10）
            start: 起始位置（1-100）
            language: 语言代码
            safe: 安全搜索（off/medium/high）
            site_search: 限定搜索站点（如 "site:example.com"）
            file_type: 限定文件类型（如 "pdf"）
            date_restrict: 时间限制（如 "d1" 一天内, "w1" 一周内, "m1" 一个月内）
            extra_params: 额外的 API 参数
            use_cache: 是否使用缓存（None 表示使用实例默认设置）
            
        Returns:
            SearchResponse 对象
        """
        # 构建查询
        full_query = query
        if site_search:
            full_query = f"site:{site_search} {query}"
        if file_type:
            full_query = f"filetype:{file_type} {full_query}"
        
        # 确定是否使用缓存
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        
        # ========== 缓存读取 ==========
        cache_key = _generate_cache_key(
            query=full_query, 
            num=num, 
            start=start,
            language=language,
            safe=safe,
            date_restrict=date_restrict
        )
        
        if should_use_cache and self._cache:
            cached_data = self._cache.get(cache_key)
            if cached_data:
                # 缓存命中，重建 SearchResponse
                print(f"[CACHE] 命中缓存: {full_query[:50]}...")
                items = [
                    SearchResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        display_link=item.get("display_link", ""),
                        html_title=item.get("html_title"),
                        html_snippet=item.get("html_snippet"),
                        formatted_url=item.get("formatted_url")
                    )
                    for item in cached_data.get("items", [])
                ]
                return SearchResponse(
                    success=True,
                    query=cached_data.get("query", query),
                    total_results=cached_data.get("total_results", 0),
                    items=items,
                    search_time=0.0,  # 缓存命中，搜索时间为 0
                    error=None
                )
        
        # ========== API 请求 ==========
        # 速率限制
        await self.limiter.acquire()
        
        # 构建参数
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": full_query,
            "num": min(num, 10),  # API 限制最多 10
            "start": max(1, min(start, 100)),  # API 限制 1-100
            "hl": language,
            "safe": safe,
        }
        
        if date_restrict:
            params["dateRestrict"] = date_restrict
        
        if extra_params:
            params.update(extra_params)
        
        # 带重试的请求
        last_exception = None
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()
                
                async with session.get(self.BASE_URL, params=params) as response:
                    search_time = time.time() - start_time
                    
                    # 检查是否需要重试的状态码
                    if response.status in DEFAULT_RETRYABLE_STATUS_CODES:
                        if attempt < self.max_retries:
                            delay = self._calculate_delay(attempt)
                            print(f"[RETRY] HTTP {response.status}，{delay:.1f}秒后重试 "
                                  f"({attempt + 1}/{self.max_retries})...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_text = await response.text()
                            return SearchResponse(
                                success=False,
                                query=query,
                                total_results=0,
                                items=[],
                                search_time=search_time,
                                error=f"API 错误 ({response.status}): {error_text}"
                            )
                    
                    if response.status != 200:
                        error_text = await response.text()
                        return SearchResponse(
                            success=False,
                            query=query,
                            total_results=0,
                            items=[],
                            search_time=search_time,
                            error=f"API 错误 ({response.status}): {error_text}"
                        )
                    
                    data = await response.json()
                    
                    # 解析结果
                    items = []
                    for item in data.get("items", []):
                        items.append(SearchResult(
                            title=item.get("title", ""),
                            link=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            display_link=item.get("displayLink", ""),
                            html_title=item.get("htmlTitle"),
                            html_snippet=item.get("htmlSnippet"),
                            formatted_url=item.get("formattedUrl")
                        ))
                    
                    # 获取总结果数
                    search_info = data.get("searchInformation", {})
                    total_results = int(search_info.get("totalResults", 0))
                    
                    # 记录 API 调用（成功即记录）
                    _record_api_call()
                    
                    # ========== 缓存写入 ==========
                    # 无论是否有结果都写入缓存，空结果也缓存以避免重复查询
                    if should_use_cache and self._cache:
                        cache_data = {
                            "query": query,
                            "full_query": full_query,
                            "total_results": total_results,
                            "has_results": len(items) > 0,  # 标记是否有结果
                            "items": [
                                {
                                    "title": item.title,
                                    "link": item.link,
                                    "snippet": item.snippet,
                                    "display_link": item.display_link,
                                    "html_title": item.html_title,
                                    "html_snippet": item.html_snippet,
                                    "formatted_url": item.formatted_url
                                }
                                for item in items
                            ],
                            "cached_at": datetime.now().isoformat(),
                            "params": {
                                "num": num,
                                "start": start,
                                "language": language,
                                "safe": safe,
                                "date_restrict": date_restrict
                            }
                        }
                        self._cache.set(cache_key, cache_data, ttl=CACHE_TTL)
                        if items:
                            print(f"[CACHE] 已写入缓存: {full_query[:50]}... ({len(items)} 条结果)")
                        else:
                            print(f"[CACHE] 已写入缓存（无结果）: {full_query[:50]}...")
                    
                    return SearchResponse(
                        success=True,
                        query=query,
                        total_results=total_results,
                        items=items,
                        search_time=search_time
                    )
                    
            except DEFAULT_RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                
                if attempt >= self.max_retries:
                    return SearchResponse(
                        success=False,
                        query=query,
                        total_results=0,
                        items=[],
                        search_time=time.time() - start_time,
                        error=f"请求失败（重试 {self.max_retries} 次后）: {type(e).__name__}: {e}"
                    )
                
                delay = self._calculate_delay(attempt)
                print(f"[RETRY] {type(e).__name__}，{delay:.1f}秒后重试 "
                      f"({attempt + 1}/{self.max_retries})...")
                await asyncio.sleep(delay)
                
            except Exception as e:
                return SearchResponse(
                    success=False,
                    query=query,
                    total_results=0,
                    items=[],
                    search_time=time.time() - start_time,
                    error=f"未知错误: {type(e).__name__}: {e}"
                )
        
        # 不应该到达这里
        return SearchResponse(
            success=False,
            query=query,
            total_results=0,
            items=[],
            search_time=time.time() - start_time,
            error=str(last_exception) if last_exception else "未知错误"
        )
    
    async def search_all(
        self,
        query: str,
        max_results: int = 50,
        **kwargs
    ) -> SearchResponse:
        """
        搜索并获取多页结果
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数（每 10 个结果消耗 1 次 API 配额）
            **kwargs: 传递给 search() 的其他参数
            
        Returns:
            合并的 SearchResponse
        """
        all_items = []
        total_results = 0
        total_time = 0
        
        for start in range(1, min(max_results, 100) + 1, 10):
            result = await self.search(
                query=query,
                num=10,
                start=start,
                **kwargs
            )
            
            if not result.success:
                if all_items:
                    # 已有部分结果，返回已获取的
                    break
                return result
            
            total_results = result.total_results
            total_time += result.search_time
            all_items.extend(result.items)
            
            if len(all_items) >= max_results or len(result.items) < 10:
                break
        
        return SearchResponse(
            success=True,
            query=query,
            total_results=total_results,
            items=all_items[:max_results],
            search_time=total_time
        )
    
    async def batch_search(
        self,
        queries: List[str],
        **kwargs
    ) -> List[SearchResponse]:
        """
        批量并发搜索
        
        Args:
            queries: 搜索关键词列表
            **kwargs: 传递给 search() 的其他参数
            
        Returns:
            SearchResponse 列表
        """
        tasks = [self.search(query=q, **kwargs) for q in queries]
        return await asyncio.gather(*tasks)


# 同步包装器，方便非异步代码使用
class GoogleSearchAPISync:
    """Google Search API 的同步包装器（支持缓存）"""
    
    def __init__(self, use_cache: bool = True, **kwargs):
        self._kwargs = kwargs
        self._kwargs["use_cache"] = use_cache
    
    def _run(self, coro):
        """运行协程"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有事件循环在运行，创建新的
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(coro)
    
    def search(self, query: str, **kwargs) -> SearchResponse:
        """同步搜索"""
        async def _search():
            async with GoogleSearchAPI(**self._kwargs) as api:
                return await api.search(query, **kwargs)
        return self._run(_search())
    
    def search_all(self, query: str, max_results: int = 50, **kwargs) -> SearchResponse:
        """同步获取多页结果"""
        async def _search():
            async with GoogleSearchAPI(**self._kwargs) as api:
                return await api.search_all(query, max_results, **kwargs)
        return self._run(_search())
    
    def batch_search(self, queries: List[str], **kwargs) -> List[SearchResponse]:
        """同步批量搜索"""
        async def _search():
            async with GoogleSearchAPI(**self._kwargs) as api:
                return await api.batch_search(queries, **kwargs)
        return self._run(_search())


# 便捷函数
async def google_search(query: str, **kwargs) -> SearchResponse:
    """快捷搜索函数"""
    async with GoogleSearchAPI() as api:
        return await api.search(query, **kwargs)


async def google_batch_search(queries: List[str], **kwargs) -> List[SearchResponse]:
    """快捷批量搜索函数"""
    async with GoogleSearchAPI() as api:
        return await api.batch_search(queries, **kwargs)


if __name__ == "__main__":
    # 测试代码
    async def main():
        print("=" * 60)
        print("Google Custom Search API 测试")
        print("=" * 60)
        
        async with GoogleSearchAPI(max_retries=3) as api:
            # 单个搜索
            print("\n[测试 1] 单个搜索")
            result = await api.search("Python programming", num=5)
            print(f"查询: {result.query}")
            print(f"成功: {result.success}")
            print(f"总结果数: {result.total_results}")
            print(f"耗时: {result.search_time:.2f}s")
            print(f"返回条数: {len(result.items)}")
            
            if result.items:
                print("\n前 3 条结果:")
                for i, item in enumerate(result.items[:3], 1):
                    print(f"  {i}. {item.title}")
                    print(f"     {item.link}")
            
            # 批量搜索
            print("\n" + "=" * 60)
            print("[测试 2] 批量并发搜索")
            queries = ["machine learning", "deep learning", "neural network"]
            results = await api.batch_search(queries, num=3)
            
            for r in results:
                print(f"\n查询: {r.query}")
                print(f"  结果数: {len(r.items)}, 耗时: {r.search_time:.2f}s")
    
    asyncio.run(main())
