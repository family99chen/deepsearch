"""
Google Custom Search API 通用模块
支持高并发请求

文档: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
"""

import asyncio
import aiohttp
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time


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


class GoogleSearchAPI:
    """Google Custom Search API 封装"""
    
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        rate_limit: int = 10
    ):
        """
        初始化 Google Search API
        
        Args:
            api_key: Google API Key（不提供则从配置文件读取）
            cx: Custom Search Engine ID（不提供则从配置文件读取）
            rate_limit: 每秒最大请求数
        """
        config = self._load_config()
        google_config = config.get("google", {})
        
        self.api_key = api_key or google_config.get("api_key")
        self.cx = cx or google_config.get("cx")
        self.rate_limit = rate_limit or google_config.get("rate_limit", 10)
        
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
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
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
        extra_params: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        执行搜索
        
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
            
        Returns:
            SearchResponse 对象
        """
        # 速率限制
        await self.limiter.acquire()
        
        # 构建查询
        full_query = query
        if site_search:
            full_query = f"site:{site_search} {query}"
        if file_type:
            full_query = f"filetype:{file_type} {full_query}"
        
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
        
        try:
            session = await self._get_session()
            start_time = time.time()
            
            async with session.get(self.BASE_URL, params=params) as response:
                search_time = time.time() - start_time
                
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
                
                return SearchResponse(
                    success=True,
                    query=query,
                    total_results=total_results,
                    items=items,
                    search_time=search_time
                )
                
        except Exception as e:
            return SearchResponse(
                success=False,
                query=query,
                total_results=0,
                items=[],
                search_time=0,
                error=str(e)
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
    """Google Search API 的同步包装器"""
    
    def __init__(self, **kwargs):
        self._async_api = GoogleSearchAPI(**kwargs)
    
    def _run(self, coro):
        """运行协程"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def search(self, query: str, **kwargs) -> SearchResponse:
        """同步搜索"""
        async def _search():
            async with GoogleSearchAPI() as api:
                return await api.search(query, **kwargs)
        return self._run(_search())
    
    def search_all(self, query: str, max_results: int = 50, **kwargs) -> SearchResponse:
        """同步获取多页结果"""
        async def _search():
            async with GoogleSearchAPI() as api:
                return await api.search_all(query, max_results, **kwargs)
        return self._run(_search())
    
    def batch_search(self, queries: List[str], **kwargs) -> List[SearchResponse]:
        """同步批量搜索"""
        async def _search():
            async with GoogleSearchAPI() as api:
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
        
        async with GoogleSearchAPI() as api:
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

