"""
SerpAPI Google Scholar 搜索模块

特性：
- MongoDB 缓存（按 query 缓存，TTL 15 天）
- API 调用次数统计（按天记录）
- 指数退避重试

官方文档: https://serpapi.com/google-scholar-api

Usage:
    from google_search_api.serpapi_google_scholar import SerpAPIScholar
    
    client = SerpAPIScholar()
    result = client.search('author:"Ju Li"')
"""

import os
import sys
import time
import json
import yaml
import hashlib
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

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
CACHE_COLLECTION = "serpapi_search_cache"
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
        storage_path = Path(__file__).parent.parent / "total_usage" / "serpapi.json"
        return UsageTracker(storage_path=storage_path, auto_save=True)
    except Exception:
        return None


def _generate_cache_key(q: str, start: int, num: int, **kwargs) -> str:
    """生成缓存键"""
    key_parts = [
        f"q:{q}",
        f"start:{start}",
        f"num:{num}",
    ]
    
    # 添加其他可能影响结果的参数
    for k, v in sorted(kwargs.items()):
        if v is not None:
            key_parts.append(f"{k}:{v}")
    
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def _record_api_call():
    """记录一次 API 调用"""
    tracker = _get_usage_tracker()
    if tracker:
        tracker.record("serpapi_scholar", count=1)


def _load_config() -> dict:
    """加载配置文件"""
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


# 加载默认 API Key
_config = _load_config()
DEFAULT_API_KEY = _config.get('serpapi', {}).get('api_key', '') or os.environ.get("SERPAPI_KEY", "")


@dataclass
class ScholarResult:
    """Scholar 搜索结果项"""
    title: str = ""
    link: str = ""
    snippet: str = ""
    publication_info: str = ""
    authors: List[Dict] = field(default_factory=list)
    cited_by_count: int = 0
    cited_by_link: str = ""
    versions_count: int = 0
    versions_link: str = ""
    position: int = 0
    raw: Dict = field(default_factory=dict)


@dataclass 
class SearchResponse:
    """搜索响应"""
    success: bool = True
    query: str = ""
    items: List[ScholarResult] = field(default_factory=list)
    total_results: int = 0
    error: str = ""
    raw: Dict = field(default_factory=dict)


class SerpAPIScholar:
    """
    SerpAPI Google Scholar 搜索客户端（支持 MongoDB 缓存）
    
    API 文档: https://serpapi.com/google-scholar-api
    """
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        verbose: bool = False,
        use_cache: bool = True,
    ):
        """
        初始化客户端
        
        Args:
            api_key: SerpAPI API Key
            max_retries: 最大重试次数
            retry_delay: 重试基础延迟（秒）
            timeout: 请求超时（秒）
            verbose: 是否打印详细信息
            use_cache: 是否使用缓存（默认 True）
        """
        self.api_key = api_key or DEFAULT_API_KEY
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verbose = verbose
        
        # 缓存配置
        self.use_cache = use_cache
        self._cache = _get_cache() if use_cache else None
        
        if not self.api_key:
            print("[WARNING] SerpAPI API Key 未配置")
    
    def search(
        self,
        q: str,
        start: int = 0,
        num: int = 10,
        hl: str = "en",
        # 高级 Scholar 参数
        cites: Optional[str] = None,
        as_ylo: Optional[int] = None,
        as_yhi: Optional[int] = None,
        scisbd: Optional[int] = None,
        cluster: Optional[str] = None,
        lr: Optional[str] = None,
        as_sdt: Optional[str] = None,
        safe: Optional[str] = None,
        filter: Optional[int] = None,
        as_vis: Optional[int] = None,
        as_rr: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> SearchResponse:
        """
        Google Scholar 搜索（支持 MongoDB 缓存）
        
        Args:
            q: 搜索查询，支持 author:, source: 等前缀
            start: 结果偏移量 (0=第1页, 10=第2页, 20=第3页...)
            num: 返回数量 (1-20, 默认10)
            hl: 语言代码 (en, zh-CN, etc.)
            cites: 文章ID，触发"被引用"搜索
            as_ylo: 起始年份
            as_yhi: 结束年份
            scisbd: 按日期排序 (1=仅摘要, 2=全部)
            cluster: 文章ID，触发"所有版本"搜索
            lr: 限制语言 (如 lang_en|lang_zh-CN)
            as_sdt: 搜索类型/过滤器
            safe: 安全搜索 (active/off)
            filter: 是否启用过滤 (0/1)
            as_vis: 排除引用 (0/1)
            as_rr: 仅显示综述文章 (0/1)
            use_cache: 是否使用缓存（None 表示使用实例默认设置）
            
        Returns:
            SearchResponse
        """
        # 确定是否使用缓存
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        
        # ========== 缓存读取 ==========
        cache_key = _generate_cache_key(
            q=q, start=start, num=num, hl=hl,
            cites=cites, as_ylo=as_ylo, as_yhi=as_yhi,
            scisbd=scisbd, cluster=cluster, lr=lr,
            as_sdt=as_sdt, safe=safe, filter=filter,
            as_vis=as_vis, as_rr=as_rr
        )
        
        if should_use_cache and self._cache:
            cached_data = self._cache.get(cache_key)
            if cached_data:
                # 缓存命中，重建 SearchResponse
                print(f"[CACHE] 命中缓存: {q[:50]}...")
                items = [
                    ScholarResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        publication_info=item.get("publication_info", ""),
                        authors=item.get("authors", []),
                        cited_by_count=item.get("cited_by_count", 0),
                        cited_by_link=item.get("cited_by_link", ""),
                        versions_count=item.get("versions_count", 0),
                        versions_link=item.get("versions_link", ""),
                        position=item.get("position", 0),
                        raw=item.get("raw", {})
                    )
                    for item in cached_data.get("items", [])
                ]
                return SearchResponse(
                    success=True,
                    query=cached_data.get("query", q),
                    items=items,
                    total_results=cached_data.get("total_results", len(items)),
                    raw={}  # 缓存不保存 raw 数据
                )
        
        # ========== API 请求 ==========
        # 构建参数
        params = {
            "engine": "google_scholar",
            "api_key": self.api_key,
            "q": q,
            "start": start,
            "num": min(max(num, 1), 20),  # 限制 1-20
            "hl": hl,
        }
        
        # 添加可选参数
        if cites:
            params["cites"] = cites
        if as_ylo:
            params["as_ylo"] = as_ylo
        if as_yhi:
            params["as_yhi"] = as_yhi
        if scisbd is not None:
            params["scisbd"] = scisbd
        if cluster:
            params["cluster"] = cluster
        if lr:
            params["lr"] = lr
        if as_sdt:
            params["as_sdt"] = as_sdt
        if safe:
            params["safe"] = safe
        if filter is not None:
            params["filter"] = filter
        if as_vis is not None:
            params["as_vis"] = as_vis
        if as_rr is not None:
            params["as_rr"] = as_rr
        
        # 额外参数
        params.update(kwargs)
        
        if self.verbose:
            print(f"[REQUEST] q={q}, start={start}, num={num}")
        
        # 发送请求
        data = self._request(params)
        
        if self.verbose:
            print(f"[RESPONSE] 获取到 {len(data.get('organic_results', []))} 条结果")
        
        # 检查错误
        if "error" in data:
            # 即使返回错误，只要 API 请求成功发出，就记录调用次数（消耗了配额）
            _record_api_call()
            return SearchResponse(
                success=False, 
                query=q, 
                error=data["error"],
                raw=data
            )
        
        # 解析结果
        items = []
        for result in data.get("organic_results", []):
            item = self._parse_result(result)
            items.append(item)
        
        # ========== 记录调用 + 缓存写入 ==========
        # API 调用成功，记录次数
        _record_api_call()
        
        # 写入缓存（无论是否有结果都缓存）
        if should_use_cache and self._cache:
            cache_data = {
                "query": q,
                "total_results": len(items),
                "has_results": len(items) > 0,
                "items": [
                    {
                        "title": item.title,
                        "link": item.link,
                        "snippet": item.snippet,
                        "publication_info": item.publication_info,
                        "authors": item.authors,
                        "cited_by_count": item.cited_by_count,
                        "cited_by_link": item.cited_by_link,
                        "versions_count": item.versions_count,
                        "versions_link": item.versions_link,
                        "position": item.position,
                        # 不缓存 raw 数据，太大
                    }
                    for item in items
                ],
                "cached_at": datetime.now().isoformat(),
                "params": {
                    "start": start,
                    "num": num,
                    "hl": hl,
                    "cites": cites,
                    "as_ylo": as_ylo,
                    "as_yhi": as_yhi,
                }
            }
            self._cache.set(cache_key, cache_data, ttl=CACHE_TTL)
            if items:
                print(f"[CACHE] 已写入缓存: {q[:50]}... ({len(items)} 条结果)")
            else:
                print(f"[CACHE] 已写入缓存（无结果）: {q[:50]}...")
        
        return SearchResponse(
            success=True,
            query=q,
            items=items,
            total_results=len(items),
            raw=data
        )
    
    def search_pages(
        self,
        q: str,
        max_results: int = 100,
        **kwargs
    ) -> SearchResponse:
        """
        分页获取多页结果
        
        Args:
            q: 搜索查询
            max_results: 最大结果数
            **kwargs: 其他参数
            
        Returns:
            合并的 SearchResponse
        """
        all_items = []
        start = 0
        
        while len(all_items) < max_results:
            result = self.search(q, start=start, num=20, **kwargs)
            
            if not result.success:
                if all_items:
                    break
                return result
            
            if not result.items:
                break
            
            all_items.extend(result.items)
            
            if len(result.items) < 20:
                break
            
            start += 20
            time.sleep(0.3)
        
        return SearchResponse(
            success=True,
            query=q,
            items=all_items[:max_results],
            total_results=len(all_items)
        )
    
    def _request(self, params: Dict) -> Dict:
        """发送请求（带重试）"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"[HTTP] GET {self.BASE_URL}")
                    print(f"[PARAMS] {params}")
                
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout
                )
                
                if self.verbose:
                    print(f"[STATUS] {response.status_code}")
                
                data = response.json()
                
                if response.status_code == 200:
                    return data
                
                # 限流重试
                if response.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"[RETRY] 429 限流，等待 {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                
                last_error = data.get("error", f"HTTP {response.status_code}")
                print(f"[ERROR] {last_error}")
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"[ERROR] 请求异常: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except json.JSONDecodeError as e:
                last_error = f"JSON 解析失败: {e}"
                print(f"[ERROR] {last_error}")
                if self.verbose:
                    print(f"[RAW] {response.text[:500]}")
        
        return {"error": last_error or "请求失败"}
    
    def _parse_result(self, result: Dict) -> ScholarResult:
        """解析单个搜索结果"""
        # 基本信息
        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        
        # 发表信息
        pub_info = result.get("publication_info", {})
        publication_info = pub_info.get("summary", "")
        
        # 作者列表
        authors = pub_info.get("authors", [])
        
        # 引用信息
        inline_links = result.get("inline_links", {})
        cited_by = inline_links.get("cited_by", {})
        cited_by_count = cited_by.get("total", 0) if isinstance(cited_by, dict) else 0
        cited_by_link = cited_by.get("link", "") if isinstance(cited_by, dict) else ""
        
        # 版本信息
        versions = inline_links.get("versions", {})
        versions_count = versions.get("total", 0) if isinstance(versions, dict) else 0
        versions_link = versions.get("link", "") if isinstance(versions, dict) else ""
        
        return ScholarResult(
            title=title,
            link=link,
            snippet=snippet,
            publication_info=publication_info,
            authors=authors,
            cited_by_count=cited_by_count,
            cited_by_link=cited_by_link,
            versions_count=versions_count,
            versions_link=versions_link,
            position=result.get("position", 0),
            raw=result
        )


# 便捷函数
def search_scholar(q: str, **kwargs) -> SearchResponse:
    """Google Scholar 搜索"""
    return SerpAPIScholar().search(q, **kwargs)


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("SerpAPI Google Scholar 测试")
    print("=" * 60)
    
    client = SerpAPIScholar(verbose=True)
    
    # 测试搜索
    query = 'author:"Ju Li"'
    print(f"\n查询: {query}\n")
    
    result = client.search(query, num=5)
    
    print("\n" + "=" * 60)
    print("解析结果:")
    print("=" * 60)
    print(f"成功: {result.success}")
    print(f"错误: {result.error}")
    print(f"结果数: {len(result.items)}")
    
    if result.items:
        print("\n论文列表:")
        for i, item in enumerate(result.items, 1):
            print(f"\n[{i}] {item.title}")
            print(f"    链接: {item.link}")
            print(f"    引用: {item.cited_by_count}")
            print(f"    发表: {item.publication_info[:60]}...")
            if item.authors:
                print(f"    作者: {[a.get('name', '') for a in item.authors]}")
