"""
SerpAPI Google Scholar 搜索模块

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
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    SerpAPI Google Scholar 搜索客户端
    
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
    ):
        """
        初始化客户端
        
        Args:
            api_key: SerpAPI API Key
            max_retries: 最大重试次数
            retry_delay: 重试基础延迟（秒）
            timeout: 请求超时（秒）
            verbose: 是否打印详细信息
        """
        self.api_key = api_key or DEFAULT_API_KEY
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verbose = verbose
        
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
        **kwargs
    ) -> SearchResponse:
        """
        Google Scholar 搜索
        
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
            
        Returns:
            SearchResponse
        """
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
