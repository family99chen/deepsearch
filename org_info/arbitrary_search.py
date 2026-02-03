"""
通用查询搜索模块（给 AI 生成的 query 使用）

功能：
- 输入 query + 个人名字 + 组织名
- 返回搜索链接列表
- 支持 MongoDB 缓存（person_relevant_link -> value.arbitrary）
"""

import sys
import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Google Search API
from google_search_api.google_search import GoogleSearchAPISync, SearchResult

# 导入 MongoDB 缓存
try:
    from localdb.insert_mongo import MongoCache
    HAS_MONGO_CACHE = True
except ImportError:
    HAS_MONGO_CACHE = False

# 导入日志
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ============ 缓存配置 ============
CACHE_COLLECTION = "person_relevant_link"
CACHE_TTL_SECONDS = 6 * 30 * 24 * 60 * 60  # 6个月


@dataclass
class ArbitraryLink:
    title: str
    url: str
    domain: str
    snippet: str
    relevance_score: float


class ArbitrarySearch:
    """通用查询搜索器（独立模块）"""

    FILTER_PATTERNS = [
        r'/login',
        r'/signin',
        r'/share',
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = True,
        use_cache: bool = True,
    ):
        self.api = GoogleSearchAPISync(
            api_key=api_key,
            cx=cx,
            max_retries=max_retries,
        )
        self.verbose = verbose
        self.use_cache = use_cache and HAS_MONGO_CACHE

        self._cache = None
        if self.use_cache:
            try:
                self._cache = MongoCache(collection_name=CACHE_COLLECTION)
                if not self._cache.is_connected():
                    if self.verbose:
                        print("[WARNING] MongoDB 缓存连接失败，将不使用缓存")
                    logger.warning("MongoDB 缓存连接失败")
                    self._cache = None
                    self.use_cache = False
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] 初始化缓存失败: {e}")
                logger.warning(f"初始化缓存失败: {e}")
                self._cache = None
                self.use_cache = False

    def _get_cache_key(self, query: str, google_scholar_url: Optional[str]) -> str:
        if google_scholar_url:
            return f"gs:{google_scholar_url.strip().lower()}"
        return f"q:{query.strip().lower()}"

    def _get_cached_links(
        self,
        query: str,
        google_scholar_url: Optional[str] = None,
    ) -> Optional[List[ArbitraryLink]]:
        if not self._cache:
            return None

        cache_key = self._get_cache_key(query, google_scholar_url)
        try:
            doc = self._cache.collection.find_one({"key": cache_key})
            if not doc:
                return None

            value = doc.get("value", {})
            if not isinstance(value, dict):
                return None

            arbitrary = value.get("arbitrary") if isinstance(value.get("arbitrary"), dict) else {}
            entry = arbitrary.get(query)
            if not entry:
                return None

            expire_at = entry.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                if self.verbose:
                    print("[CACHE] arbitrary 链接已过期，将重新搜索")
                return None

            links_data = entry.get("links", [])
            links = []
            for link_data in links_data:
                links.append(ArbitraryLink(
                    title=link_data.get("title", ""),
                    url=link_data.get("url", ""),
                    domain=link_data.get("domain", ""),
                    snippet=link_data.get("snippet", ""),
                    relevance_score=link_data.get("relevance_score", 0.0),
                ))

            if self.verbose:
                print(f"[CACHE HIT] 从缓存读取 {len(links)} 个 arbitrary 链接")
            return links
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 读取缓存失败: {e}")
            logger.warning(f"读取缓存失败: {query}, {e}")
            return None

    def _save_to_cache(
        self,
        query: str,
        links: List[ArbitraryLink],
        google_scholar_url: Optional[str] = None,
    ) -> bool:
        if not self._cache:
            return False

        cache_key = self._get_cache_key(query, google_scholar_url)
        try:
            links_data = [asdict(link) for link in links]
            from datetime import timedelta
            entry = {
                "query": query,
                "links": links_data,
                "google_scholar_url": google_scholar_url,
                "updated_at": datetime.utcnow(),
                "expire_at": datetime.utcnow() + timedelta(seconds=CACHE_TTL_SECONDS),
            }

            existing = self._cache.collection.find_one({"key": cache_key}) or {}
            value = existing.get("value", {})
            if not isinstance(value, dict):
                value = {}
            arbitrary = value.get("arbitrary")
            if not isinstance(arbitrary, dict):
                arbitrary = {}
            arbitrary[query] = entry
            value["arbitrary"] = arbitrary

            self._cache.collection.update_one(
                {"key": cache_key},
                {
                    "$set": {
                        "value": value,
                        "updated_at": datetime.utcnow(),
                    },
                    "$setOnInsert": {
                        "key": cache_key,
                        "created_at": datetime.utcnow(),
                    }
                },
                upsert=True
            )

            if self.verbose:
                print(f"[CACHE SAVE] 已缓存 {len(links)} 个 arbitrary 链接")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 保存缓存失败: {e}")
            logger.warning(f"保存缓存失败: {query}, {e}")
            return False

    def search_with_raw(
        self,
        query: str,
        person_name: Optional[str] = None,
        organization: Optional[str] = None,
        max_results: int = 20,
        force_refresh: bool = False,
        google_scholar_url: Optional[str] = None,
    ) -> Tuple[List[ArbitraryLink], Dict[str, Any]]:
        if self.verbose:
            print(f"[INFO] arbitrary 搜索: {query}")

        if self.use_cache and not force_refresh:
            cached_links = self._get_cached_links(query, google_scholar_url)
            if cached_links is not None:
                raw = {
                    "from_cache": True,
                    "success": True,
                    "query": query,
                    "total_results": len(cached_links),
                    "error": None,
                    "items": [
                        {
                            "title": link.title,
                            "link": link.url,
                            "snippet": link.snippet,
                            "display_link": link.domain,
                            "html_title": None,
                            "html_snippet": None,
                            "formatted_url": None,
                        }
                        for link in cached_links[:max_results]
                    ],
                    "google_scholar_url": google_scholar_url,
                }
                return cached_links[:max_results], raw

        result = self.api.search(query, num=min(max_results, 10))
        raw = {
            "from_cache": False,
            "success": result.success,
            "query": result.query,
            "total_results": result.total_results,
            "error": result.error,
            "items": [
                {
                    "title": item.title,
                    "link": item.link,
                    "snippet": item.snippet,
                    "display_link": item.display_link,
                    "html_title": item.html_title,
                    "html_snippet": item.html_snippet,
                    "formatted_url": item.formatted_url,
                }
                for item in result.items
            ],
            "google_scholar_url": google_scholar_url,
        }

        if not result.success:
            print(f"[ERROR] 搜索失败: {result.error}")
            logger.error(f"arbitrary 搜索失败: {query}, {result.error}")
            return [], raw

        if not result.items:
            if self.verbose:
                print("[INFO] 没有搜索结果")
            return [], raw

        links = []
        for item in result.items:
            link = self._parse_search_result(item, person_name, organization)
            if link:
                links.append(link)

        links.sort(key=lambda x: x.relevance_score, reverse=True)
        if self.use_cache and links:
            self._save_to_cache(query, links, google_scholar_url)

        return links[:max_results], raw

    def _parse_search_result(
        self,
        item: SearchResult,
        person_name: Optional[str],
        organization: Optional[str],
    ) -> Optional[ArbitraryLink]:
        url = item.link
        title = item.title
        snippet = item.snippet

        for pattern in self.FILTER_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return None

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            domain = ""

        relevance_score = self._calculate_relevance(url, title, snippet, person_name, organization)
        return ArbitraryLink(
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            relevance_score=relevance_score,
        )

    def _calculate_relevance(
        self,
        url: str,
        title: str,
        snippet: str,
        person_name: Optional[str],
        organization: Optional[str],
    ) -> float:
        score = 0.0
        if person_name and person_name.lower() in title.lower():
            score += 0.3
        name_parts = (person_name or "").lower().split()
        url_lower = url.lower()
        if any(part in url_lower for part in name_parts if len(part) > 2):
            score += 0.2
        org_parts = (organization or "").lower().split()
        if any(part in url_lower for part in org_parts if len(part) > 2):
            score += 0.2
        if person_name and person_name.lower() in snippet.lower():
            score += 0.2
        return min(score, 1.0)


def search_arbitrary_with_raw(
    query: str,
    max_results: int = 20,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    google_scholar_url: Optional[str] = None,
) -> Tuple[List[ArbitraryLink], Dict[str, Any]]:
    searcher = ArbitrarySearch(verbose=verbose, use_cache=use_cache)
    return searcher.search_with_raw(
        query=query,
        max_results=max_results,
        force_refresh=force_refresh,
        google_scholar_url=google_scholar_url,
    )

