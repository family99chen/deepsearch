"""
通过 Google Search 搜索社交媒体相关链接

功能：
- 输入组织名 + 个人名字 + 社交媒体关键词
- 返回社交媒体相关链接列表
- 支持 MongoDB 缓存，避免重复 API 调用（6个月过期）
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
class SocialMediaLink:
    """社交媒体链接"""
    title: str
    url: str
    domain: str
    snippet: str
    platform: str
    relevance_score: float


class SocialMediaSearch:
    """社交媒体搜索器（独立于 organization 模块）"""

    # 常见社交媒体域名（可扩展）
    SOCIAL_DOMAINS = [
        "linkedin.com",
        "twitter.com",
        "x.com",
        "github.com",
        "researchgate.net",
        "orcid.org",
        "scholar.google.com",
        "dblp.org",
        "scopus.com",
    ]

    # 需要过滤的页面类型
    FILTER_PATTERNS = [
        r'/login',
        r'/signin',
        r'/share',
        r'/status/',
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

        # 初始化缓存
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

    def _get_cache_key(self, person_name: str, google_scholar_url: Optional[str]) -> str:
        """生成缓存键（优先使用 Google Scholar URL 去重）"""
        if google_scholar_url:
            return f"gs:{google_scholar_url.strip().lower()}"
        return f"person:{person_name.strip().lower()}"

    def _get_cached_links(
        self,
        person_name: str,
        organization: str,
        query: str,
        google_scholar_url: Optional[str] = None,
    ) -> Optional[List[SocialMediaLink]]:
        if not self._cache:
            return None

        cache_key = self._get_cache_key(person_name, google_scholar_url)
        try:
            doc = self._cache.collection.find_one({"key": cache_key})
            if not doc:
                return None

            value = doc.get("value", {})
            if not isinstance(value, dict):
                return None

            sm_data = value.get("social_media")
            if not sm_data:
                return None
            if sm_data.get("query") != query:
                return None

            expire_at = sm_data.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                if self.verbose:
                    print(f"[CACHE] {person_name} 的 social_media 链接已过期，将重新搜索")
                return None

            links_data = sm_data.get("links", [])
            links = []
            for link_data in links_data:
                links.append(SocialMediaLink(
                    title=link_data.get("title", ""),
                    url=link_data.get("url", ""),
                    domain=link_data.get("domain", ""),
                    snippet=link_data.get("snippet", ""),
                    platform=link_data.get("platform", "other"),
                    relevance_score=link_data.get("relevance_score", 0.0),
                ))

            if self.verbose:
                print(f"[CACHE HIT] 从缓存读取 {len(links)} 个 social_media 链接")
            key_hint = google_scholar_url or person_name
            logger.info(f"缓存命中: {key_hint}, {len(links)} 个 social_media 链接")
            return links
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 读取缓存失败: {e}")
            logger.warning(f"读取缓存失败: {person_name}, {e}")
            return None

    def _save_to_cache(
        self,
        person_name: str,
        organization: str,
        links: List[SocialMediaLink],
        query: str,
        google_scholar_url: Optional[str] = None,
    ) -> bool:
        if not self._cache:
            return False

        cache_key = self._get_cache_key(person_name, google_scholar_url)
        try:
            links_data = [asdict(link) for link in links]
            from datetime import timedelta
            sm_data = {
                "query": query,
                "links": links_data,
                "source_organization": organization,
                "google_scholar_url": google_scholar_url,
                "updated_at": datetime.utcnow(),
                "expire_at": datetime.utcnow() + timedelta(seconds=CACHE_TTL_SECONDS),
            }

            self._cache.collection.update_one(
                {"key": cache_key},
                {
                    "$set": {
                        "value.social_media": sm_data,
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
                print(f"[CACHE SAVE] 已缓存 {len(links)} 个 social_media 链接")
            key_hint = google_scholar_url or person_name
            logger.info(f"缓存写入: {key_hint} @ {organization}, {len(links)} 个 social_media 链接")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 保存缓存失败: {e}")
            logger.warning(f"保存缓存失败: {person_name}, {e}")
            return False

    def _build_query(self, person_name: str, organization: str) -> str:
        # name + org + social media 关键词
        domains = " OR ".join([f"site:{d}" for d in self.SOCIAL_DOMAINS])
        return f'"{person_name}" "{organization}" (social media OR {domains})'

    def search_person_in_social_media(
        self,
        person_name: str,
        organization: str,
        max_results: int = 20,
        force_refresh: bool = False,
        google_scholar_url: Optional[str] = None,
    ) -> List[SocialMediaLink]:
        if self.verbose:
            print(f"[INFO] 搜索: {person_name} @ {organization} (social media)")
        logger.info(f"开始搜索社交媒体链接: {person_name} @ {organization}")

        query = self._build_query(person_name, organization)
        if self.use_cache and not force_refresh:
            cached_links = self._get_cached_links(
                person_name,
                organization,
                query,
                google_scholar_url,
            )
            if cached_links is not None:
                return cached_links[:max_results]

        if self.verbose:
            print(f"[DEBUG] 查询: {query}")

        result = self.api.search(query, num=min(max_results, 10))
        if not result.success:
            print(f"[ERROR] 搜索失败: {result.error}")
            logger.error(f"社交媒体搜索失败: {person_name} @ {organization}, {result.error}")
            return []

        if not result.items:
            if self.verbose:
                print("[INFO] 没有搜索结果")
            return []

        links = []
        for item in result.items:
            link = self._parse_search_result(item, person_name, organization)
            if link:
                links.append(link)

        links.sort(key=lambda x: x.relevance_score, reverse=True)
        if self.verbose:
            print(f"[INFO] 找到 {len(links)} 个相关链接")
        logger.info(f"社交媒体搜索完成: {person_name} @ {organization}, 找到 {len(links)} 个链接")

        if self.use_cache and links:
            self._save_to_cache(person_name, organization, links, query, google_scholar_url)

        return links[:max_results]

    def search_person_in_social_media_with_raw(
        self,
        person_name: str,
        organization: str,
        max_results: int = 20,
        force_refresh: bool = False,
        google_scholar_url: Optional[str] = None,
    ) -> Tuple[List[SocialMediaLink], Dict[str, Any]]:
        if self.verbose:
            print(f"[INFO] 搜索: {person_name} @ {organization} (social media)")
        logger.info(f"开始搜索社交媒体链接: {person_name} @ {organization}")

        query = self._build_query(person_name, organization)
        if self.use_cache and not force_refresh:
            cached_links = self._get_cached_links(
                person_name,
                organization,
                query,
                google_scholar_url,
            )
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

        if self.verbose:
            print(f"[DEBUG] 查询: {query}")

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
            logger.error(f"社交媒体搜索失败: {person_name} @ {organization}, {result.error}")
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
        if self.verbose:
            print(f"[INFO] 找到 {len(links)} 个相关链接")
        logger.info(f"社交媒体搜索完成: {person_name} @ {organization}, 找到 {len(links)} 个链接")

        if self.use_cache and links:
            self._save_to_cache(person_name, organization, links, query, google_scholar_url)

        return links[:max_results], raw

    def _parse_search_result(
        self,
        item: SearchResult,
        person_name: str,
        organization: str,
    ) -> Optional[SocialMediaLink]:
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

        platform = self._classify_platform(domain)
        relevance_score = self._calculate_relevance(url, title, snippet, person_name, organization)

        return SocialMediaLink(
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            platform=platform,
            relevance_score=relevance_score,
        )

    def _classify_platform(self, domain: str) -> str:
        if "linkedin.com" in domain:
            return "linkedin"
        if "twitter.com" in domain or "x.com" in domain:
            return "twitter"
        if "github.com" in domain:
            return "github"
        if "researchgate.net" in domain:
            return "researchgate"
        if "orcid.org" in domain:
            return "orcid"
        if "scholar.google.com" in domain:
            return "scholar"
        if "dblp.org" in domain:
            return "dblp"
        if "scopus.com" in domain:
            return "scopus"
        return "other"

    def _calculate_relevance(
        self,
        url: str,
        title: str,
        snippet: str,
        person_name: str,
        organization: str,
    ) -> float:
        score = 0.0
        if person_name.lower() in title.lower():
            score += 0.3
        name_parts = person_name.lower().split()
        url_lower = url.lower()
        if any(part in url_lower for part in name_parts if len(part) > 2):
            score += 0.2
        org_parts = organization.lower().split()
        if any(part in url_lower for part in org_parts if len(part) > 2):
            score += 0.2
        if person_name.lower() in snippet.lower():
            score += 0.2
        if any(domain in url_lower for domain in self.SOCIAL_DOMAINS):
            score += 0.1
        return min(score, 1.0)


# ============ 便捷函数 ============

def search_person_in_social_media(
    person_name: str,
    organization: str,
    max_results: int = 20,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    google_scholar_url: Optional[str] = None,
) -> List[SocialMediaLink]:
    searcher = SocialMediaSearch(verbose=verbose, use_cache=use_cache)
    return searcher.search_person_in_social_media(
        person_name=person_name,
        organization=organization,
        max_results=max_results,
        force_refresh=force_refresh,
        google_scholar_url=google_scholar_url,
    )


def search_person_in_social_media_with_raw(
    person_name: str,
    organization: str,
    max_results: int = 20,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    google_scholar_url: Optional[str] = None,
) -> Tuple[List[SocialMediaLink], Dict[str, Any]]:
    searcher = SocialMediaSearch(verbose=verbose, use_cache=use_cache)
    return searcher.search_person_in_social_media_with_raw(
        person_name=person_name,
        organization=organization,
        max_results=max_results,
        force_refresh=force_refresh,
        google_scholar_url=google_scholar_url,
    )

