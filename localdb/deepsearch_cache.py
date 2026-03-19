"""
DeepSearch 专用缓存集合封装。

当前包含两类缓存：
1. person_pipeline_cache: 最终人物报告缓存
2. page_analysis_cache: 单页面分析结果缓存
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .insert_mongo import MongoCache


PERSON_PIPELINE_COLLECTION = "person_pipeline_cache"
PAGE_ANALYSIS_COLLECTION = "page_analysis_cache"

PERSON_PIPELINE_CACHE_VERSION = "v2_no_extra"
PAGE_ANALYSIS_CACHE_VERSION = "v1"

PERSON_PIPELINE_CACHE_TTL = 7 * 24 * 3600
PAGE_ANALYSIS_CACHE_TTL = 7 * 24 * 3600


def _normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _normalize_url(url: Optional[str]) -> str:
    return (url or "").strip().lower()


def _stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


class PersonPipelineCache:
    """DeepSearch 总 pipeline 结果缓存。"""

    def __init__(self):
        self._cache = MongoCache(collection_name=PERSON_PIPELINE_COLLECTION)

    def is_connected(self) -> bool:
        return self._cache.is_connected()

    def _build_key(
        self,
        google_scholar_url: str,
        max_iterations: int,
        max_links: int,
        max_workers: int,
        model: str,
        backend: str,
        extra_sources: Optional[List[str]] = None,
    ) -> str:
        return (
            f"gs:{_normalize_url(google_scholar_url)}"
            f"|iter:{max_iterations}"
            f"|links:{max_links}"
            f"|workers:{max_workers}"
            f"|model:{model}"
            f"|backend:{backend}"
            f"|ver:{PERSON_PIPELINE_CACHE_VERSION}"
        )

    def get_final_result(
        self,
        google_scholar_url: str,
        max_iterations: int,
        max_links: int,
        max_workers: int,
        model: str,
        backend: str,
        extra_sources: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            return None

        key = self._build_key(
            google_scholar_url=google_scholar_url,
            max_iterations=max_iterations,
            max_links=max_links,
            max_workers=max_workers,
            model=model,
            backend=backend,
            extra_sources=extra_sources,
        )
        cached = self._cache.get(key)
        if not isinstance(cached, dict):
            return None
        final = cached.get("final")
        if not isinstance(final, dict):
            return None
        return cached

    def set_final_result(
        self,
        google_scholar_url: str,
        max_iterations: int,
        max_links: int,
        max_workers: int,
        model: str,
        backend: str,
        result: Dict[str, Any],
        extra_sources: Optional[List[str]] = None,
        stages: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.is_connected():
            return False

        key = self._build_key(
            google_scholar_url=google_scholar_url,
            max_iterations=max_iterations,
            max_links=max_links,
            max_workers=max_workers,
            model=model,
            backend=backend,
            extra_sources=extra_sources,
        )

        doc = {
            "meta": {
                "google_scholar_url": google_scholar_url,
                "max_iterations": max_iterations,
                "max_links": max_links,
                "max_workers": max_workers,
                "model": model,
                "backend": backend,
                "cache_version": PERSON_PIPELINE_CACHE_VERSION,
                "cached_at": datetime.utcnow().isoformat(),
            },
            "final": result,
            "stages": stages or {},
        }
        return self._cache.set(key, doc, ttl=PERSON_PIPELINE_CACHE_TTL)


class PageAnalysisCache:
    """单网页分析结果缓存，供多个 pipeline 共用。"""

    def __init__(self):
        self._cache = MongoCache(collection_name=PAGE_ANALYSIS_COLLECTION)

    def is_connected(self) -> bool:
        return self._cache.is_connected()

    def _build_key(self, url: str, person_name: str) -> str:
        raw = {
            "url": _normalize_url(url),
            "person_name": _normalize_text(person_name),
            "version": PAGE_ANALYSIS_CACHE_VERSION,
        }
        return f"page:{_stable_hash(raw)}"

    def get_result(self, url: str, person_name: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            return None

        key = self._build_key(url, person_name)
        cached = self._cache.get(key)
        if not isinstance(cached, dict):
            return None
        if cached.get("cache_version") != PAGE_ANALYSIS_CACHE_VERSION:
            return None
        return cached

    def set_result(self, url: str, person_name: str, result: Dict[str, Any]) -> bool:
        if not self.is_connected():
            return False

        key = self._build_key(url, person_name)
        doc = {
            "url": url,
            "person_name": person_name,
            "cache_version": PAGE_ANALYSIS_CACHE_VERSION,
            "cached_at": datetime.utcnow().isoformat(),
            "result": result,
        }
        return self._cache.set(key, doc, ttl=PAGE_ANALYSIS_CACHE_TTL)


_person_pipeline_cache: Optional[PersonPipelineCache] = None
_page_analysis_cache: Optional[PageAnalysisCache] = None


def get_person_pipeline_cache() -> PersonPipelineCache:
    global _person_pipeline_cache
    if _person_pipeline_cache is None:
        _person_pipeline_cache = PersonPipelineCache()
    return _person_pipeline_cache


def get_page_analysis_cache() -> PageAnalysisCache:
    global _page_analysis_cache
    if _page_analysis_cache is None:
        _page_analysis_cache = PageAnalysisCache()
    return _page_analysis_cache
