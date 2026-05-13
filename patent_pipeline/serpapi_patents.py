import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

try:
    from utils.usage_tracker import UsageTracker
    HAS_USAGE_TRACKER = True
except ImportError:
    HAS_USAGE_TRACKER = False

from patent_pipeline.models import PatentDetail, PatentSearchItem

CACHE_TTL_DEFAULT = 180 * 24 * 3600
SEARCH_CACHE_COLLECTION = "patent_search_cache"
DETAIL_CACHE_COLLECTION = "patent_detail_cache"


def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _cache(collection_name: str) -> Optional["MongoCache"]:
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=collection_name)
        return cache if cache.is_connected() else None
    except Exception:
        return None


def _usage_tracker() -> Optional["UsageTracker"]:
    if not HAS_USAGE_TRACKER:
        return None
    try:
        storage_path = Path(__file__).parent.parent / "total_usage" / "serpapi.json"
        return UsageTracker(storage_path=storage_path, auto_save=True)
    except Exception:
        return None


def _record_api_call(endpoint: str) -> None:
    tracker = _usage_tracker()
    if tracker:
        tracker.record(endpoint, count=1)


def _hash_key(parts: Dict[str, Any]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def patent_id_from_serpapi_link(link: str) -> str:
    try:
        query = parse_qs(urlparse(link).query)
        return (query.get("patent_id") or [""])[0]
    except Exception:
        return ""


def _parse_search_item(raw: Dict[str, Any]) -> PatentSearchItem:
    return PatentSearchItem(
        position=raw.get("position", 0) or 0,
        patent_id=raw.get("patent_id", "") or patent_id_from_serpapi_link(raw.get("serpapi_link", "")),
        patent_link=raw.get("patent_link", "") or "",
        serpapi_link=raw.get("serpapi_link", "") or "",
        title=raw.get("title", "") or "",
        snippet=raw.get("snippet", "") or "",
        priority_date=raw.get("priority_date"),
        filing_date=raw.get("filing_date"),
        publication_date=raw.get("publication_date"),
        grant_date=raw.get("grant_date"),
        inventor=raw.get("inventor", "") or "",
        assignee=raw.get("assignee", "") or "",
        publication_number=raw.get("publication_number", "") or "",
        language=raw.get("language", "") or "",
        pdf=raw.get("pdf", "") or "",
        country_status=raw.get("country_status", {}) or {},
        raw=raw,
    )


def _parse_detail(raw: Dict[str, Any], patent_id: str) -> PatentDetail:
    return PatentDetail(
        patent_id=patent_id,
        title=raw.get("title", "") or "",
        publication_number=raw.get("publication_number", "") or "",
        patent_link=(raw.get("search_metadata") or {}).get("google_patents_details_url", ""),
        pdf=raw.get("pdf", "") or "",
        inventors=raw.get("inventors", []) or [],
        assignees=raw.get("assignees", []) or [],
        abstract=raw.get("abstract", "") or "",
        claims=raw.get("claims", []) or [],
        classifications=raw.get("classifications", []) or [],
        priority_date=raw.get("priority_date"),
        filing_date=raw.get("filing_date"),
        publication_date=raw.get("publication_date"),
        grant_date=raw.get("grant_date"),
        family_id=raw.get("family_id"),
        worldwide_applications=raw.get("worldwide_applications", {}) or {},
        legal_events=raw.get("legal_events", []) or [],
        cited_by=raw.get("cited_by", {}) or {},
        patent_citations=raw.get("patent_citations", {}) or {},
        raw=raw,
    )


class SerpAPIPatents:
    BASE_URL = "https://serpapi.com/search.json"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 1.0,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        verbose: bool = False,
    ):
        config = _load_config()
        serp_config = config.get("serpapi", {})
        patent_config = config.get("patent_pipeline", {})
        ttl_days = patent_config.get("cache_ttl_days", 180)

        self.api_key = api_key or serp_config.get("api_key") or os.environ.get("SERPAPI_KEY", "")
        self.timeout = timeout if timeout is not None else serp_config.get("timeout", 30)
        self.max_retries = max_retries if max_retries is not None else serp_config.get("max_retries", 3)
        self.retry_delay = retry_delay
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl if cache_ttl is not None else int(ttl_days * 24 * 3600)
        self.verbose = verbose
        self._search_cache = _cache(SEARCH_CACHE_COLLECTION) if use_cache else None
        self._detail_cache = _cache(DETAIL_CACHE_COLLECTION) if use_cache else None

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "SerpAPI API key is not configured"}

        params = {**params, "api_key": self.api_key}
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    printable = {**params, "api_key": "<redacted>"}
                    print(f"[SerpAPI Patents] request={printable}")
                response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
                data = response.json()
                if response.status_code == 200:
                    return data
                if response.status_code == 429 and attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                last_error = data.get("error", f"HTTP {response.status_code}")
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        return {"error": last_error or "request failed"}

    def search(self, q: str, page: int = 1, num: int = 10, use_cache: Optional[bool] = None) -> Dict[str, Any]:
        should_use_cache = self.use_cache if use_cache is None else use_cache
        num = min(max(int(num), 10), 100)
        page = max(int(page), 1)
        cache_key = _hash_key({"engine": "google_patents", "q": q, "page": page, "num": num})
        if should_use_cache and self._search_cache:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                items = [_parse_search_item(item) for item in cached.get("organic_results", [])]
                return {
                    "success": True,
                    "query": q,
                    "items": items,
                    "raw": cached.get("raw", {}),
                    "from_cache": True,
                }

        params = {
            "engine": "google_patents",
            "q": q,
            "page": page,
            "num": num,
        }
        data = self._request(params)
        _record_api_call("serpapi_patents")
        if "error" in data:
            if "hasn't returned any results" in str(data["error"]):
                if should_use_cache and self._search_cache:
                    self._search_cache.set(
                        cache_key,
                        {
                            "query": q,
                            "organic_results": [],
                            "search_information": {},
                            "search_metadata": data.get("search_metadata", {}),
                            "cached_at": datetime.now().isoformat(),
                            "raw": data,
                        },
                        ttl=self.cache_ttl,
                    )
                return {"success": True, "query": q, "items": [], "raw": data, "from_cache": False}
            return {"success": False, "query": q, "items": [], "raw": data, "error": data["error"]}

        organic = data.get("organic_results", []) or []
        items = [_parse_search_item(item) for item in organic]
        if should_use_cache and self._search_cache:
            self._search_cache.set(
                cache_key,
                {
                    "query": q,
                    "organic_results": organic,
                    "search_information": data.get("search_information", {}),
                    "search_metadata": data.get("search_metadata", {}),
                    "cached_at": datetime.now().isoformat(),
                    "raw": data,
                },
                ttl=self.cache_ttl,
            )
        return {"success": True, "query": q, "items": items, "raw": data, "from_cache": False}

    def detail(self, patent_id: str, use_cache: Optional[bool] = None) -> Dict[str, Any]:
        should_use_cache = self.use_cache if use_cache is None else use_cache
        if not patent_id:
            return {"success": False, "error": "missing patent_id", "detail": None}

        cache_key = _hash_key({"engine": "google_patents_details", "patent_id": patent_id})
        if should_use_cache and self._detail_cache:
            cached = self._detail_cache.get(cache_key)
            if cached is not None:
                return {
                    "success": True,
                    "detail": _parse_detail(cached.get("raw", cached), patent_id),
                    "raw": cached.get("raw", cached),
                    "from_cache": True,
                }

        data = self._request({"engine": "google_patents_details", "patent_id": patent_id})
        _record_api_call("serpapi_patent_details")
        if "error" in data:
            return {"success": False, "error": data["error"], "detail": None, "raw": data}

        if should_use_cache and self._detail_cache:
            self._detail_cache.set(
                cache_key,
                {
                    "patent_id": patent_id,
                    "raw": data,
                    "cached_at": datetime.now().isoformat(),
                },
                ttl=self.cache_ttl,
            )
        return {
            "success": True,
            "detail": _parse_detail(data, patent_id),
            "raw": data,
            "from_cache": False,
        }
