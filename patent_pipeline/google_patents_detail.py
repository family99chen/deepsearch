import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

try:
    from proxy import get_proxy_dict
except ImportError:
    def get_proxy_dict():
        return None

from patent_pipeline.models import PatentDetail, PatentSearchItem

DETAIL_CACHE_COLLECTION = "patent_detail_cache"
CACHE_TTL_DEFAULT = 180 * 24 * 3600


def _cache() -> Optional["MongoCache"]:
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=DETAIL_CACHE_COLLECTION)
        return cache if cache.is_connected() else None
    except Exception:
        return None


def _hash_key(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _text_list(soup: BeautifulSoup, selector: str) -> List[str]:
    values = []
    for elem in soup.select(selector):
        text = elem.get_text(" ", strip=True)
        if text and text not in values:
            values.append(text)
    return values


def _first_text(soup: BeautifulSoup, selector: str) -> str:
    elem = soup.select_one(selector)
    return elem.get_text(" ", strip=True) if elem else ""


def _parse_google_patents_html(html: str, item: PatentSearchItem) -> PatentDetail:
    soup = BeautifulSoup(html, "html.parser")
    title = _first_text(soup, "h1") or item.title
    if " - Google Patents" in title:
        title = title.split(" - Google Patents", 1)[0].strip()
    if " - " in title and title.startswith(item.publication_number):
        title = title.split(" - ", 1)[1].strip()

    inventor_names = _text_list(soup, "dd[itemprop=inventor]")
    assignees = (
        _text_list(soup, "dd[itemprop=assigneeCurrent]")
        or _text_list(soup, "dd[itemprop=assigneeOriginal]")
    )
    abstract = _first_text(soup, "section[itemprop=abstract]")
    if abstract.lower().startswith("abstract "):
        abstract = abstract[len("Abstract "):].strip()
    claims = _text_list(soup, "section[itemprop=claims] div.claim-text")
    classifications = [
        {
            "code": elem.get_text(" ", strip=True),
            "description": elem.get("title", ""),
        }
        for elem in soup.select("li[itemprop=classifications] span[itemprop=Code], ul[itemprop=classifications] span[itemprop=Code]")
    ]

    return PatentDetail(
        patent_id=item.patent_id,
        title=title,
        publication_number=item.publication_number,
        patent_link=item.patent_link,
        pdf=item.pdf,
        inventors=[{"name": name} for name in inventor_names],
        assignees=assignees,
        abstract=abstract,
        claims=claims,
        classifications=classifications,
        priority_date=item.priority_date,
        filing_date=item.filing_date,
        publication_date=item.publication_date,
        grant_date=item.grant_date,
        raw={
            "source": "google_patents_page",
            "html_length": len(html),
            "fetched_at": datetime.now().isoformat(),
        },
    )


class GooglePatentsDetailFetcher:
    def __init__(
        self,
        use_cache: bool = True,
        cache_ttl: int = CACHE_TTL_DEFAULT,
        timeout: int = 30,
        verbose: bool = False,
    ):
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.verbose = verbose
        self._cache = _cache() if use_cache else None

    def _cache_key(self, item: PatentSearchItem) -> str:
        return _hash_key(f"google_patents_page|{item.patent_id or item.patent_link}")

    def _fetch_requests(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        response = requests.get(
            url,
            headers=headers,
            proxies=get_proxy_dict(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def _fetch_chromium(self, url: str) -> Optional[str]:
        driver = None
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
            from org_info.chrome_binary import resolve_chrome_binary_path

            options = Options()
            options.binary_location = resolve_chrome_binary_path()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1280,1600")
            options.add_argument("--lang=en-US")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/120.0.0.0 Safari/537.36"
            )

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            driver.get(url)
            WebDriverWait(driver, self.timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            return driver.page_source
        except Exception as exc:
            if self.verbose:
                print(f"[PatentDetail] Chromium fallback failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass

    def get_detail(self, item: PatentSearchItem) -> Optional[PatentDetail]:
        if not item.patent_link:
            return None

        cache_key = self._cache_key(item)
        if self.use_cache and self._cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict):
                return PatentDetail(**cached)

        html = None
        started = time.time()
        try:
            html = self._fetch_requests(item.patent_link)
        except Exception as exc:
            if self.verbose:
                print(f"[PatentDetail] requests failed: {type(exc).__name__}: {exc}")

        if not html:
            html = self._fetch_chromium(item.patent_link)
        if not html:
            return None

        detail = _parse_google_patents_html(html, item)
        detail.raw["elapsed_seconds"] = round(time.time() - started, 3)
        if self.use_cache and self._cache:
            self._cache.set(cache_key, detail.__dict__, ttl=self.cache_ttl)
        return detail


def fetch_details_concurrently(
    items: Iterable[PatentSearchItem],
    concurrency: int,
    use_cache: bool,
    cache_ttl: int,
    verbose: bool = False,
) -> Dict[str, Optional[PatentDetail]]:
    items = list(items)
    if not items:
        return {}
    fetcher = GooglePatentsDetailFetcher(use_cache=use_cache, cache_ttl=cache_ttl, verbose=verbose)
    results: Dict[str, Optional[PatentDetail]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {
            executor.submit(fetcher.get_detail, item): item.patent_id or item.patent_link
            for item in items
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = None
    return results
