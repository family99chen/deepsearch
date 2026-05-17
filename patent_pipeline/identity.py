import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "google_scholar_url"))

try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

from google_scholar_url.fetch_author_person_info import get_author_name
from google_scholar_url.fetch_person_organization import get_author_organization
from org_info.extract_org_info import get_profile_info

try:
    from google_scholar_url.fetch_author_google_paper_list import GoogleScholarAuthorFetcher
except ImportError:
    GoogleScholarAuthorFetcher = None

from patent_pipeline.models import Identity

ORCID_INFO_COLLECTION = "orcid_info"
SCHOLAR_PROFILE_COLLECTION = "google_scholar_person_detail"


def _cache(collection_name: str) -> Optional["MongoCache"]:
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=collection_name)
        return cache if cache.is_connected() else None
    except Exception:
        return None


def _extract_user_id(value: str) -> Optional[str]:
    if not value:
        return None
    if "://" not in value and "scholar.google" not in value and "user=" not in value:
        return value.strip()
    try:
        parsed = urlparse(value)
        return (parse_qs(parsed.query).get("user") or [None])[0]
    except Exception:
        return None


def _scholar_url(user_id_or_url: str) -> str:
    user_id = _extract_user_id(user_id_or_url)
    if not user_id:
        return user_id_or_url
    return f"https://scholar.google.com/citations?user={quote(user_id)}&hl=zh-CN"


def _value_or_none(value: Any) -> Optional[Any]:
    if value is None or value == "":
        return None
    return value


def _read_orcid_identity_from_cache(orcid_id: str) -> Dict[str, Any]:
    cache = _cache(ORCID_INFO_COLLECTION)
    if not cache:
        return {}

    full_name = cache.get_field(orcid_id, "full_name")
    if not full_name:
        given = cache.get_field(orcid_id, "given_name")
        family = cache.get_field(orcid_id, "family_name")
        credit = cache.get_field(orcid_id, "credit_name")
        full_name = credit or " ".join(part for part in [given, family] if part).strip() or None

    organization = cache.get_field(orcid_id, "primary_organization")
    if not organization:
        organizations = cache.get_field(orcid_id, "organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, dict):
                organization = first.get("name")

    return {
        "person_name": _value_or_none(full_name),
        "organization": _value_or_none(organization),
    }


def resolve_from_orcid(orcid_id: str, use_cache: bool = True) -> Optional[Identity]:
    orcid_id = (orcid_id or "").strip()
    if not orcid_id:
        return None

    cached = _read_orcid_identity_from_cache(orcid_id) if use_cache else {}
    person_name = cached.get("person_name")
    organization = cached.get("organization")

    if not person_name:
        try:
            name_data = get_author_name(orcid_id, use_cache=use_cache)
            person_name = name_data.get("full_name")
        except Exception:
            person_name = None

    if not organization:
        try:
            organization = get_author_organization(orcid_id, use_cache=use_cache)
        except Exception:
            organization = None

    if not person_name:
        return None

    return Identity(
        person_name=person_name,
        organization=organization or "",
        source="orcid",
        orcid_id=orcid_id,
        raw={"cached": cached},
    )


def _read_scholar_profile_from_cache(user_id: str) -> Dict[str, Any]:
    cache = _cache(SCHOLAR_PROFILE_COLLECTION)
    if not cache:
        return {}
    cached = cache.get(user_id)
    return cached if isinstance(cached, dict) else {}


def _fetch_scholar_profile_with_fallback(url: str, use_cache: bool) -> Dict[str, Any]:
    profile = get_profile_info(url, use_cache=use_cache, verbose=False) or {}
    if profile.get("name"):
        return profile

    if GoogleScholarAuthorFetcher is None:
        return profile

    try:
        fetcher = GoogleScholarAuthorFetcher(use_cache=use_cache, verbose=False)
        fallback = fetcher.get_profile_with_papers(url, use_cache=use_cache) or {}
        if fallback.get("name"):
            return fallback
    except Exception:
        pass
    return profile


def resolve_from_google_scholar(
    google_scholar_url: Optional[str] = None,
    user_id: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[Identity]:
    raw_identifier = google_scholar_url or user_id or ""
    resolved_user_id = _extract_user_id(raw_identifier)
    if not resolved_user_id:
        return None

    url = _scholar_url(resolved_user_id)
    profile = _read_scholar_profile_from_cache(resolved_user_id) if use_cache else {}
    if not profile.get("name"):
        profile = _fetch_scholar_profile_with_fallback(url, use_cache=use_cache)

    person_name = profile.get("name")
    organization = profile.get("affiliation")
    if not person_name or not organization:
        return None

    return Identity(
        person_name=person_name,
        organization=organization,
        source="google_scholar",
        google_scholar_url=url,
        user_id=resolved_user_id,
        raw={"profile": profile},
    )


def resolve_direct(person_name: str, organization: str) -> Optional[Identity]:
    person_name = (person_name or "").strip()
    organization = (organization or "").strip()
    if not person_name or not organization:
        return None
    return Identity(
        person_name=person_name,
        organization=organization,
        source="direct",
    )
