import re
import sys
import unicodedata
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "google_scholar_url"))

from patent_pipeline.models import Identity, PatentMatch, PatentPipelineResult
from patent_pipeline.google_patents_detail import fetch_details_concurrently
from patent_pipeline.serpapi_patents import SerpAPIPatents

try:
    from google_scholar_url.author_name_filter import is_same_author
except ImportError:
    is_same_author = None


def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value or "")
        if not unicodedata.combining(char)
    )


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"[（(].*?[）)]", " ", value)
    value = _strip_accents(value)
    value = re.sub(
        r"\b(ph\.?d\.?|m\.?d\.?|dr\.?|prof\.?|professor|inventor)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"[^A-Za-z0-9\s\-']", " ", value)
    value = re.sub(r"[-']+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _tokens(value: str) -> List[str]:
    return [token for token in normalize_name(value).split() if token]


def name_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_norm = " ".join(left_tokens)
    right_norm = " ".join(right_tokens)
    if left_norm == right_norm:
        return 1.0

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    containment = len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))
    jaccard = len(left_set & right_set) / max(1, len(left_set | right_set))

    # Avoid false positives like "Shiqi Wang" vs "Shijie Wang" from sequence ratio alone.
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        same_order_edge = left_tokens[0] == right_tokens[0] and left_tokens[-1] == right_tokens[-1]
        reversed_edge = left_tokens[0] == right_tokens[-1] and left_tokens[-1] == right_tokens[0]
        if not same_order_edge and not reversed_edge and containment < 1.0:
            sequence = min(sequence, jaccard)

    initial_ok = False
    if left_tokens and right_tokens:
        def initial_match(a: str, b: str) -> bool:
            return a == b or (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b))

        for first_left, first_right, last_left, last_right in (
            (left_tokens[0], right_tokens[0], left_tokens[-1], right_tokens[-1]),
            (left_tokens[0], right_tokens[-1], left_tokens[-1], right_tokens[0]),
        ):
            if initial_match(first_left, first_right) and last_left == last_right:
                initial_ok = True
                break

    return max(sequence, containment * 0.92, jaccard, 0.88 if initial_ok else 0.0)


def _compatible_name_edges(left: str, right: str) -> bool:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if " ".join(left_tokens) == " ".join(right_tokens):
        return True
    if len(left_tokens) == 1 or len(right_tokens) == 1:
        return left_tokens[0] == right_tokens[0]
    same_order = left_tokens[0] == right_tokens[0] and left_tokens[-1] == right_tokens[-1]
    reversed_order = left_tokens[0] == right_tokens[-1] and left_tokens[-1] == right_tokens[0]
    return same_order or reversed_order


def _organization_match(organization: str, values: Iterable[Any]) -> bool:
    org = normalize_name(organization)
    if not org:
        return False
    org_tokens = set(org.split())
    for value in values:
        candidate = normalize_name(str(value or ""))
        if not candidate:
            continue
        if org in candidate or candidate in org:
            return True
        candidate_tokens = set(candidate.split())
        if org_tokens and len(org_tokens & candidate_tokens) / max(1, min(len(org_tokens), len(candidate_tokens))) >= 0.6:
            return True
    return False


def _inventor_names(detail, item) -> List[str]:
    names: List[str] = []
    if detail:
        for inventor in detail.inventors:
            if isinstance(inventor, dict) and inventor.get("name"):
                names.append(inventor["name"])
            elif isinstance(inventor, str):
                names.append(inventor)
    if item.inventor:
        names.extend(part.strip() for part in item.inventor.split(",") if part.strip())
    return names


def _best_inventor_match(person_name: str, names: List[str]) -> Tuple[float, Optional[str]]:
    best_score = 0.0
    best_name = None
    for name in names:
        score = name_similarity(person_name, name)
        if is_same_author is not None:
            try:
                is_same, details = is_same_author(person_name, name, verbose=False)
                if is_same and _compatible_name_edges(person_name, name):
                    avg_score = float(details.get("avg_match_score") or 0.0)
                    score = max(score, avg_score, 1.0)
            except Exception:
                pass
        if score > best_score:
            best_score = score
            best_name = name
    return best_score, best_name


def _other_inventors(names: List[str], matched_name: Optional[str]) -> List[str]:
    others: List[str] = []
    for name in names:
        if not name:
            continue
        if matched_name and name == matched_name:
            continue
        if name not in others:
            others.append(name)
    return others


def _classify(score: float, min_score: float, org_match: bool) -> str:
    if score >= min_score:
        return "confirmed"
    if score >= 0.72 or (score >= 0.6 and org_match):
        return "possible"
    return "rejected"


def _result_to_dict(result: PatentPipelineResult) -> Dict[str, Any]:
    return asdict(result)


def run_patent_pipeline(
    identity: Identity,
    max_pages: Optional[int] = None,
    results_per_page: Optional[int] = None,
    detail_concurrency: Optional[int] = None,
    min_inventor_match_score: Optional[float] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    config = _load_config().get("patent_pipeline", {})
    max_pages = int(max_pages if max_pages is not None else config.get("max_pages", 1))
    results_per_page = int(results_per_page if results_per_page is not None else config.get("results_per_page", 10))
    detail_concurrency = int(
        detail_concurrency if detail_concurrency is not None else config.get("detail_concurrency", 10)
    )
    min_inventor_match_score = float(
        min_inventor_match_score
        if min_inventor_match_score is not None
        else config.get("min_inventor_match_score", 0.85)
    )
    cache_ttl = int(config.get("cache_ttl_days", 180) * 24 * 3600)

    query = identity.person_name
    client = SerpAPIPatents(use_cache=use_cache, verbose=verbose)
    confirmed: List[PatentMatch] = []
    possible: List[PatentMatch] = []
    rejected: List[PatentMatch] = []
    total_results = 0
    pages_fetched = 0
    seen_patents = set()

    for page in range(1, max(1, max_pages) + 1):
        search_result = client.search(query, page=page, num=results_per_page, use_cache=use_cache)
        if not search_result.get("success"):
            return _result_to_dict(
                PatentPipelineResult(
                    success=False,
                    identity=identity,
                    query=query,
                    confirmed=confirmed,
                    possible=possible,
                    rejected=rejected,
                    total_results=total_results,
                    pages_fetched=pages_fetched,
                    error=search_result.get("error") or "patent_search_failed",
                )
            )

        pages_fetched += 1
        items = search_result.get("items", [])
        total_results += len(items)
        if not items:
            break

        unique_items = []
        for item in items:
            patent_key = item.patent_id or item.publication_number or item.patent_link
            if patent_key in seen_patents:
                continue
            seen_patents.add(patent_key)
            unique_items.append(item)

        details_by_key = fetch_details_concurrently(
            unique_items,
            concurrency=detail_concurrency,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            verbose=verbose,
        )

        for item in unique_items:
            patent_key = item.patent_id or item.publication_number or item.patent_link
            detail = details_by_key.get(item.patent_id or item.patent_link)

            inventor_names = _inventor_names(detail, item)
            score, matched_inventor = _best_inventor_match(identity.person_name, inventor_names)
            org_values = []
            if item.assignee:
                org_values.append(item.assignee)
            if detail:
                org_values.extend(detail.assignees)
                for event in detail.legal_events:
                    for attr in event.get("attributes", []) if isinstance(event, dict) else []:
                        if isinstance(attr, dict) and attr.get("value"):
                            org_values.append(attr["value"])
            org_match = _organization_match(identity.organization, org_values)
            classification = _classify(score, min_inventor_match_score, org_match)
            match = PatentMatch(
                item=item,
                detail=detail,
                classification=classification,
                inventor_match_score=round(score, 3),
                matched_inventor=matched_inventor,
                other_inventors=_other_inventors(inventor_names, matched_inventor),
                organization_match=org_match,
            )
            if classification == "confirmed":
                confirmed.append(match)
            elif classification == "possible":
                possible.append(match)
            else:
                rejected.append(match)

    return _result_to_dict(
        PatentPipelineResult(
            success=True,
            identity=identity,
            query=query,
            confirmed=confirmed,
            possible=possible,
            rejected=rejected,
            total_results=total_results,
            pages_fetched=pages_fetched,
        )
    )
