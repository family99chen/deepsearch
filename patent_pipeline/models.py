from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Identity:
    person_name: str
    organization: str
    source: str
    orcid_id: Optional[str] = None
    google_scholar_url: Optional[str] = None
    user_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatentSearchItem:
    position: int = 0
    patent_id: str = ""
    patent_link: str = ""
    serpapi_link: str = ""
    title: str = ""
    snippet: str = ""
    priority_date: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    grant_date: Optional[str] = None
    inventor: str = ""
    assignee: str = ""
    publication_number: str = ""
    language: str = ""
    pdf: str = ""
    country_status: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatentDetail:
    patent_id: str = ""
    title: str = ""
    publication_number: str = ""
    patent_link: str = ""
    pdf: str = ""
    inventors: List[Dict[str, Any]] = field(default_factory=list)
    assignees: List[Any] = field(default_factory=list)
    abstract: str = ""
    claims: List[str] = field(default_factory=list)
    classifications: List[Dict[str, Any]] = field(default_factory=list)
    priority_date: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    grant_date: Optional[str] = None
    family_id: Optional[Any] = None
    worldwide_applications: Dict[str, Any] = field(default_factory=dict)
    legal_events: List[Dict[str, Any]] = field(default_factory=list)
    cited_by: Dict[str, Any] = field(default_factory=dict)
    patent_citations: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatentMatch:
    item: PatentSearchItem
    detail: Optional[PatentDetail]
    classification: str
    inventor_match_score: float
    matched_inventor: Optional[str] = None
    other_inventors: List[str] = field(default_factory=list)
    organization_match: bool = False


@dataclass
class PatentPipelineResult:
    success: bool
    identity: Optional[Identity] = None
    query: str = ""
    confirmed: List[PatentMatch] = field(default_factory=list)
    possible: List[PatentMatch] = field(default_factory=list)
    rejected: List[PatentMatch] = field(default_factory=list)
    total_results: int = 0
    pages_fetched: int = 0
    error: Optional[str] = None
