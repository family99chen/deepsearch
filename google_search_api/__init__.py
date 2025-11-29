"""
Google Custom Search API 模块
"""

from .google_search import (
    GoogleSearchAPI,
    GoogleSearchAPISync,
    SearchResult,
    SearchResponse,
    google_search,
    google_batch_search,
)

__all__ = [
    'GoogleSearchAPI',
    'GoogleSearchAPISync',
    'SearchResult',
    'SearchResponse',
    'google_search',
    'google_batch_search',
]

