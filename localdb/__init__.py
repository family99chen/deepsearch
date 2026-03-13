"""
本地数据库模块

提供 MongoDB 缓存功能，用于缓存 API 调用结果。
"""

from .insert_mongo import (
    MongoCache,
    get_cache,
    cache_set,
    cache_get,
    cache_delete,
    cache_exists,
)
from .deepsearch_cache import (
    PersonPipelineCache,
    PageAnalysisCache,
    get_person_pipeline_cache,
    get_page_analysis_cache,
)

__all__ = [
    "MongoCache",
    "get_cache",
    "cache_set",
    "cache_get",
    "cache_delete",
    "cache_exists",
    "PersonPipelineCache",
    "PageAnalysisCache",
    "get_person_pipeline_cache",
    "get_page_analysis_cache",
]

