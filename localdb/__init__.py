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

__all__ = [
    "MongoCache",
    "get_cache",
    "cache_set",
    "cache_get",
    "cache_delete",
    "cache_exists",
]

