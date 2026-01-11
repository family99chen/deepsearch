"""
LLM 模块

提供统一的 LLM 调用接口，支持云端 API 和本地模型
支持同步和异步两种模式，支持高并发批量查询
"""

# 同步接口
from llm.llm_factory import (
    LLMFactory,
    query,
    chat,
    LLMType,
)

# 异步接口
from llm.llm_factory import (
    LLMFactoryAsync,
    query_async,
    chat_async,
    batch_query_async,
)

# 具体客户端（同步）
from llm.api import LLMApiClient
from llm.local import LLMLocalClient

# 具体客户端（异步）
from llm.api import LLMApiClientAsync
from llm.local import LLMLocalClientAsync

__all__ = [
    # 同步
    "LLMFactory",
    "query",
    "chat",
    "LLMType",
    "LLMApiClient",
    "LLMLocalClient",
    # 异步
    "LLMFactoryAsync",
    "query_async",
    "chat_async",
    "batch_query_async",
    "LLMApiClientAsync",
    "LLMLocalClientAsync",
]
