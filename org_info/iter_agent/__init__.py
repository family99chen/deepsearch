"""
Iter Agent 模块

异步迭代式深挖网页获取目标人物信息
"""

from org_info.iter_agent.pagereader import (
    PageReader,
    PageReadResult,
    read_page,
    read_page_sync,
)

from org_info.iter_agent.iteragent import (
    IterAgent,
    AgentResult,
    IterationResult,
    IterationHistory,
    search_person_in_org,
    search_person_in_org_sync,
)

__all__ = [
    # PageReader
    "PageReader",
    "PageReadResult",
    "read_page",           # 异步
    "read_page_sync",      # 同步
    # IterAgent
    "IterAgent",
    "AgentResult",
    "IterationResult",
    "IterationHistory",
    "search_person_in_org",       # 异步
    "search_person_in_org_sync",  # 同步
]
