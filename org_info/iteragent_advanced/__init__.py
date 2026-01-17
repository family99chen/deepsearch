"""
Iteragent Advanced 模块

三层架构的智能网页浏览代理：
- Brain: 决策中心，负责思考和规划
- PageExecuter: 执行器，负责与浏览器交互
- Summarizer: 总结器，负责提取和总结页面内容
"""

from .brain import Brain, BrainResult, search_person
from .pageexecuter import (
    PageExecuter, 
    PageState, 
    Action, 
    ActionType,
    InteractiveElement,
    ElementType,
)
from .summarizer import Summarizer, PageSummary, summarize_page

__all__ = [
    # Brain
    "Brain",
    "BrainResult",
    "search_person",
    # PageExecuter
    "PageExecuter",
    "PageState",
    "Action",
    "ActionType",
    "InteractiveElement",
    "ElementType",
    # Summarizer
    "Summarizer",
    "PageSummary",
    "summarize_page",
]





