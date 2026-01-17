"""
Brain 模块

主脑，负责：
1. 接收页面状态和总结
2. 决定下一步动作
3. 管理历史记录
4. 判断任务是否完成

不负责具体执行
"""

import sys
import asyncio
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============ 加载配置 ============
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

def _load_config() -> Dict[str, Any]:
    """加载配置文件"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

_config = _load_config()
_advanced_config = _config.get("iteragent_advanced", {})

# 默认配置
DEFAULT_MAX_ITERATIONS = _advanced_config.get("max_iterations", 10)
DEFAULT_MAX_HISTORY_IN_PROMPT = _advanced_config.get("max_history_in_prompt", 5)

from llm import query_async

# 导入共享 driver（用于标签页切换）
from org_info.shared_driver import switch_to_tab

# 支持直接运行和模块导入两种方式
try:
    from .pageexecuter import PageExecuter, PageState, Action, ActionType, InteractiveElement, ElementType
    from .summarizer import Summarizer, PageSummary
except ImportError:
    from pageexecuter import PageExecuter, PageState, Action, ActionType, InteractiveElement, ElementType
    from summarizer import Summarizer, PageSummary

try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============ 提示词 ============

BRAIN_PROMPT = """你是一个智能网页浏览代理。你的任务是找到关于 "{person_name}" 的个人信息。

## 当前状态

### 当前页面
URL: {current_url}
标题: {current_title}
页面类型: {page_type}
页面摘要: {page_summary}

### 已收集的信息
{collected_info}

### 浏览历史（最近 {history_count} 页）
{history}

### 当前页面可用操作
{available_actions}

## 你的任务
根据当前状态，决定下一步操作。你可以：
1. 点击某个链接（如果看起来能找到更多信息）
2. 使用搜索框搜索目标人物
3. 返回上一页
4. 结束任务（如果已收集到足够信息或确定找不到）

## 输出格式（严格按此格式，只输出一行）
ACTION: [动作类型] [参数]

动作类型：
- CLICK [元素ID] - 点击链接或按钮，如: ACTION: CLICK link_5 或 ACTION: CLICK btn_18
- SEARCH [元素ID] [关键词] - 在搜索框中搜索，如: ACTION: SEARCH search_1 John Smith
- BACK - 返回上一页，如: ACTION: BACK
- DONE [原因] - 结束任务，如: ACTION: DONE 已找到足够信息

注意：翻页按钮通常是 btn_xx（如 btn_18: Next），要翻页请用 CLICK btn_xx

## 重要规则（必须遵守）
1. **禁止重复操作**：如果历史记录显示你已经在当前页面做过某个操作（如已搜索过），不要再做同样的操作
2. **搜索结果页翻页**：如果当前页面是搜索结果，但没有找到目标人物，或者不确定是否是当前目标人物（检查url是否包含人物名字），应该点击 Next/下一页 按钮翻页查看更多结果
3. **查看 URL**：如果链接 URL 包含目标人物的名字，优先点击该链接
4. **避免无效操作**：如果当前页面已经有搜索结果，不要重复搜索，而是浏览结果或翻页
5. **BACK的使用**: 如果当前页面不是目标人物而是其他人物且有历史记录，说明上一步选择错误，应及时BACK
6. **相似名字**: 注意相似姓名，不要把相似姓名的人判断为目标人物了

## 决策建议
- 如果是第一次访问且有搜索框，尝试搜索目标人物姓名
- 如果已经搜索过，查看搜索结果中的链接或点击翻页按钮
- 如果看到人员列表，寻找目标人物的链接（看 URL 中的名字）
- 如果当前页面无关且没有有用的链接，考虑返回或结束
- 最多尝试 {max_iterations} 次操作

请直接输出 ACTION 行，不要有其他解释。"""


@dataclass
class HistoryEntry:
    """历史记录条目"""
    url: str                        # 访问的 URL
    title: str                      # 页面标题
    page_type: str                  # 页面类型
    page_summary: str               # 页面摘要
    person_info: str                # 提取到的信息
    action_taken: str               # 采取的动作（如 CLICK link_5）
    action_detail: str = ""         # 动作详情（如链接文字、目标URL等）
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BrainResult:
    """Brain 最终结果"""
    success: bool                   # 是否成功找到信息
    person_name: str                # 目标人物
    collected_info: str             # 收集到的所有信息
    pages_visited: int              # 访问的页面数
    history: List[HistoryEntry]     # 完整历史
    final_reason: str               # 结束原因


class Brain:
    """
    主脑 - 决策中心
    
    负责思考和决策，指挥 PageExecuter 和 Summarizer
    """
    
    def __init__(
        self,
        max_iterations: int = None,
        max_history_in_prompt: int = None,
        verbose: bool = True,
    ):
        # 使用配置文件中的值作为默认值
        if max_iterations is None:
            max_iterations = DEFAULT_MAX_ITERATIONS
        if max_history_in_prompt is None:
            max_history_in_prompt = DEFAULT_MAX_HISTORY_IN_PROMPT
        self.max_iterations = max_iterations
        self.max_history_in_prompt = max_history_in_prompt
        self.verbose = verbose
        
        self.executer = PageExecuter(verbose=verbose)
        self.summarizer = Summarizer(verbose=verbose)
        
        self._history: List[HistoryEntry] = []
        self._collected_info: List[str] = []
        self._visited_urls: set = set()
    
    async def run(
        self,
        start_url: str,
        person_name: str,
        tab_handle: str = None,
    ) -> BrainResult:
        """
        运行主循环
        
        Args:
            start_url: 起始 URL
            person_name: 目标人物姓名
            tab_handle: 指定的标签页 handle（用于隔离）
        
        Returns:
            BrainResult
        """
        # 如果指定了标签页，切换到该标签页并绑定给 PageExecuter
        if tab_handle:
            switch_to_tab(tab_handle)
            self.executer.set_tab(tab_handle)
        
        if self.verbose:
            print("=" * 60)
            print(f"[Brain] 开始搜索: {person_name}")
            print(f"[Brain] 起始 URL: {start_url}")
            if tab_handle:
                print(f"[Brain] 使用标签页: {tab_handle[:15]}...")
            print("=" * 60)
        
        # 设置任务给 PageExecuter
        self.executer.set_task(f"查找关于 {person_name} 的信息")
        
        try:
            # 初始导航
            action = Action(type=ActionType.NAVIGATE, target=start_url)
            state = await self.executer.execute(action)
            
            if not state.success:
                return BrainResult(
                    success=False,
                    person_name=person_name,
                    collected_info="",
                    pages_visited=0,
                    history=[],
                    final_reason=f"无法访问起始页面: {state.error}",
                )
            
            # 主循环
            for iteration in range(self.max_iterations):
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"[Brain] 迭代 {iteration + 1}/{self.max_iterations}")
                    print(f"[Brain] 当前 URL: {state.url}")
                    print(f"[Brain] 页面标题: {state.title}")
                    print(f"[Brain] 收到元素数量: {len(state.elements)}")
                    if state.elements:
                        print(f"[Brain] 元素列表:")
                        for elem in state.elements[:10]:
                            print(f"  - {elem.id} [{elem.type.value}]: {elem.text[:40]}")
                
                # 1. 总结当前页面
                summary = await self.summarizer.summarize(
                    html_content=state.html_content,
                    url=state.url,
                    title=state.title,
                    person_name=person_name,
                )
                
                if self.verbose:
                    print(f"\n[Brain] === 页面总结 ===")
                    print(f"  页面类型: {summary.page_type}")
                    print(f"  包含目标信息: {summary.has_target_info}")
                    if summary.person_info:
                        print(f"  人物信息: {summary.person_info[:200]}")
                    print(f"  页面摘要: {summary.page_summary}")
                    print(f"[Brain] === 总结结束 ===\n")
                
                # 2. 收集信息
                if summary.has_target_info and summary.person_info:
                    self._collected_info.append(f"[来源: {state.url}]\n{summary.person_info}")
                    if self.verbose:
                        print(f"[Brain] ✓ 收集到新信息")
                
                # 3. 决定下一步
                next_action = await self._decide_next_action(
                    state=state,
                    summary=summary,
                    person_name=person_name,
                )
                
                # 获取动作的详细信息
                action_detail = self._get_action_detail(next_action, state.elements)
                
                if self.verbose:
                    print(f"[Brain] 决策: {next_action.type.value} {next_action.target} {next_action.value}")
                    if action_detail:
                        print(f"[Brain] 详情: {action_detail}")
                
                # 4. 记录历史
                self._history.append(HistoryEntry(
                    url=state.url,
                    title=state.title,
                    page_type=summary.page_type,
                    page_summary=summary.page_summary,
                    person_info=summary.person_info if summary.has_target_info else "",
                    action_taken=f"{next_action.type.value} {next_action.target} {next_action.value}".strip(),
                    action_detail=action_detail,
                ))
                self._visited_urls.add(state.url)
                
                # 5. 检查是否结束
                if next_action.type == ActionType.DONE:
                    return BrainResult(
                        success=bool(self._collected_info),
                        person_name=person_name,
                        collected_info=self._merge_collected_info(),
                        pages_visited=len(self._history),
                        history=self._history,
                        final_reason=next_action.value or "任务完成",
                    )
                
                # 6. 执行动作
                state = await self.executer.execute(next_action)
                
                if not state.success:
                    if self.verbose:
                        print(f"[Brain] ✗ 执行失败: {state.error}")
                    # 失败后不盲目返回，而是重新获取当前页面状态
                    state = await self.executer.get_state()
                    # 如果回到了 chrome:// 页面，重新导航到起始 URL
                    if state.url.startswith("chrome://"):
                        if self.verbose:
                            print(f"[Brain] 检测到空白页，重新导航到起始 URL")
                        state = await self.executer.execute(Action(type=ActionType.NAVIGATE, target=start_url))
            
            # 达到最大迭代次数
            return BrainResult(
                success=bool(self._collected_info),
                person_name=person_name,
                collected_info=self._merge_collected_info(),
                pages_visited=len(self._history),
                history=self._history,
                final_reason="达到最大迭代次数",
            )
            
        finally:
            self.executer.close()
    
    async def _decide_next_action(
        self,
        state: PageState,
        summary: PageSummary,
        person_name: str,
    ) -> Action:
        """决定下一步动作"""
        
        # 构建可用操作列表
        available_actions = self._format_available_actions(state.elements)
        
        # 构建历史摘要
        history_text = self._format_history()
        
        # 构建已收集信息
        collected_text = self._format_collected_info()
        
        prompt = BRAIN_PROMPT.format(
            person_name=person_name,
            current_url=state.url,
            current_title=state.title,
            page_type=summary.page_type,
            page_summary=summary.page_summary,
            collected_info=collected_text if collected_text else "暂无",
            history_count=len(self._history),
            history=history_text if history_text else "无（这是第一页）",
            available_actions=available_actions,
            max_iterations=self.max_iterations,
        )
        
        if self.verbose:
            print(f"\n[Brain] === 可用操作 ===")
            print(available_actions)
            print(f"[Brain] === 可用操作结束 ===\n")
        
        response = await query_async(prompt, temperature=0.3, verbose=False)
        
        if self.verbose:
            print(f"[Brain] LLM 响应: {response[:200]}")
        
        return self._parse_action(response, state.elements)
    
    def _format_available_actions(self, elements: List[InteractiveElement]) -> str:
        """格式化可用操作"""
        lines = []
        
        # 搜索框
        search_elements = [e for e in elements if e.type == ElementType.SEARCH]
        if search_elements:
            lines.append("【搜索框】（可使用 SEARCH 命令）")
            for e in search_elements[:3]:
                placeholder = f'placeholder="{e.placeholder}"' if e.placeholder else ""
                lines.append(f"  {e.id}: {placeholder}")
        else:
            lines.append("【搜索框】无（此页面没有搜索框，不能使用 SEARCH 命令）")
        
        # 链接（限制数量，显示详细信息）
        link_elements = [e for e in elements if e.type == ElementType.LINK]
        if link_elements:
            lines.append(f"【链接】(共 {len(link_elements)} 个)")
            for e in link_elements[:20]:
                text = e.text[:40] if e.text else "(无文字)"
                # 显示链接地址，帮助 Brain 判断
                href_short = e.href[-50:] if e.href and len(e.href) > 50 else (e.href or "")
                if href_short:
                    lines.append(f"  {e.id}: {text} -> {href_short}")
                else:
                    lines.append(f"  {e.id}: {text}")
        else:
            lines.append("【链接】无")
        
        # 按钮（特别标注翻页按钮）
        btn_elements = [e for e in elements if e.type == ElementType.BUTTON]
        if btn_elements:
            lines.append(f"【按钮】（可使用 CLICK 命令）")
            for e in btn_elements[:10]:
                text = e.text[:30]
                # 标注翻页按钮
                if text.lower() in ['next', 'previous', 'last', 'first', '下一页', '上一页']:
                    lines.append(f"  {e.id}: {text} ← 翻页按钮")
                else:
                    lines.append(f"  {e.id}: {text}")
        
        if not elements:
            return "无可用操作（页面可能加载失败）"
        
        return '\n'.join(lines)
    
    def _get_action_detail(self, action: Action, elements: List[InteractiveElement]) -> str:
        """获取动作的详细信息（链接文字、目标URL等）"""
        if action.type == ActionType.DONE:
            return ""
        
        if action.type == ActionType.BACK:
            return "返回上一页"
        
        if action.type == ActionType.NAVIGATE:
            return f"导航到: {action.target}"
        
        if action.type == ActionType.SEARCH:
            # 找到搜索框元素
            for elem in elements:
                if elem.id == action.target:
                    placeholder = elem.placeholder or "搜索框"
                    return f'在 "{placeholder}" 中搜索 "{action.value}"'
            return f'搜索 "{action.value}"'
        
        if action.type == ActionType.CLICK:
            # 找到点击的元素
            for elem in elements:
                if elem.id == action.target:
                    text = elem.text[:50] if elem.text else "(无文字)"
                    if elem.type == ElementType.LINK and elem.href:
                        # 链接：显示文字和目标URL
                        href_short = elem.href[-60:] if len(elem.href) > 60 else elem.href
                        return f'点击链接 "{text}" -> {href_short}'
                    elif elem.type == ElementType.BUTTON:
                        # 按钮：显示按钮文字
                        return f'点击按钮 "{text}"'
                    else:
                        return f'点击 "{text}"'
            return f"点击 {action.target}"
        
        return ""
    
    def _format_history(self) -> str:
        """格式化历史记录"""
        if not self._history:
            return ""
        
        # 只显示最近几条
        recent = self._history[-self.max_history_in_prompt:]
        lines = []
        
        for i, entry in enumerate(recent, 1):
            info_mark = "✓有信息" if entry.person_info else ""
            lines.append(f"{i}. [{entry.page_type}] {entry.title[:40]} {info_mark}")
            lines.append(f"   URL: {entry.url[:80]}")
            lines.append(f"   执行的操作: {entry.action_taken}")
            # 显示详细信息
            if entry.action_detail:
                lines.append(f"   操作详情: {entry.action_detail}")
            # 如果是搜索操作，特别标注
            if 'search' in entry.action_taken.lower():
                lines.append(f"   ⚠️ 注意：已在此页面执行过搜索")
        
        return '\n'.join(lines)
    
    def _format_collected_info(self) -> str:
        """格式化已收集的信息"""
        if not self._collected_info:
            return ""
        
        # 合并并去重
        return '\n\n'.join(self._collected_info[-3:])  # 只显示最近 3 条
    
    def _merge_collected_info(self) -> str:
        """合并所有收集的信息"""
        if not self._collected_info:
            return ""
        return '\n\n---\n\n'.join(self._collected_info)
    
    def _parse_action(self, response: str, elements: List[InteractiveElement]) -> Action:
        """解析 LLM 返回的动作"""
        # 查找 ACTION: 行
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.upper().startswith('ACTION:'):
                action_str = line.split(':', 1)[1].strip()
                return self._parse_action_string(action_str, elements)
        
        # 没找到，尝试直接解析
        return self._parse_action_string(response.strip(), elements)
    
    def _parse_action_string(self, action_str: str, elements: List[InteractiveElement]) -> Action:
        """解析动作字符串"""
        parts = action_str.split(None, 2)  # 最多分 3 部分
        
        if not parts:
            return Action(type=ActionType.DONE, value="无法解析动作")
        
        action_type = parts[0].upper()
        
        if action_type == 'CLICK':
            target = parts[1] if len(parts) > 1 else ""
            return Action(type=ActionType.CLICK, target=target)
        
        elif action_type == 'SEARCH':
            if len(parts) >= 3:
                target = parts[1]
                query = parts[2]
            elif len(parts) == 2:
                # 可能只给了关键词，自动找搜索框
                target = ""
                query = parts[1]
            else:
                return Action(type=ActionType.DONE, value="搜索参数不完整")
            
            # 如果没指定搜索框，找第一个
            if not target or not target.startswith('search_'):
                for e in elements:
                    if e.type == ElementType.SEARCH:
                        target = e.id
                        break
            
            return Action(type=ActionType.SEARCH, target=target, value=query)
        
        elif action_type == 'BACK':
            return Action(type=ActionType.BACK)
        
        elif action_type == 'DONE':
            reason = ' '.join(parts[1:]) if len(parts) > 1 else "任务完成"
            return Action(type=ActionType.DONE, value=reason)
        
        elif action_type == 'NAVIGATE':
            url = parts[1] if len(parts) > 1 else ""
            return Action(type=ActionType.NAVIGATE, target=url)
        
        else:
            # 尝试智能匹配
            if 'search' in action_str.lower():
                return Action(type=ActionType.DONE, value="无法解析搜索动作")
            return Action(type=ActionType.DONE, value=f"未知动作: {action_type}")


# ============ 便捷函数 ============

async def search_person(
    start_url: str,
    person_name: str,
    max_iterations: int = None,  # 使用 config.yaml 中的配置
    verbose: bool = True,
    tab_handle: str = None,
) -> BrainResult:
    """
    快捷搜索函数
    
    Args:
        start_url: 起始 URL
        person_name: 目标人物姓名
        max_iterations: 最大迭代次数
        verbose: 是否打印详细日志
        tab_handle: 指定的标签页 handle（用于隔离）
    """
    brain = Brain(max_iterations=max_iterations, verbose=verbose)
    return await brain.run(start_url, person_name, tab_handle=tab_handle)


# ============ 测试 ============

if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("Brain 测试")
        print(f"配置: max_iterations={DEFAULT_MAX_ITERATIONS}, max_history_in_prompt={DEFAULT_MAX_HISTORY_IN_PROMPT}")
        print("=" * 60)
        
        result = await search_person(
            start_url="https://www.cityu.edu.hk/",
            person_name="Wang Shiqi",
            # max_iterations 使用 config.yaml 中的配置
        )
        
        print("\n" + "=" * 60)
        print("最终结果")
        print("=" * 60)
        print(f"成功: {result.success}")
        print(f"访问页面数: {result.pages_visited}")
        print(f"结束原因: {result.final_reason}")
        print(f"\n收集到的信息:\n{result.collected_info}")
    
    asyncio.run(main())

