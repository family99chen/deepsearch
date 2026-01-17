"""
Iter Agent 模块（异步版本）

迭代式深挖某个 link，获取目标人物在组织中的信息
使用 LLM 进行多轮分析和决策
每轮迭代内的多个页面并发读取
"""

import sys
import yaml
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入 PageReader
from org_info.iter_agent.pagereader import PageReader, PageReadResult

# 导入共享 driver（用于标签页切换）
from org_info.shared_driver import switch_to_tab, get_current_tab

# 导入异步 LLM
from llm import query_async

# 导入日志
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============ 配置加载 ============

def _load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ============ 提示词模板 ============

SYNTHESIZE_PROMPT = """你是一个信息综合分析专家。请根据多轮迭代收集到的信息，综合分析关于 "{person_name}" 的信息。

## 收集到的信息：
{collected_info}

## 任务：
1. 综合所有信息，生成一份关于 "{person_name}" 的完整报告

## 输出格式：
REPORT:
[综合报告内容，包括：姓名、职位、组织、联系方式（邮箱、电话）、社交媒体链接、研究方向等]

请直接输出结果。"""


DECIDE_CONTINUE_PROMPT = """你是一个决策专家。请根据当前的搜索状态，决定是否需要继续深入搜索。

## 目标人物：{person_name}
## 当前迭代轮次：{current_iteration} / {max_iterations}
## 待探索链接数：{pending_links}
## 已收集信息数：{info_count}

## 已收集的信息摘要：
{info_summary}

## 任务：
判断是否需要继续搜索。考虑因素：
- 已收集的信息是否足够完整写一份该任务报告（姓名、职位、组织、联系方式、研究方向等）
- 是否还有未探索的有价值链接
- 是否已达到迭代上限

## 输出：
只输出 "CONTINUE" 或 "STOP"，不要有其他内容。"""


@dataclass
class IterationResult:
    """单次迭代结果"""
    iteration: int                    # 迭代轮次
    pages_read: int                   # 读取的页面数
    info_found: List[str]             # 找到的信息
    new_links: List[str]              # 发现的新链接


@dataclass
class IterationHistory:
    """迭代历史记录"""
    iteration: int                    # 迭代轮次
    urls_visited: List[str]           # 本轮访问的 URL
    info_collected: List[str]         # 本轮收集的信息
    links_discovered: List[str]       # 本轮发现的新链接


@dataclass
class AgentResult:
    """Agent 最终结果"""
    success: bool                     # 是否成功
    person_name: str                  # 目标人物
    start_url: str                    # 起始 URL
    iterations: int                   # 迭代轮次
    pages_visited: int                # 访问的页面数
    collected_info: List[str]         # 收集到的所有信息（包含社交媒体链接作为文本）
    final_report: str                 # 最终综合报告
    visited_urls: List[str]           # 访问过的 URL
    iteration_history: List[IterationHistory]  # 迭代历史
    execution_time: float             # 执行时间（秒）


class IterAgent:
    """
    异步迭代式信息收集 Agent
    
    从初始 URL 开始，迭代深挖获取目标人物信息
    每轮迭代内的多个页面并发读取
    """
    
    def __init__(
        self,
        max_iterations: int = None,
        max_links_per_page: int = None,
        max_concurrent: int = 3,
        verbose: bool = True,
    ):
        """
        初始化 Agent
        
        Args:
            max_iterations: 最大迭代次数（k）
            max_links_per_page: 每个页面最多返回的链接数（n）
            max_concurrent: 最大并发数
            verbose: 是否打印详细日志
        """
        config = _load_config()
        agent_config = config.get("iter_agent", {})
        
        self.max_iterations = max_iterations or agent_config.get("max_iterations", 3)
        self.max_links_per_page = max_links_per_page or agent_config.get("max_links_per_page", 3)
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self._tab_handle = None  # 绑定的标签页（运行时设置）
        
        # PageReader 实例（稍后在 run() 中创建，以便传入 tab_handle）
        self.page_reader = None
        
        if self.verbose:
            print(f"[IterAgent] 初始化: max_iterations={self.max_iterations}, "
                  f"max_links_per_page={self.max_links_per_page}, "
                  f"max_concurrent={self.max_concurrent}")
        
        logger.info(f"IterAgent 初始化: k={self.max_iterations}, n={self.max_links_per_page}")
    
    def close(self):
        """关闭资源"""
        try:
            if self.page_reader:
                self.page_reader.close()
        except Exception:
            pass
    
    def __del__(self):
        self.close()
    
    async def run(
        self,
        start_url: str,
        person_name: str,
        tab_handle: str = None,
    ) -> AgentResult:
        """
        异步执行迭代式信息收集
        
        Args:
            start_url: 起始 URL
            person_name: 目标人物姓名
            tab_handle: 指定的标签页 handle（用于隔离，可选）
            
        Returns:
            AgentResult 对象
        """
        start_time = datetime.now()
        
        # 如果指定了标签页，切换到该标签页
        self._tab_handle = tab_handle
        if tab_handle:
            switch_to_tab(tab_handle)
        
        # 创建 PageReader（传入 tab_handle 以实现标签页隔离）
        self.page_reader = PageReader(
            max_links=self.max_links_per_page,
            verbose=self.verbose,
            tab_handle=tab_handle,
        )
        
        if self.verbose:
            print()
            print("=" * 60)
            print(f"[IterAgent] 开始深挖")
            print(f"[IterAgent] 目标人物: {person_name}")
            print(f"[IterAgent] 起始 URL: {start_url}")
            print("=" * 60)
        
        logger.info(f"IterAgent 开始: {person_name}, URL: {start_url}")
        
        # 状态变量
        collected_info: List[str] = []
        visited_urls: Set[str] = set()
        pending_urls: List[str] = [start_url]
        iteration_history: List[IterationHistory] = []
        
        try:
            for iteration in range(1, self.max_iterations + 1):
                if not pending_urls:
                    if self.verbose:
                        print(f"\n[IterAgent] 迭代 {iteration}: 没有待访问的链接，停止")
                    break
                
                if self.verbose:
                    print()
                    print(f"{'=' * 60}")
                    print(f"[IterAgent] 迭代 {iteration}/{self.max_iterations}")
                    print(f"[IterAgent] 待访问链接: {len(pending_urls)}")
                    print(f"{'=' * 60}")
                
                # 本轮要访问的 URL
                urls_to_visit = [u for u in pending_urls if u not in visited_urls]
                pending_urls.clear()
                
                if not urls_to_visit:
                    if self.verbose:
                        print(f"[IterAgent] 所有链接已访问过，停止")
                    break
                
                # 真正并行读取页面（多标签页 + LLM 并行分析）
                round_info: List[str] = []
                round_links: List[str] = []
                
                if self.verbose:
                    print(f"\n[IterAgent] 并行获取 {len(urls_to_visit)} 个页面...")
                    for i, url in enumerate(urls_to_visit, 1):
                        print(f"  {i}. {url[:70]}...")
                
                # 使用批量并行方法（网络请求并行 + LLM 分析并行）
                results = await self.page_reader.read_pages_batch(urls_to_visit, person_name)
                
                # 处理结果
                for url, result in zip(urls_to_visit, results):
                    visited_urls.add(url)
                    
                    if not result.success:
                        if self.verbose:
                            print(f"[IterAgent] ✗ 页面失败: {url[:50]}... ({result.error})")
                        continue
                    
                    if result.success and result.contains_person_info:
                        if result.person_info:
                            info_text = f"[来源: {url}]\n{result.person_info}"
                            round_info.append(info_text)
                            collected_info.append(info_text)
                            
                            if self.verbose:
                                print(f"[IterAgent] ✓ 找到信息: {result.person_info[:100]}...")
                    
                    # 收集可探索链接（过滤已访问的）
                    if self.verbose:
                        print(f"[IterAgent] PageReader 返回 relevant_links: {result.relevant_links}")
                    
                    if result.relevant_links:
                        new_links = [l for l in result.relevant_links if l not in visited_urls]
                        round_links.extend(new_links)
                        
                        if self.verbose:
                            if new_links:
                                print(f"[IterAgent] ✓ 过滤后保留 {len(new_links)} 个新链接")
                                for i, link in enumerate(new_links, 1):
                                    print(f"    {i}. {link[:80]}")
                            else:
                                # 所有链接都被过滤了，打印原因
                                print(f"[IterAgent] ⚠ LLM 返回 {len(result.relevant_links)} 个链接，全部被过滤:")
                                for link in result.relevant_links:
                                    status = "已访问" if link in visited_urls else "其他原因"
                                    print(f"    - {link[:60]}... ({status})")
                    else:
                        if self.verbose:
                            print(f"[IterAgent] ⚠ PageReader 未返回任何 relevant_links")
                
                # 记录迭代历史
                history = IterationHistory(
                    iteration=iteration,
                    urls_visited=urls_to_visit.copy(),
                    info_collected=round_info.copy(),
                    links_discovered=round_links.copy(),
                )
                iteration_history.append(history)
                
                # 去重并添加到待访问队列
                seen = set(pending_urls)
                for link in round_links:
                    if link not in seen and link not in visited_urls:
                        pending_urls.append(link)
                        seen.add(link)
                
                if self.verbose:
                    print(f"\n[IterAgent] 迭代 {iteration} 完成:")
                    print(f"  - 读取页面: {len(urls_to_visit)}")
                    print(f"  - 找到信息: {len(round_info)} 条")
                    print(f"  - 新链接: {len(pending_urls)} 个")
                    print(f"  - 累计信息: {len(collected_info)} 条")
                
                # 判断是否继续
                if collected_info and await self._should_stop(
                    person_name=person_name,
                    current_iteration=iteration,
                    pending_links=len(pending_urls),
                    collected_info=collected_info,
                ):
                    if self.verbose:
                        print(f"\n[IterAgent] LLM 判断信息已足够，停止迭代")
                    break
            
            # 生成最终报告
            final_report = await self._synthesize_report(person_name, collected_info)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.verbose:
                print()
                print("=" * 60)
                print("[IterAgent] 完成")
                print("=" * 60)
                print(f"迭代轮次: {len(iteration_history)}")
                print(f"访问页面: {len(visited_urls)}")
                print(f"收集信息: {len(collected_info)} 条")
                print(f"执行时间: {execution_time:.1f} 秒")
            
            logger.info(f"IterAgent 完成: {person_name}, 迭代 {len(iteration_history)} 轮, "
                       f"访问 {len(visited_urls)} 页, 收集 {len(collected_info)} 条信息")
            
            return AgentResult(
                success=len(collected_info) > 0,
                person_name=person_name,
                start_url=start_url,
                iterations=len(iteration_history),
                pages_visited=len(visited_urls),
                collected_info=collected_info,
                final_report=final_report,
                visited_urls=list(visited_urls),
                iteration_history=iteration_history,
                execution_time=execution_time,
            )
        
        finally:
            self.close()
    
    async def _should_stop(
        self,
        person_name: str,
        current_iteration: int,
        pending_links: int,
        collected_info: List[str],
    ) -> bool:
        """判断是否应该停止迭代"""
        if pending_links == 0:
            return True
        
        if not collected_info:
            return False
        
        info_summary = "\n".join(collected_info[-5:])
        if len(info_summary) > 2000:
            info_summary = info_summary[:2000] + "..."
        
        prompt = DECIDE_CONTINUE_PROMPT.format(
            person_name=person_name,
            current_iteration=current_iteration,
            max_iterations=self.max_iterations,
            pending_links=pending_links,
            info_count=len(collected_info),
            info_summary=info_summary,
        )
        
        try:
            response = await query_async(prompt, temperature=0.1, verbose=False)
            return "STOP" in response.upper()
        except Exception as e:
            logger.warning(f"LLM 决策失败: {e}")
            return False
    
    async def _synthesize_report(
        self,
        person_name: str,
        collected_info: List[str],
    ) -> str:
        """综合生成最终报告"""
        if not collected_info:
            return f"未找到关于 {person_name} 的信息。"
        
        all_info = "\n\n".join(collected_info)
        
        if len(all_info) > 6000:
            all_info = all_info[:6000] + "\n...(内容过长已截断)"
        
        prompt = SYNTHESIZE_PROMPT.format(
            person_name=person_name,
            collected_info=all_info,
        )
        
        try:
            response = await query_async(prompt, temperature=0.3, verbose=False)
            
            if "REPORT:" in response:
                report_start = response.find("REPORT:") + 7
                report_end = response.find("MISSING_INFO:")
                if report_end == -1:
                    report = response[report_start:].strip()
                else:
                    report = response[report_start:report_end].strip()
                return report
            
            return response
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return f"信息收集完成，但报告生成失败: {e}\n\n原始信息:\n" + "\n".join(collected_info[:3])


# ============ 便捷函数 ============

async def search_person_in_org(
    start_url: str,
    person_name: str,
    max_iterations: int = None,
    max_links_per_page: int = None,
    max_concurrent: int = 3,
    verbose: bool = True,
    tab_handle: str = None,
) -> AgentResult:
    """
    异步在组织网站中搜索人物信息
    
    Args:
        start_url: 起始 URL
        person_name: 目标人物姓名
        max_iterations: 最大迭代次数
        max_links_per_page: 每页最多链接数
        max_concurrent: 最大并发数
        verbose: 是否打印详细日志
        tab_handle: 指定的标签页 handle（用于隔离）
        
    Returns:
        AgentResult 对象
    """
    agent = IterAgent(
        max_iterations=max_iterations,
        max_links_per_page=max_links_per_page,
        max_concurrent=max_concurrent,
        verbose=verbose,
    )
    return await agent.run(start_url, person_name, tab_handle=tab_handle)


# 同步版本（兼容）
def search_person_in_org_sync(
    start_url: str,
    person_name: str,
    max_iterations: int = None,
    max_links_per_page: int = None,
    verbose: bool = True,
) -> AgentResult:
    """同步版本"""
    return asyncio.run(search_person_in_org(
        start_url, person_name, max_iterations, max_links_per_page, verbose=verbose
    ))


if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("Iter Agent 测试 (异步版本)")
        print("=" * 60)
        
        #test_url = "https://www.cityu.edu.hk/directories/people/academic"
        #test_name = "AHMED Irfan"
        test_url = "https://www.scut.edu.cn/new/"
        test_name = "CAI Yutao"
        
        print(f"\n起始 URL: {test_url}")
        print(f"目标人物: {test_name}")
        
        result = await search_person_in_org(
            start_url=test_url,
            person_name=test_name,
            max_iterations=10,
            max_links_per_page=2,
            max_concurrent=2,
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("最终报告")
        print("=" * 60)
        print(result.final_report)
        
        print("\n" + "=" * 60)
        print("迭代历史")
        print("=" * 60)
        for h in result.iteration_history:
            print(f"\n迭代 {h.iteration}:")
            print(f"  访问 URL: {len(h.urls_visited)} 个")
            print(f"  收集信息: {len(h.info_collected)} 条")
            print(f"  发现链接: {len(h.links_discovered)} 个")
        
        print("\n" + "=" * 60)
        print("统计")
        print("=" * 60)
        print(f"成功: {result.success}")
        print(f"迭代轮次: {result.iterations}")
        print(f"访问页面: {result.pages_visited}")
        print(f"收集信息: {len(result.collected_info)} 条")
        print(f"执行时间: {result.execution_time:.1f} 秒")
    
    asyncio.run(main())
