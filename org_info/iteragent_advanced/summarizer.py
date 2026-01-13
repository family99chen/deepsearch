"""
Summarizer 模块

负责总结页面内容，提取关键信息
Brain 看到的是 Summarizer 总结后的内容，而不是原始 HTML
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from bs4 import BeautifulSoup

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm import query_async

try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============ 提示词 ============

SUMMARIZE_PROMPT = """你是一个网页内容分析专家。请分析以下网页内容，提取关于 "{person_name}" 的信息。

## 网页 URL：{url}
## 网页标题：{title}

## 网页内容：
{content}

## 任务：
1. 判断这个页面是否包含关于 "{person_name}" 的信息
2. 如果包含，提取关于此人的所有信息（职位、邮箱、电话、研究方向、教育背景、获奖等）
3. 简要描述这个页面是什么类型的页面（个人主页？人员列表？搜索结果？其他？）

## 输出格式：
PAGE_TYPE: [页面类型，如：个人主页 / 人员列表 / 搜索结果 / 组织介绍 / 其他]
HAS_TARGET_INFO: [YES/NO]
PERSON_INFO: 
[如果有信息，按条列出；如果没有，写"无"]
PAGE_SUMMARY:
[简要描述页面内容，50字以内]

请直接输出，不要有其他解释。"""


@dataclass
class PageSummary:
    """页面总结结果"""
    url: str                        # 页面 URL
    title: str                      # 页面标题
    page_type: str                  # 页面类型
    has_target_info: bool           # 是否包含目标人物信息
    person_info: str                # 提取到的人物信息
    page_summary: str               # 页面简要描述
    error: Optional[str] = None     # 错误信息


class Summarizer:
    """
    页面总结器
    
    负责将原始 HTML 转换为 Brain 可理解的总结
    """
    
    def __init__(
        self,
        max_content_length: int = 30000,
        verbose: bool = True,
    ):
        self.max_content_length = max_content_length
        self.verbose = verbose
    
    async def summarize(
        self,
        html_content: str,
        url: str,
        title: str,
        person_name: str,
    ) -> PageSummary:
        """
        总结页面内容
        
        Args:
            html_content: 原始 HTML
            url: 页面 URL
            title: 页面标题
            person_name: 目标人物姓名
        
        Returns:
            PageSummary
        """
        if self.verbose:
            print(f"[Summarizer] 总结页面: {url[:80]}...")
        
        # 提取纯文本
        text_content = self._extract_text(html_content, url)
        
        if not text_content:
            return PageSummary(
                url=url,
                title=title,
                page_type="未知",
                has_target_info=False,
                person_info="",
                page_summary="页面内容为空",
                error="无法提取页面内容",
            )
        
        # 调用 LLM 总结
        try:
            summary = await self._llm_summarize(
                content=text_content,
                url=url,
                title=title,
                person_name=person_name,
            )
            return summary
            
        except Exception as e:
            logger.error(f"LLM 总结失败: {e}")
            return PageSummary(
                url=url,
                title=title,
                page_type="未知",
                has_target_info=False,
                person_info="",
                page_summary="",
                error=f"LLM 总结失败: {e}",
            )
    
    def _extract_text(self, html: str, base_url: str) -> str:
        """从 HTML 提取文本内容"""
        if not html:
            return ""
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'noscript', 
                         'iframe', 'svg', 'path', 'meta', 'link']):
            tag.decompose()
        
        # 将链接转换为 [text](url) 格式
        from urllib.parse import urljoin
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            
            if href.startswith('http'):
                full_url = href
            else:
                full_url = urljoin(base_url, href)
            
            if not full_url.startswith('http'):
                continue
            
            if text:
                a.replace_with(f"[{text}]({full_url})")
        
        # 提取文本
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # 截断
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n...(内容过长已截断)"
        
        return text
    
    async def _llm_summarize(
        self,
        content: str,
        url: str,
        title: str,
        person_name: str,
    ) -> PageSummary:
        """调用 LLM 总结"""
        prompt = SUMMARIZE_PROMPT.format(
            person_name=person_name,
            url=url,
            title=title,
            content=content,
        )
        
        response = await query_async(prompt, temperature=0.3, verbose=False)
        
        if self.verbose:
            print(f"[Summarizer] LLM 响应长度: {len(response)} 字符")
        
        # 解析响应
        return self._parse_response(response, url, title)
    
    def _parse_response(self, response: str, url: str, title: str) -> PageSummary:
        """解析 LLM 响应"""
        page_type = "未知"
        has_target_info = False
        person_info_lines = []
        page_summary = ""
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_upper = line.upper()
            
            if 'PAGE_TYPE' in line_upper and ':' in line:
                page_type = line.split(':', 1)[1].strip()
                current_section = None
                
            elif 'HAS_TARGET_INFO' in line_upper and ':' in line:
                value = line.split(':', 1)[1].strip().upper()
                has_target_info = 'YES' in value
                current_section = None
                
            elif 'PERSON_INFO' in line_upper and ':' in line:
                current_section = 'person_info'
                value = line.split(':', 1)[1].strip()
                if value and value != '无' and value.lower() != 'none':
                    person_info_lines.append(value)
                    
            elif 'PAGE_SUMMARY' in line_upper and ':' in line:
                current_section = 'page_summary'
                value = line.split(':', 1)[1].strip()
                if value:
                    page_summary = value
                    
            elif current_section == 'person_info':
                if line in ('无', 'None', 'N/A', '-'):
                    continue
                # 去掉列表前缀
                clean = line
                for prefix in ('- ', '* ', '• ', '· '):
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                        break
                if clean:
                    person_info_lines.append(clean)
                    
            elif current_section == 'page_summary':
                page_summary += " " + line
        
        person_info = '\n'.join(person_info_lines)
        
        if self.verbose:
            print(f"[Summarizer] 页面类型: {page_type}")
            print(f"[Summarizer] 包含目标信息: {has_target_info}")
            if person_info:
                preview = person_info[:100].replace('\n', ' | ')
                print(f"[Summarizer] 人物信息: {preview}...")
        
        return PageSummary(
            url=url,
            title=title,
            page_type=page_type,
            has_target_info=has_target_info,
            person_info=person_info.strip(),
            page_summary=page_summary.strip()[:200],
        )


# ============ 便捷函数 ============

async def summarize_page(
    html_content: str,
    url: str,
    title: str,
    person_name: str,
    verbose: bool = True,
) -> PageSummary:
    """快捷总结函数"""
    summarizer = Summarizer(verbose=verbose)
    return await summarizer.summarize(html_content, url, title, person_name)


# ============ 测试 ============

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 60)
        print("Summarizer 测试")
        print("=" * 60)
        
        # 模拟 HTML
        test_html = """
        <html>
        <head><title>Dr. John Smith - CityU</title></head>
        <body>
            <h1>Dr. John Smith</h1>
            <p>Associate Professor, Department of Computer Science</p>
            <p>Email: johnsmith@cityu.edu.hk</p>
            <p>Research: Machine Learning, NLP</p>
            <a href="/people">Back to People</a>
        </body>
        </html>
        """
        
        result = await summarize_page(
            html_content=test_html,
            url="https://www.cityu.edu.hk/people/johnsmith",
            title="Dr. John Smith - CityU",
            person_name="John Smith",
        )
        
        print(f"\n页面类型: {result.page_type}")
        print(f"包含信息: {result.has_target_info}")
        print(f"人物信息:\n{result.person_info}")
        print(f"页面摘要: {result.page_summary}")
    
    asyncio.run(test())

