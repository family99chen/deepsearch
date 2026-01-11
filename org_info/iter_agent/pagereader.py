"""
Page Reader 模块（异步版本）

负责爬取网页内容，提取文本和链接
使用 undetected-chromedriver 绕过反爬虫检测
使用 LLM 分析页面内容，判断是否包含目标人物信息

注意：在无图形界面的服务器上运行时，会自动启动 Xvfb 虚拟显示器
"""

import sys
import os
import re
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

# undetected-chromedriver（专门用于绕过反爬虫）
import undetected_chromedriver as uc

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入异步 LLM
from llm import query_async

# 导入日志
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============ 常量定义 ============

# Chrome 二进制路径（由 Selenium Manager 自动下载）
CHROME_BINARY_PATH = "/root/.cache/selenium/chrome/linux64/143.0.7499.192/chrome"

# 社交媒体/联系方式域名（这些链接应该作为 person_info 而不是 relevant_links）
CONTACT_DOMAINS = [
    "facebook.com", "fb.com",
    "linkedin.com",
    "twitter.com", "x.com",
    "researchgate.net",
    "orcid.org",
    "github.com",
    "google.com/citations",  # Google Scholar
    "scholar.google.com",
    "academia.edu",
    "instagram.com",
    "youtube.com",
    "weibo.com",
    "zhihu.com",
]


# ============ Xvfb 虚拟显示器管理 ============

def _setup_virtual_display():
    """在无图形界面的服务器上自动启动 Xvfb 虚拟显示器"""
    if os.environ.get("DISPLAY"):
        return None
    
    try:
        display_num = 99
        for i in range(99, 199):
            lock_file = f"/tmp/.X{i}-lock"
            if not os.path.exists(lock_file):
                display_num = i
                break
        
        display = f":{display_num}"
        xvfb_proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = display
        time.sleep(1)
        return xvfb_proc
        
    except FileNotFoundError:
        raise RuntimeError(
            "Xvfb 未安装。请安装: apt-get install xvfb\n"
            "或使用 xvfb-run 运行: xvfb-run python your_script.py"
        )
    except Exception as e:
        raise RuntimeError(f"无法启动 Xvfb: {e}")


# 全局 Xvfb 进程
_xvfb_process = None


# ============ 提示词模板 ============

ANALYZE_PAGE_PROMPT = """你是一个网页内容分析专家。请分析以下网页内容，判断是否包含关于 "{person_name}" 的个人信息。

## 网页内容：
{page_content}

## 任务：
1. 判断这个页面是否包含关于 "{person_name}" 的信息
2. 如果包含，提取关于此人的所有且尽可能多的相关信息（如职位、研究方向、联系方式、邮箱、研究领域、奖项等）
3. 从页面中找出最多 {max_links} 个可能包含此人更多信息的**其他页面**链接

## 重要说明：
- 只提取**属于 {person_name} 本人的**社交媒体链接（如此人的 LinkedIn 个人页面、个人 Twitter 账号、个人 ResearchGate 页面、ORCID 等），放入 PERSON_INFO
- **不要提取组织/机构/大学的官方社交媒体账号**（如学校的 Facebook 主页、大学的 Twitter 账号等，这些不是个人的）
- 请判断当前网页有哪些链接可能可以有助于找到此人更多信息，把它们放到RELEVANT_LINKS中，优先放属于当前网站组织的网页链接
- RELEVANT_LINKS 的页面不要求一定是此人的网页，但需要有助于缩小搜索范围，比如能列出全体staff的网页

## 输出格式（严格按照此格式）：
CONTAINS_INFO: [YES/NO]
PERSON_INFO: [提取到的信息，包括此人本人的社交媒体链接作为文本；如果没有，写 "无"]
RELEVANT_LINKS:
- [链接1的完整URL]
- [链接2的完整URL]
...
（如果没有相关链接，写 "无"）

请直接输出结果，不要有其他解释。"""


@dataclass
class PageReadResult:
    """页面读取结果"""
    url: str                          # 页面 URL
    success: bool                     # 是否成功读取
    contains_person_info: bool        # 是否包含目标人物信息
    person_info: str                  # 提取到的人物信息（包含社交媒体链接作为文本）
    relevant_links: List[str]         # 可探索的链接
    page_title: str                   # 页面标题
    error: Optional[str] = None       # 错误信息


class PageReader:
    """
    异步页面读取器
    
    使用 undetected-chromedriver 绕过反爬虫检测
    采用"先访问主页再访问目标页"策略获取 cookies
    使用 LLM 分析是否包含目标人物信息
    """
    
    def __init__(
        self,
        max_links: int = 3,
        timeout: int = 30,
        max_content_length: int = 50000,
        verbose: bool = True,
    ):
        self.max_links = max_links
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.verbose = verbose
        
        self._driver = None
        self._warmed_domains: Set[str] = set()
        self._lock = asyncio.Lock()  # 用于保护 driver 的并发访问
    
    def _get_driver(self):
        """获取或创建 undetected-chromedriver（同步方法，内部使用）"""
        global _xvfb_process
        
        if self._driver is None:
            if _xvfb_process is None:
                _xvfb_process = _setup_virtual_display()
                if _xvfb_process and self.verbose:
                    print(f"[PageReader] 已启动 Xvfb 虚拟显示器 (DISPLAY={os.environ.get('DISPLAY')})")
            
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            
            if Path(CHROME_BINARY_PATH).exists():
                options.binary_location = CHROME_BINARY_PATH
            
            try:
                self._driver = uc.Chrome(options=options)
                self._driver.set_page_load_timeout(self.timeout)
                
                if self.verbose:
                    print("[PageReader] undetected-chromedriver 已初始化")
                logger.info("undetected-chromedriver 初始化成功")
                
            except Exception as e:
                logger.error(f"Driver 初始化失败: {e}")
                raise RuntimeError(f"无法初始化 undetected-chromedriver: {e}")
        
        return self._driver
    
    def _get_domain(self, url: str) -> str:
        """提取 URL 的域名"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _is_contact_link(self, url: str) -> bool:
        """判断是否是社交媒体/联系方式链接"""
        url_lower = url.lower()
        for domain in CONTACT_DOMAINS:
            if domain in url_lower:
                return True
        return False
    
    def _warm_up_domain(self, url: str):
        """预热域名：先访问主页获取 cookies"""
        domain = self._get_domain(url)
        
        if domain in self._warmed_domains:
            return
        
        driver = self._get_driver()
        
        if self.verbose:
            print(f"[PageReader] 预热域名: {domain}")
        
        try:
            driver.get(domain)
            time.sleep(3)
            self._warmed_domains.add(domain)
            
            if self.verbose:
                print(f"[PageReader] 域名预热完成")
            logger.info(f"域名预热完成: {domain}")
            
        except Exception as e:
            logger.warning(f"域名预热失败: {domain}, {e}")
    
    def close(self):
        """关闭浏览器和虚拟显示器"""
        global _xvfb_process
        
        if self._driver:
            try:
                self._driver.quit()
                if self.verbose:
                    print("[PageReader] 浏览器已关闭")
            except Exception as e:
                logger.warning(f"关闭浏览器时出错: {e}")
            finally:
                self._driver = None
        
        if _xvfb_process:
            try:
                _xvfb_process.terminate()
                _xvfb_process.wait(timeout=5)
                if self.verbose:
                    print("[PageReader] Xvfb 已关闭")
            except Exception:
                pass
            finally:
                _xvfb_process = None
    
    def __del__(self):
        self.close()
    
    def _fetch_page_sync(self, url: str) -> Tuple[str, str]:
        """同步获取页面内容（内部使用）"""
        self._warm_up_domain(url)
        driver = self._get_driver()
        
        try:
            driver.get(url)
            time.sleep(4)
            
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
                time.sleep(0.5)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(0.5)
            except Exception:
                pass
            
            html = driver.page_source
            
            if len(html) < 1000 and ("Incapsula" in html or "Access Denied" in html):
                return "", "页面被反爬虫系统拦截"
            
            return html, ""
            
        except Exception as e:
            error_msg = str(e)[:200]
            logger.error(f"页面获取失败: {url}, {error_msg}")
            return "", f"页面获取失败: {error_msg}"
    
    async def read_page(
        self,
        url: str,
        person_name: str,
    ) -> PageReadResult:
        """
        异步读取页面并分析
        
        Args:
            url: 页面 URL
            person_name: 目标人物姓名
        """
        if self.verbose:
            print(f"[PageReader] 读取: {url}")
        
        # 使用锁保护 driver 访问（Selenium 不支持并发）
        async with self._lock:
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            html, error = await loop.run_in_executor(
                None, self._fetch_page_sync, url
            )
        
        if error:
            logger.error(f"页面获取失败: {url}, {error}")
            return PageReadResult(
                url=url,
                success=False,
                contains_person_info=False,
                person_info="",
                relevant_links=[],
                page_title="",
                error=error,
            )
        
        if not html:
            return PageReadResult(
                url=url,
                success=False,
                contains_person_info=False,
                person_info="",
                relevant_links=[],
                page_title="",
                error="页面内容为空",
            )
        
        # 解析 HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        page_content = self._extract_text_content(soup, url)
        all_links = self._extract_links(soup, url)
        
        if self.verbose:
            print(f"[PageReader] 标题: {title[:50] if title else 'N/A'}")
            print(f"[PageReader] 内容长度: {len(page_content)} 字符")
            print(f"[PageReader] 链接数: {len(all_links)}")
        
        # 使用 LLM 分析页面
        try:
            result = await self._analyze_with_llm(
                page_content=page_content,
                all_links=all_links,
                page_url=url,
                person_name=person_name,
            )
            result.url = url
            result.success = True
            result.page_title = title
            return result
            
        except Exception as e:
            logger.error(f"LLM 分析失败: {e}")
            return PageReadResult(
                url=url,
                success=True,
                contains_person_info=False,
                person_info="",
                relevant_links=[],
                page_title=title,
                error=f"LLM 分析失败: {e}",
            )
    
    def _extract_text_content(self, soup: BeautifulSoup, base_url: str = "") -> str:
        """提取页面文本内容，保留链接 URL"""
        soup_copy = BeautifulSoup(str(soup), 'html.parser')
        
        # 移除不需要的标签
        for tag in soup_copy(['script', 'style', 'nav', 'footer', 'header', 
                              'noscript', 'iframe', 'svg', 'path']):
            tag.decompose()
        
        # 将 <a> 标签转换为 [text](url) 格式，让 LLM 能看到链接
        for a in soup_copy.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            
            # 转换为完整 URL
            if base_url and not href.startswith('http'):
                full_url = urljoin(base_url, href)
            else:
                full_url = href
            
            # 跳过无效链接
            if not full_url.startswith('http'):
                continue
            if any(x in full_url.lower() for x in ['javascript:', 'mailto:', 'tel:']):
                continue
            
            # 替换为 markdown 格式: [text](url)
            if text:
                a.replace_with(f"[{text}]({full_url})")
            else:
                a.replace_with(f"({full_url})")
        
        text = soup_copy.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n...(内容过长已截断)"
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """提取页面中的链接"""
        links = []
        seen_urls = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            
            full_url = urljoin(base_url, href)
            
            if not full_url.startswith('http'):
                continue
            if full_url in seen_urls:
                continue
            if any(x in full_url.lower() for x in ['javascript:', 'mailto:', 'tel:']):
                continue
            # 过滤锚点（但保留 # 后面有内容的）
            if '#' in full_url and full_url.split('#')[1] == '':
                continue
            # 过滤文件下载链接
            if any(full_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.png', '.jpg', '.jpeg', '.gif']):
                continue
            
            seen_urls.add(full_url)
            links.append({
                'url': full_url,
                'text': text[:100] if text else "",
            })
        
        return links
    
    async def _analyze_with_llm(
        self,
        page_content: str,
        all_links: List[Dict[str, str]],
        page_url: str,
        person_name: str,
    ) -> PageReadResult:
        """使用 LLM 分析页面内容（异步）"""
        
        prompt = ANALYZE_PAGE_PROMPT.format(
            person_name=person_name,
            page_content=page_content[:30000],
            max_links=self.max_links,
        )
        
        if self.verbose:
            print(f"[PageReader] 调用 LLM 分析...")
        
        response = await query_async(prompt, temperature=0.3, verbose=False)
        
        if self.verbose:
            print(f"\n[PageReader] === LLM 原始响应 ===")
            print(response[:1500] if len(response) > 1500 else response)
            print(f"[PageReader] === 响应结束 ===\n")
        
        # 解析响应（鲁棒版本）
        contains_info = False
        person_info_lines = []
        relevant_links = []
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测区域标记（不区分大小写，允许冒号前后有空格）
            line_upper = line.upper()
            
            if 'CONTAINS_INFO' in line_upper and ':' in line:
                value = line.split(':', 1)[1].strip()
                contains_info = 'YES' in value.upper()
                current_section = None  # 重置区域
                
            elif 'PERSON_INFO' in line_upper and ':' in line:
                current_section = 'person_info'
                value = line.split(':', 1)[1].strip()
                if value and value != '无' and value != 'None' and value.lower() != 'n/a':
                    person_info_lines.append(value)
                    
            elif 'RELEVANT_LINKS' in line_upper and ':' in line:
                current_section = 'links'
                # 检查同一行是否有链接
                value = line.split(':', 1)[1].strip()
                if value.startswith('http'):
                    if not self._is_contact_link(value):
                        relevant_links.append(value)
                
            elif current_section == 'person_info':
                # 收集所有非空内容，只跳过明确的"无"
                if line in ('无', 'None', 'N/A', 'n/a', '-'):
                    continue
                # 去掉常见列表前缀
                clean_line = line
                for prefix in ('- ', '* ', '• ', '· '):
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):]
                        break
                # 去掉数字列表前缀（如 "1. ", "2) "）
                if len(clean_line) > 2 and clean_line[0].isdigit():
                    if clean_line[1] in '.):' or (len(clean_line) > 2 and clean_line[1].isdigit() and clean_line[2] in '.):'):
                        idx = clean_line.find(' ')
                        if idx > 0:
                            clean_line = clean_line[idx+1:]
                person_info_lines.append(clean_line)
                    
            elif current_section == 'links':
                # 提取链接（支持多种格式）
                link = line
                # 去掉列表前缀
                for prefix in ('- ', '* ', '• ', '· '):
                    if link.startswith(prefix):
                        link = link[len(prefix):]
                        break
                # 去掉数字前缀
                if len(link) > 2 and link[0].isdigit() and link[1] in '.):':
                    link = link[2:].strip()
                
                link = link.strip()
                
                # 跳过"无"
                if link in ('无', 'None', 'N/A', 'n/a', '-', ''):
                    continue
                    
                # 只接受 http 开头的完整链接
                if link.startswith('http'):
                    if not self._is_contact_link(link):
                        relevant_links.append(link)
        
        # 合并 person_info
        person_info = '\n'.join(person_info_lines)
        
        # 如果 LLM 没有返回链接，使用启发式方法（只返回 1 条）
        if not relevant_links and all_links:
            fallback_links = self._select_links_heuristic(all_links, person_name)
            relevant_links = fallback_links[:1]  # fallback 只取 1 条
        
        relevant_links = relevant_links[:self.max_links]
        
        if self.verbose:
            print(f"[PageReader] 包含信息: {contains_info}")
            if person_info:
                preview = person_info[:200].replace('\n', ' | ')
                print(f"[PageReader] 解析到信息: {preview}{'...' if len(person_info) > 200 else ''}")
            else:
                print(f"[PageReader] ⚠ 未解析到 person_info")
            print(f"[PageReader] 可探索链接: {len(relevant_links)} 个")
            if relevant_links:
                print(f"[PageReader] === 解析到的链接 ===")
                for i, link in enumerate(relevant_links, 1):
                    print(f"  {i}. {link}")
                print(f"[PageReader] === 链接结束 ===")
            else:
                print(f"[PageReader] ⚠ 未解析到任何可探索链接")
        
        return PageReadResult(
            url="",
            success=True,
            contains_person_info=contains_info,
            person_info=person_info.strip(),
            relevant_links=relevant_links,
            page_title="",
        )
    
    def _select_links_heuristic(
        self,
        all_links: List[Dict[str, str]],
        person_name: str,
    ) -> List[str]:
        """启发式选择相关链接"""
        scored_links = []
        name_parts = person_name.lower().split()
        
        for link in all_links:
            url = link['url'].lower()
            text = link['text'].lower()
            score = 0
            
            for part in name_parts:
                if len(part) > 2:
                    if part in url:
                        score += 3
                    if part in text:
                        score += 2
            
            keywords = ['profile', 'people', 'faculty', 'staff', 'team', 'member', 'about', 'directory', 'person']
            for kw in keywords:
                if kw in url:
                    score += 1
            
            if score > 0:
                scored_links.append((score, link['url']))
        
        scored_links.sort(key=lambda x: x[0], reverse=True)
        return [url for _, url in scored_links[:self.max_links]]


# ============ 便捷函数 ============

async def read_page(
    url: str,
    person_name: str,
    max_links: int = 3,
    verbose: bool = True,
) -> PageReadResult:
    """异步读取页面并分析"""
    reader = PageReader(max_links=max_links, verbose=verbose)
    try:
        return await reader.read_page(url, person_name)
    finally:
        reader.close()


# 同步版本（兼容）
def read_page_sync(
    url: str,
    person_name: str,
    max_links: int = 3,
    verbose: bool = True,
) -> PageReadResult:
    """同步读取页面并分析"""
    return asyncio.run(read_page(url, person_name, max_links, verbose))


if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("Page Reader 测试 (异步版本)")
        print("=" * 60)
        
        test_url = "https://www.cityu.edu.hk/directories/people/academic"
        test_name = "AHMED Irfan"
        
        print(f"\n测试 URL: {test_url}")
        print(f"目标人物: {test_name}")
        print()
        
        result = await read_page(test_url, test_name, max_links=3)
        
        print("\n" + "=" * 60)
        print("结果")
        print("=" * 60)
        print(f"成功: {result.success}")
        print(f"包含信息: {result.contains_person_info}")
        print(f"页面标题: {result.page_title}")
        
        if result.person_info:
            print(f"\n提取到的信息:")
            print(result.person_info[:500])
        
        if result.contact_links:
            print(f"\n联系方式链接 ({len(result.contact_links)} 个):")
            for link in result.contact_links:
                print(f"  - {link}")
        
        if result.relevant_links:
            print(f"\n站内链接 ({len(result.relevant_links)} 个):")
            for link in result.relevant_links:
                print(f"  - {link}")
        
        if result.error:
            print(f"\n错误: {result.error}")
    
    asyncio.run(main())
