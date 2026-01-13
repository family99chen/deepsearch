"""
Page Executer 模块

执行器，负责：
1. 执行 Brain 的指令（点击、输入、导航等）
2. 使用 LLM 智能识别可交互元素
3. 返回页面状态

使用 LLM 来分析页面结构，识别搜索框、重要链接等
"""

import sys
import os
import re
import time
import asyncio
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver as uc

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm import query_async

try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============ 常量 ============

CHROME_BINARY_PATH = "/root/.cache/selenium/chrome/linux64/143.0.7499.192/chrome"


# ============ 提示词 ============

EXTRACT_ELEMENTS_PROMPT = """你是一个网页分析专家。请分析以下网页的 HTML 结构，识别可交互元素。

## 当前任务：{task}

## 网页 URL：{url}
## 网页标题：{title}

## 候选元素列表：
{candidates}

## 任务：
从上述候选元素中，识别以下类型的元素：

1. **搜索输入框**：只有标记为 [输入框] 的元素才能是搜索框！其他都不是。
2. **重要链接**：包括以下几类（最多选15个）：
   - 人员相关链接（如 People、Staff、Directory、具体人名等）
   - 分页链接（如 Next、Previous、下一页、上一页、数字页码 1/2/3/4/...、>、>>、First、Last 等）
   - 导航链接（如 About、Home、Departments 等）
3. **按钮**：包括搜索按钮、提交按钮、翻页按钮等（注意：写着 "Search" 但不是 [输入框] 的元素是按钮，不是搜索框）

## 输出格式（严格按此格式）：
SEARCH_INPUTS: [元素编号列表，只能选 [输入框] 类型，如 1,2 或 无]
IMPORTANT_LINKS: [元素编号列表，最多15个，如 2,4,6,8 或 无]
NAV_BUTTONS: [元素编号列表，如 7,9 或 无]

请直接输出，不要有其他解释。"""


# ============ 数据结构 ============

class ElementType(str, Enum):
    """元素类型"""
    LINK = "link"           # 可点击链接
    BUTTON = "button"       # 按钮
    INPUT = "input"         # 输入框
    SEARCH = "search"       # 搜索框
    SELECT = "select"       # 下拉菜单


class ActionType(str, Enum):
    """动作类型"""
    NAVIGATE = "navigate"   # 访问 URL
    CLICK = "click"         # 点击元素
    TYPE = "type"           # 输入文字（不提交）
    SEARCH = "search"       # 搜索（输入 + 回车）
    SCROLL = "scroll"       # 滚动
    WAIT = "wait"           # 等待
    BACK = "back"           # 返回上一页
    DONE = "done"           # 任务完成


@dataclass
class InteractiveElement:
    """可交互元素"""
    id: str                                     # 唯一标识: "elem_1", "elem_2"
    type: ElementType                           # 元素类型
    text: str                                   # 元素文本
    tag: str = ""                               # HTML 标签
    href: Optional[str] = None                  # 链接地址
    placeholder: Optional[str] = None           # 输入框提示
    options: List[str] = field(default_factory=list)  # 下拉选项


@dataclass
class Action:
    """Brain 发出的指令"""
    type: ActionType
    target: str = ""        # 元素 ID 或 URL
    value: str = ""         # 输入的文字


@dataclass
class PageState:
    """页面状态 - 返回给 Brain"""
    url: str                                    # 当前 URL
    title: str                                  # 页面标题
    html_content: str                           # 原始 HTML（给 Summarizer）
    elements: List[InteractiveElement]          # 可交互元素
    success: bool = True                        # 动作是否成功
    error: Optional[str] = None                 # 错误信息


# ============ Xvfb 管理 ============

_xvfb_process = None

def _setup_virtual_display():
    """在无图形界面的服务器上启动 Xvfb"""
    global _xvfb_process
    
    if os.environ.get("DISPLAY"):
        return
    
    try:
        display_num = 99
        for i in range(99, 199):
            if not os.path.exists(f"/tmp/.X{i}-lock"):
                display_num = i
                break
        
        display = f":{display_num}"
        _xvfb_process = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = display
        time.sleep(1)
        print(f"[PageExecuter] Xvfb 已启动 (DISPLAY={display})")
        
    except FileNotFoundError:
        raise RuntimeError("Xvfb 未安装，请执行: apt-get install xvfb")


# ============ PageExecuter ============

class PageExecuter:
    """
    页面执行器
    
    使用 LLM 智能识别可交互元素
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_candidates: int = 1000,
        verbose: bool = True,
    ):
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.verbose = verbose
        
        self._driver = None
        self._element_map: Dict[str, Any] = {}  # id -> WebElement
        self._warmed_domains: set = set()
        self._lock = asyncio.Lock()
        self._task: str = ""  # 当前任务（目标人物名等）
        self._url_history: List[str] = []  # URL 历史栈，用于 back
    
    def set_task(self, task: str):
        """设置当前任务，用于更精确地提取元素"""
        self._task = task
    
    def _get_driver(self):
        """获取或创建浏览器"""
        if self._driver is None:
            _setup_virtual_display()
            
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            
            if Path(CHROME_BINARY_PATH).exists():
                options.binary_location = CHROME_BINARY_PATH
            
            self._driver = uc.Chrome(options=options)
            self._driver.set_page_load_timeout(self.timeout)
            
            if self.verbose:
                print("[PageExecuter] 浏览器已启动")
        
        return self._driver
    
    async def execute(self, action: Action) -> PageState:
        """
        执行动作，返回新的页面状态（异步）
        """
        if self.verbose:
            print(f"[PageExecuter] 执行: {action.type.value} {action.target} {action.value}")
        
        try:
            if action.type == ActionType.NAVIGATE:
                return await self._navigate(action.target)
            elif action.type == ActionType.CLICK:
                return await self._click(action.target)
            elif action.type == ActionType.TYPE:
                return await self._type(action.target, action.value)
            elif action.type == ActionType.SEARCH:
                return await self._search(action.target, action.value)
            elif action.type == ActionType.SCROLL:
                return await self._scroll(action.value)
            elif action.type == ActionType.WAIT:
                return await self._wait(float(action.value) if action.value else 2)
            elif action.type == ActionType.BACK:
                return await self._back()
            else:
                return await self.get_state()
                
        except Exception as e:
            error_msg = str(e)[:200]
            logger.error(f"执行失败: {action.type.value}, {error_msg}")
            return PageState(
                url=self._get_current_url(),
                title="",
                html_content="",
                elements=[],
                success=False,
                error=error_msg,
            )
    
    async def get_state(self) -> PageState:
        """获取当前页面状态（异步，使用 LLM 识别元素）"""
        async with self._lock:
            loop = asyncio.get_event_loop()
            url, title, html = await loop.run_in_executor(
                None, self._get_page_info_sync
            )
        
        if not html:
            return PageState(
                url=url,
                title=title,
                html_content="",
                elements=[],
                success=False,
                error="页面内容为空",
            )
        
        # 使用 LLM 智能识别元素
        elements = await self._extract_elements_with_llm(html, url, title)
        
        return PageState(
            url=url,
            title=title,
            html_content=html,
            elements=elements,
            success=True,
        )
    
    def _get_page_info_sync(self) -> Tuple[str, str, str]:
        """同步获取页面基本信息"""
        driver = self._get_driver()
        try:
            url = driver.current_url
            title = driver.title or ""
            html = driver.page_source
            return url, title, html
        except Exception as e:
            logger.error(f"获取页面信息失败: {e}")
            return "", "", ""
    
    def _get_current_url(self) -> str:
        """安全获取当前 URL"""
        try:
            return self._driver.current_url if self._driver else ""
        except:
            return ""
    
    def _warm_up_domain_sync(self, url: str):
        """预热域名（同步）"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain in self._warmed_domains:
            return
        
        driver = self._get_driver()
        
        if self.verbose:
            print(f"[PageExecuter] 预热域名: {domain}")
        
        try:
            driver.get(domain)
            time.sleep(3)
            self._warmed_domains.add(domain)
        except Exception as e:
            logger.warning(f"域名预热失败: {e}")
    
    async def _navigate(self, url: str) -> PageState:
        """访问 URL"""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._navigate_sync, url)
        
        return await self.get_state()
    
    def _navigate_sync(self, url: str):
        """同步导航"""
        self._warm_up_domain_sync(url)
        driver = self._get_driver()
        driver.get(url)
        time.sleep(4)
        
        # 滚动触发懒加载
        try:
            driver.execute_script("window.scrollTo(0, 300);")
            time.sleep(0.5)
            driver.execute_script("window.scrollTo(0, 0);")
        except:
            pass
    
    async def _click(self, element_id: str) -> PageState:
        """点击元素"""
        if element_id not in self._element_map:
            return PageState(
                url=self._get_current_url(),
                title="",
                html_content="",
                elements=[],
                success=False,
                error=f"元素不存在: {element_id}",
            )
        
        async with self._lock:
            loop = asyncio.get_event_loop()
            success, error = await loop.run_in_executor(
                None, self._click_sync, element_id
            )
        
        if not success:
            return PageState(
                url=self._get_current_url(),
                title="",
                html_content="",
                elements=[],
                success=False,
                error=error,
            )
        
        return await self.get_state()
    
    def _click_sync(self, element_id: str) -> Tuple[bool, str]:
        """同步点击 - 对于链接直接导航到 URL"""
        element = self._element_map[element_id]
        driver = self._get_driver()
        
        # 记录当前 URL 到历史
        current_url = driver.current_url
        if not self._url_history or self._url_history[-1] != current_url:
            self._url_history.append(current_url)
        
        # 尝试获取链接的 href
        href = None
        try:
            href = element.get_attribute("href")
        except:
            pass
        
        # 如果是链接且有 href，直接导航
        if href and href.startswith("http"):
            try:
                if self.verbose:
                    print(f"[PageExecuter] 直接导航到: {href[:60]}")
                driver.get(href)
                time.sleep(3)
                return True, ""
            except Exception as e:
                return False, f"导航失败: {e}"
        
        # 否则尝试正常点击（按钮等）
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.3)
            element.click()
            time.sleep(2)
            return True, ""
        except Exception as e:
            # 尝试 JS 点击
            try:
                driver.execute_script("arguments[0].click();", element)
                time.sleep(2)
                return True, ""
            except Exception as e2:
                return False, f"点击失败: {e2}"
    
    async def _type(self, element_id: str, text: str) -> PageState:
        """输入文字"""
        if element_id not in self._element_map:
            return PageState(
                url=self._get_current_url(),
                title="",
                html_content="",
                elements=[],
                success=False,
                error=f"元素不存在: {element_id}",
            )
        
        async with self._lock:
            loop = asyncio.get_event_loop()
            success, error = await loop.run_in_executor(
                None, self._type_sync, element_id, text
            )
        
        if not success:
            return PageState(
                url=self._get_current_url(),
                title="",
                html_content="",
                elements=[],
                success=False,
                error=error,
            )
        
        return await self.get_state()
    
    def _type_sync(self, element_id: str, text: str) -> Tuple[bool, str]:
        """同步输入"""
        element = self._element_map[element_id]
        try:
            element.clear()
            element.send_keys(text)
            time.sleep(0.5)
            return True, ""
        except Exception as e:
            return False, f"输入失败: {e}"
    
    async def _search(self, element_id: str, query: str) -> PageState:
        """搜索 = 输入 + 回车"""
        # 如果没指定元素，找搜索框
        if element_id not in self._element_map:
            search_id = self._find_search_element()
            if search_id:
                element_id = search_id
            else:
                return PageState(
                    url=self._get_current_url(),
                    title="",
                    html_content="",
                    elements=[],
                    success=False,
                    error="找不到搜索框",
                )
        
        async with self._lock:
            loop = asyncio.get_event_loop()
            success, error = await loop.run_in_executor(
                None, self._search_sync, element_id, query
            )
        
        if not success:
            return PageState(
                url=self._get_current_url(),
                title="",
                html_content="",
                elements=[],
                success=False,
                error=error,
            )
        
        return await self.get_state()
    
    def _search_sync(self, element_id: str, query: str) -> Tuple[bool, str]:
        """同步搜索 - 处理新窗口情况，保持单窗口模式"""
        element = self._element_map[element_id]
        driver = self._get_driver()
        
        # 记录当前窗口和 URL
        original_handle = driver.current_window_handle
        original_url = driver.current_url
        old_handles = set(driver.window_handles)
        
        # 记录到历史
        if not self._url_history or self._url_history[-1] != original_url:
            self._url_history.append(original_url)
        
        try:
            element.clear()
            element.send_keys(query)
            time.sleep(0.3)
            element.send_keys(Keys.RETURN)
            time.sleep(3)
            
            # 检查是否打开了新窗口
            new_handles = set(driver.window_handles)
            new_windows = new_handles - old_handles
            
            if new_windows:
                # 有新窗口，获取新窗口的 URL，然后关闭新窗口，在原窗口导航
                new_handle = list(new_windows)[0]
                driver.switch_to.window(new_handle)
                time.sleep(2)
                new_url = driver.current_url
                
                if self.verbose:
                    print(f"[PageExecuter] 搜索打开了新窗口，获取 URL: {new_url[:60]}")
                
                # 关闭新窗口，回到原窗口
                driver.close()
                driver.switch_to.window(original_handle)
                
                # 在原窗口导航到搜索结果
                driver.get(new_url)
                time.sleep(3)
            else:
                # 没有新窗口，等待页面加载
                time.sleep(2)
            
            return True, ""
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str:
                if self.verbose:
                    print(f"[PageExecuter] 搜索超时，但可能已成功，继续执行")
                time.sleep(2)
                return True, ""
            return False, f"搜索失败: {e}"
    
    def _find_search_element(self) -> Optional[str]:
        """查找搜索框元素"""
        for eid in self._element_map:
            if "search" in eid:
                return eid
        return None
    
    async def _scroll(self, direction: str = "down") -> PageState:
        """滚动页面"""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._scroll_sync, direction)
        return await self.get_state()
    
    def _scroll_sync(self, direction: str):
        """同步滚动"""
        driver = self._get_driver()
        if direction == "up":
            driver.execute_script("window.scrollBy(0, -500);")
        else:
            driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(0.5)
    
    async def _wait(self, seconds: float) -> PageState:
        """等待"""
        await asyncio.sleep(seconds)
        return await self.get_state()
    
    async def _back(self) -> PageState:
        """返回上一页"""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._back_sync)
        return await self.get_state()
    
    def _back_sync(self):
        """同步返回（使用 URL 历史栈）"""
        driver = self._get_driver()
        
        # 从历史栈返回
        if self._url_history:
            prev_url = self._url_history.pop()
            if self.verbose:
                print(f"[PageExecuter] 返回到: {prev_url[:60]}")
            driver.get(prev_url)
            time.sleep(2)
        else:
            # 没有历史，尝试浏览器后退
            driver.back()
            time.sleep(2)
    
    # ============ LLM 元素识别 ============
    
    async def _extract_elements_with_llm(
        self,
        html: str,
        url: str,
        title: str,
    ) -> List[InteractiveElement]:
        """
        使用 LLM 智能识别可交互元素
        
        流程：
        1. 提取所有候选元素（输入框、链接、按钮）
        2. 让 LLM 判断哪些是搜索框、重要链接等
        3. 返回分类后的元素列表
        """
        # 1. 提取候选元素
        candidates, candidate_elements = self._extract_candidates_sync()
        
        if not candidates:
            if self.verbose:
                print("[PageExecuter] 未找到候选元素")
            return []
        
        if self.verbose:
            print(f"[PageExecuter] 候选元素: {len(candidates)} 个")
        
        # 2. 调用 LLM 分析
        prompt = EXTRACT_ELEMENTS_PROMPT.format(
            task=self._task or "浏览网页，识别可交互元素",
            url=url,
            title=title,
            candidates=self._format_candidates(candidates),
        )
        
        try:
            response = await query_async(prompt, temperature=0.2, verbose=False)
            
            if self.verbose:
                print(f"[PageExecuter] LLM 响应: {response[:200]}")
            
            # 3. 解析 LLM 响应
            elements = self._parse_llm_response(response, candidates, candidate_elements)
            
            if self.verbose:
                search_count = sum(1 for e in elements if e.type == ElementType.SEARCH)
                link_count = sum(1 for e in elements if e.type == ElementType.LINK)
                btn_count = sum(1 for e in elements if e.type == ElementType.BUTTON)
                print(f"[PageExecuter] 识别元素: search={search_count}, link={link_count}, btn={btn_count}")
            
            return elements
            
        except Exception as e:
            logger.error(f"LLM 元素识别失败: {e}")
            # 降级：使用规则提取
            return self._fallback_extract(candidates, candidate_elements)
    
    def _extract_candidates_sync(self) -> Tuple[List[Dict], Dict[int, Any]]:
        """
        同步提取候选元素（宽松模式，让 LLM 来判断）
        
        返回:
            candidates: 候选元素信息列表
            candidate_elements: 编号 -> WebElement 映射
        """
        driver = self._get_driver()
        base_url = driver.current_url
        candidates = []
        candidate_elements = {}
        idx = 0
        
        # 1. 提取输入框（只过滤 hidden 类型）
        try:
            inputs = driver.find_elements(By.TAG_NAME, "input")
            for inp in inputs[:50]:
                try:
                    input_type = inp.get_attribute("type") or "text"
                    # 只过滤真正不可交互的类型
                    if input_type == "hidden":
                        continue
                    
                    idx += 1
                    placeholder = inp.get_attribute("placeholder") or ""
                    name = inp.get_attribute("name") or ""
                    input_id = inp.get_attribute("id") or ""
                    aria_label = inp.get_attribute("aria-label") or ""
                    
                    candidates.append({
                        "idx": idx,
                        "tag": "input",
                        "type": input_type,
                        "text": name or input_id or aria_label or placeholder,
                        "placeholder": placeholder,
                    })
                    candidate_elements[idx] = inp
                except:
                    continue
        except Exception as e:
            logger.warning(f"提取输入框失败: {e}")
        
        # 2. 提取链接（几乎不过滤，让 LLM 判断）
        try:
            links = driver.find_elements(By.TAG_NAME, "a")
            for link in links[:self.max_candidates]:
                try:
                    href = link.get_attribute("href") or ""
                    text = link.text.strip()
                    
                    # 只过滤无效链接
                    if not href:
                        continue
                    if href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
                        continue
                    
                    # 相对路径转绝对路径
                    if not href.startswith("http"):
                        href = urljoin(base_url, href)
                    
                    # 没有文字时尝试获取其他属性
                    if not text:
                        text = link.get_attribute("aria-label") or ""
                    if not text:
                        text = link.get_attribute("title") or ""
                    if not text:
                        # 尝试获取子元素的 alt 文本（图片链接）
                        try:
                            img = link.find_element(By.TAG_NAME, "img")
                            text = img.get_attribute("alt") or ""
                        except:
                            pass
                    
                    # 仍然没有文字就用 href 的最后部分
                    if not text:
                        text = href.split("/")[-1][:30] or "[link]"
                    
                    idx += 1
                    candidates.append({
                        "idx": idx,
                        "tag": "a",
                        "text": text[:80],
                        "href": href,
                    })
                    candidate_elements[idx] = link
                except:
                    continue
        except Exception as e:
            logger.warning(f"提取链接失败: {e}")
        
        # 3. 提取按钮（宽松模式）
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons[:50]:
                try:
                    text = btn.text.strip()
                    if not text:
                        text = btn.get_attribute("aria-label") or ""
                    if not text:
                        text = btn.get_attribute("title") or ""
                    if not text:
                        text = btn.get_attribute("value") or ""
                    if not text:
                        text = "[button]"
                    
                    idx += 1
                    candidates.append({
                        "idx": idx,
                        "tag": "button",
                        "text": text[:50],
                    })
                    candidate_elements[idx] = btn
                except:
                    continue
        except Exception as e:
            logger.warning(f"提取按钮失败: {e}")
        
        # 4. 提取可点击的 div/span（有 onclick 或 role=button）
        try:
            clickables = driver.find_elements(By.CSS_SELECTOR, "[onclick], [role='button'], [tabindex='0']")
            for elem in clickables[:30]:
                try:
                    tag = elem.tag_name.lower()
                    if tag in ["a", "button", "input"]:  # 已处理
                        continue
                    
                    text = elem.text.strip()[:50]
                    if not text:
                        text = elem.get_attribute("aria-label") or "[clickable]"
                    
                    idx += 1
                    candidates.append({
                        "idx": idx,
                        "tag": tag,
                        "text": text,
                    })
                    candidate_elements[idx] = elem
                except:
                    continue
        except Exception as e:
            logger.warning(f"提取可点击元素失败: {e}")
        
        if self.verbose:
            print(f"[PageExecuter] 提取到 {len(candidates)} 个候选元素")
            # 显示各类型数量
            inputs = [c for c in candidates if c["tag"] == "input"]
            links = [c for c in candidates if c["tag"] == "a"]
            buttons = [c for c in candidates if c["tag"] == "button"]
            others = [c for c in candidates if c["tag"] not in ["input", "a", "button"]]
            print(f"  - 输入框: {len(inputs)}, 链接: {len(links)}, 按钮: {len(buttons)}, 其他: {len(others)}")
            if inputs:
                print(f"  - 输入框详情:")
                for inp in inputs[:5]:
                    print(f"      {inp['idx']}. type={inp.get('type', '')} placeholder={inp.get('placeholder', '')[:30]}")
        
        return candidates, candidate_elements
    
    def _format_candidates(self, candidates: List[Dict]) -> str:
        """格式化候选元素列表"""
        lines = []
        for c in candidates[:1000]:  # 限制数量
            if c["tag"] == "input":
                input_type = c.get("type", "text")
                # 只有可输入类型才标记为 [输入框]，其他标记为具体类型
                if input_type in ("text", "search", "email", "url", "tel", "password", ""):
                    line = f'{c["idx"]}. [输入框 type={input_type}] placeholder="{c.get("placeholder", "")}" name="{c.get("text", "")}"'
                else:
                    # radio、checkbox、submit 等标记为具体类型
                    line = f'{c["idx"]}. [input type={input_type}] name="{c.get("text", "")}"'
            elif c["tag"] == "a":
                line = f'{c["idx"]}. [链接] "{c["text"]}" -> {c.get("href", "")[:60]}'
            elif c["tag"] == "button":
                line = f'{c["idx"]}. [按钮] "{c["text"]}"'
            else:
                line = f'{c["idx"]}. [{c["tag"]}] {c.get("text", "")}'
            lines.append(line)
        return '\n'.join(lines)
    
    def _parse_llm_response(
        self,
        response: str,
        candidates: List[Dict],
        candidate_elements: Dict[int, Any],
    ) -> List[InteractiveElement]:
        """解析 LLM 响应，构建元素列表"""
        self._element_map.clear()
        elements = []
        
        # 解析各类元素编号
        search_ids = set()
        link_ids = set()
        btn_ids = set()
        
        for line in response.strip().split('\n'):
            line = line.strip().upper()
            
            if 'SEARCH_INPUT' in line and ':' in line:
                value = line.split(':', 1)[1].strip()
                search_ids = self._parse_id_list(value)
                
            elif 'IMPORTANT_LINK' in line and ':' in line:
                value = line.split(':', 1)[1].strip()
                link_ids = self._parse_id_list(value)
                
            elif 'NAV_BUTTON' in line and ':' in line:
                value = line.split(':', 1)[1].strip()
                btn_ids = self._parse_id_list(value)
        
        # 构建元素列表
        elem_counter = 0
        
        # 添加搜索框
        for idx in search_ids:
            if idx in candidate_elements:
                elem_counter += 1
                eid = f"search_{elem_counter}"
                self._element_map[eid] = candidate_elements[idx]
                
                cand = next((c for c in candidates if c["idx"] == idx), {})
                elements.append(InteractiveElement(
                    id=eid,
                    type=ElementType.SEARCH,
                    tag="input",
                    text=cand.get("text", ""),
                    placeholder=cand.get("placeholder", ""),
                ))
        
        # 添加重要链接
        for idx in link_ids:
            if idx in candidate_elements:
                elem_counter += 1
                eid = f"link_{elem_counter}"
                self._element_map[eid] = candidate_elements[idx]
                
                cand = next((c for c in candidates if c["idx"] == idx), {})
                elements.append(InteractiveElement(
                    id=eid,
                    type=ElementType.LINK,
                    tag="a",
                    text=cand.get("text", ""),
                    href=cand.get("href", ""),
                ))
        
        # 添加按钮
        for idx in btn_ids:
            if idx in candidate_elements:
                elem_counter += 1
                eid = f"btn_{elem_counter}"
                self._element_map[eid] = candidate_elements[idx]
                
                cand = next((c for c in candidates if c["idx"] == idx), {})
                elements.append(InteractiveElement(
                    id=eid,
                    type=ElementType.BUTTON,
                    tag="button",
                    text=cand.get("text", ""),
                ))
        
        return elements
    
    def _parse_id_list(self, value: str) -> set:
        """解析逗号分隔的 ID 列表，支持 [1,2,3] 或 1,2,3 格式"""
        if '无' in value or 'NONE' in value.upper() or not value:
            return set()
        
        # 去掉方括号
        value = value.replace('[', '').replace(']', '')
        
        ids = set()
        for part in re.split(r'[,\s]+', value):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
        return ids
    
    def _fallback_extract(
        self,
        candidates: List[Dict],
        candidate_elements: Dict[int, Any],
    ) -> List[InteractiveElement]:
        """降级：规则提取"""
        self._element_map.clear()
        elements = []
        elem_counter = 0
        
        for cand in candidates:
            idx = cand["idx"]
            if idx not in candidate_elements:
                continue
            
            elem_counter += 1
            
            if cand["tag"] == "input":
                # 简单规则判断搜索框
                text_lower = (cand.get("placeholder", "") + cand.get("text", "")).lower()
                is_search = any(kw in text_lower for kw in ["search", "query", "搜索", "查找"])
                
                if is_search:
                    eid = f"search_{elem_counter}"
                    etype = ElementType.SEARCH
                else:
                    eid = f"input_{elem_counter}"
                    etype = ElementType.INPUT
                
                self._element_map[eid] = candidate_elements[idx]
                elements.append(InteractiveElement(
                    id=eid,
                    type=etype,
                    tag="input",
                    text=cand.get("text", ""),
                    placeholder=cand.get("placeholder", ""),
                ))
                
            elif cand["tag"] == "a":
                eid = f"link_{elem_counter}"
                self._element_map[eid] = candidate_elements[idx]
                elements.append(InteractiveElement(
                    id=eid,
                    type=ElementType.LINK,
                    tag="a",
                    text=cand.get("text", ""),
                    href=cand.get("href", ""),
                ))
                
            elif cand["tag"] == "button":
                eid = f"btn_{elem_counter}"
                self._element_map[eid] = candidate_elements[idx]
                elements.append(InteractiveElement(
                    id=eid,
                    type=ElementType.BUTTON,
                    tag="button",
                    text=cand.get("text", ""),
                ))
        
        return elements[:50]  # 限制数量
    
    def close(self):
        """关闭浏览器"""
        global _xvfb_process
        
        if self._driver:
            try:
                self._driver.quit()
                if self.verbose:
                    print("[PageExecuter] 浏览器已关闭")
            except:
                pass
            self._driver = None
        
        if _xvfb_process:
            try:
                _xvfb_process.terminate()
                _xvfb_process.wait(timeout=5)
            except:
                pass
            _xvfb_process = None
    
    def __del__(self):
        self.close()


# ============ 测试 ============

if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("PageExecuter 测试 (LLM 版本)")
        print("=" * 60)
        
        executer = PageExecuter()
        
        try:
            action = Action(type=ActionType.NAVIGATE, target="https://www.cityu.edu.hk")
            state = await executer.execute(action)
            
            print(f"\nURL: {state.url}")
            print(f"Title: {state.title}")
            print(f"Elements: {len(state.elements)} 个")
            
            print("\n识别到的元素:")
            for elem in state.elements:
                print(f"  - [{elem.id}] {elem.type.value}: {elem.text[:50]}")
            
        finally:
            executer.close()
    
    asyncio.run(test())
