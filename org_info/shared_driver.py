"""
共享 Chrome Driver 管理器

解决问题：
- 多个模块（PageReader, PageExecuter）各自创建 driver 会冲突
- undetected-chromedriver 不支持多实例并发

方案：
- 全局单例 driver
- 支持多标签页真正并行（网络请求并行，内容获取串行）
- 使用锁保护关键操作

并行原理：
- 同时在多个标签页发起导航请求
- 网络请求是并行的（最耗时的部分）
- 然后串行切换到每个标签页获取内容（这部分很快）

使用方式：
    # 单页面方式
    from org_info.shared_driver import get_shared_driver, release_driver
    driver = get_shared_driver()
    driver.get(url)
    html = driver.page_source
    
    # 多页面并行方式
    from org_info.shared_driver import fetch_pages_parallel
    results = await fetch_pages_parallel([url1, url2, url3])
    # results = [(url1, html1), (url2, html2), (url3, html3)]
"""

import os
import time
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager, asynccontextmanager

# undetected-chromedriver
import undetected_chromedriver as uc

# 导入日志
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============ 常量 ============

CHROME_BINARY_PATH = "/root/.cache/selenium/chrome/linux64/143.0.7499.192/chrome"


def _get_chrome_version_main() -> Optional[int]:
    """尝试从 Chrome 可执行文件读取主版本号（例如 143）"""
    if not Path(CHROME_BINARY_PATH).exists():
        return None
    try:
        result = subprocess.run(
            [CHROME_BINARY_PATH, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = (result.stdout or result.stderr).strip()
        # 形如: "Google Chrome 143.0.7499.192"
        for token in output.split():
            if token and token[0].isdigit():
                major = token.split(".", 1)[0]
                return int(major)
    except Exception:
        return None
    return None


# ============ Xvfb 管理 ============

class XvfbManager:
    """Xvfb 虚拟显示器管理器（单例）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._process: Optional[subprocess.Popen] = None
        self._display: Optional[str] = None
    
    def start(self) -> str:
        """启动 Xvfb，返回 DISPLAY 环境变量值"""
        if self._process is not None:
            return self._display
        
        # 如果已有 DISPLAY，直接使用
        if os.environ.get("DISPLAY"):
            self._display = os.environ["DISPLAY"]
            return self._display
        
        # 使用 PID 生成唯一的 display number，避免多进程冲突
        # PID % 100 + 100，范围 100-199，避开常用的 :0, :1, :99
        pid = os.getpid()
        display_num = (pid % 100) + 100
        
        # 如果还是冲突，尝试其他号码
        for offset in range(100):
            candidate = display_num + offset
            if candidate > 255:
                candidate = 100 + (candidate % 100)
            lock_file = f"/tmp/.X{candidate}-lock"
            if not os.path.exists(lock_file):
                # 先创建自己的锁文件，防止竞态
                try:
                    Path(lock_file).touch(exist_ok=False)
                    display_num = candidate
                    break
                except FileExistsError:
                    # 另一个进程抢先创建了，继续找
                    continue
        
        self._display = f":{display_num}"
        
        try:
            self._process = subprocess.Popen(
                ["Xvfb", self._display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.environ["DISPLAY"] = self._display
            time.sleep(1)
            logger.info(f"Xvfb 已启动: DISPLAY={self._display}")
            return self._display
            
        except FileNotFoundError:
            raise RuntimeError("Xvfb 未安装。请运行: apt-get install xvfb")
        except Exception as e:
            raise RuntimeError(f"无法启动 Xvfb: {e}")
    
    def stop(self):
        """停止 Xvfb"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                pass
            finally:
                self._process = None
                # 清理锁文件（如果是我们创建的）
                if self._display:
                    lock_file = f"/tmp/.X{self._display[1:]}-lock"
                    try:
                        Path(lock_file).unlink(missing_ok=True)
                    except Exception:
                        pass
                    # 清除环境变量，确保下次 start() 会重新启动 Xvfb
                    if os.environ.get("DISPLAY") == self._display:
                        del os.environ["DISPLAY"]
                    self._display = None
                logger.info("Xvfb 已停止")


# ============ 共享 Driver 管理器 ============

class SharedDriverManager:
    """
    共享 Chrome Driver 管理器（单例）
    
    特性：
    - 全局单例，所有模块共享一个 driver
    - 支持多标签页隔离（每个并行任务有独立的标签页）
    - 线程安全
    
    标签页隔离使用方式：
        tab_handle = manager.acquire_tab()  # 获取独立标签页
        try:
            manager.switch_to_tab(tab_handle)  # 切换到该标签页
            driver.get(url)  # 操作
            # ... 更多操作，都在这个标签页内
        finally:
            manager.release_tab(tab_handle)  # 释放标签页
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._driver: Optional[uc.Chrome] = None
        self._xvfb = XvfbManager()
        self._warmed_domains: set = set()
        self._driver_lock = asyncio.Lock()  # 异步锁
        self._sync_lock = threading.Lock()  # 同步锁
        self._ref_count = 0  # 引用计数
        
        # 标签页管理
        self._main_tab: Optional[str] = None  # 主标签页（不分配给任务）
        self._available_tabs: List[str] = []  # 可用的标签页池
        self._in_use_tabs: Dict[str, bool] = {}  # 正在使用的标签页
        self._tab_lock = threading.Lock()  # 标签页分配锁
    
    def _create_driver(self) -> uc.Chrome:
        """创建 Chrome driver（使用文件锁避免多进程冲突）"""
        import fcntl
        import random
        
        # 确保 Xvfb 运行
        self._xvfb.start()
        
        # 使用文件锁序列化 Chrome 初始化（undetected-chromedriver 不支持并行初始化）
        lock_file = Path("/tmp/.chrome_init.lock")
        
        # 随机延迟，减少锁争抢
        time.sleep(random.uniform(0.1, 0.5))
        
        with open(lock_file, "w") as f:
            logger.info(f"等待 Chrome 初始化锁...")
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 获取排他锁
            try:
                options = uc.ChromeOptions()
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--window-size=1920,1080")
                
                if Path(CHROME_BINARY_PATH).exists():
                    options.binary_location = CHROME_BINARY_PATH

                version_main = _get_chrome_version_main()
                if version_main:
                    logger.info(f"检测到 Chrome 主版本: {version_main}")
                else:
                    logger.info("未检测到 Chrome 主版本，将由 uc 自动选择驱动")

                driver = uc.Chrome(options=options, version_main=version_main)
                driver.set_page_load_timeout(30)
                
                logger.info("共享 Chrome driver 已创建")
                return driver
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
    
    def get_driver(self) -> uc.Chrome:
        """
        获取共享的 driver 实例（同步方法）
        
        Returns:
            Chrome driver 实例
        """
        with self._sync_lock:
            if self._driver is None:
                self._driver = self._create_driver()
            self._ref_count += 1
            return self._driver
    
    async def get_driver_async(self) -> uc.Chrome:
        """
        获取共享的 driver 实例（异步方法）
        
        Returns:
            Chrome driver 实例
        """
        async with self._driver_lock:
            if self._driver is None:
                # 在线程池中创建 driver（避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                self._driver = await loop.run_in_executor(None, self._create_driver)
                # 记录主标签页
                self._main_tab = self._driver.current_window_handle
            self._ref_count += 1
            return self._driver
    
    # ============ 标签页隔离管理 ============
    
    def acquire_tab(self) -> str:
        """
        获取一个独立的标签页（同步）
        
        如果池中有可用标签页则复用，否则创建新标签页。
        
        Returns:
            标签页 handle
        """
        driver = self.get_driver()
        
        with self._tab_lock:
            # 确保主标签页已记录
            if self._main_tab is None:
                self._main_tab = driver.current_window_handle
            
            # 尝试从池中获取
            if self._available_tabs:
                tab_handle = self._available_tabs.pop()
                self._in_use_tabs[tab_handle] = True
                logger.info(f"复用标签页: {tab_handle[:20]}...")
                return tab_handle
            
            # 创建新标签页
            driver.execute_script("window.open('about:blank');")
            new_handle = driver.window_handles[-1]
            self._in_use_tabs[new_handle] = True
            logger.info(f"创建新标签页: {new_handle[:20]}...")
            return new_handle
    
    async def acquire_tab_async(self) -> str:
        """获取一个独立的标签页（异步）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.acquire_tab)
    
    def release_tab(self, tab_handle: str, close: bool = False):
        """
        释放标签页
        
        Args:
            tab_handle: 标签页 handle
            close: 是否关闭标签页（默认 False，放回池中复用）
        """
        with self._tab_lock:
            if tab_handle in self._in_use_tabs:
                del self._in_use_tabs[tab_handle]
                
                if close:
                    # 关闭标签页
                    try:
                        driver = self._driver
                        if driver:
                            current = driver.current_window_handle
                            driver.switch_to.window(tab_handle)
                            driver.close()
                            # 切回原来的标签页或主标签页
                            if current != tab_handle and current in driver.window_handles:
                                driver.switch_to.window(current)
                            elif self._main_tab in driver.window_handles:
                                driver.switch_to.window(self._main_tab)
                            elif driver.window_handles:
                                driver.switch_to.window(driver.window_handles[0])
                        logger.info(f"关闭标签页: {tab_handle[:20]}...")
                    except Exception as e:
                        logger.warning(f"关闭标签页失败: {e}")
                else:
                    # 放回池中复用
                    self._available_tabs.append(tab_handle)
                    logger.info(f"标签页放回池中: {tab_handle[:20]}...")
        
        self.release()
    
    def switch_to_tab(self, tab_handle: str):
        """
        切换到指定标签页
        
        Args:
            tab_handle: 标签页 handle
        """
        if self._driver:
            self._driver.switch_to.window(tab_handle)
    
    def get_current_tab(self) -> Optional[str]:
        """获取当前标签页 handle"""
        if self._driver:
            return self._driver.current_window_handle
        return None
    
    def release(self):
        """释放 driver 引用（不会立即关闭）"""
        with self._sync_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
    
    def warm_up_domain(self, url: str):
        """预热域名（同步）"""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain in self._warmed_domains:
            return
        
        driver = self.get_driver()
        try:
            driver.get(domain)
            time.sleep(2)
            self._warmed_domains.add(domain)
            logger.info(f"域名预热完成: {domain}")
        except Exception as e:
            logger.warning(f"域名预热失败: {domain}, {e}")
        finally:
            self.release()
    
    async def warm_up_domain_async(self, url: str):
        """预热域名（异步）"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.warm_up_domain, url)
    
    def is_domain_warmed(self, url: str) -> bool:
        """检查域名是否已预热"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        return domain in self._warmed_domains
    
    def close_driver_only(self):
        """只关闭 driver，保持 Xvfb 运行（用于 driver 崩溃后重建）"""
        with self._sync_lock:
            if self._driver:
                try:
                    self._driver.quit()
                    logger.info("Chrome driver 已关闭（保持 Xvfb）")
                except Exception as e:
                    logger.warning(f"关闭 driver 时出错: {e}")
                finally:
                    self._driver = None
                    self._ref_count = 0
                    self._warmed_domains.clear()
    
    def close(self):
        """关闭 driver 和 Xvfb（完全清理）"""
        self.close_driver_only()
        self._xvfb.stop()
    
    @contextmanager
    def driver_context(self):
        """同步上下文管理器"""
        driver = self.get_driver()
        try:
            yield driver
        finally:
            self.release()
    
    @asynccontextmanager
    async def driver_context_async(self):
        """异步上下文管理器"""
        driver = await self.get_driver_async()
        try:
            yield driver
        finally:
            self.release()
    
    def fetch_pages_parallel_sync(
        self, 
        urls: List[str], 
        wait_time: float = 3.0,
        scroll: bool = True,
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        多标签页并行获取页面内容（同步版本）
        
        原理：
        1. 为每个 URL 创建一个标签页
        2. 同时在所有标签页发起导航（网络请求并行）
        3. 等待所有页面加载
        4. 串行切换到每个标签页获取内容
        
        Args:
            urls: URL 列表
            wait_time: 等待页面加载的时间（秒）
            scroll: 是否滚动页面触发懒加载
        
        Returns:
            [(url, html, error), ...] 列表
        """
        if not urls:
            return []
        
        driver = self.get_driver()
        original_window = driver.current_window_handle
        results: List[Tuple[str, str, Optional[str]]] = []
        tab_handles: List[Tuple[str, str]] = []  # [(handle, url), ...]
        
        try:
            # Step 1: 为每个 URL 创建标签页并发起导航
            for i, url in enumerate(urls):
                try:
                    if i == 0:
                        # 第一个 URL 使用当前标签页
                        driver.get(url)
                        tab_handles.append((driver.current_window_handle, url))
                    else:
                        # 创建新标签页
                        driver.execute_script("window.open('');")
                        new_handle = driver.window_handles[-1]
                        driver.switch_to.window(new_handle)
                        driver.get(url)
                        tab_handles.append((new_handle, url))
                except Exception as e:
                    logger.warning(f"打开标签页失败: {url}, {e}")
                    results.append((url, "", str(e)))
            
            # Step 2: 等待所有页面加载
            time.sleep(wait_time)
            
            # Step 3: 串行获取每个标签页的内容
            for handle, url in tab_handles:
                try:
                    driver.switch_to.window(handle)
                    
                    # 滚动触发懒加载
                    if scroll:
                        try:
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
                            time.sleep(0.3)
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                            time.sleep(0.3)
                        except:
                            pass
                    
                    html = driver.page_source
                    
                    # 检查是否被反爬
                    if len(html) < 1000 and ("Incapsula" in html or "Access Denied" in html):
                        results.append((url, html, "页面被反爬虫拦截"))
                    else:
                        results.append((url, html, None))
                        
                except Exception as e:
                    logger.error(f"获取页面内容失败: {url}, {e}")
                    results.append((url, "", str(e)))
            
            # Step 4: 关闭多余的标签页，只保留原始标签页
            for handle, _ in tab_handles:
                if handle != original_window:
                    try:
                        driver.switch_to.window(handle)
                        driver.close()
                    except:
                        pass
            
            # 切回原始标签页
            try:
                driver.switch_to.window(original_window)
            except:
                # 如果原始标签页也被关了，就用第一个
                if driver.window_handles:
                    driver.switch_to.window(driver.window_handles[0])
            
            return results
            
        except Exception as e:
            logger.error(f"多标签页并行获取失败: {e}")
            # 返回错误结果
            for url in urls:
                if not any(r[0] == url for r in results):
                    results.append((url, "", str(e)))
            return results
        
        finally:
            self.release()
    
    async def fetch_pages_parallel(
        self, 
        urls: List[str], 
        wait_time: float = 3.0,
        scroll: bool = True,
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        多标签页并行获取页面内容（异步版本）
        
        Args:
            urls: URL 列表
            wait_time: 等待页面加载的时间（秒）
            scroll: 是否滚动页面触发懒加载
        
        Returns:
            [(url, html, error), ...] 列表
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.fetch_pages_parallel_sync(urls, wait_time, scroll)
        )


# ============ 全局实例 ============

_manager: Optional[SharedDriverManager] = None


def get_manager() -> SharedDriverManager:
    """获取全局管理器实例"""
    global _manager
    if _manager is None:
        _manager = SharedDriverManager()
    return _manager


def get_shared_driver() -> uc.Chrome:
    """获取共享的 driver（同步）"""
    return get_manager().get_driver()


async def get_shared_driver_async() -> uc.Chrome:
    """获取共享的 driver（异步）"""
    return await get_manager().get_driver_async()


def release_driver():
    """释放 driver 引用"""
    get_manager().release()


def close_shared_driver():
    """关闭共享 driver 和 Xvfb（完全清理）"""
    get_manager().close()


def reset_shared_driver():
    """只关闭 driver，保持 Xvfb 运行（用于 driver 崩溃后重建新的）"""
    get_manager().close_driver_only()


def warm_up_domain(url: str):
    """预热域名（同步）"""
    get_manager().warm_up_domain(url)


async def warm_up_domain_async(url: str):
    """预热域名（异步）"""
    await get_manager().warm_up_domain_async(url)


def is_domain_warmed(url: str) -> bool:
    """检查域名是否已预热"""
    return get_manager().is_domain_warmed(url)


# ============ 标签页隔离 ============

def acquire_tab() -> str:
    """
    获取一个独立的标签页
    
    用于并行任务隔离：每个任务在自己的标签页内操作，互不干扰。
    
    Returns:
        标签页 handle
        
    Example:
        tab = acquire_tab()
        try:
            switch_to_tab(tab)
            driver.get(url)
            # ... 所有操作都在这个标签页内
        finally:
            release_tab(tab)
    """
    return get_manager().acquire_tab()


async def acquire_tab_async() -> str:
    """获取一个独立的标签页（异步）"""
    return await get_manager().acquire_tab_async()


def release_tab(tab_handle: str, close: bool = False):
    """
    释放标签页
    
    Args:
        tab_handle: 标签页 handle
        close: 是否关闭标签页（默认 False，放回池中复用）
    """
    get_manager().release_tab(tab_handle, close)


def switch_to_tab(tab_handle: str):
    """切换到指定标签页"""
    get_manager().switch_to_tab(tab_handle)


def get_current_tab() -> Optional[str]:
    """获取当前标签页 handle"""
    return get_manager().get_current_tab()


@contextmanager
def tab_context(close_on_exit: bool = False):
    """
    标签页上下文管理器（同步）
    
    自动获取和释放标签页，确保操作隔离。
    
    Args:
        close_on_exit: 退出时是否关闭标签页（默认 False，放回池中复用）
    
    Example:
        with tab_context() as tab:
            switch_to_tab(tab)
            driver = get_shared_driver()
            driver.get(url)
            # ... 操作
        # 自动释放标签页
    """
    tab = acquire_tab()
    try:
        switch_to_tab(tab)
        yield tab
    finally:
        release_tab(tab, close=close_on_exit)


@asynccontextmanager
async def tab_context_async(close_on_exit: bool = False):
    """
    标签页上下文管理器（异步）
    
    Args:
        close_on_exit: 退出时是否关闭标签页
    """
    tab = await acquire_tab_async()
    try:
        switch_to_tab(tab)
        yield tab
    finally:
        release_tab(tab, close=close_on_exit)


# ============ 多标签页并行获取 ============

def fetch_pages_parallel_sync(
    urls: List[str], 
    wait_time: float = 3.0,
    scroll: bool = True,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    多标签页并行获取页面内容（同步）
    
    Args:
        urls: URL 列表
        wait_time: 等待页面加载的时间（秒）
        scroll: 是否滚动页面触发懒加载
    
    Returns:
        [(url, html, error), ...] 列表
    """
    return get_manager().fetch_pages_parallel_sync(urls, wait_time, scroll)


async def fetch_pages_parallel(
    urls: List[str], 
    wait_time: float = 3.0,
    scroll: bool = True,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    多标签页并行获取页面内容（异步）
    
    真正的并行：网络请求同时发起，只有内容获取是串行的。
    
    Args:
        urls: URL 列表
        wait_time: 等待页面加载的时间（秒）
        scroll: 是否滚动页面触发懒加载
    
    Returns:
        [(url, html, error), ...] 列表
        
    Example:
        results = await fetch_pages_parallel([
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ])
        for url, html, error in results:
            if error:
                print(f"Error: {url} - {error}")
            else:
                print(f"Got {len(html)} chars from {url}")
    """
    return await get_manager().fetch_pages_parallel(urls, wait_time, scroll)


# ============ 便捷上下文管理器 ============

@contextmanager
def shared_driver_context():
    """同步上下文管理器"""
    with get_manager().driver_context() as driver:
        yield driver


@asynccontextmanager
async def shared_driver_context_async():
    """异步上下文管理器"""
    async with get_manager().driver_context_async() as driver:
        yield driver


# ============ 测试 ============

if __name__ == "__main__":
    print("=" * 60)
    print("共享 Driver 管理器测试")
    print("=" * 60)
    
    # 测试同步获取
    print("\n[Test 1] 同步获取 driver")
    with shared_driver_context() as driver:
        driver.get("https://www.google.com")
        print(f"  标题: {driver.title}")
    
    # 测试多次获取（应该返回同一个实例）
    print("\n[Test 2] 多次获取应返回同一实例")
    d1 = get_shared_driver()
    d2 = get_shared_driver()
    print(f"  同一实例: {d1 is d2}")
    release_driver()
    release_driver()
    
    # 清理
    print("\n[Test 3] 关闭")
    close_shared_driver()
    print("  已关闭")

