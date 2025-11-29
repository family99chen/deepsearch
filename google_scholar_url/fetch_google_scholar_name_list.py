"""
Google Scholar 作者搜索模块
用于接收作者名字，使用 Selenium 模拟用户搜索，返回所有匹配作者的个人主页 URL
"""

import time
import random
from typing import List, Dict, Optional
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class GoogleScholarAuthorScraper:
    """Google Scholar 作者搜索爬虫类"""
    
    BASE_URL = "https://scholar.google.com/citations"
    
    def __init__(self, headless: bool = True, proxy: Optional[str] = None):
        """
        初始化爬虫
        
        Args:
            headless: 是否使用无头模式
            proxy: 代理地址，格式如 "http://127.0.0.1:7890"
        """
        self.driver = None
        self.headless = headless
        self.proxy = proxy
        
    def _init_driver(self) -> webdriver.Chrome:
        """初始化 Chrome WebDriver"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        # 基本配置
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # 模拟真实浏览器
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # 设置 User-Agent
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # 代理设置
        if self.proxy:
            chrome_options.add_argument(f"--proxy-server={self.proxy}")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # 修改 webdriver 属性，防止被检测
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        
        return driver
    
    def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """随机延迟，模拟人类行为"""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def search_author(self, author_name: str, max_results: int = 20) -> List[Dict]:
        """
        搜索作者并返回作者信息列表
        
        Args:
            author_name: 作者名字
            max_results: 最大返回结果数
            
        Returns:
            包含作者信息的字典列表，每个字典包含:
            - name: 作者姓名
            - url: 作者个人主页 URL
            - affiliation: 所属机构
            - email: 邮箱域名（如果有）
            - cited_by: 被引用次数
            - interests: 研究兴趣列表
        """
        if self.driver is None:
            self.driver = self._init_driver()
        
        # 构建搜索 URL
        encoded_name = quote_plus(author_name)
        search_url = f"{self.BASE_URL}?view_op=search_authors&mauthors={encoded_name}&hl=zh-CN"
        
        print(f"[INFO] 正在搜索作者: {author_name}")
        print(f"[INFO] 搜索 URL: {search_url}")
        
        try:
            self.driver.get(search_url)
            self._random_delay(2, 4)
            
            # 等待搜索结果加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.gsc_1usr, div.gs_med"))
            )
            
        except TimeoutException:
            print("[WARNING] 页面加载超时，可能没有搜索结果或被限制访问")
            return []
        
        authors = []
        page_num = 1
        
        while len(authors) < max_results:
            print(f"[INFO] 正在解析第 {page_num} 页...")
            
            # 解析当前页面的作者信息
            page_authors = self._parse_author_results()
            
            if not page_authors:
                print("[INFO] 没有更多结果")
                break
            
            authors.extend(page_authors)
            print(f"[INFO] 已获取 {len(authors)} 个作者信息")
            
            # 检查是否有下一页
            if len(authors) >= max_results:
                break
                
            if not self._go_to_next_page():
                break
            
            page_num += 1
            self._random_delay(2, 4)
        
        return authors[:max_results]
    
    def _parse_author_results(self) -> List[Dict]:
        """解析当前页面的作者搜索结果"""
        authors = []
        
        # Google Scholar 作者搜索结果的选择器
        author_cards = self.driver.find_elements(By.CSS_SELECTOR, "div.gsc_1usr")
        
        for card in author_cards:
            try:
                author_info = {}
                
                # 获取作者姓名和链接
                name_element = card.find_element(By.CSS_SELECTOR, "h3.gs_ai_name a")
                author_info["name"] = name_element.text.strip()
                author_info["url"] = name_element.get_attribute("href")
                
                # 获取所属机构
                try:
                    affiliation_element = card.find_element(By.CSS_SELECTOR, "div.gs_ai_aff")
                    author_info["affiliation"] = affiliation_element.text.strip()
                except NoSuchElementException:
                    author_info["affiliation"] = ""
                
                # 获取邮箱域名
                try:
                    email_element = card.find_element(By.CSS_SELECTOR, "div.gs_ai_eml")
                    author_info["email"] = email_element.text.strip()
                except NoSuchElementException:
                    author_info["email"] = ""
                
                # 获取被引用次数
                try:
                    cited_element = card.find_element(By.CSS_SELECTOR, "div.gs_ai_cby")
                    cited_text = cited_element.text.strip()
                    # 提取数字，格式如 "被引用次数：12345"
                    author_info["cited_by"] = cited_text
                except NoSuchElementException:
                    author_info["cited_by"] = ""
                
                # 获取研究兴趣
                try:
                    interests_elements = card.find_elements(By.CSS_SELECTOR, "div.gs_ai_int a")
                    author_info["interests"] = [elem.text.strip() for elem in interests_elements]
                except NoSuchElementException:
                    author_info["interests"] = []
                
                authors.append(author_info)
                print(f"  - 找到作者: {author_info['name']} | {author_info['affiliation']}")
                
            except Exception as e:
                print(f"[WARNING] 解析作者信息失败: {e}")
                continue
        
        return authors
    
    def _go_to_next_page(self) -> bool:
        """
        尝试跳转到下一页
        
        Returns:
            是否成功跳转到下一页
        """
        try:
            # 查找下一页按钮
            next_button = self.driver.find_element(
                By.CSS_SELECTOR, 
                "button.gs_btnPR:not([disabled])"
            )
            next_button.click()
            self._random_delay(1, 2)
            return True
        except NoSuchElementException:
            return False
        except Exception as e:
            print(f"[WARNING] 翻页失败: {e}")
            return False
    
    def get_author_urls(self, author_name: str, max_results: int = 20) -> List[str]:
        """
        搜索作者并只返回 URL 列表
        
        Args:
            author_name: 作者名字
            max_results: 最大返回结果数
            
        Returns:
            作者个人主页 URL 列表
        """
        authors = self.search_author(author_name, max_results)
        return [author["url"] for author in authors if author.get("url")]
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def fetch_google_scholar_author_urls(
    author_name: str,
    max_results: int = 20,
    headless: bool = True,
    proxy: Optional[str] = None
) -> List[str]:
    """
    便捷函数：搜索作者并返回 URL 列表
    
    Args:
        author_name: 作者名字
        max_results: 最大返回结果数
        headless: 是否使用无头模式
        proxy: 代理地址
        
    Returns:
        作者个人主页 URL 列表
    """
    with GoogleScholarAuthorScraper(headless=headless, proxy=proxy) as scraper:
        return scraper.get_author_urls(author_name, max_results)


def fetch_google_scholar_authors(
    author_name: str,
    max_results: int = 20,
    headless: bool = True,
    proxy: Optional[str] = None
) -> List[Dict]:
    """
    便捷函数：搜索作者并返回完整信息列表
    
    Args:
        author_name: 作者名字
        max_results: 最大返回结果数
        headless: 是否使用无头模式
        proxy: 代理地址
        
    Returns:
        包含作者完整信息的字典列表
    """
    with GoogleScholarAuthorScraper(headless=headless, proxy=proxy) as scraper:
        return scraper.search_author(author_name, max_results)


# 使用示例
if __name__ == "__main__":
    # 示例：搜索作者 "Yann LeCun"
    test_name = "Yann LeCun"
    
    print("=" * 60)
    print(f"搜索作者: {test_name}")
    print("=" * 60)
    
    # 方式1：使用便捷函数获取 URL 列表
    urls = fetch_google_scholar_author_urls(test_name, max_results=5, headless=True)
    
    print("\n获取到的作者主页 URL:")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")
    
    print("\n" + "=" * 60)
    
    # 方式2：使用类获取完整信息
    with GoogleScholarAuthorScraper(headless=True) as scraper:
        authors = scraper.search_author(test_name, max_results=5)
        
        print("\n作者详细信息:")
        for author in authors:
            print(f"\n  姓名: {author['name']}")
            print(f"  主页: {author['url']}")
            print(f"  机构: {author['affiliation']}")
            print(f"  邮箱: {author['email']}")
            print(f"  引用: {author['cited_by']}")
            print(f"  兴趣: {', '.join(author['interests'])}")

