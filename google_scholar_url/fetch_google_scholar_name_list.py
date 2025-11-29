"""
Google Scholar 作者搜索模块
支持两种模式：
1. 首次运行：使用 Selenium 手动登录并保存 cookies
2. 后续运行：使用保存的 cookies 通过 requests 访问（可在服务器运行）
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Optional
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# 添加项目根目录到 path，以便导入 proxy 模块
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/google_scholar_url/', 1)[0])

# 导入代理模块
try:
    from proxy import configure_session, get_selenium_proxy_args, is_proxy_enabled
    PROXY_AVAILABLE = True
except ImportError:
    PROXY_AVAILABLE = False
    def configure_session(session): return session
    def get_selenium_proxy_args(): return []
    def is_proxy_enabled(): return False

# Selenium 相关（仅首次登录需要）
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class GoogleScholarAuthorScraper:
    """Google Scholar 作者搜索爬虫"""
    
    BASE_URL = "https://scholar.google.com/citations"
    # 使用脚本所在目录的路径，而不是当前工作目录
    COOKIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_cookies.json")
    
    def __init__(self, cookies_path: Optional[str] = None):
        """
        初始化爬虫
        
        Args:
            cookies_path: cookies 文件路径，默认为脚本所在目录下的 google_cookies.json
        """
        self.cookies_path = cookies_path or self.COOKIES_FILE
        self.cookies_valid = False  # cookies 是否有效
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        
        # 配置代理（如果启用）
        configure_session(self.session)
        
        # 尝试加载已保存的 cookies
        self.cookies_valid = self._load_cookies()
    
    def _load_cookies(self) -> bool:
        """
        加载保存的 cookies
        
        Returns:
            是否成功加载有效的 cookies
        """
        if not os.path.exists(self.cookies_path):
            return False
            
        try:
            with open(self.cookies_path, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            
            # 检查 cookies 是否过期
            expired, expiry_info = self._check_cookies_expiry(cookies)
            
            if expired:
                print(f"[WARNING] Cookies 已过期！{expiry_info}")
                print("[WARNING] 请重新运行登录流程获取新的 cookies")
                return False
            
            # 加载 cookies 到 session
            for cookie in cookies:
                self.session.cookies.set(
                    cookie['name'], 
                    cookie['value'], 
                    domain=cookie.get('domain', '.google.com')
                )
            
            print(f"[INFO] 已加载 cookies: {self.cookies_path}")
            if expiry_info:
                print(f"[INFO] {expiry_info}")
            return True
            
        except Exception as e:
            print(f"[WARNING] 加载 cookies 失败: {e}")
            return False
    
    def _check_cookies_expiry(self, cookies: list) -> tuple:
        """
        检查关键登录 cookies 是否过期
        
        Args:
            cookies: cookies 列表
            
        Returns:
            (是否过期, 过期信息字符串)
        """
        # Google 关键登录 cookies
        key_cookies = ['SID', 'HSID', 'SSID', 'APISID', 'SAPISID']
        current_time = time.time()
        
        found_key_cookies = {}
        earliest_expiry = None
        earliest_name = None
        
        for cookie in cookies:
            name = cookie.get('name', '')
            expiry = cookie.get('expiry')
            
            if name in key_cookies:
                found_key_cookies[name] = expiry
                
                if expiry:
                    if earliest_expiry is None or expiry < earliest_expiry:
                        earliest_expiry = expiry
                        earliest_name = name
        
        # 检查是否找到关键 cookies
        if not found_key_cookies:
            return True, "未找到登录 cookies，可能未登录"
        
        # 检查是否有过期的
        for name, expiry in found_key_cookies.items():
            if expiry and expiry < current_time:
                expired_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expiry))
                return True, f"Cookie '{name}' 已于 {expired_time} 过期"
        
        # 计算最近的过期时间
        if earliest_expiry:
            days_left = (earliest_expiry - current_time) / 86400
            expiry_date = time.strftime('%Y-%m-%d', time.localtime(earliest_expiry))
            
            if days_left < 7:
                return False, f"⚠️ Cookie '{earliest_name}' 将在 {days_left:.1f} 天后过期 ({expiry_date})，建议尽快更新"
            else:
                return False, f"Cookies 有效期至 {expiry_date}（还剩 {days_left:.0f} 天）"
        
        return False, ""
    
    def _save_cookies(self, cookies: list):
        """保存 cookies 到文件"""
        with open(self.cookies_path, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Cookies 已保存到: {self.cookies_path}")
    
    def login_and_save_cookies(self, wait_time: int = 60):
        """
        打开浏览器让用户手动登录，然后保存 cookies
        
        Args:
            wait_time: 等待用户登录的时间（秒）
        """
        if not SELENIUM_AVAILABLE:
            print("[ERROR] 需要安装 selenium: pip install selenium")
            return False
        
        print("=" * 60)
        print("即将打开浏览器，请在浏览器中：")
        print("1. 登录你的 Google 账号")
        print("2. 访问 Google Scholar 确认可以正常使用")
        print(f"3. 完成后等待 {wait_time} 秒自动保存，或手动关闭浏览器")
        print("=" * 60)
        
        chrome_options = Options()
        # 不使用无头模式，让用户可以手动操作
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        # 添加代理参数（如果启用）
        for arg in get_selenium_proxy_args():
            chrome_options.add_argument(arg)
            print(f"[INFO] Selenium 使用代理")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # 打开 Google Scholar
            driver.get("https://scholar.google.com/")
            
            print(f"\n[INFO] 请在浏览器中登录，你有 {wait_time} 秒时间...")
            print("[INFO] 登录完成后可以提前关闭浏览器")
            
            # 等待用户登录
            for i in range(wait_time, 0, -10):
                print(f"[INFO] 剩余等待时间: {i} 秒")
                time.sleep(10)
                # 检查浏览器是否还开着
                try:
                    _ = driver.title
                except:
                    break
            
            # 获取并保存 cookies
            cookies = driver.get_cookies()
            self._save_cookies(cookies)
            
            # 同时加载到当前 session
            for cookie in cookies:
                self.session.cookies.set(
                    cookie['name'], 
                    cookie['value'], 
                    domain=cookie.get('domain', '.google.com')
                )
            
            print("[SUCCESS] 登录成功，cookies 已保存！")
            return True
            
        except Exception as e:
            print(f"[ERROR] 登录过程出错: {e}")
            return False
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def search_author(self, author_name: str, max_results: int = 20) -> List[Dict]:
        """
        搜索作者并返回作者信息列表（支持翻页）
        
        Args:
            author_name: 作者名字
            max_results: 最大返回结果数
            
        Returns:
            包含作者信息的字典列表
        """
        import re
        
        encoded_name = quote_plus(author_name)
        base_url = f"{self.BASE_URL}?view_op=search_authors&mauthors={encoded_name}&hl=zh-CN"
        
        print(f"[INFO] 正在搜索作者: {author_name}")
        print(f"[INFO] 目标获取数量: {max_results}")
        
        all_authors = []
        current_url = base_url
        page_num = 1
        
        while len(all_authors) < max_results:
            print(f"[INFO] 正在获取第 {page_num} 页...")
            
            try:
                response = self.session.get(current_url, timeout=15)
                response.raise_for_status()
                time.sleep(2)  # 礼貌性延迟，避免被封
                
            except requests.RequestException as e:
                print(f"[ERROR] 请求失败: {e}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析当前页的作者
            page_authors = self._parse_author_results(soup)
            
            if not page_authors:
                # 如果没有结果，检查是否被拦截或需要验证
                if self._check_blocked(soup, response.text):
                    print("[ERROR] 访问被限制，请重新登录获取 cookies")
                    print("[提示] 运行: python fetch_google_scholar_name_list.py")
                    self.cookies_valid = False  # 标记 cookies 无效
                    break
                print("[INFO] 没有更多结果")
                break
            
            all_authors.extend(page_authors)
            print(f"[INFO] 已获取 {len(all_authors)} 个作者")
            
            # 检查是否已经够了
            if len(all_authors) >= max_results:
                break
            
            # 检查是否还有下一页（当前页不足10条说明到底了）
            if len(page_authors) < 10:
                print("[INFO] 已到最后一页")
                break
            
            # 从页面中提取下一页链接
            next_url = self._extract_next_page_url(soup, response.text)
            if not next_url:
                print("[INFO] 未找到下一页链接")
                break
            
            current_url = next_url
            page_num += 1
            
            # 额外延迟，避免请求过快
            time.sleep(1)
        
        return all_authors[:max_results]
    
    def _extract_next_page_url(self, soup: BeautifulSoup, html_text: str) -> Optional[str]:
        """
        从页面中提取下一页的 URL
        
        Args:
            soup: BeautifulSoup 对象
            html_text: 原始 HTML 文本
            
        Returns:
            下一页 URL，如果没有则返回 None
        """
        import re
        
        # 方法1：查找下一页按钮的链接
        next_button = soup.select_one('button.gs_btnPR')
        if next_button:
            # 检查按钮的 onclick 属性
            onclick = next_button.get('onclick', '')
            if onclick:
                # 提取 URL
                match = re.search(r"window\.location='([^']+)'", onclick)
                if match:
                    href = match.group(1).replace('\\x3d', '=').replace('\\x26', '&')
                    if href.startswith('/'):
                        return f"https://scholar.google.com{href}"
                    return href
        
        # 方法2：从 HTML 中查找包含 after_author 的链接
        # Google Scholar 的下一页按钮通常在一个特定的 pattern 中
        pattern = r'href="(/citations\?[^"]*after_author[^"]*)"'
        match = re.search(pattern, html_text)
        if match:
            href = match.group(1).replace('\\x3d', '=').replace('\\x26', '&')
            return f"https://scholar.google.com{href}"
        
        # 方法3：查找 navigate 函数调用
        pattern = r"navigate\('([^']*after_author[^']*)'\)"
        match = re.search(pattern, html_text)
        if match:
            href = match.group(1).replace('\\x3d', '=').replace('\\x26', '&')
            if href.startswith('/'):
                return f"https://scholar.google.com{href}"
            return href
        
        return None
    
    def _check_blocked(self, soup: BeautifulSoup, html_text: str) -> bool:
        """
        检查是否被 Google Scholar 拦截
        
        Args:
            soup: BeautifulSoup 对象
            html_text: 原始 HTML 文本
            
        Returns:
            是否被拦截
        """
        # 检查是否有验证码
        if 'captcha' in html_text.lower() or 'recaptcha' in html_text.lower():
            return True
        
        # 检查是否是异常流量页面
        if 'unusual traffic' in html_text.lower():
            return True
        
        # 检查是否是登录页面
        if 'accounts.google.com' in html_text:
            return True
        
        # 检查是否完全没有搜索结果区域
        if 'gsc_1usr' not in html_text and 'gs_ai_name' not in html_text:
            return True
        
        return False
    
    def _parse_author_results(self, soup: BeautifulSoup) -> List[Dict]:
        """解析当前页面的作者搜索结果"""
        authors = []
        
        # Google Scholar 作者卡片选择器
        author_cards = soup.select('div.gsc_1usr')
        
        if not author_cards:
            return []
        
        for card in author_cards:
            try:
                author_info = {}
                
                # 作者姓名和链接
                name_elem = card.select_one('h3.gs_ai_name a')
                if name_elem:
                    author_info["name"] = name_elem.text.strip()
                    href = name_elem.get('href', '')
                    if href.startswith('/'):
                        author_info["url"] = f"https://scholar.google.com{href}"
                    else:
                        author_info["url"] = href
                else:
                    continue
                
                # 所属机构
                aff_elem = card.select_one('div.gs_ai_aff')
                author_info["affiliation"] = aff_elem.text.strip() if aff_elem else ""
                
                # 邮箱域名
                email_elem = card.select_one('div.gs_ai_eml')
                author_info["email"] = email_elem.text.strip() if email_elem else ""
                
                # 被引用次数
                cited_elem = card.select_one('div.gs_ai_cby')
                author_info["cited_by"] = cited_elem.text.strip() if cited_elem else ""
                
                # 研究兴趣
                interest_elems = card.select('div.gs_ai_int a')
                author_info["interests"] = [elem.text.strip() for elem in interest_elems]
                
                authors.append(author_info)
                print(f"  - 找到: {author_info['name']} | {author_info['affiliation']}")
                
            except Exception as e:
                print(f"[WARNING] 解析失败: {e}")
                continue
        
        return authors
    
    def get_author_urls(self, author_name: str, max_results: int = 20) -> List[str]:
        """只返回作者主页 URL 列表"""
        authors = self.search_author(author_name, max_results)
        return [a["url"] for a in authors if a.get("url")]


def main():
    """主函数"""
    scraper = GoogleScholarAuthorScraper()
    
    # 检查 cookies 状态（不存在、过期、或无效都需要重新登录）
    if not scraper.cookies_valid:
        print("-" * 40)
        print("[INFO] 需要登录获取 cookies")
        print("-" * 40)
        scraper.login_and_save_cookies(wait_time=120)
    
    # 搜索作者
    test_name = "JUN LI"
    
    print("\n" + "=" * 60)
    print(f"搜索作者: {test_name}")
    print("=" * 60)
    
    authors = scraper.search_author(test_name, max_results=30)  # 可调整数量
    
    if authors:
        print("\n" + "-" * 40)
        print("搜索结果:")
        print("-" * 40)
        for i, author in enumerate(authors, 1):
            print(f"\n[{i}] {author['name']}")
            print(f"    主页: {author['url']}")
            print(f"    机构: {author['affiliation']}")
            print(f"    引用: {author['cited_by']}")
            print(f"    兴趣: {', '.join(author['interests'])}")
    else:
        print("\n[WARNING] 未获取到结果")


if __name__ == "__main__":
    main()

