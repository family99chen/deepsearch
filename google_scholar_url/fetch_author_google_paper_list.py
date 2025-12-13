"""
从 Google Scholar 个人页面获取作者的论文列表

特性：
- 使用假 cookies，无需登录
- 支持指数退避重试
- 支持代理配置
- 自动分页获取所有论文
- 支持 MongoDB 缓存
"""

import os
import sys
import time
import random
import string
import requests
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path，以便导入 proxy 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入代理模块
try:
    from proxy import configure_session
except ImportError:
    def configure_session(session): return session

# 导入重试模块
try:
    from utils.retry import DEFAULT_RETRYABLE_EXCEPTIONS, DEFAULT_RETRYABLE_STATUS_CODES
except ImportError:
    DEFAULT_RETRYABLE_EXCEPTIONS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )
    DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# 导入缓存模块
HAS_CACHE = False
try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    pass

# 缓存配置
CACHE_COLLECTION = "google_scholar_person_detail"
CACHE_TTL = 180 * 24 * 3600  # 6个月


def _get_cache() -> Optional["MongoCache"]:
    """获取缓存实例"""
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=CACHE_COLLECTION)
        return cache if cache.is_connected() else None
    except Exception:
        return None


def generate_random_string(length: int, chars: str = None) -> str:
    """生成随机字符串"""
    if chars is None:
        chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_fake_cookies() -> Dict[str, str]:
    """
    生成假的 Google cookies
    让请求看起来像是登录用户
    
    Returns:
        cookies 字典
    """
    timestamp = int(time.time())
    
    fake_cookies = {
        # 主要登录 cookies
        'SID': generate_random_string(68, string.ascii_letters + string.digits + '-_'),
        'HSID': generate_random_string(11, string.ascii_letters + string.digits),
        'SSID': generate_random_string(11, string.ascii_letters + string.digits),
        'APISID': generate_random_string(32, string.ascii_letters + string.digits + '-_'),
        'SAPISID': generate_random_string(43, string.ascii_letters + string.digits + '-_/'),
        
        # Google Scholar 特定 cookies
        'GSP': f'LM={timestamp}:S={generate_random_string(16)}',
        'NID': f'{random.randint(500, 520)}=' + generate_random_string(170, string.ascii_letters + string.digits + '-_='),
        
        # 首选项 cookies
        'SEARCH_SAMESITE': 'CgQIz5sB',
        'AEC': generate_random_string(70, string.ascii_letters + string.digits + '-_'),
        
        # 1P_JAR cookie（包含日期）
        '1P_JAR': time.strftime('%Y-%m-%d-%H', time.gmtime()),
        
        # 其他常见 cookies
        'SIDCC': generate_random_string(76, string.ascii_letters + string.digits + '-_'),
        '__Secure-1PSID': generate_random_string(68, string.ascii_letters + string.digits + '-_.'),
        '__Secure-3PSID': generate_random_string(68, string.ascii_letters + string.digits + '-_.'),
        '__Secure-1PAPISID': generate_random_string(43, string.ascii_letters + string.digits + '-_/'),
        '__Secure-3PAPISID': generate_random_string(43, string.ascii_letters + string.digits + '-_/'),
        
        # CONSENT cookie
        'CONSENT': f'PENDING+{random.randint(100, 999)}',
        
        # Scholar 偏好
        'GOOGLE_ABUSE_EXEMPTION': generate_random_string(80, string.ascii_letters + string.digits + '-_='),
    }
    
    return fake_cookies


class GoogleScholarProfileScraper:
    """Google Scholar 个人主页文章爬虫（使用假 cookies + 指数退避重试 + 缓存）"""
    
    def __init__(
        self, 
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """
        初始化爬虫
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            use_cache: 是否使用缓存
            verbose: 是否打印详细日志
        """
        # 重试配置
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_cache = use_cache
        self.verbose = verbose
        
        # 缓存实例
        self._cache = _get_cache() if use_cache else None
        
        # 创建 session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        
        # 配置代理（如果启用）
        configure_session(self.session)
        
        # 加载假 cookies
        self._load_fake_cookies()
    
    def _load_fake_cookies(self):
        """加载假 cookies"""
        fake_cookies = generate_fake_cookies()
        for name, value in fake_cookies.items():
            self.session.cookies.set(name, value, domain='.google.com')
        if self.verbose:
            print(f"[INFO] 已加载假 cookies ({len(fake_cookies)} 个)")
    
    def _parse_profile_info(self, soup: BeautifulSoup, user_id: str) -> Dict[str, Any]:
        """
        从页面解析作者个人信息
        
        Args:
            soup: BeautifulSoup 对象
            user_id: Google Scholar user ID
            
        Returns:
            包含个人信息的字典
        """
        profile = {
            "user_id": user_id,
            "url": f"https://scholar.google.com/citations?user={user_id}",
        }
        
        # 作者名字
        name_elem = soup.select_one('div#gsc_prf_in')
        if name_elem:
            profile["name"] = name_elem.text.strip()
        
        # 组织/机构
        affiliation_elem = soup.select_one('div.gsc_prf_il')
        if affiliation_elem:
            # 可能有链接，取纯文本
            profile["affiliation"] = affiliation_elem.get_text(strip=True)
        
        # 邮箱验证信息
        email_elem = soup.select_one('div#gsc_prf_ivh')
        if email_elem:
            profile["verified_email"] = email_elem.get_text(strip=True)
        
        # 研究兴趣/领域
        interests = []
        interest_elems = soup.select('a.gsc_prf_inta')
        for elem in interest_elems:
            interests.append(elem.text.strip())
        profile["interests"] = interests
        
        # 头像 URL
        photo_elem = soup.select_one('img#gsc_prf_pup-img')
        if photo_elem:
            profile["photo_url"] = photo_elem.get('src', '')
        
        # 引用统计（从表格中提取）
        # 表格结构: 引用次数 | h-index | i10-index
        #          全部    | 全部     | 全部
        #          近5年   | 近5年    | 近5年
        stats_table = soup.select('td.gsc_rsb_std')
        if len(stats_table) >= 6:
            profile["citations"] = {
                "total": self._safe_int(stats_table[0].text),
                "since_year": self._safe_int(stats_table[1].text),
            }
            profile["h_index"] = {
                "total": self._safe_int(stats_table[2].text),
                "since_year": self._safe_int(stats_table[3].text),
            }
            profile["i10_index"] = {
                "total": self._safe_int(stats_table[4].text),
                "since_year": self._safe_int(stats_table[5].text),
            }
        
        return profile
    
    def _safe_int(self, text: str) -> int:
        """安全转换为整数"""
        try:
            return int(text.strip().replace(',', ''))
        except (ValueError, AttributeError):
            return 0
    
    def _parse_coauthors(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        解析共同作者列表
        
        Args:
            soup: BeautifulSoup 对象
            
        Returns:
            共同作者列表，每个包含 name, user_id, url, affiliation
        """
        coauthors = []
        
        # 共同作者在右侧边栏
        coauthor_elems = soup.select('li.gsc_rsb_aa')
        
        for elem in coauthor_elems:
            coauthor = {}
            
            # 名字和链接
            name_link = elem.select_one('a')
            if name_link:
                coauthor["name"] = name_link.text.strip()
                href = name_link.get('href', '')
                if 'user=' in href:
                    # 提取 user_id
                    parsed = parse_qs(urlparse(href).query)
                    coauthor["user_id"] = parsed.get('user', [''])[0]
                    coauthor["url"] = f"https://scholar.google.com{href}"
            
            # 机构
            affiliation_elem = elem.select_one('span.gsc_rsb_a_ext')
            if affiliation_elem:
                coauthor["affiliation"] = affiliation_elem.text.strip()
            
            if coauthor.get("name"):
                coauthors.append(coauthor)
        
        return coauthors
    
    def _request_with_retry(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """
        带指数退避重试的请求
        
        Args:
            url: 请求 URL
            timeout: 超时时间
            
        Returns:
            Response 对象，如果失败返回 None
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=timeout)
                
                # 检查是否需要重试的状态码
                if response.status_code in DEFAULT_RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        print(f"[RETRY] HTTP {response.status_code}，{delay:.1f}秒后重试 "
                              f"({attempt + 1}/{self.max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[ERROR] HTTP {response.status_code}，重试次数已用尽")
                        return None
                
                response.raise_for_status()
                return response
                
            except DEFAULT_RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                
                if attempt >= self.max_retries:
                    print(f"[ERROR] 请求失败，重试次数已用尽: {type(e).__name__}: {e}")
                    return None
                
                delay = self._calculate_delay(attempt)
                print(f"[RETRY] {type(e).__name__}，{delay:.1f}秒后重试 "
                      f"({attempt + 1}/{self.max_retries})...")
                time.sleep(delay)
                
            except requests.RequestException as e:
                print(f"[ERROR] 请求失败: {e}")
                return None
        
        return None
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟时间
        
        Args:
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            延迟时间（秒）
        """
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay
        )
        # 添加随机抖动 (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter
    
    def _extract_user_id(self, profile_url: str) -> Optional[str]:
        """从 URL 中提取 user_id"""
        parsed = urlparse(profile_url)
        params = parse_qs(parsed.query)
        return params.get('user', [None])[0]
    
    def get_profile_with_papers(self, profile_url: str, use_cache: bool = None) -> Dict[str, Any]:
        """
        获取作者完整信息（个人信息 + 共同作者 + 论文列表），支持缓存
        
        Args:
            profile_url: 作者 Google Scholar 主页 URL
            use_cache: 是否使用缓存（None 表示使用实例默认设置）
            
        Returns:
            完整的作者信息字典
        """
        user_id = self._extract_user_id(profile_url)
        if not user_id:
            print(f"[ERROR] 无法从 URL 中提取 user ID: {profile_url}")
            return {}
        
        # 确定是否使用缓存
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        
        # 尝试从缓存读取
        if should_use_cache and self._cache:
            cached_data = self._cache.get(user_id)
            if cached_data:
                if self.verbose:
                    print(f"[CACHE] 命中缓存: {user_id}")
                return cached_data
        
        if self.verbose:
            print(f"[INFO] 正在获取作者完整信息, user_id: {user_id}")
        
        # 获取第一页（包含个人信息）
        first_page_url = f"https://scholar.google.com/citations?user={user_id}&hl=zh-CN&cstart=0&pagesize=100"
        response = self._request_with_retry(first_page_url)
        
        if response is None:
            print(f"[ERROR] 无法获取作者页面: {user_id}")
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 检查是否被封锁
        if self._check_blocked(response.text):
            print("[ERROR] 访问被限制")
            return {}
        
        # 解析个人信息
        profile = self._parse_profile_info(soup, user_id)
        
        # 解析共同作者
        profile["coauthors"] = self._parse_coauthors(soup)
        
        # 解析论文（从第一页开始）
        all_papers = []
        paper_rows = soup.select('tr.gsc_a_tr')
        for row in paper_rows:
            paper = self._parse_paper_row(row)
            if paper:
                all_papers.append(paper)
        
        # 获取后续页面的论文
        cstart = 100
        pagesize = 100
        
        while len(paper_rows) >= pagesize:
            time.sleep(2)  # 礼貌性延迟
            
            url = f"https://scholar.google.com/citations?user={user_id}&hl=zh-CN&cstart={cstart}&pagesize={pagesize}"
            if self.verbose:
                print(f"[INFO] 正在获取第 {cstart // pagesize + 1} 页...")
            
            response = self._request_with_retry(url)
            if response is None:
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            paper_rows = soup.select('tr.gsc_a_tr')
            
            if not paper_rows:
                break
            
            for row in paper_rows:
                paper = self._parse_paper_row(row)
                if paper:
                    all_papers.append(paper)
            
            if self.verbose:
                print(f"[INFO] 已获取 {len(all_papers)} 篇论文")
            
            cstart += pagesize
        
        profile["papers"] = all_papers
        profile["paper_count"] = len(all_papers)
        profile["fetched_at"] = datetime.now().isoformat()
        
        # 写入缓存
        if should_use_cache and self._cache:
            self._cache.set(user_id, profile, ttl=CACHE_TTL)
            if self.verbose:
                print(f"[CACHE] 已写入缓存: {user_id}")
        
        return profile
    
    def get_all_papers(self, profile_url: str) -> List[dict]:
        """
        获取作者主页的所有论文（带重试）
        
        Args:
            profile_url: 作者 Google Scholar 主页 URL
            
        Returns:
            论文列表，每个论文包含 title, year, citations, authors, venue 等
        """
        # 解析 URL 获取 user ID
        user_id = self._extract_user_id(profile_url)
        
        if not user_id:
            print(f"[ERROR] 无法从 URL 中提取 user ID: {profile_url}")
            return []
        
        # 尝试从缓存读取
        if self.use_cache and self._cache:
            cached_data = self._cache.get(user_id)
            if cached_data and "papers" in cached_data:
                if self.verbose:
                    print(f"[CACHE] 命中缓存: {user_id} ({len(cached_data['papers'])} 篇论文)")
                return cached_data["papers"]
        
        if self.verbose:
            print(f"[INFO] 正在获取作者文章列表, user_id: {user_id}")
        
        all_papers = []
        cstart = 0  # 分页起始位置
        pagesize = 100  # 每页数量
        consecutive_failures = 0  # 连续失败次数
        max_consecutive_failures = 2  # 最大连续失败次数
        first_page_soup = None  # 保存第一页用于提取个人信息
        
        while True:
            # 构建请求 URL
            url = (f"https://scholar.google.com/citations?"
                   f"user={user_id}&hl=zh-CN&cstart={cstart}&pagesize={pagesize}")
            
            if self.verbose:
                print(f"[INFO] 正在获取第 {cstart // pagesize + 1} 页 (cstart={cstart})...")
            
            # 使用带重试的请求
            response = self._request_with_retry(url)
            
            if response is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"[ERROR] 连续 {consecutive_failures} 次请求失败，停止获取")
                    break
                print(f"[WARNING] 请求失败，跳过当前页 ({consecutive_failures}/{max_consecutive_failures})")
                cstart += pagesize
                continue
            
            # 重置连续失败计数
            consecutive_failures = 0
            
            # 礼貌性延迟
            time.sleep(2)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 检查是否被封锁
            if self._check_blocked(response.text):
                print("[ERROR] 访问被限制，尝试刷新 cookies...")
                # 重新生成假 cookies 并重试一次
                self._load_fake_cookies()
                response = self._request_with_retry(url)
                if response is None or self._check_blocked(response.text):
                    print("[ERROR] 刷新 cookies 后仍被限制，停止获取")
                    break
                soup = BeautifulSoup(response.text, 'html.parser')
            
            # 保存第一页的 soup，用于后续提取个人信息
            if cstart == 0:
                first_page_soup = soup
            
            # 解析论文列表
            paper_rows = soup.select('tr.gsc_a_tr')
            
            if not paper_rows:
                if self.verbose:
                    print("[INFO] 没有更多论文")
                break
            
            for row in paper_rows:
                paper = self._parse_paper_row(row)
                if paper:
                    all_papers.append(paper)
            
            if self.verbose:
                print(f"[INFO] 已获取 {len(all_papers)} 篇论文")
            
            # 检查是否还有下一页
            if len(paper_rows) < pagesize:
                if self.verbose:
                    print("[INFO] 已获取全部论文")
                break
            
            # 检查是否有"显示更多"按钮被禁用
            show_more = soup.select_one('button#gsc_bpf_more')
            if show_more and show_more.get('disabled'):
                if self.verbose:
                    print("[INFO] 已到最后一页")
                break
            
            cstart += pagesize
            time.sleep(1)  # 额外延迟
        
        # 如果获取到了论文，缓存完整信息（使用已保存的第一页，无需重新请求）
        if all_papers and self.use_cache and self._cache and first_page_soup:
            profile = self._parse_profile_info(first_page_soup, user_id)
            profile["coauthors"] = self._parse_coauthors(first_page_soup)
            profile["papers"] = all_papers
            profile["paper_count"] = len(all_papers)
            profile["fetched_at"] = datetime.now().isoformat()
            self._cache.set(user_id, profile, ttl=CACHE_TTL)
            if self.verbose:
                print(f"[CACHE] 已写入缓存: {user_id}")
        
        return all_papers
    
    def _check_blocked(self, html_text: str) -> bool:
        """
        检查是否被 Google Scholar 封锁
        
        Args:
            html_text: HTML 文本
            
        Returns:
            是否被封锁
        """
        # 检查是否有验证码
        if 'captcha' in html_text.lower() or 'recaptcha' in html_text.lower():
            return True
        
        # 检查是否是异常流量页面
        if 'unusual traffic' in html_text.lower():
            return True
        
        # 检查是否是登录页面（但页面内容不含论文）
        if 'accounts.google.com' in html_text and 'gsc_a_at' not in html_text:
            return True
        
        return False
    
    def _parse_paper_row(self, row) -> Optional[dict]:
        """
        解析单行论文信息
        
        Args:
            row: BeautifulSoup 的 tr 元素
            
        Returns:
            论文字典或 None
        """
        paper = {}
        
        # 标题
        title_elem = row.select_one('a.gsc_a_at')
        if title_elem:
            paper['title'] = title_elem.text.strip()
            # 获取论文详情链接
            href = title_elem.get('href', '')
            if href:
                paper['detail_url'] = f"https://scholar.google.com{href}"
        else:
            return None
        
        # 作者和期刊信息
        gray_text = row.select('div.gs_gray')
        if len(gray_text) >= 1:
            paper['authors'] = gray_text[0].text.strip()
        if len(gray_text) >= 2:
            paper['venue'] = gray_text[1].text.strip()
        
        # 引用次数
        cite_elem = row.select_one('a.gsc_a_ac')
        if cite_elem and cite_elem.text.strip():
            paper['citations'] = cite_elem.text.strip()
        else:
            paper['citations'] = '0'
        
        # 年份
        year_elem = row.select_one('span.gsc_a_h.gsc_a_hc.gs_ibl')
        if year_elem:
            paper['year'] = year_elem.text.strip()
        
        return paper


def get_google_scholar_papers(
    profile_url: str, 
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    use_cache: bool = True,
) -> List[dict]:
    """
    便捷函数：获取 Google Scholar 作者的所有论文（带重试和缓存）
    
    Args:
        profile_url: 作者 Google Scholar 主页 URL
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        use_cache: 是否使用缓存
        
    Returns:
        论文列表
    """
    scraper = GoogleScholarProfileScraper(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        use_cache=use_cache,
    )
    return scraper.get_all_papers(profile_url)


def get_google_scholar_profile(
    profile_url: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    便捷函数：获取 Google Scholar 作者的完整信息（个人信息 + 共同作者 + 论文）
    
    Args:
        profile_url: 作者 Google Scholar 主页 URL
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        use_cache: 是否使用缓存
        
    Returns:
        完整的作者信息字典，包含：
        - user_id: Google Scholar ID
        - name: 作者名字
        - affiliation: 机构
        - interests: 研究兴趣列表
        - citations: {"total": int, "since_year": int}
        - h_index: {"total": int, "since_year": int}
        - i10_index: {"total": int, "since_year": int}
        - coauthors: 共同作者列表
        - papers: 论文列表
        - paper_count: 论文数量
        - fetched_at: 抓取时间
    """
    scraper = GoogleScholarProfileScraper(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        use_cache=use_cache,
    )
    return scraper.get_profile_with_papers(profile_url)


if __name__ == "__main__":
    # 示例用法
    test_url = "https://scholar.google.com/citations?hl=zh-CN&user=GUaAHccAAAAJ"
    
    print("=" * 60)
    print("Google Scholar 论文列表获取工具（假 Cookies 模式）")
    print("=" * 60)
    print(f"目标 URL: {test_url}")
    print()
    
    papers = get_google_scholar_papers(test_url)
    
    print()
    print("=" * 60)
    print(f"共获取 {len(papers)} 篇论文")
    print("=" * 60)
    
    # 显示前 10 篇
    for i, paper in enumerate(papers[:10], 1):
        print(f"\n[{i}] {paper.get('title', 'N/A')}")
        print(f"    作者: {paper.get('authors', 'N/A')}")
        print(f"    期刊: {paper.get('venue', 'N/A')}")
        print(f"    年份: {paper.get('year', 'N/A')}")
        print(f"    引用: {paper.get('citations', '0')}")
    
    if len(papers) > 10:
        print(f"\n... 还有 {len(papers) - 10} 篇论文未显示")
